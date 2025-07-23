#!/usr/bin/env python3
"""DonkeyCar を操作するためのスクリプト。

Usage:
    manage.py drive [--model=<model>] [--type=(linear|categorical|...)]
    manage.py calibrate

Options:
    -h --help          この画面を表示する。
"""

from docopt import docopt
import logging
import os

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter, TubWiper
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, RCReceiver
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.pipeline.augmentations import ImageAugmentation

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class DriveMode:
    """AI とユーザー運転を切り替えるヘルパークラス。"""

    def __init__(self, cfg):
        """クラスを初期化する。

        Args:
            cfg: 設定オブジェクト。
        """
        self.cfg = cfg

    def run(self, mode, user_angle, user_throttle, pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle
        elif mode == 'local_angle':
            return pilot_angle if pilot_angle else 0.0, user_throttle
        else:
            return pilot_angle if pilot_angle else 0.0, \
                   pilot_throttle * self.cfg.AI_THROTTLE_MULT if \
                       pilot_throttle else 0.0


class PilotCondition:
    """誰が運転するかを判断するヘルパークラス。"""

    def run(self, mode):
        """パイロットを実行するかどうかを返す。

        Args:
            mode: ユーザーモードかどうかを示す文字列。

        Returns:
            bool: ユーザーでなければ ``True``。
        """
        return mode != 'user'


def drive(cfg, model_path=None, model_type=None):
    """各種パーツを組み合わせて最小構成のロボットカーを構築する。

    ここでは 1 台のカメラと Web またはジョイスティックコントローラ、
    自動運転(オートパイロット)および tubwriter を使用する。

    各パーツは Vehicle ループ内でジョブとして実行され、`threaded` フラグ
    に応じて ``run`` または ``run_threaded`` が呼び出される。すべてのパーツ
    は ``cfg.DRIVE_LOOP_HZ`` で指定されたフレームレートで順番に更新され、
    各パーツが速やかに処理を終えることを前提としている。パーツは名前付き
    の出力と入力を持つことができ、フレームワークは同じ名前の入力を要求
    するパーツへ自動的に出力を渡す。

    Args:
        cfg: 設定オブジェクト。
        model_path: 事前学習済みモデルのパス。
        model_type: モデルのタイプ。

    Returns:
        None
    """
    logger.info(f'PID番号: {os.getpid()}')
    car = dk.vehicle.Vehicle()
    # カメラを追加
    inputs = []
    if cfg.DONKEY_GYM:
        from donkeycar.parts.dgym import DonkeyGymEnv 
        cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST,
                           env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF,
                           delay=cfg.SIM_ARTIFICIAL_LATENCY)
        inputs = ['angle', 'throttle', 'brake']
    elif cfg.CAMERA_TYPE == "PICAM":
        from donkeycar.parts.camera import PiCamera
        cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                       image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE,
                       vflip=cfg.CAMERA_VFLIP, hflip=cfg.CAMERA_HFLIP)
    elif cfg.CAMERA_TYPE == "WEBCAM":
        from donkeycar.parts.camera import Webcam
        cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                     image_d=cfg.IMAGE_DEPTH)
    elif cfg.CAMERA_TYPE == "CVCAM":
        from donkeycar.parts.cv import CvCam
        cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                    image_d=cfg.IMAGE_DEPTH)
    elif cfg.CAMERA_TYPE == "CSIC":
        from donkeycar.parts.camera import CSICamera
        cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                        image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE,
                        gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
    elif cfg.CAMERA_TYPE == "V4L":
        from donkeycar.parts.camera import V4LCamera
        cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                        image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
    elif cfg.CAMERA_TYPE == "MOCK":
        from donkeycar.parts.camera import MockCamera
        cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H,
                         image_d=cfg.IMAGE_DEPTH)
    elif cfg.CAMERA_TYPE == "IMAGE_LIST":
        from donkeycar.parts.camera import ImageListCamera
        cam = ImageListCamera(path_mask=cfg.PATH_MASK)
    else:
        raise Exception("未知のカメラタイプ: %s" % cfg.CAMERA_TYPE)

    car.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=True)

    # コントローラを追加
    if cfg.USE_RC:
        rc_steering = RCReceiver(cfg.STEERING_RC_GPIO, invert=True)
        rc_throttle = RCReceiver(cfg.THROTTLE_RC_GPIO)
        rc_wiper = RCReceiver(cfg.DATA_WIPER_RC_GPIO, jitter=0.05, no_action=0)
        car.add(rc_steering, outputs=['user/angle', 'user/angle_on'])
        car.add(rc_throttle, outputs=['user/throttle', 'user/throttle_on'])
        car.add(rc_wiper, outputs=['user/wiper', 'user/wiper_on'])
        ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT,
                                 mode=cfg.WEB_INIT_MODE)
        # Web コントローラはユーザーモードのみ設定し、角度とスロットルは使用しない。
        car.add(ctr, inputs=['cam/image_array'],
                outputs=['webcontroller/angle', 'webcontroller/throttle',
                         'user/mode', 'recording'],
                threaded=True)

    else:
        if cfg.USE_JOYSTICK_AS_DEFAULT:
            from donkeycar.parts.controller import get_js_controller
            ctr = get_js_controller(cfg)
            if cfg.USE_NETWORKED_JS:
                from donkeycar.parts.controller import JoyStickSub
                netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                car.add(netwkJs, threaded=True)
                ctr.js = netwkJs
        else:
            ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT,
                                     mode=cfg.WEB_INIT_MODE)
        car.add(ctr,
                inputs=['cam/image_array'],
                outputs=['user/angle', 'user/throttle', 'user/mode',
                         'recording'],
                threaded=True)

    # ユーザー運転か AI かを判断するパート
    car.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    # オートパイロットを追加
    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
    if model_path:
        kl = dk.utils.get_model_by_type(model_type, cfg)
        kl.load(model_path=model_path)
        inputs = ['cam/image_array']
        # クロップや台形マスクなどの画像変換を追加
        if hasattr(cfg, 'TRANSFORMATIONS') and cfg.TRANSFORMATIONS:
            outputs = ['cam/image_array_trans']
            car.add(ImageAugmentation(cfg, 'TRANSFORMATIONS'),
                    inputs=inputs, outputs=outputs)
            inputs = outputs

        outputs = ['pilot/angle', 'pilot/throttle']
        car.add(kl, inputs=inputs, outputs=outputs, run_condition='run_pilot')

    # 車両に反映する入力を選択
    car.add(DriveMode(cfg=cfg),
            inputs=['user/mode', 'user/angle', 'user/throttle',
                    'pilot/angle', 'pilot/throttle'],
            outputs=['angle', 'throttle'])

    # 駆動系のセットアップ
    if cfg.DONKEY_GYM or cfg.DRIVE_TRAIN_TYPE == "MOCK":
        pass
    else:
        steering_controller = PCA9685(cfg.STEERING_CHANNEL,
                                      cfg.PCA9685_I2C_ADDR,
                                      busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=cfg.STEERING_LEFT_PWM,
                               right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL,
                                      cfg.PCA9685_I2C_ADDR,
                                      busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                               max_pulse=cfg.THROTTLE_FORWARD_PWM,
                               zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                               min_pulse=cfg.THROTTLE_REVERSE_PWM)

        car.add(steering, inputs=['angle'])
        car.add(throttle, inputs=['throttle'])

    # データ保存用の Tub を追加
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode']
    types = ['image_array', 'float', 'float', 'str']

    # 新しいディレクトリに記録するか既存に追記するか
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    tub_writer = TubWriter(base_path=tub_path, inputs=inputs, types=types)
    car.add(tub_writer, inputs=inputs, outputs=["tub/num_records"],
            run_condition='recording')
    if not model_path and cfg.USE_RC:
        tub_wiper = TubWiper(tub_writer.tub, num_records=cfg.DRIVE_LOOP_HZ)
        car.add(tub_wiper, inputs=['user/wiper_on'])
    # 車を起動する
    car.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


def calibrate(cfg):
    """RC コントローラーのみで構成された補助車両を起動し、値を表示する。

    RC リモコンには一般的にスロットルやステアリング(steering)用の調整つま
    みが付いており、このループではコントローラーを実行して値を表示すること
    で PWM 信号をセンタリングできるようにする。リモコンに 3 つ目のチャン
    ネルがある場合は、記録中に不要なデータを除外するために使用できるので、
    その値もここで表示する。

    Args:
        cfg: 設定オブジェクト。

    Returns:
        None
    """
    donkey_car = dk.vehicle.Vehicle()

    # RC レシーバーを作成
    rc_steering = RCReceiver(cfg.STEERING_RC_GPIO, invert=True)
    rc_throttle = RCReceiver(cfg.THROTTLE_RC_GPIO)
    rc_wiper = RCReceiver(cfg.DATA_WIPER_RC_GPIO, jitter=0.05, no_action=0)
    donkey_car.add(rc_steering, outputs=['user/angle', 'user/steering_on'])
    donkey_car.add(rc_throttle, outputs=['user/throttle', 'user/throttle_on'])
    donkey_car.add(rc_wiper, outputs=['user/wiper', 'user/wiper_on'])

    # シェルに出力するプロッターパートを作成
    class Plotter:
        def run(self, angle, steering_on, throttle, throttle_on, wiper, wiper_on):
            print('角度=%+5.4f, steering_on=%1d, スロットル=%+5.4f, '
                  'throttle_on=%1d ワイパー=%+5.4f, wiper_on=%1d' %
                  (angle, steering_on, throttle, throttle_on, wiper, wiper_on))

    # プロッターパートを追加
    donkey_car.add(Plotter(), inputs=['user/angle', 'user/steering_on',
                                      'user/throttle', 'user/throttle_on',
                                      'user/wiper', 'user/wiper_on'])
    # ネットワーク負荷を抑えるため 5Hz で実行
    donkey_car.start(rate_hz=10, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    if args['drive']:
        drive(cfg, model_path=args['--model'], model_type=args['--type'])
    elif args['calibrate']:
        calibrate(cfg)
