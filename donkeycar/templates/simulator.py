#!/usr/bin/env python3
"""ドンキー4カーを運転するためのスクリプト。

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help               この画面を表示します。
    --js                    物理ジョイスティックを使用します。
    -f --file=<file>        tub ファイルのパスを1行に1つ記述したテキストファイル。複数回指定可能。
    --meta=<key:value>      この走行に関するメタデータを表すキー／値の文字列。複数回指定可能。
    --myconfig=filename     使用する myconfig ファイルを指定します。
                            [既定値: myconfig.py]
"""
import os
import time

from docopt import docopt
import numpy as np

import donkeycar as dk

from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, JoystickController, WebFpv
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.utils import *


def drive(cfg, model_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[]):
    """多くのパーツから動作するロボット車両を構築する。

    各パーツは Vehicle ループ内でジョブとして実行され、コンストラクタの
    フラグ ``threaded`` に応じて ``run`` もしくは ``run_threaded`` メソッド
    が呼び出される。すべてのパーツは ``cfg.DRIVE_LOOP_HZ`` で定められた
    フレームレートで順番に更新され、各パーツが適切な時間内に処理を終える
    ことを前提としている。パーツには名前付きの出力と入力を持たせることが
    でき、フレームワークは同名の入力を要求するパーツへその値を自動的に渡す。

    Args:
        cfg: 設定オブジェクト。
        model_path: 読み込むモデルのパス。
        use_joystick: ジョイスティックを使用するかどうか。
        model_type: 使用するモデルタイプ。
        camera_type: カメラタイプ ``single`` または ``stereo``。
        meta: 追加メタデータのリスト。

    Returns:
        なし
    """

    if cfg.DONKEY_GYM:
        # シミュレータが CUDA を使用するため、同時に CUDA を利用すると
        # たいていリソース不足になる。そのため donkey_gym では CUDA を
        # 無効化する。
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    # 車を初期化
    V = dk.vehicle.Vehicle()

    from donkeycar.parts.dgym import DonkeyGymEnv

    inputs = []

    cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF, delay=cfg.SIM_ARTIFICIAL_LATENCY)
    threaded = True
    inputs = ['angle', 'throttle', 'brake']

    V.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=threaded)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        # よりパワーを得るには max_throttle を 1.0 に近づける
        # 操舵(steering) の反応を抑えたい場合は steering_scale を 1.0 より小さくする
        if cfg.CONTROLLER_TYPE == "MM1":
            from donkeycar.parts.robohat import RoboHATController            
            ctr = RoboHATController(cfg)
        elif "custom" == cfg.CONTROLLER_TYPE:
            #
            # `donkey createjs` コマンドで作成されたカスタムコントローラ
            #
            from my_joystick import MyJoystickController
            ctr = MyJoystickController(
                throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
            ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
        else:
            from donkeycar.parts.controller import get_js_controller

            ctr = get_js_controller(cfg)

            if cfg.USE_NETWORKED_JS:
                from donkeycar.parts.controller import JoyStickSub
                netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                V.add(netwkJs, threaded=True)
                ctr.js = netwkJs
        
        V.add(ctr, 
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    else:
        # この Web コントローラはステアリング、スロットル、モードなどを
        # 操作できる Web サーバーを生成する
        ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
        
        V.add(ctr,
          inputs=['cam/image_array', 'tub/num_records'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # このスロットルフィルタは ESC のリバースで一度の入力で後退できるようにする
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    # パイロットモジュールを実行すべきか判断する。
    # run_condition が真偽値しか受け取れないため必要となる
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    class LedConditionLogic:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
            # 点滅速度を返す。0 で消灯、-1 で常時点灯、正の値で点滅間隔を表す

            if track_loc is not None:
                led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                return -1

            if model_file_changed:
                led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                return 0.1
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if recording_alert:
                led.set_rgb(*recording_alert)
                return self.cfg.REC_COUNT_ALERT_BLINK_RATE
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if behavior_state is not None and model_type == 'behavior':
                r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                led.set_rgb(r, g, b)
                return -1 # 常時点灯

            if recording:
                return -1 # 常時点灯
            elif mode == 'user':
                return 1
            elif mode == 'local_angle':
                return 0.5
            elif mode == 'local':
                return 0.1
            return 0

    if cfg.HAVE_RGB_LED and not cfg.DONKEY_GYM:
        from donkeycar.parts.led_status import RGB_LED
        led = RGB_LED(cfg.LED_PIN_R, cfg.LED_PIN_G, cfg.LED_PIN_B, cfg.LED_INVERT)
        led.set_rgb(cfg.LED_R, cfg.LED_G, cfg.LED_B)

        V.add(LedConditionLogic(cfg), inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
              outputs=['led/blink_rate'])

        V.add(led, inputs=['led/blink_rate'])

    def get_record_alert_color(num_records):
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print("記録数", num_records, "件")

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE and isinstance(ctr, JoystickController):
        # サークルボタンを使用していない場合、記録数表示を強制するために流用する
        def show_record_acount_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        ctr.set_button_down_trigger('circle', show_record_acount_status)

    # FPV プレビューを使用する。クロップ後の画像またはフルフレームを表示できる
    if cfg.USE_FPV:
        V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)

    # 行動状態 (Behavioral state)
    if cfg.TRAIN_BEHAVIORS:
        bh = BehaviorPart(cfg.BEHAVIOR_LIST)
        V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
        try:
            ctr.set_button_down_trigger('L1', bh.increment_state)
        except:
            pass

        inputs = ['cam/image_array', "behavior/one_hot_state_array"]
    else:
        inputs=['cam/image_array']

    def load_model(kl, model_path):
        start = time.time()
        print('モデルを読み込み中', model_path)
        kl.load(model_path)
        print('読み込み完了 %s 秒' % (str(time.time() - start)) )

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('モデルの重みを読み込み中', weights_path)
            kl.model.load_weights(weights_path)
            print('読み込み完了 %s 秒' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print('ERR>> 重みの読み込みに問題があります', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('モデル JSON を読み込み中', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('JSON の読み込み完了 %s 秒' % (str(time.time() - start)) )
        except Exception as e:
            print(e)
            print("ERR>> モデル JSON の読み込みに問題があります", json_fnm)

    if model_path:
        # モデルが指定されている場合、まず適切な Keras パートを作成する
        kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path or '.uff' in model_path or 'tflite' in model_path or '.pkl' in model_path:
            # 拡張子が .h5 の場合
            # モデルファイルからすべて読み込む
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif '.json' in model_path:
            # 拡張子が .json の場合
            # そこからモデルを読み込み、対応する
            # 重みのみを持つ .wts ファイルを探す
            load_model_json(kl, model_path)
            weights_path = model_path.replace('.json', '.weights')
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace('.json', '.weights')
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> モデルファイルの拡張子が不明です!!")
            return

        # このパートは接続されていれば LED に変化を通知する
        V.add(FileWatcher(model_path, verbose=True), outputs=['modelfile/modified'])

        # 以下のパーツは AI が動作している間のみモデルファイルをリロードし、ユーザー走行を妨げないようにする
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        V.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs=['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        V.add(kl, inputs=inputs,
              outputs=outputs,
              run_condition='run_pilot')
    
    if cfg.STOP_SIGN_DETECTOR:
        from donkeycar.parts.object_detector.stop_sign_detector import StopSignDetector
        V.add(StopSignDetector(cfg.STOP_SIGN_MIN_SCORE, cfg.STOP_SIGN_SHOW_BOUNDING_BOX), inputs=['cam/image_array', 'pilot/throttle'], outputs=['pilot/throttle', 'cam/image_array'])

    # どの入力を車の制御に使うか選択する
    class DriveMode:
        def run(self, mode,
                    user_angle, user_throttle, user_brake,
                    pilot_angle, pilot_throttle, pilot_brake):
            if mode == 'user':
                return user_angle, user_throttle, user_brake

            elif mode == 'local_angle':
                return pilot_angle if pilot_angle else 0.0, user_throttle, user_brake

            else:
                return pilot_angle if pilot_angle else 0.0, pilot_throttle * cfg.AI_THROTTLE_MULT if pilot_throttle else 0.0, pilot_brake if pilot_brake else 0.0

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle', 'user/brake',
                  'pilot/angle', 'pilot/throttle', 'pilot/brake'],
          outputs=['angle', 'throttle', 'brake'])


    # レースで AI モード開始時に車を加速させるための処理
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)

    V.add(aiLauncher,
        inputs=['user/mode', 'throttle'],
        outputs=['throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)


    class AiRunCondition:
        """AI が稼働中かを知らせる真偽値パート。"""
        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    # AI モード時の録画設定
    class AiRecordingCondition:
        """AI モードでは常に True を返し、そうでなければユーザーの録画設定を尊重する。"""
        def run(self, mode, recording):
            if mode == 'user':
                return recording
            return True

    if cfg.RECORD_DURING_AI:
        V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

    # データを保存する tub を追加

    inputs=['cam/image_array',
            'user/angle', 'user/throttle',
            'user/mode']

    types=['image_array',
           'float', 'float',
           'str']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    # 新しい記録を別ディレクトリに保存するか既存に追加するかを決める
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])

    if type(ctr) is LocalWebController:
        if cfg.DONKEY_GYM:
            print("http://localhost:%d にアクセスして車を操作できます" % cfg.WEB_CONTROL_PORT)
        else:
            print("<your hostname.local>:%d にアクセスして車を操作できます" % cfg.WEB_CONTROL_PORT)
    elif isinstance(ctr, JoystickController):
        print("ジョイスティックを動かして車を操作できます")
        ctr.set_tub(tub_writer.tub)
        ctr.print_controls()

    # 車両を実行する
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
              model_type=model_type, camera_type=camera_type,
              meta=args['--meta'])
    elif args['train']:
        print('python train.py を使用してください。\n')

