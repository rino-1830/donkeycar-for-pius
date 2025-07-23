#!/usr/bin/env python3
"""
ドンキー2カーを走らせるためのスクリプト。

Usage:
    manage.py (drive)


Options:
    -h --help          この画面を表示。
"""
import os
import time

from docopt import docopt

import donkeycar as dk

# パーツのインポート
from donkeycar.parts.controller import LocalWebController, \
    JoystickController, WebFpv
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts import pins
from donkeycar.utils import *

from socket import gethostname

def drive(cfg ):
    """複数の部品を組み合わせて動作するロボット車両を構築する。

    各部品は Vehicle ループのジョブとして実行され、`threaded` フラグに応じて
    `run` または `run_threaded` メソッドが呼び出される。すべての部品は
    `cfg.DRIVE_LOOP_HZ` で指定されたフレームレートで順に更新され、各部品が
    適切な時間内に処理を終えることを前提としている。部品には入出力の名前を
    付けることができ、フレームワークは同じ名前を要求する部品へその値を渡す。

    Args:
        cfg: 走行設定を保持する ``dk.config`` オブジェクト。

    Returns:
        None
    """

    # 車両を初期化
    V = dk.vehicle.Vehicle()

    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT)
    V.add(ctr,
          inputs=['cam/image_array', 'tub/num_records'],
          outputs=['angle', 'throttle', 'user/mode', 'recording'],
          threaded=True)

    # このスロットルフィルターによりESCのリバースを一度の操作で行える
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['throttle'], outputs=['throttle'])

    drive_train = None

    # ドライブトレインの設定
    if cfg.DONKEY_GYM or cfg.DRIVE_TRAIN_TYPE == "MOCK":
        pass

    elif cfg.DRIVE_TRAIN_TYPE == "PWM_STEERING_THROTTLE":
        #
        # サーボとESCを備えたRCカー用のドライブトレイン。
        # ステアリング(サーボ)に ``PwmPin`` を使用し、
        # スロットル(ESC)には2つ目の ``PwmPin`` を使用する。
        #
        from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PulseController
        dt = cfg.PWM_STEERING_THROTTLE
        steering_controller = PulseController(
            pwm_pin=pins.pwm_pin_by_id(dt["PWM_STEERING_PIN"]),
            pwm_scale=dt["PWM_STEERING_SCALE"],
            pwm_inverted=dt["PWM_STEERING_INVERTED"])
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=dt["STEERING_LEFT_PWM"],
                               right_pulse=dt["STEERING_RIGHT_PWM"])

        throttle_controller = PulseController(
            pwm_pin=pins.pwm_pin_by_id(dt["PWM_THROTTLE_PIN"]),
            pwm_scale=dt["PWM_THROTTLE_SCALE"],
            pwm_inverted=dt['PWM_THROTTLE_INVERTED'])
        throttle = PWMThrottle(controller=throttle_controller,
                               max_pulse=dt['THROTTLE_FORWARD_PWM'],
                               zero_pulse=dt['THROTTLE_STOPPED_PWM'],
                               min_pulse=dt['THROTTLE_REVERSE_PWM'])

        drive_train = dict()
        drive_train['steering'] = steering
        drive_train['throttle'] = throttle

        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)

    elif cfg.DRIVE_TRAIN_TYPE == "I2C_SERVO":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
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

        drive_train = dict()
        drive_train['steering'] = steering
        drive_train['throttle'] = throttle
        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)

    elif cfg.DRIVE_TRAIN_TYPE == "MM1":
        from donkeycar.parts.robohat import RoboHATDriver
        drive_train = RoboHATDriver(cfg)
        V.add(drive_train, inputs=['angle', 'throttle'])

    # TODO: モンキーパッチは好ましくない!!!
    ctr.drive_train = drive_train
    ctr.drive_train_type = cfg.DRIVE_TRAIN_TYPE

    print(
        f"http://{gethostname()}.local:{ctr.port}/calibrate にアクセスして"
        f"キャリブレーションを行ってください"
    )

    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        drive(cfg)
