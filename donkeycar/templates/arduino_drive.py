#!/usr/bin/env python3
"""
ドンキー2カーを走行させるためのスクリプト。

Arduino を駆動系として利用する車両でドライブループを実装する方法と、
動作デモとしてジョイスティックで車を制御する方法を示す。

Usage:
    manage.py (drive)

Options:
    -h --help          この画面を表示する。
"""
import os
import time

from docopt import docopt

import donkeycar as dk
from donkeycar.parts.actuator import ArduinoFirmata, ArdPWMSteering, ArdPWMThrottle
from donkeycar.parts.controller import get_js_controller


def drive(cfg):
    """複数のパーツからなるロボット車両を構築して動作させる。

    各パーツは ``threaded`` フラグに応じて ``run`` または ``run_threaded``
    メソッドが呼び出され、 ``cfg.DRIVE_LOOP_HZ`` で指定されたフレームレート
    で順次更新される。パーツは名前付き入出力を持つことができ、同じ名前を
    求めるパーツへその値が渡される。

    Args:
        cfg: 車両の設定オブジェクト。

    Returns:
        None
    """

    # 車両を初期化する
    V = dk.vehicle.Vehicle()
    ctr = get_js_controller(cfg)
    V.add(ctr,
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # 駆動系のセットアップ
    arduino_controller = ArduinoFirmata(
        servo_pin=cfg.STEERING_ARDUINO_PIN, esc_pin=cfg.THROTTLE_ARDUINO_PIN)
    steering = ArdPWMSteering(controller=arduino_controller,
                              left_pulse=cfg.STEERING_ARDUINO_LEFT_PWM,
                              right_pulse=cfg.STEERING_ARDUINO_RIGHT_PWM)

    throttle = ArdPWMThrottle(controller=arduino_controller,
                              max_pulse=cfg.THROTTLE_ARDUINO_FORWARD_PWM,
                              zero_pulse=cfg.THROTTLE_ARDUINO_STOPPED_PWM,
                              min_pulse=cfg.THROTTLE_ARDUINO_REVERSE_PWM)

    V.add(steering, inputs=['user/angle'])
    V.add(throttle, inputs=['user/throttle'])

    # 車両を走らせる
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        drive(cfg)
