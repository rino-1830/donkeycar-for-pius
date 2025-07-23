#!/usr/bin/env python3
"""Donkey 2 カーを走らせるためのスクリプト。

Usage:
    manage.py (drive)

Options:
    -h --help          この画面を表示します。
"""
import os
import time

from docopt import docopt

import donkeycar as dk
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle


def drive(cfg):
    """多数のパーツから構成されるロボット車両を組み立てて走行させる。

    各パートは ``threaded`` 引数に応じて ``run`` または ``run_threaded`` を
    呼び出しながら Vehicle ループでジョブとして動作する。すべてのパートは
    ``cfg.DRIVE_LOOP_HZ`` で指定されたフレームレートに従って順に更新され、
    各パートが遅滞なく処理を終えることを前提としている。パートには名前付き
    入出力を持たせることができ、フレームワークは同名の入力を要求するパート
    へ自動的に値を渡す。

    Args:
        cfg: 走行設定を含む設定オブジェクト。
    """

    # 車を初期化する
    V = dk.vehicle.Vehicle()
    
    class MyController:
        """一定のステアリングとスロットルを出力する簡易コントローラー。"""
        def run(self):
            steering = 0.0
            throttle = 0.1
            return steering, throttle

    V.add(MyController(), outputs=['angle', 'throttle'])

    # ドライブトレインの設定
    steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=cfg.STEERING_LEFT_PWM, 
                                    right_pulse=cfg.STEERING_RIGHT_PWM)
    
    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                    zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                    min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])
    
    # 20 秒間車を走らせる
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    
    if args['drive']:
        drive(cfg)
