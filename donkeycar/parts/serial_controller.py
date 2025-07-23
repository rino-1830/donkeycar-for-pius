#!/usr/bin/env python3
"""Arduinoからの信号を読み取り、ステアリング(steering)とスロットル(throttle)の
出力へ変換するスクリプト。

Arduinoの入力信号範囲: 0 ～ 200。
出力範囲: -1.00 ～ 1.00。
"""

import serial
import time

class SerialController:
    """Arduinoの信号から車両制御値を取得するシリアルコントローラ。"""
    def __init__(self):
        """インスタンスを初期化しシリアルポートを開く。"""
        print("シリアルコントローラーを起動します")

        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.serial = serial.Serial('/dev/ttyS0', 115200, timeout=1) # シリアルポート - ノートPC: 'COM3', Arduino: '/dev/ttyACM0'


    def update(self):
        """起動時に遅延し、シリアル入力を継続的に処理する。"""
        # 起動直後のクラッシュを防ぐため待機
        print("シリアルコントローラーを準備中")
        time.sleep(3)

        while True:
            line = str(self.serial.readline().decode()).strip('\n').strip('\r')
            output = line.split(", ")
            if len(output) == 2:
                if output[0].isnumeric() and output[1].isnumeric():
                    self.angle = (float(output[0])-1500)/500
                    self.throttle = (float(output[1])-1500)/500
                    if self.throttle > 0.01:
                        self.recording = True
                        print("録画中")
                    else:
                        self.recording = False
                    time.sleep(0.01)

    def run(self, img_arr=None):
        """スレッド化せずに現在の制御値を取得する。"""
        return self.run_threaded()

    def run_threaded(self, img_arr=None):
        """最新のステアリング角度とスロットルを返す。"""
        #print("信号:", self.angle, self.throttle)
        return self.angle, self.throttle, self.mode, self.recording
