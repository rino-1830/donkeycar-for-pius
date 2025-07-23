from datetime import datetime
import donkeycar as dk
import re
import time

class TeensyRCin:
    """TeensyボードからRC入力を取得するパーツ。"""

    def __init__(self):
        """インスタンスを初期化する。"""
        self.inSteering = 0.0
        self.inThrottle = 0.0

        self.sensor = dk.parts.actuator.Teensy(0)

        TeensyRCin.LEFT_ANGLE = -1.0
        TeensyRCin.RIGHT_ANGLE = 1.0
        TeensyRCin.MIN_THROTTLE = -1.0
        TeensyRCin.MAX_THROTTLE =  1.0

        TeensyRCin.LEFT_PULSE = 496.0
        TeensyRCin.RIGHT_PULSE = 242.0
        TeensyRCin.MAX_PULSE = 496.0
        TeensyRCin.MIN_PULSE = 242.0


        self.on = True

    def map_range(self, x, X_min, X_max, Y_min, Y_max):
        """値の範囲を線形変換する。

        Args:
            x (float): 変換する値。
            X_min (float): 元の範囲の最小値。
            X_max (float): 元の範囲の最大値。
            Y_min (float): 変換後の範囲の最小値。
            Y_max (float): 変換後の範囲の最大値。

        Returns:
            float: 変換後の値。
        """
        X_range = X_max - X_min
        Y_range = Y_max - Y_min
        XY_ratio = X_range/Y_range

        return ((x-X_min) / XY_ratio + Y_min)

    def update(self):
        """センサーから値を読み取り、ステアリングとスロットルを更新する。"""

        rcin_pattern = re.compile('^I +([.0-9]+) +([.0-9]+).*$')

        while self.on:
            start = datetime.now()

            l = self.sensor.teensy_readline()

            while l:
                # print("mw TeensyRCin 行= " + l.decode('utf-8'))
                m = rcin_pattern.match(l.decode('utf-8'))

                if m:
                    i = float(m.group(1))
                    if i == 0.0:
                        self.inSteering = 0.0
                    else:
                        i = i / (1000.0 * 1000.0)  # 秒単位
                        i *= self.sensor.frequency * 4096.0
                        self.inSteering = self.map_range(i,
                                                         TeensyRCin.LEFT_PULSE, TeensyRCin.RIGHT_PULSE,
                                                         TeensyRCin.LEFT_ANGLE, TeensyRCin.RIGHT_ANGLE)

                    k = float(m.group(2))
                    if k == 0.0:
                        self.inThrottle = 0.0
                    else:
                        k = k / (1000.0 * 1000.0)  # 秒単位
                        k *= self.sensor.frequency * 4096.0
                        self.inThrottle = self.map_range(k,
                                                         TeensyRCin.MIN_PULSE, TeensyRCin.MAX_PULSE,
                                                         TeensyRCin.MIN_THROTTLE, TeensyRCin.MAX_THROTTLE)

                    # print("一致 %.1f  %.1f  %.1f  %.1f" % (i, self.inSteering, k, self.inThrottle))
                l = self.sensor.teensy_readline()

            stop = datetime.now()
            s = 0.01 - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

    def run_threaded(self):
        """スレッド実行用に現在の値を返す。"""

        return self.inSteering, self.inThrottle

    def shutdown(self):
        """スレッド停止を指示し、後処理を行う。"""

        # スレッドを停止するフラグを設定
        self.on = False
        print('TeensyRCin を停止します')
        time.sleep(.5)

