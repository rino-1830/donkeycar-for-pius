"""pigpio を用いたエンコーダ部品群."""

from donkeycar.utilities.deprecated import deprecated
import pigpio
import time

#
# このファイルの内容は非推奨です。
# 代わりに ``donkeycar.parts.tachometer`` と
# ``donkeycar.parts.odometer`` を利用してください。
#

@deprecated("donkeycar.parts.odometer.Odometer を使用してください")
class OdomDist(object):
    """オドメトリのティックから移動距離を計算するクラス."""
    def __init__(self, mm_per_tick, debug=False):
        """インスタンスを生成する.

        Args:
            mm_per_tick (float): 1ティックあたりの距離(mm)。
            debug (bool, optional): デバッグ出力を有効にするかどうか。
                デフォルトは ``False``。
        """

        self.mm_per_tick = mm_per_tick
        self.m_per_tick = mm_per_tick / 1000.0
        self.meters = 0
        self.last_time = time.time()
        self.meters_per_second = 0
        self.debug = debug
        self.prev_ticks = 0

    def run(self, ticks, throttle):
        """距離と速度を計算する.

        Args:
            ticks: 起動してからの累積ティック数。
            throttle: 正または負の速度判定に使用するスロットル値。

        Returns:
            ``float``: 走行距離 (m)。
            ``float``: 現在の速度 (m/s)。
            ``float``: 直近区間の距離 (m)。
        """
        new_ticks = ticks - self.prev_ticks
        self.prev_ticks = ticks

        # 前回の時刻を保存してタイマーをリセットする
        start_time = self.last_time
        end_time = time.time()
        self.last_time = end_time
        
        # 経過時間と移動距離を計算する
        seconds = end_time - start_time
        distance = new_ticks * self.m_per_tick
        if throttle < 0.0:
            distance = distance * -1.0
        velocity = distance / seconds
        
        # オドメータの値を更新する
        self.meters += distance
        self.meters_per_second = velocity

        # デバッグ用のコンソール出力
        if self.debug:
            print('秒数:', seconds)
            print('変化量:', distance)
            print('速度:', velocity)

            print('距離 (m):', self.meters)
            print('速度 (m/s):', self.meters_per_second)

        return self.meters, self.meters_per_second, distance


@deprecated("donkeycar.parts.tachometer.GpioTachometer を使用してください")
class PiPGIOEncoder:
    """``pigpio`` を使用したエンコーダ."""

    def __init__(self, pin, pi):
        """インスタンスを生成する.

        Args:
            pin (int): 監視する GPIO ピン番号。
            pi (pigpio.pi): ``pigpio`` の ``pi`` オブジェクト。
        """

        self.pin = pin
        self.pi = pi
        self.pi.set_mode(pin, pigpio.INPUT)
        self.pi.set_pull_up_down(pin, pigpio.PUD_UP)
        self.cb = pi.callback(self.pin, pigpio.FALLING_EDGE, self._cb)
        self.count = 0

    def _cb(self, pin, level, tick):
        """割り込みコールバックでカウントを増加させる."""

        self.count += 1

    def run(self):
        """現在のカウントを返す.

        Returns:
            int: 読み取ったパルス数。
        """

        return self.count

    def shutdown(self):
        """ピンとコールバックを解放する."""

        if self.cb != None:
            self.cb.cancel()
            self.cb = None

        self.pi.stop()


if __name__ == "__main__":
    pi = pigpio.pi()
    e = PiPGIOEncoder(4, pi)
    while True:
        time.sleep(0.1)
        e.run()


