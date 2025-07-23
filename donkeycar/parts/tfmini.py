import time
import serial
import logging

"""ポーリング遅延に関する注意

ポーリング遅延を入れないと、少しだけ値を出力した後に失敗します。これはおそらく、
シリアルポートの入力バッファが十分な速さでクリアされないためです。58 行目の
self.ser.reset_input_buffer() を呼ぶと多少は改善しますが、完全には解決しませ
ん。車速が 4m/s の場合、ポーリング遅延を 10ms にすると系統的な誤差が 4cm 発生
する点に注意してください。
"""

class TFMini:
    """TFMini および TFMini-Plus 距離センサーを扱うクラス。

    配線やインストール方法については
    https://github.com/TFmini/TFmini-RaspberryPi を参照してください。

    Attributes:
        ser (serial.Serial): デバイスとのシリアル接続。
        poll_delay (float): ポーリングの待ち時間。
        dist (int): センチメートル単位の距離値。
    """

    def __init__(self, port="/dev/serial0", baudrate=115200, poll_delay=0.01,
                 init_delay=0.1):
        """センサーを初期化する。

        Args:
            port (str): シリアルポート名。
            baudrate (int): ボーレート。
            poll_delay (float): ポーリング間隔（秒）。
            init_delay (float): 初期化時の待機時間（秒）。
        """
        self.ser = serial.Serial(port, baudrate)
        self.poll_delay = poll_delay

        self.dist = 0

        if not self.ser.is_open:
            self.ser.close()  # まだ開いている場合、二重に開かないよう閉じる
            self.ser.open()

        self.logger = logging.getLogger(__name__)

        self.logger.info("TFMiniを初期化")
        time.sleep(init_delay)

    def update(self):
        """ポーリングを繰り返して距離を更新する。"""
        while self.ser.is_open:
            self.poll()
            if self.poll_delay > 0:
                time.sleep(self.poll_delay)

    def poll(self):
        """センサーからデータを読み取り距離を更新する。"""
        try:
            count = self.ser.in_waiting
            if count > 8:
                recv = self.ser.read(9)
                self.ser.reset_input_buffer()

                if recv[0] == 0x59 and recv[1] == 0x59:
                    dist = recv[2] + recv[3] * 256
                    strength = recv[4] + recv[5] * 256

                    if strength > 0:
                        self.dist = dist

                    self.ser.reset_input_buffer()

        except Exception as e:
            self.logger.error(e)


    def run_threaded(self):
        """スレッド実行時に距離を返す。"""
        return self.dist

    def run(self):
        """ポーリングを一度実行して距離を返す。"""
        self.poll()
        return self.dist

    def shutdown(self):
        """シリアルポートを閉じる。"""
        self.ser.close()

if __name__ == "__main__":
    lidar = TFMini(poll_delay=0.1)

    for i in range(100):
        data = lidar.run()
        print(i, data)

    lidar.shutdown()
    