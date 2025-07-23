import time
from typing import Tuple, Optional

from donkeycar.utilities.circular_buffer import CircularBuffer


class Odometer:
    """タコメータからの回転数を距離と速度へ変換するクラス。

    回転数から算出される速度は、指定した回数分の読み取りで平均化する
    ことができます。
    """
    def __init__(self, distance_per_revolution: float, smoothing_count: int = 1,
                 debug: bool = False):
        """Odometerオブジェクトを初期化する。

        Args:
            distance_per_revolution (float): 1回転あたりの移動距離。
            smoothing_count (int, optional): 速度平滑化に用いる読み取り回数。
            debug (bool, optional): デバッグモードの有無。
        """
        self.distance_per_revolution: float = distance_per_revolution
        self.timestamp: float = 0
        self.revolutions: float = 0
        self.running: bool = True
        self.queue = CircularBuffer(smoothing_count if smoothing_count >= 1 else 1)
        self.debug = debug
        self.reading = (0, 0, None)  # 距離、速度、タイムスタンプ

    def poll(self, revolutions: int, timestamp: Optional[float] = None) -> None:
        """回転数から距離と速度を計算して内部状態を更新する。

        Args:
            revolutions (int): 現在までの総回転数。
            timestamp (float, optional): 計測時刻。未指定の場合は現在時刻。
        """
        if self.running:
            if timestamp is None:
                timestamp = time.time()
            distance = revolutions * self.distance_per_revolution

            # 速度を平滑化する
            velocity = 0
            if self.queue.count > 0:
                lastDistance, lastVelocity, lastTimestamp = self.queue.tail()
                if timestamp > lastTimestamp:
                    velocity = (distance - lastDistance) / (timestamp - lastTimestamp)
            self.queue.enqueue((distance, velocity, timestamp))

            #
            # Python の代入はアトミックでありスレッドセーフ
            #
            self.reading = (distance, velocity, timestamp)

    def update(self) -> None:
        """内部スレッドループで定期的に ``poll`` を呼び出す。"""
        while self.running:
            self.poll(self.revolutions, None)
            time.sleep(0)  # 他のスレッドに実行を譲る

    def run_threaded(
        self, revolutions: float = 0, timestamp: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """スレッド環境向けに ``poll`` を呼び出す。

        Args:
            revolutions (float, optional): 現在までの回転数。
            timestamp (float, optional): 計測時刻。

        Returns:
            Tuple[float, float, float]: 距離、速度、タイムスタンプ。
        """
        if self.running:
            self.revolutions = revolutions
            self.timestamp = timestamp if timestamp is not None else time.time()

            return self.reading
        return 0, 0, self.timestamp

    def run(
        self, revolutions: float = 0, timestamp: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """ ``poll`` を即座に実行して結果を返す。

        Args:
            revolutions (float, optional): 現在までの回転数。
            timestamp (float, optional): 計測時刻。

        Returns:
            Tuple[float, float, float]: 距離、速度、タイムスタンプ。
        """
        if self.running:
            self.timestamp = timestamp if timestamp is not None else time.time()
            self.poll(revolutions, self.timestamp)

            return self.reading
        return 0, 0, self.timestamp

    def shutdown(self) -> None:
        """オドメーターの動作を停止する。"""
        self.running = False