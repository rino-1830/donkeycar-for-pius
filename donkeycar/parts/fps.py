"""FPS計測用のユーティリティクラスを提供するモジュール。"""

import time
import logging

logger = logging.getLogger(__name__)


class FrequencyLogger(object):
    """周波数の現在値、最小値および最大値を記録するクラス。

    Attributes:
        last_timestamp (float | None): 前回計測した時刻。
        counter (int): 1秒間の呼び出し回数を数えるカウンタ。
        fps (int): 現在計算されたFPS。
        fps_list (List[int]): 計測したFPSの履歴。
        last_debug_timestamp (float | None): デバッグ情報を最後に出力した時刻。
        debug_interval (int): デバッグ情報を表示する間隔（秒単位）。
    """

    def __init__(self, debug_interval: int = 10):
        """コンストラクタ。

        Args:
            debug_interval (int, optional): デバッグ情報を表示する間隔（秒単位）。
        """
        self.last_timestamp = None
        self.counter = 0
        self.fps = 0
        self.fps_list = []

        self.last_debug_timestamp = None
        self.debug_interval = debug_interval

    def run(self) -> tuple[int, list[int]]:
        """FPSを計測し、必要に応じてデバッグ情報を表示する。

        このメソッドはフレームごとに呼び出すことを想定しており、
        1秒ごとにFPSを計算して履歴に追加する。``debug_interval``で
        指定した秒数ごとに、現在のFPSをログへ出力する。

        Returns:
            tuple[int, list[int]]: 最新のFPSと過去のFPSのリスト。
        """
        if self.last_timestamp is None:
            self.last_timestamp = time.time()

        if time.time() - self.last_timestamp > 1:
            self.fps = self.counter
            self.fps_list.append(self.counter)
            self.counter = 0
            self.last_timestamp = time.time()
        else:
            self.counter += 1

        # 周波数情報をシェルへ表示する
        if self.last_debug_timestamp is None:
            self.last_debug_timestamp = time.time()

        if time.time() - self.last_debug_timestamp > self.debug_interval:
            logger.info(f"現在のFPS = {self.fps}")
            self.last_debug_timestamp = time.time()

        return self.fps, self.fps_list

    def shutdown(self):
        """終了処理を行い、統計情報をログに出力する。"""

        if self.fps_list:
            logger.info(
                f"FPS（最小/最大） = {min(self.fps_list):2d} / {max(self.fps_list):2d}"
            )
            logger.info(f"FPS一覧 = {self.fps_list}")
