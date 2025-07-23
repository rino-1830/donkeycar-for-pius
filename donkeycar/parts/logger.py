import logging
from typing import List, Set, Dict, Tuple, Optional


class LoggerPart:
    """指定された値を車両のメモリに記録するパーツ。"""
    def __init__(self, inputs: List[str], level: str = "INFO", rate: int = 1, logger=None):
        """初期設定を行う。

        Args:
            inputs: 監視する項目名のリスト。
            level: 使用するログレベル名。
            rate: 同じ値を何回の呼び出しごとに記録するか。
            logger: 利用するロガー名。未指定の場合は ``LoggerPart`` を使用する。
        """
        self.inputs = inputs
        self.rate = rate
        self.level = logging._nameToLevel.get(level, logging.INFO)
        self.logger = logging.getLogger(logger if logger is not None else "LoggerPart")

        self.values = {}
        self.count = 0
        self.running = True

    def run(self, *args):
        """入力値を記録する。

        Args:
            *args: ``inputs`` に対応する値の並び。
        """
        if self.running and args is not None and len(args) == len(self.inputs):
            self.count = (self.count + 1) % (self.rate + 1)
            for i in range(len(self.inputs)):
                field = self.inputs[i]
                value = args[i]
                old_value = self.values.get(field)
                if old_value != value:
                    # 常に変更を記録する
                    self.logger.log(self.level, f"{field} の値が {old_value} -> {value}")
                    self.values[field] = value
                elif self.count >= self.rate:
                    self.logger.log(self.level, f"{field} の値 {value}")

    def shutdown(self):
        """ロギングを停止する。"""
        self.running = False
