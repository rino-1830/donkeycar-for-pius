
class ThrottleFilter(object):
    """リバース入力に応じて自動で逆スロットルを作動させるフィルタ。"""

    def __init__(self):
        """インスタンスを初期化する。"""
        self.reverse_triggered = False
        self.last_throttle = 0.0

    def run(self, throttle_in):
        """スロットル入力を処理し、自動逆転を制御する。

        Args:
            throttle_in (float | None): 入力スロットル値。

        Returns:
            float | None: 処理後のスロットル値。
        """
        if throttle_in is None:
            return throttle_in

        throttle_out = throttle_in

        if throttle_out < 0.0:
            if not self.reverse_triggered and self.last_throttle < 0.0:
                throttle_out = 0.0
                self.reverse_triggered = True
        else:
            self.reverse_triggered = False

        self.last_throttle = throttle_out
        return throttle_out

    def shutdown(self):
        """後処理は特に行わない。"""
        pass
