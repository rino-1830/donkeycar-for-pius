import math
from donkeycar.utils import map_frange, sign

class VelocityNormalize:
    """速度を0から1の範囲に正規化するクラス。"""

    def __init__(self, min_speed: float, max_speed: float,
                 min_normal_speed: float = 0.1) -> None:
        """コンストラクタ。

        Args:
            min_speed: 車両が停止する速度よりも上の最小速度。
            max_speed: 車両の最高速度（目標値の場合もある）。
            min_normal_speed: ``min_speed`` に対応する正規化スロットル値。
        """
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_normal_speed = min_normal_speed

    def run(self, speed: float) -> float:
        """速度を正規化して返す。

        Args:
            speed: 実際の速度。負値は後退を表す。

        Returns:
            正規化された速度。
        """
        s = sign(speed)
        speed = abs(speed)
        if speed < self.min_speed:
            return 0.0
        if speed >= self.max_speed:
            return s * 1.0
        return s * map_frange(
            speed,
            self.min_speed, self.max_speed,
            self.min_normal_speed, 1.0)

    def shutdown(self):
        pass


class VelocityUnnormalize:
    """正規化速度を実速度に変換するクラス。"""

    def __init__(self, min_speed: float, max_speed: float,
                 min_normal_speed: float = 0.1) -> None:
        """コンストラクタ。

        Args:
            min_speed: 車両が停止する速度よりも上の最小速度。
            max_speed: 車両の最高速度。
            min_normal_speed: ``min_speed`` に対応する正規化スロットル値。
        """
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_normal_speed = min_normal_speed

    def run(self, speed: float) -> float:
        """正規化速度を実速度に変換して返す。

        Args:
            speed: 正規化された速度。負値は後退を表す。

        Returns:
            実速度。
        """
        s = sign(speed)
        speed = abs(speed)
        if speed < self.min_normal_speed:
            return 0.0
        if speed >= 1.0:
            return s * 1.0
        return s * map_frange(
            speed,
            self.min_normal_speed, 1.0,
            self.min_speed, self.max_speed)

    def shutdown(self):
        pass


class StepSpeedController:
    """単純なステップ制御で速度を調整するクラス。"""

    def __init__(self, min_speed: float, max_speed: float,
                 throttle_step: float = 1 / 255,
                 min_throttle: float = 0) -> None:
        """コンストラクタ。

        Args:
            min_speed: 車両が停止しない最小安定速度。
            max_speed: 最大スロットル時の速度。
            throttle_step: ``min_throttle`` から ``1.0`` までのステップ幅。
            min_throttle: ``min_speed`` に対応するスロットル値。
        """
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_throttle = min_throttle
        self.step_size = throttle_step
    
    def run(self, throttle: float, speed: float, target_speed: float) -> float:
        """現在速度と目標速度からスロットルを更新する。

        Args:
            throttle: 現在のスロットル値（-1〜1）。
            speed: 現在の速度。負値は後退を表す。
            target_speed: 目標速度。

        Returns:
            更新後のスロットル値。
        """
        if speed is None or target_speed is None:
            # 制御する速度が無いのでそのまま返す
            return throttle

        target_direction = sign(target_speed)
        direction = sign(speed)

        target_speed = abs(target_speed)
        speed = abs(speed)

        # 最小速度未満はゼロとみなす
        if target_speed < self.min_speed:
            return 0

        # 方向転換や停止状態からの発進時には
        # フィードフォワードでスロットルを推定し
        # すぐに動作範囲へ入るようにする
        if direction != target_direction:
            # 速度が高すぎてすぐに反転できない場合は減速する
            if speed > self.min_speed:
                return 0
            
            # 目標速度に達するための初期スロットルを算出
            return target_direction * map_frange(target_speed, self.min_speed, self.max_speed, self.min_throttle, 1.0)

        # スロットルを調整
        if speed > target_speed:
            # 速すぎるので減速
            throttle -= self.step_size
        elif speed > target_speed:
            # 遅すぎるので加速
            throttle += self.step_size

        return target_direction * throttle
