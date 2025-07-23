"""行動パーツの定義.

このモジュールは車両の状態を管理するクラス `BehaviorPart` を提供します。
"""


class BehaviorPart(object):
    """複数の状態を管理するクラス。

    Attributes:
        states (list[str]): 状態名のリスト。
        active_state (int): 現在アクティブな状態のインデックス。
        one_hot_state_array (list[float]): 現在の状態を示すワンホット配列。
    """

    def __init__(self, states):
        """インスタンスを初期化する。

        Args:
            states (list[str]): 状態名を並べたリスト。
        """
        print("動作状態一覧:", states)
        self.states = states
        self.active_state = 0
        self.one_hot_state_array = [0.0 for _ in states]
        self.one_hot_state_array[0] = 1.0

    def increment_state(self):
        """状態を一つ進める。"""
        self.one_hot_state_array[self.active_state] = 0.0
        self.active_state += 1
        if self.active_state >= len(self.states):
            self.active_state = 0
        self.one_hot_state_array[self.active_state] = 1.0
        print("現在の状態:", self.states[self.active_state])

    def decrement_state(self):
        """状態を一つ戻す。"""
        self.one_hot_state_array[self.active_state] = 0.0
        self.active_state -= 1
        if self.active_state < 0:
            self.active_state = len(self.states) - 1
        self.one_hot_state_array[self.active_state] = 1.0
        print("現在の状態:", self.states[self.active_state])

    def set_state(self, iState):
        """状態を指定した値に設定する。

        Args:
            iState (int): 設定する状態のインデックス。
        """
        self.one_hot_state_array[self.active_state] = 0.0
        self.active_state = iState
        self.one_hot_state_array[self.active_state] = 1.0
        print("現在の状態:", self.states[self.active_state])

    def run(self):
        """現在の状態情報を返す。

        Returns:
            Tuple[int, str, list[float]]: アクティブ状態のインデックス、
            状態名、ワンホット配列のタプル。
        """
        return (
            self.active_state,
            self.states[self.active_state],
            self.one_hot_state_array,
        )

    def shutdown(self):
        """終了処理。"""
        pass
