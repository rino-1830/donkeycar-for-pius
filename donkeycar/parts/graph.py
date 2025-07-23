import numpy as np
import cv2


class Graph(object):
    """入力値を画像上にプロットするクラス。

    (x, y) と (b, g, r) のペアを要素とするリストを受け取り、指定された座標に
    色を描画する。x の値が画像幅を超えるとグラフは消去され、描画は左端から
    再開される。x は時間のように単調に増加する値を想定している。
    """
    def __init__(self, res=(200, 200, 3)):
        """コンストラクタ。

        Args:
            res (tuple[int, int, int], optional): 生成する画像サイズ。
                ``(高さ, 幅, チャンネル数)`` を指定する。デフォルトは
                ``(200, 200, 3)``。
        """
        self.img = np.zeros(res)
        self.prev = 0

    def clamp(self, val, lo, hi):
        """値を範囲内に収める。

        Args:
            val (float): 制限したい値。
            lo (float): 下限値。
            hi (float): 上限値。

        Returns:
            int: 範囲内に収めた整数値。
        """

        if val < lo:
            val = lo
        elif val > hi:
            val = hi
        return int(val)

    def run(self, values):
        """値のリストを受け取りグラフを更新する。

        Args:
            values (list[tuple[tuple[int, int], tuple[float, float, float]]] | None):
                座標と色のペアのリスト。``None`` を指定すると前回の画像を返す。

        Returns:
            numpy.ndarray: 更新後の画像。
        """

        if values is None:
            return self.img

        for coord, col in values:
            x = coord[0] % self.img.shape[1]
            y = self.clamp(coord[1], 0, self.img.shape[0] - 1)
            self.img[y, x] = col

        if abs(self.prev - x) > self.img.shape[1] / 2:
            self.img = np.zeros_like(self.img)

        self.prev = x

        return self.img

    def shutdown(self):
        """終了処理を行う。"""
        pass
