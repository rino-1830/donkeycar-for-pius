"""パスの記録・読込・描画およびCTE計算を行うモジュール."""

import pickle
import math
import logging
import pathlib
import numpy
from PIL import Image, ImageDraw

from donkeycar.parts.transform import PIDController
from donkeycar.utils import norm_deg, dist, deg2rad, arr_to_img, is_number_type


class AbstractPath:
    """パスを記録・操作する基底クラス."""

    def __init__(self, min_dist=1.):
        self.path = []  # (x,y) タプルのリスト
        self.min_dist = min_dist
        self.x = math.inf
        self.y = math.inf

    def run(self, recording, x, y):
        """記録モードなら現在位置をパスに追加する."""

        if recording:
            d = dist(x, y, self.x, self.y)
            if d > self.min_dist:
                logging.info(f"パス地点 ({x},{y})")
                self.path.append((x, y))
                self.x = x
                self.y = y
        return self.path

    def length(self):
        return len(self.path)

    def is_empty(self):
        return 0 == self.length()

    def is_loaded(self):
        return not self.is_empty()

    def get_xy(self):
        return self.path

    def reset(self):
        self.path = []
        return True

    def save(self, filename):
        return False

    def load(self, filename):
        return False


class CsvPath(AbstractPath):
    """CSV 形式でパスを保存・読み込むクラス."""

    def __init__(self, min_dist=1.):
        super().__init__(min_dist)

    def save(self, filename):
        if self.length() > 0:
            with open(filename, 'w') as outfile:
                for (x, y) in self.path:
                    outfile.write(f"{x}, {y}\n")
            return True
        else:
            return False

    def load(self, filename):
        path = pathlib.Path(filename)
        if path.is_file():
            with open(filename, "r") as infile:
                self.path = []
                for line in infile:
                    xy = [float(i.strip()) for i in line.strip().split(sep=",")]
                    self.path.append((xy[0], xy[1]))
            return True
        else:
            logging.info(f"ファイル '{filename}' が存在しません")
            return False

        self.recording = False

class CsvThrottlePath(AbstractPath):
    """スロットル値付きでパスを保存するクラス."""
    def __init__(self, min_dist: float = 1.0) -> None:
        super().__init__(min_dist)
        self.throttles = []

    def run(self, recording: bool, x: float, y: float, throttle: float) -> tuple:
        if recording:
            d = dist(x, y, self.x, self.y)
            if d > self.min_dist:
                logging.info(f"パス地点: ({x},{y}) スロットル: {throttle}")
                self.path.append((x, y))
                self.throttles.append(throttle)
                self.x = x
                self.y = y
        return self.path, self.throttles

    def reset(self) -> bool:
        super().reset()
        self.throttles = []
        return True

    def save(self, filename: str) -> bool:
        if self.length() > 0:
            with open(filename, 'w') as outfile:
                for (x, y), v in zip(self.path, self.throttles):
                    outfile.write(f"{x}, {y}, {v}\n")
            return True
        else:
            return False

    def load(self, filename: str) -> bool:
        path = pathlib.Path(filename)
        if path.is_file():
            with open(filename, "r") as infile:
                self.path = []
                for line in infile:
                    xy = [float(i.strip()) for i in line.strip().split(sep=",")]
                    self.path.append((xy[0], xy[1]))
                    self.throttles.append(xy[2])
            return True
        else:
            logging.warning(f"ファイル '{filename}' が存在しません")
            return False


class RosPath(AbstractPath):
    """pickle を用いてパスを保存・読み込むクラス."""
    def __init__(self, min_dist=1.):
        super().__init__(self, min_dist)

    def save(self, filename):
        outfile = open(filename, 'wb')
        pickle.dump(self.path, outfile)
        return True

    def load(self, filename):
        infile = open(filename, 'rb')
        self.path = pickle.load(infile)
        self.recording = False
        return True

class PImage(object):
    """画像を保持し各フレームで再利用するユーティリティ."""
    def __init__(self, resolution=(500, 500), color="white", clear_each_frame=False):
        self.resolution = resolution
        self.color = color
        self.img = Image.new('RGB', resolution, color=color)
        self.clear_each_frame = clear_each_frame

    def run(self):
        if self.clear_each_frame:
            self.img = Image.new('RGB', self.resolution, color=self.color)

        return self.img


class OriginOffset(object):
    """再起動せずに原点を変更するための部品."""

    def __init__(self, debug=False):
        self.debug = debug
        self.ox = 0.0
        self.oy = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.reset = None

    def run(self, x, y, closest_pt):
        """原点からの相対位置を算出する.

        Args:
            x: 現在の横方向位置。
            y: 現在の縦方向位置。
            closest_pt: 現在のCTEまたは最寄り点インデックス。

        Returns:
            tuple: 平行移動後の ``x`` ``y`` と更新された最寄り点インデックス。
        """
        if is_number_type(x) and is_number_type(y):
            # originがNoneなら現在位置を原点として設定する
            if self.reset:
                self.ox = x
                self.oy = y

            self.last_x = x
            self.last_y = y
        else:
            logging.debug("OriginOffset は数値以外を無視します")

        # 与えられた位置を原点分だけ平行移動する
        pos = (0, 0)
        if self.last_x is not None and self.last_y is not None and self.ox is not None and self.oy is not None:
            pos = (self.last_x - self.ox, self.last_y - self.oy)
        if self.debug:
            logging.info(f"位置 pos/x = {pos[0]}, pos/y = {pos[1]}")

        # CTEアルゴリズムの探索開始インデックスをリセットする
        if self.reset:
            if self.debug:
                logging.info(f"cte/closest_pt = {closest_pt} -> None")
            closest_pt = None

        # リセットのラッチを解除する
        self.reset = False

        return pos[0], pos[1], closest_pt

    def set_origin(self, x, y):
        logging.info(f"原点を ({x}, {y}) にリセットします")
        self.ox = x
        self.oy = y

    def reset_origin(self):
        """次に受け取る値を原点として設定する."""
        self.ox = None
        self.oy = None
        self.reset = True

    def init_to_last(self):
        self.set_origin(self.last_x, self.last_y)


class PathPlot(object):
    """パスを画像に描画するクラス."""
    def __init__(self, scale=1.0, offset=(0., 0.0)):
        self.scale = scale
        self.offset = offset

    def plot_line(self, sx, sy, ex, ey, draw, color):
        """スケールを適用してラインを描画する."""
        draw.line((sx,sy, ex, ey), fill=color, width=1)

    def run(self, img, path):
        """パスを描画して画像を返す."""
        
        if type(img) is numpy.ndarray:
            stacked_img = numpy.stack((img,)*3, axis=-1)
            img = arr_to_img(stacked_img)

        if path:
            draw = ImageDraw.Draw(img)
            color = (255, 0, 0)
            for iP in range(0, len(path) - 1):
                ax, ay = path[iP]
                bx, by = path[iP + 1]

                #
                # 北に進むとyが増えるためスケールで補正する
                #
                self.plot_line(ax * self.scale + self.offset[0],
                            ay * -self.scale + self.offset[1],
                            bx * self.scale + self.offset[0],
                            by * -self.scale + self.offset[1],
                            draw,
                            color)
        return img


class PlotCircle(object):
    """画像上に円を描画するクラス."""
    def __init__(self,  scale=1.0, offset=(0., 0.0), radius=4, color = (0, 255, 0)):
        self.scale = scale
        self.offset = offset
        self.radius = radius
        self.color = color

    def plot_circle(self, x, y, rad, draw, color, width=1):
        """スケールを適用して円を描画する."""
        sx = x - rad
        sy = y - rad
        ex = x + rad
        ey = y + rad

        draw.ellipse([(sx, sy), (ex, ey)], fill=color)


    def run(self, img, x, y):
        """指定座標に円を描画して画像を返す."""

        draw = ImageDraw.Draw(img)
        self.plot_circle(x * self.scale + self.offset[0],
                        y * -self.scale + self.offset[1],  # 北に進むとyが増える
                        self.radius,
                        draw, 
                        self.color)

        return img

from donkeycar.la import Line3D, Vec3

class CTE(object):
    """クロストラックエラーを計算するクラス."""

    def __init__(self, look_ahead=1, look_behind=1, num_pts=None) -> None:
        self.num_pts = num_pts
        self.look_ahead = look_ahead
        self.look_behind = look_behind

    #
    # (x, y) からの距離が最小となるパス要素のインデックスを求める。
    # 複数の要素が同距離の場合は最初の要素を採用する。
    #
    def nearest_pt(self, path, x, y, from_pt=0, num_pts=None):
        from_pt = from_pt if from_pt is not None else 0
        num_pts = num_pts if num_pts is not None else len(path)
        num_pts = min(num_pts, len(path))
        if num_pts < 0:
            logging.error("num_pts は負であってはいけません。")
            return None, None, None

        min_pt = None
        min_dist = None
        min_index = None
        for j in range(num_pts):
            i = (j + from_pt) % len(path)
            p = path[i]
            d = dist(p[0], p[1], x, y)
            if min_dist is None or d < min_dist:
                min_pt = p
                min_dist = d
                min_index = i
        return min_pt, min_index, min_dist


    # TODO: 指定した開始点から指定数まで探索し最も近い2点を求めるよう改良する。
    #       高速化だけでなく交差する経路にも対応可能となる。
    def nearest_two_pts(self, path, x, y):
        if path is None or len(path) < 2:
            logging.error("path が None のため最近傍点を計算できません")
            return None, None

        distances = []
        for iP, p in enumerate(path):
            d = dist(p[0], p[1], x, y)
            distances.append((d, iP, p))
        distances.sort(key=lambda elem : elem[0])

        # ひとつ前のポイントを区間の始点とする
        iA = (distances[0][1] - 1) % len(path)
        a = path[iA]

        # 次のポイントを区間の終点とする
        iB = (iA + 2) % len(path)
        b = path[iB]
        
        return a, b

    def nearest_waypoints(self, path, x, y, look_ahead=1, look_behind=1, from_pt=0, num_pts=None):
        """最寄り点周辺のウェイポイントを取得する。

        Args:
            path: ``(x, y)`` のリスト。
            x: 確認したい点の x 座標。
            y: 確認したい点の y 座標。
            from_pt: 探索を開始するインデックス。
            num_pts: 探索する最大ポイント数。
            look_ahead: 最寄り点より先のウェイポイント数。
            look_behind: 最寄り点より後ろのウェイポイント数。

        Returns:
            tuple: 先頭インデックス、最寄り点インデックス、最後のインデックス。
        """
        if path is None or len(path) < 2:
            logging.error("path が None のため最近傍点を計算できません")
            return None, None

        if look_ahead < 0:
            logging.error("look_ahead は非負の数でなければなりません")
            return None, None
        if look_behind < 0:
            logging.error("look_behind は非負の数でなければなりません")
            return None, None
        if (look_ahead + look_behind) > len(path):
            logging.error("指定したウェイポイント数を満たすほど path が長くありません")
            return None, None

        _pt, i, _distance = self.nearest_pt(path, x, y, from_pt, num_pts)

        # 区間の開始インデックスを求める
        a = (i + len(path) - look_behind) % len(path)

        # 区間の終了インデックスを求める
        b = (i + look_ahead) % len(path)

        return a, i, b

    def nearest_track(self, path, x, y, look_ahead=1, look_behind=1, from_pt=0, num_pts=None):
        """最寄り点を挟む線分を取得する。

        Args:
            path: ``(x, y)`` のリスト。
            x: 確認したい点の x 座標。
            y: 確認したい点の y 座標。
            from_pt: 探索を開始するインデックス。
            num_pts: 探索する最大ポイント数。
            look_ahead: 最寄り点より先のウェイポイント数。
            look_behind: 最寄り点より後ろのウェイポイント数。

        Returns:
            tuple: 線分の始点、終点、最寄り点インデックス。
        """

        a, i, b = self.nearest_waypoints(path, x, y, look_ahead, look_behind, from_pt, num_pts)

        return (path[a], path[b], i) if a is not None and b is not None else (None, None, None)

    def run(self, path, x, y, from_pt=None):
        """CTE を計算するメインルーチン。

        Returns:
            tuple: CTE と最寄り点のインデックス。
        """
        cte = 0.
        i = from_pt

        a, b, i = self.nearest_track(path, x, y, 
                                     look_ahead=self.look_ahead, look_behind=self.look_behind, 
                                     from_pt=from_pt, num_pts=self.num_pts)
        
        if a and b:
            logging.info(f"最寄り: ({a[0]}, {a[1]}) -> ({x}, {y})")
            a_v = Vec3(a[0], 0., a[1])
            b_v = Vec3(b[0], 0., b[1])
            p_v = Vec3(x, 0., y)
            line = Line3D(a_v, b_v)
            err = line.vector_to(p_v)
            sign = 1.0
            cp = line.dir.cross(err.normalized())
            if cp.y > 0.0 :
                sign = -1.0
            cte = err.mag() * sign            
        else:
            logging.info(f"({x},{y}) に近い点がありません")
        return cte, i


class PID_Pilot(object):
    """PID とパス情報を用いてステアリングとスロットルを決定するクラス."""

    def __init__(
            self,
            pid: PIDController,
            throttle: float,
            use_constant_throttle: bool = False,
            min_throttle: float = None) -> None:
        self.pid = pid
        self.throttle = throttle
        self.use_constant_throttle = use_constant_throttle
        self.variable_speed_multiplier = 1.0
        self.min_throttle = min_throttle if min_throttle is not None else throttle

    def run(self, cte: float, throttles: list, closest_pt_idx: int) -> tuple:
        """CTE から操舵角とスロットルを計算する."""

        steer = self.pid.run(cte)
        if self.use_constant_throttle or throttles is None or closest_pt_idx is None:
            throttle = self.throttle
        elif throttles[closest_pt_idx] * self.variable_speed_multiplier < self.min_throttle:
            throttle = self.min_throttle
        else:
            throttle = throttles[closest_pt_idx] * self.variable_speed_multiplier
        logging.info(f"CTE: {cte} ステア: {steer} スロットル: {throttle}")
        return steer, throttle
