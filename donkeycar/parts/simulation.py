#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物理的な車両がなくても Donkeycar を試すためのパーツ群。"""

import random
import numpy as np


class MovingSquareTelemetry:
    """シミュレーション用の跳ね回る四角形の座標を生成する。"""
    def __init__(self, max_velocity=29,
                 x_min=10, x_max=150,
                 y_min=10, y_max=110):
        """インスタンスを初期化する。

        Args:
            max_velocity: 正規化された最大速度。
            x_min: x 座標の最小値。
            x_max: x 座標の最大値。
            y_min: y 座標の最小値。
            y_max: y 座標の最大値。
        """

        self.velocity = random.random() * max_velocity

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        self.x_direction = random.random() * 2 - 1
        self.y_direction = random.random() * 2 - 1

        self.x = random.random() * x_max
        self.y = random.random() * y_max

        self.tel = self.x, self.y

    def run(self):
        """位置を更新し現在の座標を返す。"""

        # 移動
        self.x += self.x_direction * self.velocity
        self.y += self.y_direction * self.velocity

        # 壁に当たったら跳ね返る
        if self.y < self.y_min or self.y > self.y_max:
            self.y_direction *= -1
        if self.x < self.x_min or self.x > self.x_max:
            self.x_direction *= -1

        return int(self.x), int(self.y)

    def update(self):
        """非スレッド実行時に位置情報を更新する。"""

        self.tel = self.run()

    def run_threaded(self):
        """最新の座標を返す。"""

        return self.tel


class SquareBoxCamera:
    """四角形のボックスを描いた画像を返すダミーカメラ。

    学習アルゴリズムが学習できるかを試す際に使用できる。
    """

    def __init__(self, resolution=(120, 160), box_size=4, color=(255, 0, 0)):
        """カメラの設定を初期化する。

        Args:
            resolution: 画像サイズ ``(高さ, 幅)``。
            box_size: 描画するボックスの一辺の長さ。
            color: ボックスの RGB カラー。
        """

        self.resolution = resolution
        self.box_size = box_size
        self.color = color

    def run(self, x, y, box_size=None, color=None):
        """指定された座標にボックスを描いた画像を生成する。

        Args:
            x: ボックス中心の x 座標。
            y: ボックス中心の y 座標。
            box_size: 使用するボックスサイズ。未指定時は ``self.box_size`` を利用する。
            color: ボックスの色。未指定時は ``self.color`` を利用する。

        Returns:
            numpy.ndarray: ボックスを描画した画像。"""
        radius = int((box_size or self.box_size)/2)
        color = color or self.color
        frame = np.zeros(shape=self.resolution + (3,))
        frame[y - radius: y + radius,
              x - radius: x + radius, :] = color
        return frame