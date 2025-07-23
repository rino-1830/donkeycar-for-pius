"""ライダー(Lidar)。"""
#
# glob をインストールしておくこと: "pip3 install glob2"
# Adafruit RPLidar ドライバーをインストールしておくこと:
#   pip install Adafruit_CircuitPython_RPLIDAR
#
import logging
import sys
import time
import math
import pickle
import serial
import numpy as np
from donkeycar.utils import norm_deg, dist, deg2rad, arr_to_img
from PIL import Image, ImageDraw

logger = logging.getLogger("donkeycar.parts.lidar")

CLOCKWISE = 1
COUNTER_CLOCKWISE = -1


def limit_angle(angle):
    """角度を0から360の範囲に収める。"""
    while angle < 0:
        angle += 360 
    while angle > 360:
        angle -= 360
    return angle


def angle_in_bounds(angle, min_angle, max_angle):
    """角度が指定された範囲内かを判定する。

    Args:
        angle (float): 判定対象の角度。
        min_angle (float): 範囲の最小角度。
        max_angle (float): 範囲の最大角度。

    Returns:
        bool: 角度が範囲内なら ``True``。
    """
    if min_angle <= max_angle:
        return min_angle <= angle <= max_angle
    else:
        # min_angle < max_angle の場合、範囲が 0 度を跨ぐため
        # 2 つの範囲に分割して判定する
        return (min_angle <= angle <= 360) or (max_angle >= angle >= 0)


class RPLidar2(object):
    """RPLidar 用ドライバー。

    参考: https://github.com/Ezward/rplidar

    Notes:
        実測では 1 秒間に 7 回のスキャンと 1846 回の測定を行う。
    """
    def __init__(self,
                 min_angle = 0.0, max_angle = 360.0,
                 min_distance = sys.float_info.min,
                 max_distance = sys.float_info.max,
                 forward_angle = 0.0,
                 angle_direction=CLOCKWISE,
                 batch_ms=50,  # run() 内のループ時間(ms)
                 debug=False):
        
        self.lidar = None
        self.port = None
        self.on = False

        help = []
        if min_distance < 0:
            help.append("min_distance は 0以上でなければなりません")

        if max_distance <= 0:
            help.append("max_distance は 0より大きくなければなりません")

        if min_angle < 0 or min_angle > 360:
            help.append("min_angle は 0以上360以下で指定してください")

        if max_angle <= 0 or max_angle > 360:
            help.append("max_angle は 0より大きく360以下で指定してください")

        if forward_angle < 0 or forward_angle > 360:
            help.append("forward_angle は 0以上360以下で指定してください")

        if angle_direction != CLOCKWISE and \
           angle_direction != COUNTER_CLOCKWISE:
            help.append("angle-direction は 1(時計回り) または -1(反時計回り) で指定してください")  # noqa

        if len(help) > 0:
            msg = "RPLidar を開始できません。コンストラクタへの引数が不正です: "
            raise ValueError(msg + " ".join(help))

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.forward_angle = forward_angle
        self.spin_reverse = (args.angle_direction != CLOCKWISE)
        self.measurements = [] # (distance, angle, time, scan, index) のリスト

        from adafruit_rplidar import RPLidar
        import glob
        
        #
        # Lidar が接続されているシリアルポートを探す
        #
        port_found = False
        temp_list = glob.glob ('/dev/ttyUSB*')
        result = []
        for a_port in temp_list:
            try:
                s = serial.Serial(a_port)
                s.close()
                result.append(a_port)
                port_found = True
            except serial.SerialException:
                pass
        if not port_found:
            raise RuntimeError("RPLidar が接続されていません。")

        # 初期化
        self.port = result[0]
        self.lidar = RPLidar(None, self.port, timeout=3)
        self.lidar.clear_input()
        time.sleep(1)

        self.measurement_count = 0  # 1 スキャン内の測定数
        self.measurement_index = 0  # 次に書き込む測定のインデックス
        self.full_scan_count = 0
        self.full_scan_index = 0
        self.total_measurements = 0
        self.iter_measurements = self.lidar.iter_measurements()
        self.measurement_batch_ms = batch_ms

        self.running = True

    def poll(self):
        """計測を1件取得してバッファへ格納する。"""
        if self.running:
            try:
                #
                # 測定値を1件読み込む
                #
                new_scan, quality, angle, distance = next(self.iter_measurements)  # noqa
                                        
                now = time.time()
                self.total_measurements += 1

                # 新しいスキャンの開始をチェック
                if new_scan:
                    self.full_scan_count += 1
                    self.full_scan_index = 0
                    self.measurement_count = self.measurement_index  # このスキャンの測定数
                    self.measurement_index = 0   # 次のスキャンの書き込み開始位置
                    
                #
                # RPLidar は時計回りに回転するが、角度は反時計回りで増加させたい
                #
                if self.spin_reverse:
                    angle = (360.0 - (angle % 360.0)) % 360.0
                
                # 0 度が前方になるよう補正
                angle = (angle - self.forward_angle + 360.0) % 360.0
            
                # 角度と距離で測定値をフィルタリング
                if angle_in_bounds(angle, self.min_angle, self.max_angle):
                    if distance >= self.min_distance and distance <= self.max_distance:
                        #
                        # 測定値は (distance, angle, time, scan, index) の
                        # タプルとして保持される。
                        # distance : 距離(mm)。0 は無効値を示す。
                        # angle    : 角度(度)。
                        # time     : 取得時刻(秒)。
                        # scan     : この測定が属するスキャン番号。
                        # index    : スキャン内でのインデックス。
                        #
                        # scan と index の組み合わせで測定を一意に識別できる。
                        # ドライバーは最新 360 度分の測定を循環バッファで保持する。
                        # run_threaded() を高頻度で呼び出すと、前回のスキャンと
                        # 重複する測定値が返ることがある。scan:index を比較することで
                        # どの測定が新しいかを判定できる。
                        # また取得時刻も含まれるため、移動体では古い測定を無視したり、
                        # 表示をフェードアウトさせたりすることが可能。
                        measurement = (distance, angle, now,
                                        self.full_scan_count, self.full_scan_index)
                        
                        # バッファが足りなければ拡張し、そうでなければ上書き
                        if self.measurement_index >= len(self.measurements):
                            self.measurements.append(measurement)
                            self.measurement_count = self.measurement_index + 1
                        else:
                            self.measurements[self.measurement_index] = measurement  # noqa
                        self.measurement_index += 1
                        self.full_scan_index += 1
                            
            except serial.serialutil.SerialException:
                logger.error('RPLidar から SerialException が発生しました。')

    def update(self):
        """連続して計測し、統計情報をログに出力する。"""
        start_time = time.time()
        while self.running:
            self.poll()
            time.sleep(0)  # 他のスレッドに実行を譲る
        total_time = time.time() - start_time
        scan_rate = self.full_scan_count / total_time
        measurement_rate = self.total_measurements / total_time
        logger.info("RPLidar 全体の計測時間: {time} 秒".format(time=total_time))
        logger.info("RPLidar スキャン回数: {count} 回".format(count=self.full_scan_count))
        logger.info("RPLidar 測定数: {count} 回".format(count=self.total_measurements))
        logger.info("RPLidar スキャンレート: {rate} 回/秒".format(rate=scan_rate))
        logger.info("RPLidar 測定レート: {rate} 件/秒".format(rate=measurement_rate))

    def run_threaded(self):
        """最新の計測データを返す。"""
        if self.running:
            return self.measurements
        return []
    
    def run(self):
        """一定時間計測して結果を返す。"""
        if not self.running:
            return []
        #
        # 指定時間分計測して結果を返す
        #
        batch_time = time.time() + self.measurement_batch_ms / 1000.0
        while True:
            self.poll()
            time.sleep(0)  # 他のスレッドに実行を譲る
            if time.time() >= batch_time:
                break
        return self.measurements

    def shutdown(self):
        """センサーを停止してリソースを解放する。"""
        self.running = False
        time.sleep(2)
        if self.lidar is not None:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
            self.lidar = None


class RPLidar(object):
    """Adafruit 製 RPLidar ドライバー。

    https://github.com/adafruit/Adafruit_CircuitPython_RPLIDAR
    """
    def __init__(self, lower_limit = 0, upper_limit = 360, debug=False):
        from adafruit_rplidar import RPLidar

        # RPLidar を初期化
        PORT_NAME = "/dev/ttyUSB0"

        import glob
        port_found = False
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        temp_list = glob.glob ('/dev/ttyUSB*')
        result = []
        for a_port in temp_list:
            try:
                s = serial.Serial(a_port)
                s.close()
                result.append(a_port)
                port_found = True
            except serial.SerialException:
                pass
        if port_found:
            self.port = result[0]
            self.distances = [] # 距離の測定値リスト
            self.angles = [] # 距離測定に対応する角度リスト
            self.lidar = RPLidar(None, self.port, timeout=3)
            self.lidar.clear_input()
            time.sleep(1)
            self.on = True
        else:
            logger.error("RPLidarが接続されていません")
            raise RuntimeError("RPLidarが接続されていません")

    def update(self):
        """スキャンを取得して距離と角度を更新する。"""
        scans = self.lidar.iter_scans(550)
        while self.on:
            try:
                for scan in scans:
                    self.distances = [item[2] for item in scan]
                    self.angles = [item[1] for item in scan]
            except serial.serialutil.SerialException:
                logger.error('Lidar から SerialException が発生しました')

    def run_threaded(self):
        """角度順に並べ替えた距離リストを返す。"""
        sorted_distances = []
        if (self.angles != []) and (self.distances != []):
            angs = np.copy(self.angles)
            dists = np.copy(self.distances)

            filter_angs = angs[(angs > self.lower_limit) & (angs < self.upper_limit)]
            filter_dist = dists[(angs > self.lower_limit) & (angs < self.upper_limit)]  # 角度で距離を抽出

            angles_ind = np.argsort(filter_angs)         # filter_angs をソートするインデックス
            if angles_ind != []:
                sorted_distances = np.argsort(filter_dist) # 角度順に距離を並べ替え
        return sorted_distances

    def shutdown(self):
        """センサーを停止する。"""
        self.on = False
        time.sleep(2)
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()


class YDLidar(object):
    """PyLidar3 を利用した YDLidar ドライバー。

    https://pypi.org/project/PyLidar3/
    """
    def __init__(self, port='/dev/ttyUSB0'):
        """デバイスに接続してスキャンを開始する。"""
        import PyLidar3
        self.port = port
        self.distances = [] # 距離の測定値リスト
        self.angles = [] # 測定値に対応する角度リスト
        self.lidar = PyLidar3.YdLidarX4(port)
        if(self.lidar.Connect()):
            logger.debug(self.lidar.GetDeviceInfo())
            self.gen = self.lidar.StartScanning()
        else:
            logger.error("YDLidar への接続エラー")
            raise RuntimeError("YDLidar への接続エラー")
        self.on = True


    def init(self, port='/dev/ttyUSB0'):
        """再初期化してスキャンジェネレータを返す。"""
        import PyLidar3
        logger.debug("Lidar を起動します...")
        self.port = port
        self.distances = [] # 距離の測定値リスト
        self.angles = [] # 測定値に対応する角度リスト
        self.lidar = PyLidar3.YdLidarX4(port)
        if(self.lidar.Connect()):
            logger.debug(self.lidar.GetDeviceInfo())
            gen = self.lidar.StartScanning()
            return gen
        else:
            logger.error("YDLidar への接続エラー")
            raise RuntimeError("YDLidar への接続エラー")
        self.on = True

    def update(self, lidar, debug = False):
        """スキャンを読み取り必要に応じて返す。"""
        while self.on:
            try:
                self.data = next(lidar)
                for angle in range(0,360):
                    if(self.data[angle]>1000):
                        self.angles = [angle]
                        self.distances = [self.data[angle]]
                if debug:
                    return self.distances, self.angles
            except serial.serialutil.SerialException:
                logger.error('Lidar から SerialException が発生しました。')

    def run_threaded(self):
        """距離と角度の一覧を返す。"""
        return self.distances, self.angles

    def shutdown(self):
        """センサーを停止する。"""
        self.on = False
        time.sleep(2)
        self.lidar.StopScanning()
        self.lidar.Disconnect()


class LidarPlot(object):
    """生のライダー計測値を画像に描画するクラス。"""
    PLOT_TYPE_LINE = 0
    PLOT_TYPE_CIRC = 1
    def __init__(self, resolution=(500,500),
        max_dist=1000, # mm 単位
        radius_plot=3,
        plot_type=PLOT_TYPE_CIRC):
        self.frame = Image.new('RGB', resolution)
        self.max_dist = max_dist
        self.rad = radius_plot
        self.resolution = resolution
        if plot_type == self.PLOT_TYPE_CIRC:
            self.plot_fn = self.plot_circ
        else:
            self.plot_fn = self.plot_line
            
    def plot_line(self, img, dist, theta, max_dist, draw):
        """距離を線で描画する。"""
        center = (img.width / 2, img.height / 2)
        max_pixel = min(center[0], center[1])
        dist = dist / max_dist * max_pixel
        if dist < 0 :
            dist = 0
        elif dist > max_pixel:
            dist = max_pixel
        theta = np.radians(theta)
        sx = math.cos(theta) * dist + center[0]
        sy = math.sin(theta) * dist + center[1]
        ex = math.cos(theta) * (dist + self.rad) + center[0]
        ey = math.sin(theta) * (dist + self.rad) + center[1]
        fill = 128
        draw.line((sx,sy, ex, ey), fill=(fill, fill, fill), width=1)
        
    def plot_circ(self, img, dist, theta, max_dist, draw):
        """距離を円で描画する。"""
        center = (img.width / 2, img.height / 2)
        max_pixel = min(center[0], center[1])
        dist = dist / max_dist * max_pixel
        if dist < 0 :
            dist = 0
        elif dist > max_pixel:
            dist = max_pixel
        theta = np.radians(theta)
        sx = int(math.cos(theta) * dist + center[0])
        sy = int(math.sin(theta) * dist + center[1])
        ex = int(math.cos(theta) * (dist + 2 * self.rad) + center[0])
        ey = int(math.sin(theta) * (dist + 2 * self.rad) + center[1])
        fill = 128

        draw.ellipse((min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey)),
                     fill=(fill, fill, fill))

    def plot_scan(self, img, distances, angles, max_dist, draw):
        """1回のスキャン結果を画像に描画する。"""
        for dist, angle in zip(distances, angles):
            self.plot_fn(img, dist, angle, max_dist, draw)
            
    def run(self, distances, angles):
        """距離と角度のリストを受け取り画像を返す。"""
        self.frame = Image.new('RGB', self.resolution, (255, 255, 255))
        draw = ImageDraw.Draw(self.frame)
        self.plot_scan(self.frame, distances, angles, self.max_dist, draw)
        return self.frame

    def shutdown(self):
        """後始末用のメソッド。"""
        pass


def mark_line(draw_context, cx, cy,
              distance_px, theta_degrees,
              mark_color, mark_px):
    """線分を描画する。"""
    theta = np.radians(theta_degrees)
    sx = cx + math.cos(theta) * distance_px
    sy = cy - math.sin(theta) * distance_px
    ex = cx + math.cos(theta) * (distance_px + mark_px)
    ey = cy - math.sin(theta) * (distance_px + mark_px)
    draw_context.line((sx, sy, ex, ey), fill=mark_color, width=1)


def mark_circle(draw_context, cx, cy,
                distance_px, theta_degrees,
                mark_color, mark_px):
    """円形のマークを描画する。"""
    theta = np.radians(theta_degrees)
    sx = int(cx + math.cos(theta) * (distance_px + mark_px))
    sy = int(cy - math.sin(theta) * (distance_px + mark_px))
    draw_context.ellipse(
        (sx - mark_px, sy - mark_px, sx + mark_px, sy + mark_px),
        fill=mark_color)


def plot_polar_point(draw_context, bounds, mark_fn, mark_color, mark_px,
                     distance, theta, max_distance,
                     angle_direction=COUNTER_CLOCKWISE, rotate_plot=0):
    """1 点の極座標データを描画する。

    Args:
        draw_context: PIL の描画コンテキスト。
        bounds: 描画領域の矩形 ``(left, top, right, bottom)``。
        mark_fn: 点を描画する関数。
        mark_color: マークの色。
        mark_px: マークの大きさ(px)。
        distance: 距離(mm)。
        theta: 角度(度)。
        max_distance: 表示上の最大距離。
        angle_direction: 角度の増加方向。
        rotate_plot: 描画時に回転させる角度。

    """

    if distance < 0 or distance > max_distance:
        return  # don't print out of range pixels

    left, top, right, bottom = bounds
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    max_pixel = min(cx, cy)
    distance_px = distance / max_distance * max_pixel
    
    theta = (theta + rotate_plot) % 360.0
        
    if angle_direction != COUNTER_CLOCKWISE:
        theta = (360.0 - (theta % 360.0)) % 360.0
        
    mark_fn(draw_context, cx, cy, distance_px, theta, mark_color, mark_px)


def plot_polar_points(draw_context, bounds, mark_fn, mark_color, mark_px,
                    measurements, max_distance,
                    angle_direction=COUNTER_CLOCKWISE, rotate_plot=0):
    """複数の極座標データを画像に描画する。

    Args:
        draw_context: PIL の描画コンテキスト。
        bounds: 描画領域の矩形。
        mark_fn: 点を描画する関数。
        mark_color: マークの色。
        mark_px: マークの大きさ(px)。
        measurements: ``(distance, angle)`` のリスト。
        max_distance: 表示上の最大距離。
        angle_direction: 角度の増加方向。
        rotate_plot: 描画時に回転させる角度。
    """
    # plot each measurement
    for distance, angle in measurements:
        plot_polar_point(draw_context, bounds, mark_fn, mark_color, mark_px,
                              distance, angle, max_distance,
                              angle_direction, rotate_plot)


def plot_polar_bounds(draw_context, bounds, color,
                      angle_direction=COUNTER_CLOCKWISE, rotate_plot=0):
    """極座標の枠線を描画する。

    Args:
        draw_context: PIL の描画コンテキスト。
        bounds: 描画領域の矩形。
        color: 線の色。
        angle_direction: 角度の増加方向。
        rotate_plot: 描画時に回転させる角度。
    """

    left, top, right, bottom = bounds
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    max_pixel = min(cx, cy)
    
    #
    # draw the zero heading axis
    #
    # correct the angle for direction of scan
    theta = rotate_plot
    if angle_direction != COUNTER_CLOCKWISE:
        theta = (360.0 - (theta % 360.0)) % 360.0
    # draw the axis line
    theta = np.radians(theta)
    sx = cx + math.cos(theta) * max_pixel
    sy = cy - math.sin(theta) * max_pixel
    draw_context.ellipse(
        (cx - max_pixel, cy - max_pixel, cx + max_pixel, cy + max_pixel),
        outline=color)


def plot_polar_angle(draw_context, bounds, color, theta,
                     angle_direction=COUNTER_CLOCKWISE, rotate_plot=0):
    """特定の角度を示す線を描画する。

    Args:
        draw_context: PIL の描画コンテキスト。
        bounds: 描画領域の矩形。
        color: 線の色。
        theta: 描画する角度(度)。
        angle_direction: 角度の増加方向。
        rotate_plot: 描画時に回転させる角度。
    """

    left, top, right, bottom = bounds
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    max_pixel = min(cx, cy)
    
    #
    # draw the zero heading axis
    #
    # correct the angle for direction of scan
    theta += rotate_plot
    if angle_direction != COUNTER_CLOCKWISE:
        theta = (360.0 - (theta % 360.0)) % 360.0
        
    # draw the angle line
    theta = np.radians(theta)
    sx = cx + math.cos(theta) * max_pixel
    sy = cy - math.sin(theta) * max_pixel
    draw_context.line((cx, cy, sx, sy), fill=color, width=1)


class LidarPlot2(object):
    """スキャンデータを画像として描画するクラス。

    Args:
        resolution (tuple): 画像サイズ ``(width, height)``。
        plot_type (int): PLOT_TYPE_CIRCLE または PLOT_TYPE_LINE。
        mark_px (int): 点の大きさ(px)。
        max_dist (int): 描画する最大距離(mm)。
        angle_direction (int): 角度の増加方向。
        rotate_plot (float): 描画時に回転させる角度。
    """
    PLOT_TYPE_LINE = 0
    PLOT_TYPE_CIRCLE = 1
    def __init__(self,
                 resolution=(500,500),
                 plot_type=PLOT_TYPE_CIRCLE,
                 mark_px=3,
                 max_dist=4000, # mm 単位
                 angle_direction=COUNTER_CLOCKWISE, 
                 rotate_plot=0,
                 background_color=(224, 224, 224),
                 border_color=(128, 128, 128),
                 point_color=(255, 64, 64)):
        
        self.frame = Image.new('RGB', resolution)
        self.mark_px = mark_px
        self.max_distance = max_dist
        self.resolution = resolution
        if plot_type == self.PLOT_TYPE_CIRCLE:
            self.mark_fn = mark_circle
        else:
            self.mark_fn = mark_line
        self.angle_direction = angle_direction
        self.rotate_plot = rotate_plot
        
        self.background_color = background_color
        self.border_color = border_color
        self.point_color = point_color

    def run(self, measurements):
        '''
        draw measurements to a PIL image and output the pil image
        measurements: list of polar coordinates as (distance, angle) tuples
        '''
            
        self.frame = Image.new('RGB', self.resolution, (255, 255, 255))
        bounds = (0, 0, self.frame.width, self.frame.height)
        draw = ImageDraw.Draw(self.frame)
        
        # 背景を描画
        draw.rectangle(bounds, fill=self.background_color)

        # 外枠とゼロ度線を描画
        plot_polar_bounds(draw, bounds, self.border_color,
                          self.angle_direction, self.rotate_plot)
        plot_polar_angle(draw, bounds, self.border_color, 0,
                         self.angle_direction, self.rotate_plot)
        
        # 計測点を描画
        plot_polar_points(
            draw, bounds, self.mark_fn, self.point_color, self.mark_px,
            [(distance, angle) for distance, angle, _, _, _ in measurements],
            self.max_distance, self.angle_direction, self.rotate_plot)
        
        return self.frame

    def shutdown(self):
        pass


class BreezySLAM(object):
    """BreezySLAM を利用した SLAM 実装。

    https://github.com/simondlevy/BreezySLAM
    """
    def __init__(self, MAP_SIZE_PIXELS=500, MAP_SIZE_METERS=10):
        """コンストラクタ."""
        from breezyslam.algorithms import RMHC_SLAM
        from breezyslam.sensors import Laser

        laser_model = Laser(scan_size=360, scan_rate_hz=10.,
                            detection_angle_degrees=360,
                            distance_no_detection_mm=12000)
        MAP_QUALITY=5
        self.slam = RMHC_SLAM(laser_model,
                              MAP_SIZE_PIXELS, MAP_SIZE_METERS, MAP_QUALITY)
    
    def run(self, distances, angles, map_bytes):
        """SLAM を実行し位置を返す。"""
        self.slam.update(distances, scan_angles_degrees=angles)
        x, y, theta = self.slam.getpos()

        if map_bytes is not None:
            self.slam.getmap(map_bytes)

        return x, y, deg2rad(norm_deg(theta))

    def shutdown(self):
        """後始末用のメソッド。"""
        pass


class BreezyMap(object):
    """BreezySLAM が生成するビットマップ。"""
    def __init__(self, MAP_SIZE_PIXELS=500):
        """コンストラクタ."""
        self.mapbytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)

    def run(self):
        """マップデータを返す。"""
        return self.mapbytes

    def shutdown(self):
        """後始末用のメソッド。"""
        pass


class MapToImage(object):

    def __init__(self, resolution=(500, 500)):
        """画像サイズを指定して初期化する。"""
        self.resolution = resolution

    def run(self, map_bytes):
        """マップデータを画像に変換して返す。"""
        np_arr = np.array(map_bytes).reshape(self.resolution)
        return arr_to_img(np_arr)

    def shutdown(self):
        """後始末用のメソッド。"""
        pass


if __name__ == "__main__":
    import argparse
    import cv2
    import json
    from threading import Thread
    
    def convert_from_image_to_cv2(img: Image) -> np.ndarray:
        """PIL 画像を OpenCV 形式に変換する。"""
        # OpenCV の BGR 形式が必要な場合は下記を使用
        # return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return np.asarray(img)
    
    # 引数の解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rate", type=float, default=20,
                        help="1 秒あたりのスキャン数")
    parser.add_argument("-n", "--number", type=int, default=40,
                        help="取得するスキャン数")
    parser.add_argument("-a", "--min-angle", type=float, default=0,
                        help="保存する最小角度(度)")
    parser.add_argument("-A", "--max-angle", type=float, default=360,
                        help="保存する最大角度(度)")
    parser.add_argument("-d", "--min-distance", type=float, default=sys.float_info.min,  # noqa
                        help="保存する最小距離")
    parser.add_argument("-D", "--max-distance", type=float, default=4000,
                        help="保存する最大距離")
    parser.add_argument("-f", "--forward-angle", type=float, default=0.0,
                        help="前方方向を表す角度")
    parser.add_argument("-s", "--angle-direction", type=int, default=COUNTER_CLOCKWISE,  # noqa
                        help="角度が増加する方向 (1: 時計回り, -1: 反時計回り)")  # noqa
    parser.add_argument("-p", "--rotate-plot", type=float, default=0.0,
                        help="プロットを回転する角度(度)")  # noqa
    parser.add_argument("-t", "--threaded", action='store_true', help="スレッドモードで実行")

    # コマンドライン引数を読み込む
    args = parser.parse_args()
    
    help = []
    if args.rate < 1:
        help.append("-r/--rate は 1以上で指定してください")
        
    if args.number < 1:
        help.append("-n/--number は 1以上で指定してください")
        
    if args.min_distance < 0:
        help.append("-d/--min-distance は 0以上で指定してください")

    if args.max_distance <= 0:
        help.append("-D/--max-distance は 0より大きく指定してください")
        
    if args.min_angle < 0 or args.min_angle > 360:
        help.append("-a/--min-angle は 0以上360以下で指定してください")

    if args.max_angle <= 0 or args.max_angle > 360:
        help.append("-A/--max-angle は 0より大きく360以下で指定してください")
      
    if args.forward_angle < 0 or args.forward_angle > 360:
        help.append("-f/--forward-angle は 0以上360以下で指定してください")
        
    if args.angle_direction != CLOCKWISE and \
       args.angle_direction != COUNTER_CLOCKWISE:
        help.append("-s/--angle-direction は 1(時計回り) または -1(反時計回り) を指定してください")  # noqa
        
    if args.rotate_plot < 0 or args.rotate_plot > 360:
        help.append("-p/--rotate-plot は 0以上360以下で指定してください")
        
    if len(help) > 0:
        parser.print_help()
        for h in help:
            print("  " + h)
        sys.exit(1)
        
    lidar_thread = None
    lidar = None
    
    try:
        scan_count = 0
        seconds_per_scan = 1.0 / args.rate
        scan_time = time.time() + seconds_per_scan

        #
        # Lidar パーツの生成
        #
        lidar = RPLidar2(
            min_angle=args.min_angle, max_angle=args.max_angle,
            min_distance=args.min_distance, max_distance=args.max_distance,
            forward_angle=args.forward_angle,
            angle_direction=args.angle_direction,
            batch_ms=1000.0/args.rate)
        
        #
        # Lidar プロッターの生成
        #
        plotter = LidarPlot2(plot_type=LidarPlot2.PLOT_TYPE_CIRCLE,
                             max_dist=args.max_distance,
                             angle_direction=args.angle_direction,
                             rotate_plot=args.rotate_plot,
                             background_color=(32, 32, 32),
                             border_color=(128, 128, 128),
                             point_color=(64, 255, 64))        
        #
        # スレッドを起動し、ウィンドウを開く
        #
        cv2.namedWindow("lidar")
        if args.threaded:
            lidar_thread = Thread(target=lidar.update, args=())
            lidar_thread.start()
            cv2.startWindowThread()
        
        while scan_count < args.number:
            start_time = time.time()

            # スキャンを出力
            scan_count += 1

            # 最新のスキャンを取得して描画
            if args.threaded:
                measurements = lidar.run_threaded()
            else:
                measurements = lidar.run()
            
            img = plotter.run(measurements)
            
            # 画像をウィンドウに表示
            cv2img = convert_from_image_to_cv2(img)
            cv2.imshow("lidar", cv2img)
            
            if not args.threaded:
                key = cv2.waitKey(1) & 0xFF
                if 27 == key or key == ord('q') or key == ord('Q'):
                    break

            # バックグラウンドスレッドに実行を譲る
            sleep_time = seconds_per_scan - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                time.sleep(0)  # 他のスレッドに実行を譲る

    except KeyboardInterrupt:
        print('早期終了します。')
    except Exception as e:
        print(e)
        exit(1)
    finally:
        if lidar is not None:
            lidar.shutdown()
            plotter.shutdown()
            cv2.destroyAllWindows()
        if lidar_thread is not None:
            lidar_thread.join()  # スレッド終了を待つ
