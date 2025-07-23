import argparse
from functools import reduce
import logging
import operator
import threading
import time

import pynmea2
import serial
import utm

from donkeycar.parts.serial_port import SerialPort
from donkeycar.parts.text_writer import CsvLogger

logger = logging.getLogger(__name__)


class GpsNmeaPositions:
    """NMEAセンテンスの配列を(x, y)座標の配列へ変換するDonkeycar部品。

    Args:
        debug (bool): デバッグ情報を出力するかどうか。
    """
    def __init__(self, debug=False):
        self.debug = debug

    def run(self, lines):
        positions = []
        if lines:
            for ts, nmea in lines:
                position = parseGpsPosition(nmea, self.debug)
                if position:
                    # (ts, x, y) を出力。経度をx、緯度をyとする
                    positions.append((ts, position[0], position[1]))
        return positions

    def update(self):
        pass

    def run_threaded(self, lines):
        return self.run(lines)

class GpsLatestPosition:
    """最新の有効なGPS位置を返す部品。

    Args:
        debug (bool): デバッグ情報を出力するかどうか。
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.position = None

    def run(self, positions):
        if positions is not None and len(positions) > 0:
            self.position = positions[-1]
        return self.position

class GpsPosition:
    """シリアルポートからNMEA行を読み取り、位置情報へ変換するDonkeycar部品。

    Args:
        serial (SerialPort): 使用するシリアルポート。
        debug (bool): デバッグ情報を出力するかどうか。
    """
    def __init__(self, serial:SerialPort, debug = False) -> None:
        self.line_reader = SerialLineReader(serial)
        self.debug = debug
        self.position_reader = GpsNmeaPositions()
        self.position = None
        self._start()

    def _start(self):
        # 少なくとも1つのGPS位置が得られるまで待つ
        while self.position is None:
            logger.info("GPSの位置情報を待機中")
            self.position = self.run()

    def run_once(self, lines):
        positions = self.GpsNmeaPositions.run(lines)
        if positions is not None and len(positions) > 0:
            self.position = positions[-1]
            if self.debug:
                logger.info(f"UTM東距離 = {self.position[0]}, UTM北距離 = {self.position[1]}")
        return self.position

    def run(self):
        lines = line_reader.run()
        return self.run_once(lines)

    def run_threaded(self):
        lines = line_reader.run_threaded()
        return self.run_once(lines)

    def update(self):
        self.line_reader.update()

    def shutdown(self):
        return self.line_reader.shutdown()


class GpsPlayer:
    """NMEAセンテンスのログを再生する部品。

    NMEAロガーで記録されたセンテンスを再生し、必要に応じて現在のNMEAを
    そのまま通過させます。

    Args:
        nmea_logger (CsvLogger): 再生に使用するNMEAログ。
    """
    def __init__(self, nmea_logger:CsvLogger):
        self.nmea = nmea_logger
        self.index = -1
        self.starttime = None
        self.running = False

    def start(self):
        self.running = True
        self.starttime = None  # run()が最初に呼ばれたときに設定される
        self.index = -1
        return self

    def stop(self):
        self.running = False
        return self

    def run(self, playing, nmea_sentences):
        """自動運転モードかつ再生中であればNMEAを再生する。

        Args:
            playing (bool): 録音済みNMEAを再生する場合は``True``。
            nmea_sentences (List[str]): 再生しない場合にそのまま通過させるライブNMEA。

        Returns:
            Tuple[bool, List[str]]: ``playing``の状態と結果のNMEAセンテンス一覧。
        """
        if self.running and playing:
            # 再生中であれば記録済みNMEAを返す
            nmea = self.run_once(time.time())
            return True, nmea

        # 再生していない場合は受け取ったNMEAをそのまま返す
        return False, nmea_sentences

    def run_once(self, now):
        """指定された時刻までのNMEAセンテンスを収集する。

        Args:
            now (float): 収集を行う現在時刻（秒）。

        Returns:
            List[Tuple[float, str]]: 収集されたNMEAセンテンス。
        """
        nmea_sentences = []
        if self.running:
            # starttimeが未設定ならリセット
            if self.starttime is None:
                print("GPSプレーヤーの開始時刻をリセットします。")
                self.starttime = now

            # 最初のNMEAセンテンスを取得して記録時間を得る
            start_nmea = self.nmea.get(0)
            if start_nmea is not None:
                #
                # 次のNMEAセンテンスを取得し、時間内なら再生する。
                # 次のセンテンスが無い場合は最初に戻る。
                #
                start_nmea_time = float(start_nmea[0])
                offset_nmea_time = 0
                within_time = True
                while within_time:
                    next_nmea = None
                    if self.index >= self.nmea.length():
                        # 最後まで到達したら最初に戻る
                        self.index = 0
                        self.starttime += offset_nmea_time
                        next_nmea = self.nmea.get(0)
                    else:
                        next_nmea = self.nmea.get(self.index + 1)

                    if next_nmea is None:
                        self.index += 1  # 無効なセンテンスを飛ばす
                    else:
                        next_nmea_time = float(next_nmea[0])
                        offset_nmea_time = (next_nmea_time - start_nmea_time)
                        next_nmea_time = self.starttime + offset_nmea_time
                        within_time = next_nmea_time <= now
                        if within_time:
                            nmea_sentences.append((next_nmea_time, next_nmea[1]))
                            self.index += 1
        return nmea_sentences


def parseGpsPosition(line, debug=False):
    """GPSモジュールからの1行を解析して位置を得る。

    Args:
        line (str): 解析するNMEAセンテンス。
        debug (bool): デバッグ情報を出力するかどうか。

    Returns:
        Optional[Tuple[float, float]]: UTM座標(東距離, 北距離)。解析できない場合は``None``。
    """
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
        
    #
    # '$'で始まりチェックサムで終わる必要がある
    #
    if '$' != line[0]:
        logger.info("NMEAに行頭がありません")
        return None
        
    if '*' != line[-3]:
        logger.info("NMEAにチェックサムがありません")
        return None
        
    nmea_checksum = parse_nmea_checksum(line) # ## チェックサムの16進数値
    nmea_msg = line[1:-3]      # '$'と'*##'を除いたメッセージ
    nmea_parts = nmea_msg.split(",")
    message = nmea_parts[0]
    if (message == "GPRMC") or (message == "GNRMC"):   
        #
        # '$GPRMC,003918.00,A,3806.92281,N,12235.64362,W,0.090,,060322,,,D*67'のような形式
        # GPRMC = 推奨される最小限のGPS/Transitデータ
        #
        # チェックサムが正しいか確認
        #
        calculated_checksum = calculate_nmea_checksum(line)
        if nmea_checksum != calculated_checksum:
            logger.info(f"NMEAのチェックサムが一致しません: {nmea_checksum} != {calculated_checksum}")
            return None

        #
        # 既知のパーサと比較してこのパーサを検証する
        # TODO: 多くの例外が発生するようならライブラリに切り替える。
        #       このパーサが十分機能するなら軽量なのでそのまま使用する。
        #
        if debug:
            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError as e:
                logger.error('NMEAの解析エラー: {}'.format(e))
                return None

        # GPSの位置確定データを読む方法も有効
        if nmea_parts[2] == 'V':
            # V = Warning。おそらく衛星が見えていない状態...
            logger.info("GPS受信機の警告: 位置が無効です。無効な位置を無視します。")
        else:
            #
            # テキスト形式のNMEA位置を度に変換
            #
            longitude = nmea_to_degrees(nmea_parts[5], nmea_parts[6])
            latitude = nmea_to_degrees(nmea_parts[3], nmea_parts[4])

            if debug:
                if msg.longitude != longitude:
                    print(f"経度が一致しません {msg.longitude} != {longitude}")
                if msg.latitude != latitude:
                    print(f"緯度が一致しません {msg.latitude} != {latitude}")

            #
            # 度単位の位置をローカルのメートル単位に変換
            #
            utm_position = utm.from_latlon(latitude, longitude)
            if debug:
                logger.info(f"UTM東距離 = {utm_position[0]}, UTM北距離 = {utm_position[1]}")
            
            # 緯度経度を浮動小数点の度単位で返す
            return float(utm_position[0]), float(utm_position[1])
    else:
        # 位置情報ではないメッセージ、または無効な文字列
        # print(f"行を無視します: {line}")
        pass
    return None


def parse_nmea_checksum(nmea_line):
    """NMEA行からチェックサムを取得する。

    Args:
        nmea_line (str): 先頭が``$``で終端が``*##``を含む完全なNMEA行。

    Returns:
        int: 本文から計算したチェックサム。
    """
    return int(nmea_line[-2:], 16)  # チェックサムの16進数値
    
    
def calculate_nmea_checksum(nmea_line):
    """NMEA行の本文からチェックサムを計算する。

    Args:
        nmea_line (str): ``$``で始まり``*##``で終わる完全なNMEA行。

    Returns:
        int: 計算されたチェックサム。
    """
    # 
    # メッセージ中の全ての文字の排他的論理和を取り、1バイトのチェックサムを算出する
    # 先頭の`$`と末尾の`*##`は計算に含めない
    #
    return reduce(operator.xor, map(ord, nmea_line[1:-3]), 0)


def nmea_to_degrees(gps_str, direction):
    """GPS座標文字列を度単位の浮動小数点数に変換する。

    Args:
        gps_str (str): ``DDDMM.MMMMM``形式の座標文字列。
        direction (str): 北緯/南緯または東経/西経を示す文字。``N`` ``S`` ``E`` ``W`` のいずれか。

    Returns:
        float: 度単位で表した座標値。
    """
    if not gps_str or gps_str == "0":
        return 0
        
    #
    # 度と分を取り出し、分を結合する
    #
    parts = gps_str.split(".")
    degrees_str = parts[0][:-2]        # 0〜3桁の数値になる
    minutes_str = parts[0][-2:]        # 常に2桁になる
    if 2 == len(parts):
        minutes_str += "." + parts[1]  # 分の整数部と小数部を結合する
    
    #
    # 度を浮動小数点数に変換
    #
    degrees = 0.0
    if len(degrees_str) > 0:
        degrees = float(degrees_str)
    
    #
    # 分を度に変換
    #
    minutes = 0.0
    if len(minutes_str) > 0:
        minutes = float(minutes_str) / 60
        
    #
    # 度と分を合計し、方位で符号を適用する
    #
    return (degrees + minutes) * (-1 if direction in ['W', 'S'] else 1)
    

#
# 以下は `__main__` 実行時の簡易テストコードで、位置のログ取得や
# ウェイポイントの記録を行うことができます
#
if __name__ == "__main__":
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import sys
    import readchar
    from donkeycar.parts.serial_port import SerialPort, SerialLineReader


    def stats(data):
        """浮動小数点数リストの統計量を計算する。

        Args:
            data (List[float]): 対象データ。

        Returns:
            Optional[Stats]: ``data``が空の場合は``None``。
        """
        if not data:
            return None
        count = len(data)
        min = None
        max = None
        sum = 0
        for x in data:
            if min is None or x < min:
                min = x
            if max is None or x > max:
                max = x
            sum += x
        mean = sum / count
        sum_errors_squared = 0
        for x in data:
            error = x - mean
            sum_errors_squared += (error * error)
        std_deviation = math.sqrt(sum_errors_squared / count)
        return Stats(count, sum, min, max, mean, std_deviation)

    class Stats:
        """データ集合の統計量を表すクラス。"""
        def __init__(self, count, sum, min, max, mean, std_deviation):
            self.count = count
            self.sum = sum
            self.min = min
            self.max = max
            self.mean = mean
            self.std_deviation = std_deviation

    class Waypoint:
        """複数のサンプルから作成されるウェイポイントを表す。

        非軸合わせ（回転した）楕円体としてモデル化し、GPSのように
        ノイズの多いソースから得た点群を扱う。
        """
        def __init__(self, samples, nstd=1.0):
            """サンプルから楕円体をフィッティングする。

            Args:
                samples (List[Tuple[float, float, float]]): 位置サンプル一覧。
                nstd (float): 標準偏差の何倍で楕円を描くか。
            """
            
            # サンプルからx軸とy軸の値を抽出する
            self.x = [w[1] for w in samples]
            self.y = [w[2] for w in samples]
            
            # 各軸の統計値を計算する
            self.x_stats = stats(self.x)
            self.y_stats = stats(self.y)

            #
            # 回転楕円を用いてサンプル点群を近似する。
            # xとyは互いに独立ではないため、非軸合わせの楕円を利用する。
            #
            def eigsorted(cov):
                """共分散行列から固有値と固有ベクトルを取得し、大きい順に並べる。"""
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                return eigenvalues[order], eigenvectors[:, order]

            # x軸とy軸の共分散行列を計算する
            self.cov = np.cov(self.x, self.y)

            # 共分散行列から固有値と固有ベクトルを取得する
            self.eigenvalues, self.eigenvectors = eigsorted(self.cov)

            # 標準偏差の倍率に応じた楕円を計算する
            self.theta = np.degrees(np.arctan2(*self.eigenvectors[:, 0][::-1]))
            self.width, self.height = 2 * nstd * np.sqrt(self.eigenvalues)

        def is_inside(self, x, y):
            """指定された(x, y)がウェイポイントの楕円内にあるか判定する。"""
            # if (x >= self.x_stats.min) and (x <= self.x_stats.max):
            #     if (y >= self.y_stats.min) and (y <= self.y_stats.max):
            #         return True
            # return False
            # if (x >= (self.x_stats.mean - self.x_stats.std_deviation)) and (x <= (self.x_stats.mean + self.x_stats.std_deviation)):
            #     if (y >= (self.y_stats.mean - self.y_stats.std_deviation)) and (y <= (self.y_stats.mean + self.y_stats.std_deviation)):
            #         return True
            # return False
            cos_theta = math.cos(self.theta)
            sin_theta = math.sin(self.theta)
            x_translated = x - self.x_stats.mean
            y_translated = y - self.y_stats.mean
            #
            # テスト点を楕円の座標系（中心）へ移動し、回転させて通常の楕円判定を行う
            #
            part1 = ((cos_theta * x_translated + sin_theta * y_translated) / self.width)**2
            part2 = ((sin_theta * x_translated - cos_theta * y_translated) / self.height)**2
            return (part1 + part2) <= 1

        def is_in_range(self, x, y):
            """収集したサンプル範囲内にあるかを判定する。"""
            return (x >= self.x_stats.min) and \
                   (x <= self.x_stats.max) and \
                   (y >= self.y_stats.min) and \
                   (y <= self.y_stats.max)
            
        def is_in_std(self, x, y, std_multiple=1.0):
            """各軸の標準偏差の指定倍以内か判定する。"""
            x_std = self.x_stats.std_deviation * std_multiple
            y_std = self.y_stats.std_deviation * std_multiple
            return (x >= (self.x_stats.mean - x_std)) and \
                   (x <= (self.x_stats.mean + x_std)) and \
                   (y >= (self.y_stats.mean - y_std)) and \
                   (y <= (self.y_stats.mean + y_std))

        def show(self):
            """ウェイポイントの楕円を描画して表示する。"""
            from matplotlib.patches import Ellipse
            import matplotlib.pyplot as plt
            ax = plt.subplot(111, aspect='equal')
            self.plot()
            plt.show()
            
        def plot(self):
            """ウェイポイントの楕円を描画する。"""
            from matplotlib.patches import Ellipse, Rectangle
            import matplotlib.pyplot as plt
            # MatplotlibのFigureとAxisを定義
            ax = plt.subplot(111, aspect='equal')
            
            # 収集した値をプロット
            plt.scatter(self.x, self.y)
            
            # 重心をプロット
            plt.plot(self.x_stats.mean, self.y_stats.mean, marker="+", markeredgecolor="green", markerfacecolor="green")
            
            # 測定範囲をプロット
            bounds = Rectangle(
                (self.x_stats.min, self.y_stats.min), 
                self.x_stats.max - self.x_stats.min, 
                self.y_stats.max - self.y_stats.min,
                alpha=0.5,
                edgecolor='red',
                fill=False,
                visible=True)
            ax.add_artist(bounds)

            # 楕円をプロット
            ellipse = Ellipse(xy=(self.x_stats.mean, self.y_stats.mean),
                          width=self.width, height=self.height,
                          angle=self.theta)
            ellipse.set_alpha(0.25)
            ellipse.set_facecolor('green')
            ax.add_artist(ellipse)

    def is_in_waypoint_range(waypoints, x, y):
        """座標がサンプル範囲内か判定する。

        Args:
            waypoints (List[Waypoint]): ウェイポイントのリスト。
            x (float): 検査するX座標。
            y (float): 検査するY座標。

        Returns:
            Tuple[bool, int]: 範囲内かどうかとヒットしたウェイポイントのインデックス。
        """
        i = 0
        for waypoint in waypoints:
            if waypoint.is_in_range(x, y):
                return True, i
            i += 1
        return False, -1

    def is_in_waypoint_std(waypoints, x, y, std):
        """座標が指定標準偏差以内か判定する。

        Args:
            waypoints (List[Waypoint]): ウェイポイントのリスト。
            x (float): 検査するX座標。
            y (float): 検査するY座標。
            std (float): 許容する標準偏差の倍率。

        Returns:
            Tuple[bool, int]: 判定結果と該当ウェイポイントのインデックス。
        """
        i = 0
        for waypoint in waypoints:
            if waypoint.is_in_std(x, y, std):
                return True, i
            i += 1
        return False, -1

    def is_in_waypoint(waypoints, x, y):
        """座標が楕円内にあるか判定する。

        Args:
            waypoints (List[Waypoint]): ウェイポイントのリスト。
            x (float): 検査するX座標。
            y (float): 検査するY座標。

        Returns:
            Tuple[bool, int]: 判定結果と該当ウェイポイントのインデックス。
        """
        i = 0
        for waypoint in waypoints:
            if waypoint.is_inside(x, y):
                return True, i
            i += 1
        return False, -1


    def plot(waypoints):
        """ウェイポイントの楕円を描画する。"""
        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt
        ax = plt.subplot(111, aspect='equal')
        for waypoint in waypoints:
            waypoint.plot()
        plt.show()


    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--serial", type=str, required=True, help="シリアルポートのアドレス (例: '/dev/tty.usbmodem1411')")
    parser.add_argument("-b", "--baudrate", type=int, default=9600, help="シリアルポートのボーレート")
    parser.add_argument("-t", "--timeout", type=float, default=0.5, help="シリアルポートのタイムアウト(秒)")
    parser.add_argument("-sp", '--samples', type=int, default=5, help="ウェイポイントごとのサンプル数")
    parser.add_argument("-wp", "--waypoints", type=int, default=0, help="収集するウェイポイント数。0なら位置のみ記録")
    parser.add_argument("-nstd", "--nstd", type=float, default=1.0, help="楕円の標準偏差倍率")
    parser.add_argument("-th", "--threaded", action='store_true', help="スレッドモードで実行する")
    parser.add_argument("-db", "--debug", action='store_true', help="詳細ログを有効にする")
    args = parser.parse_args()

    if args.waypoints < 0:
        print("ウェイポイントを収集するには1以上を指定し、0を指定すると位置のみを記録します")
        parser.print_help()
        sys.exit(0)

    if args.samples <= 0:
        print("ウェイポイントあたりのサンプル数は0より大きくなければなりません")
        parser.print_help()
        sys.exit(0)

    if args.nstd <= 0:
        print("ウェイポイントの倍率は0より大きくなければなりません")
        parser.print_help()
        sys.exit(0)

    if args.timeout <= 0:
        print("タイムアウトは0より大きくなければなりません")
        parser.print_help()
        sys.exit(0)

    update_thread = None
    line_reader = None

    waypoint_count = args.waypoints      # パス上のウェイポイント数
    samples_per_waypoint = args.samples  # 各ウェイポイントの測定回数
    waypoints = []
    waypoint_samples = []

    try:
        serial_port = SerialPort(args.serial, baudrate=args.baudrate, timeout=args.timeout)
        line_reader = SerialLineReader(serial_port, max_lines=args.samples, debug=args.debug)
        position_reader = GpsNmeaPositions(args.debug)

        #
        # スレッド処理を開始し、プロットを表示するためのスレッドウィンドウを開く
        #
        if args.threaded:
            update_thread = threading.Thread(target=line_reader.update, args=())
            update_thread.start()

        def read_gps():
            """GPSから位置情報を取得する補助関数。"""
            lines = line_reader.run_threaded() if args.threaded else line_reader.run()
            positions = position_reader.run(lines)
            return positions

        ts = time.time()
        state = "prompt" if waypoint_count > 0 else ""
        while line_reader.running:
            readings = read_gps()
            if readings:
                print("")
                if state == "prompt":
                    print(f"ウェイポイント#{len(waypoints)+1}に移動し、スペースバーとEnterでサンプリング開始。他のキーで記録のみを開始します")
                    state = "move"
                elif state == "move":
                    key_press = readchar.readchar()  # キー入力を取得
                    if key_press == ' ':
                        waypoint_samples = []
                        line_reader.clear()  # バッファされた読み取りを破棄
                        state = "sampling"
                    else:
                        state = ""  # 単に記録を開始
                elif state == "sampling":
                    waypoint_samples += readings
                    count = len(waypoint_samples)
                    print(f"これまでに{count}件収集しました...")
                    if count > samples_per_waypoint:
                        print(f"...完了。ウェイポイント#{len(waypoints)+1}のサンプルを{count}件収集しました")
                        #
                        # ウェイポイントを回転楕円体としてモデル化する。
                        # これは測定点群の95%信頼区間を表す。
                        #
                        waypoint = Waypoint(waypoint_samples, nstd=args.nstd)
                        waypoints.append(waypoint)
                        if len(waypoints) < waypoint_count:
                            state = "prompt"
                        else:
                            state = "test_prompt"
                            if args.debug:
                                plot(waypoints)
                elif state == "test_prompt":
                    print("ウェイポイントの記録が完了しました。周囲を歩き回り、ウェイポイントに入ったときに確認してください")
                    state = "test"
                elif state == "test":
                    for ts, x, y in readings:
                        print(f"現在位置は({x}, {y})です")
                        hit, index = is_in_waypoint_range(waypoints, x, y)
                        if hit:
                            print(f"ウェイポイント{index + 1}のサンプル範囲内にいます")
                        std_deviation = 1.0
                        hit, index = is_in_waypoint_std(waypoints, x, y, std_deviation)
                        if hit:
                            print(f"ウェイポイント{index + 1}の中心から{std_deviation}標準偏差以内です")
                        hit, index = is_in_waypoint(waypoints, x, y)
                        if hit:
                            print(f"ウェイポイント{index + 1}の楕円内にいます")
                else:
                    # 測位結果のみ記録する
                    for position in readings:
                        ts, x, y = position
                        print(f"現在位置は({x}, {y})です")
            else:
                if time.time() > (ts + 0.5):
                    print(".", end="")
                    ts = time.time()
    finally:
        if line_reader:
            line_reader.shutdown()
        if update_thread is not None:
            update_thread.join()  # スレッド終了待ち

