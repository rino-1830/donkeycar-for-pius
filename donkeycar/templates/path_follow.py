#!/usr/bin/env python3
"""ドンキーカーで走行しながら経路を記録し、
オートパイロットでその経路を辿るためのスクリプト。
ホイールエンコーダーや Intel T265 に対応。

Usage:
    manage.py (drive) [--js] [--log=INFO] [--camera=(single|stereo)]


Options:
    -h --help          この画面を表示する。
    --js               物理ジョイスティックを使用する。
    -f --file=<file>   1 行につき 1 つの tub ファイルへのパスを含むテキストファイル。複数回指定可能。
    --meta=<key:value> この走行に関するメタ情報を表すキーと値の文字列。複数回指定可能。

起動時はユーザーモード。
- 経路を"学習"する手順:
  - ``python manage.py drive`` を実行。
  - ユーザーモードで開始されるので手動運転しながらウェイポイントを記録。
  - スタート地点とゴール地点はなるべく近づけて経路を記録。
  - 経路が気に入らない場合はリセットボタンを押すと、原点をリセットし経路をメモリから消去。車両を物理的な原点に戻して再度記録開始。
  - 経路が気に入ったらセーブボタンを押して保存。
- オートパイロットの手順:
  - 経路を記録するか読み込む(ロードボタンを選択)。
  - ジョイスティックまたは Web UI からオートパイロットモードを選択すると、車両は記録済みの経路を走行し始める。
  - ユーザーモードに戻すとオートパイロットを停止。
  - 保存済み経路を再利用するには次の操作を行う:
    - リセットボタンで経路を消去し原点をリセット。
    - ロードボタンで経路を読み込む。
    - 車両を物理的な原点に戻す。

"""
from distutils.log import debug
import os
import logging

#
# tensorflow の後にインポートすると問題が起きるのを避けるため、早めに cv2 をインポートする
# 詳細は https://github.com/opencv/opencv/issues/14884#issuecomment-599852128 を参照
#
import time

try:
    import cv2
except:
    pass


from docopt import docopt

import donkeycar as dk
from donkeycar.parts.controller import JoystickController
from donkeycar.parts.path import CsvThrottlePath, PathPlot, CTE, PID_Pilot, \
    PlotCircle, PImage, OriginOffset
from donkeycar.parts.transform import PIDController
from donkeycar.parts.kinematics import TwoWheelSteeringThrottle
from donkeycar.templates.complete import add_odometry, add_camera, \
    add_user_controller, add_drivetrain, add_simulator, add_imu, DriveMode, \
    UserPilotCondition, ToggleRecording
from donkeycar.parts.logger import LoggerPart
from donkeycar.parts.transform import Lambda
from donkeycar.parts.explode import ExplodeDict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def drive(cfg, use_joystick=False, camera_type='single'):
    """複数のパーツからロボット車両を組み立てて走行させる。

    それぞれのパーツは Vehicle ループ内でジョブとして実行され、コンストラクター
    の ``threaded`` フラグによって ``run`` か ``run_threaded`` が呼び出される。
    すべてのパーツは ``cfg.DRIVE_LOOP_HZ`` で指定されたフレームレートで順に更新さ
    れ、各パーツがタイムリーに処理を終えることを前提としている。パーツは出力名と
    入力名を持つことができ、フレームワークが同名の入出力を自動的に接続する。

    Args:
        cfg: 設定オブジェクト。
        use_joystick (bool): ジョイスティックを使うかどうか。
        camera_type (str): ``single`` または ``stereo`` を指定。

    """

    is_differential_drive = cfg.DRIVE_TRAIN_TYPE.startswith("DC_TWO_WHEEL")

    # 車両オブジェクトを初期化
    V = dk.vehicle.Vehicle()

    if cfg.HAVE_SOMBRERO:
        from donkeycar.utils import Sombrero
        s = Sombrero()
   
    # コンソールログ用にロギングを初期化
    if cfg.HAVE_CONSOLE_LOGGING:
        logger.setLevel(logging.getLevelName(cfg.LOGGING_LEVEL))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(cfg.LOGGING_FORMAT))
        logger.addHandler(ch)

    if cfg.HAVE_MQTT_TELEMETRY:
        from donkeycar.parts.telemetry import MqttTelemetry
        tel = MqttTelemetry(cfg)

    #
    # シミュレータを使用する場合はここで設定
    #
    add_simulator(V, cfg)

    #
    # IMU 設定
    #
    add_imu(V, cfg)

    #
    # オドメトリ・タコメータ・速度制御
    #
    if cfg.HAVE_ODOM:
        #
        # エンコーダ・オドメトリ・姿勢推定を設定
        #
        add_odometry(V, cfg)
    else:
        # オドメトリが無いことを示すため T265 にはキャリブレーションを渡さない
        cfg.WHEEL_ODOM_CALIB = None

        # RS_T265 パーツの入力要件を満たすためのダミーパーツ
        class NoOdom():
            def run(self):
                return 0.0

        V.add(NoOdom(), outputs=['enc/vel_m_s'])

    if cfg.HAVE_T265:
        from donkeycar.parts.realsense2 import RS_T265
        if cfg.HAVE_ODOM and not os.path.exists(cfg.WHEEL_ODOM_CALIB):
            print("T265 でオドメトリを使用する場合は json ファイルが必要です。"
                  "テンプレートにサンプルがあります。")
            print("cp donkeycar/donkeycar/templates/calibration_odometry.json .")
            exit(1)

        #
        # 注意: T265 の画像出力は Python API では壊れており修正されないため、画像は出力できない
        #
        rs = RS_T265(image_output=False, calib_filename=cfg.WHEEL_ODOM_CALIB, device_id=cfg.REALSENSE_T265_ID)
        V.add(rs, inputs=['enc/vel_m_s'], outputs=['rs/pos', 'rs/vel', 'rs/acc'], threaded=True)

        #
        # realsense T265 の位置ストリームから 2 次元座標を取り出してマップに使用する。
        # T265 の AR 座標系を、東が正の X、北が正の Y となる一般的な上から見た POSE 座標系に変換する。
        #
        class PosStream:
            def run(self, pos):
                # y: 上向き, x: 右向き, z: 前後(前進が負)
                return -pos.z, -pos.x

        V.add(PosStream(), inputs=['rs/pos'], outputs=['pos/x', 'pos/y'])

    #
    # GPS の出力 ['pos/x', 'pos/y']
    #
    gps_player = add_gps(V, cfg)

    #
    # メインカメラを設定
    #
    add_camera(V, cfg, camera_type)

    #
    # ユーザー入力コントローラーを追加
    # - Web コントローラーを追加する
    # - 設定されていればジョイスティックコントローラーも追加
    #
    has_input_controller = hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    ctr = add_user_controller(V, cfg, use_joystick, input_image = 'map/image')

    #
    # Web ボタンを個別のキーと値に展開
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    #
    # このパーツは車両を原点にリセットする。既知の原点に車両を置き、
    # コントローラーの ``cfg.RESET_ORIGIN_BTN`` を押すとマッピングにオフセットを
    # 与えることができる。
    #
    origin_reset = OriginOffset(cfg.PATH_DEBUG)
    V.add(origin_reset, inputs=['pos/x', 'pos/y', 'cte/closest_pt'], outputs=['pos/x', 'pos/y', 'cte/closest_pt'])


    #
    # ユーザーモードとオートパイロットモードの実行条件を維持
    #
    V.add(UserPilotCondition(),
          inputs=['user/mode', "cam/image_array", "cam/image_array"],
          outputs=['run_user', "run_pilot", "ui/image_array"])


    # 経路オブジェクト。距離が変化し、かつ ``cfg.PATH_MIN_DIST`` メートル以上進むと経路を記録する。
    # フォローモード時は記録しない（詳細は後述）。
    path = CsvThrottlePath(min_dist=cfg.PATH_MIN_DIST)
    V.add(path, inputs=['recording', 'pos/x', 'pos/y', 'user/throttle'], outputs=['path', 'throttles'])

    #
    # 姿勢をログに記録
    #
    # if cfg.DONKEY_GYM:
    #     lpos = LoggerPart(inputs=['dist/left', 'dist/right', 'dist', 'pos/pos_x', 'pos/pos_y', 'yaw'], level="INFO", logger="simulator")
    #     V.add(lpos, inputs=lpos.inputs)
    # if cfg.HAVE_ODOM:
    #     if cfg.HAVE_ODOM_2:
    #         lpos = LoggerPart(inputs=['enc/left/distance', 'enc/right/distance', 'enc/left/timestamp', 'enc/right/timestamp'], level="INFO", logger="odometer")
    #         # V.add(lpos, inputs=lpos.inputs)
    #     lpos = LoggerPart(inputs=['enc/distance', 'enc/timestamp'], level="INFO", logger="odometer")
    #     V.add(lpos, inputs=lpos.inputs)
    #     lpos = LoggerPart(inputs=['pos/x', 'pos/y', 'pos/steering'], level="INFO", logger="kinematics")
    #     V.add(lpos, inputs=lpos.inputs)

    def save_path():
        if path.length() > 0:
            if path.save(cfg.PATH_FILENAME):
                print("その経路を {} に保存しました".format(cfg.PATH_FILENAME))

                # 記録された GPS データも保存
                if gps_player:
                    gps_player.nmea.save()
            else:
                print("経路を保存できませんでした。myconfig.py の PATH_FILENAME が正しいか確認してください")
        else:
            print("保存する経路がありません。まず経路を記録してください")

    def load_path():
        if os.path.exists(cfg.PATH_FILENAME) and path.load(cfg.PATH_FILENAME):
           print("経路を {} から読み込みました".format(cfg.PATH_FILENAME))

           # 経路を読み込んだので、保存されていた GPS データも読み込む
           if gps_player:
               gps_player.stop().nmea.load()
               gps_player.start()
        else:
           print("経路を読み込めませんでした。事前に保存されているか確認してください")

    def erase_path():
        origin_reset.reset_origin()
        if path.reset():
            print("原点と経路をリセットしました。新しい経路を記録できます")
            if gps_player:
                gps_player.stop().nmea.reset()
        else:
            print("原点をリセットしました。新しい経路を記録できます")

    def reset_origin():
        """有効な姿勢を (0, 0) にリセットする。"""
        origin_reset.reset_origin()
        print("現在位置を原点としてリセットしました")

        # 記録されていた GPS データを最初から再生し直す
        if gps_player:
            gps_player.start()


    # 経路が読み込まれている場合はフォローモードになり、記録は行わない
    if os.path.exists(cfg.PATH_FILENAME):
        load_path()

    # マッピング用の画像オブジェクト
    img = PImage(clear_each_frame=True)
    V.add(img, outputs=['map/image'])

    # PathPlot は画像上に経路を描画する

    plot = PathPlot(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET)
    V.add(plot, inputs=['map/image', 'path'], outputs=['map/image'])

    # 経路と現在位置からクロストラックエラーを計算
    cte = CTE(look_ahead=cfg.PATH_LOOK_AHEAD, look_behind=cfg.PATH_LOOK_BEHIND, num_pts=cfg.PATH_SEARCH_LENGTH)
    V.add(cte, inputs=['path', 'pos/x', 'pos/y', 'cte/closest_pt'], outputs=['cte/error', 'cte/closest_pt'], run_condition='run_pilot')

    # クロストラックエラーと PID 定数を用いて経路へ復帰するよう操舵を計算
    pid = PIDController(p=cfg.PID_P, i=cfg.PID_I, d=cfg.PID_D)
    pilot = PID_Pilot(pid, cfg.PID_THROTTLE, cfg.USE_CONSTANT_THROTTLE, min_throttle=cfg.PID_THROTTLE)
    V.add(pilot, inputs=['cte/error', 'throttles', 'cte/closest_pt'], outputs=['pilot/steering', 'pilot/throttle'], run_condition="run_pilot")

    def dec_pid_d():
        pid.Kd -= cfg.PID_D_DELTA
        logging.info("pid: d- %f" % pid.Kd)

    def inc_pid_d():
        pid.Kd += cfg.PID_D_DELTA
        logging.info("pid: d+ %f" % pid.Kd)

    def dec_pid_p():
        pid.Kp -= cfg.PID_P_DELTA
        logging.info("pid: p- %f" % pid.Kp)

    def inc_pid_p():
        pid.Kp += cfg.PID_P_DELTA
        logging.info("pid: p+ %f" % pid.Kp)


    recording_control = ToggleRecording(cfg.AUTO_RECORD_ON_THROTTLE, cfg.RECORD_DURING_AI)
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])


    #
    # ユーザー操作を扱うボタン類を追加する
    # ボタン名は設定で決まり、ゲームコントローラー(ジョイスティック)または
    # Web UI のボタンを指す。Web UI には ``web/w1`` 〜 ``web/w5`` の 5 つの
    # プログラム可能なボタンがあり、run_condition にボタン名を設定したパーツを
    # 追加するだけで、押されたときにそのパーツが実行される。
    #
    have_joystick = ctr is not None and isinstance(ctr, JoystickController)

    # 経路を保存するトリガー。コースを一周したら保存してプロセスを終了し、
    # 再起動すると経路が読み込まれる。
    if cfg.SAVE_PATH_BTN:
        print(f"保存ボタンは {cfg.SAVE_PATH_BTN} です")
        if cfg.SAVE_PATH_BTN.startswith("web/w"):
            V.add(Lambda(lambda: save_path()), run_condition=cfg.SAVE_PATH_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.SAVE_PATH_BTN, save_path)

    # コントローラーから経路を再読み込みできるようにする
    if cfg.LOAD_PATH_BTN:
        print(f"ロードボタンは {cfg.LOAD_PATH_BTN} です")
        if cfg.LOAD_PATH_BTN.startswith("web/w"):
            V.add(Lambda(lambda: load_path()), run_condition=cfg.LOAD_PATH_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.LOAD_PATH_BTN, load_path)

    # 保存済み経路をメモリから消去するトリガー。ファイルは削除しない
    if cfg.ERASE_PATH_BTN:
        print(f"消去ボタンは {cfg.ERASE_PATH_BTN} です")
        if cfg.ERASE_PATH_BTN.startswith("web/w"):
            V.add(Lambda(lambda: erase_path()), run_condition=cfg.ERASE_PATH_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.ERASE_PATH_BTN, erase_path)

    # 現在位置を基準に原点をリセットするトリガー
    if cfg.RESET_ORIGIN_BTN:
        print(f"原点リセットボタンは {cfg.RESET_ORIGIN_BTN} です")
        if cfg.RESET_ORIGIN_BTN.startswith("web/w"):
            V.add(Lambda(lambda: reset_origin()), run_condition=cfg.RESET_ORIGIN_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.RESET_ORIGIN_BTN, reset_origin)

    # 記録のオンオフを切り替えるボタン
    if cfg.TOGGLE_RECORDING_BTN:
        print(f"録画切替ボタンは {cfg.TOGGLE_RECORDING_BTN} です")
        if cfg.TOGGLE_RECORDING_BTN.startswith("web/w"):
            V.add(Lambda(lambda: recording_control.toggle_recording()), run_condition=cfg.TOGGLE_RECORDING_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.TOGGLE_RECORDING_BTN, recording_control.toggle_recording)

    # PID 定数を調整するボタン群
    if cfg.DEC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"PID P 減少ボタンは {cfg.DEC_PID_P_BTN} です")
        if cfg.DEC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_p()), run_condition=cfg.DEC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_P_BTN, dec_pid_p)
    if cfg.INC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"PID P 増加ボタンは {cfg.INC_PID_P_BTN} です")
        if cfg.INC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_p()), run_condition=cfg.INC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_P_BTN, inc_pid_p)
    if cfg.DEC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"PID D 減少ボタンは {cfg.DEC_PID_D_BTN} です")
        if cfg.DEC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_d()), run_condition=cfg.DEC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_D_BTN, dec_pid_d)
    if cfg.INC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"PID D 増加ボタンは {cfg.INC_PID_D_BTN} です")
        if cfg.INC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_d()), run_condition=cfg.INC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_D_BTN, inc_pid_d)


    #
    # ユーザーモードかオートパイロットかに応じて、操舵とスロットルの入力源を選択
    #
    V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/steering', 'user/throttle',
                  'pilot/steering', 'pilot/throttle'],
          outputs=['steering', 'throttle'])

    # V.add(LoggerPart(['user/mode', 'steering', 'throttle'], logger="drivemode"), inputs=['user/mode', 'steering', 'throttle'])

    #
    # 差動駆動で旋回するため、操舵量に応じて左右モーターのスロットルを分配
    #
    if is_differential_drive:
        V.add(TwoWheelSteeringThrottle(),
            inputs=['throttle', 'steering'],
            outputs=['left/throttle', 'right/throttle'])

    #
    # ドライブトレインをセットアップ
    #
    add_drivetrain(V, cfg)


    #
    # OLED ディスプレイの設定
    #
    if cfg.USE_SSD1306_128_32:
        from donkeycar.parts.oled import OLEDPart
        auto_record_on_throttle = cfg.USE_JOYSTICK_AS_DEFAULT and cfg.AUTO_RECORD_ON_THROTTLE
        oled_part = OLEDPart(cfg.SSD1306_128_32_I2C_ROTATION, cfg.SSD1306_RESOLUTION, auto_record_on_throttle)
        V.add(oled_part, inputs=['recording', 'tub/num_records', 'user/mode'], outputs=[], threaded=True)


    # ジョイスティックの操作方法を表示
    if ctr is not None and isinstance(ctr, JoystickController):
        ctr.print_controls()

    #
    # 車両の動きを地図画像に描画
    #
    loc_plot = PlotCircle(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET, color = "blue")
    V.add(loc_plot, inputs=['map/image', 'pos/x', 'pos/y'], outputs=['map/image'], run_condition='run_pilot')

    loc_plot = PlotCircle(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET, color = "green")
    V.add(loc_plot, inputs=['map/image', 'pos/x', 'pos/y'], outputs=['map/image'], run_condition='run_user')


    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
        max_loop_count=cfg.MAX_LOOPS)


def add_gps(V, cfg):
    """GPS パーツを追加する。

    Args:
        V: ``Vehicle`` インスタンス。
        cfg: 設定オブジェクト。

    Returns:
        Optional[GpsPlayer]: 録音した NMEA データを再生するためのプレーヤー。
    """

    if cfg.HAVE_GPS:
        from donkeycar.parts.serial_port import SerialPort, SerialLineReader
        from donkeycar.parts.gps import GpsNmeaPositions, GpsLatestPosition, GpsPlayer
        from donkeycar.parts.pipe import Pipe
        from donkeycar.parts.text_writer import CsvLogger

        #
        # 以下のパーツを構成する
        # - シリアルポートから NMEA ラインを読み取る
        # - または記録済みファイルを再生する
        # - NMEA ラインを位置情報に変換する
        # - 最新の位置を取得する
        #
        serial_port = SerialPort(cfg.GPS_SERIAL, cfg.GPS_SERIAL_BAUDRATE)
        nmea_reader = SerialLineReader(serial_port)
        V.add(nmea_reader, outputs=['gps/nmea'], threaded=True)

        # NMEA 文章を保存して後で再生できるようにするパート
        nmea_player = None
        if cfg.GPS_NMEA_PATH:
            nmea_writer = CsvLogger(cfg.GPS_NMEA_PATH, separator='\t', field_count=2)
            V.add(nmea_writer, inputs=['recording', 'gps/nmea'], outputs=['gps/recorded/nmea'])  # ユーザーモード時のみ NMEA を記録
            nmea_player = GpsPlayer(nmea_writer)
            V.add(nmea_player, inputs=['run_pilot', 'gps/nmea'], outputs=['gps/playing', 'gps/nmea'])  # オートパイロット時のみ再生

        gps_positions = GpsNmeaPositions(debug=cfg.GPS_DEBUG)
        V.add(gps_positions, inputs=['gps/nmea'], outputs=['gps/positions'])
        gps_latest_position = GpsLatestPosition(debug=cfg.GPS_DEBUG)
        V.add(gps_latest_position, inputs=['gps/positions'], outputs=['gps/timestamp', 'gps/utm/longitude', 'gps/utm/latitude'])

        # GPS UTM 座標を姿勢の値としてリネーム
        V.add(Pipe(), inputs=['gps/utm/longitude', 'gps/utm/latitude'], outputs=['pos/x', 'pos/y'])

        return nmea_player


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    log_level = args['--log'] or "INFO"
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('無効なログレベル: %s' % log_level)
    logging.basicConfig(level=numeric_level)


    if args['drive']:
        drive(cfg, use_joystick=args['--js'], camera_type=args['--camera'])
