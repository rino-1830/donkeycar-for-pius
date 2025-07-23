#!/usr/bin/env python3
"""
コンピュータビジョンを用いた自動運転のためのスクリプト。

Usage:
    manage.py (drive) [--js] [--log=INFO] [--camera=(single|stereo)] [--myconfig=<filename>]


Options:
    -h --help          この画面を表示します。
    --js               物理ジョイスティックを使用します。
    --myconfig=filename     使用する myconfig ファイルを指定します。
                            [default: myconfig.py]
"""
import logging

from docopt import docopt
from simple_pid import PID

import donkeycar as dk
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.line_follower import LineFollower
from donkeycar.templates.complete import add_odometry, add_camera, \
    add_user_controller, add_drivetrain, add_simulator, add_imu, DriveMode, \
    UserPilotCondition, ToggleRecording
from donkeycar.parts.logger import LoggerPart
from donkeycar.parts.transform import Lambda
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.controller import JoystickController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def drive(cfg, use_joystick=False, camera_type='single', meta=[]):
    """複数のパーツから構成されるロボット車両を組み立てて走行させる。

    各パーツは Vehicle ループ内でジョブとして実行され、`threaded` フラグに
    応じて ``run`` もしくは ``run_threaded`` が呼び出される。すべてのパーツ
    は ``cfg.DRIVE_LOOP_HZ`` で指定されたフレームレートで順次更新される。

    Args:
        cfg: 設定を保持するオブジェクト。
        use_joystick: ジョイスティックを使用する場合は ``True``。
        camera_type: 使用するカメラタイプ。
        meta: 追加のメタデータを含むリスト。
    """
    
    # 車を初期化
    V = dk.vehicle.Vehicle()

    #
    # シミュレーターを使用する場合の設定
    #
    add_simulator(V, cfg)

    #
    # メインカメラを設定
    #
    add_camera(V, cfg, camera_type)

    #
    # ユーザー入力コントローラを追加
    # - Web コントローラが追加される
    # - 設定されていればジョイスティックコントローラも追加
    #
    has_input_controller = hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    ctr = add_user_controller(V, cfg, use_joystick, input_image = 'ui/image_array')

    #
    # Web ボタンをメモリ上で個別のキーと値に展開
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    #
    # ユーザー操作と自動運転の状態を追跡
    #
    V.add(UserPilotCondition(show_pilot_image=getattr(cfg, 'OVERLAY_IMAGE', False)),
          inputs=['user/mode', "cam/image_array", "cv/image_array"],
          outputs=['run_user', "run_pilot", "ui/image_array"])

    #
    # cv_controller で使用する PID コントローラ
    #
    pid = PID(Kp=cfg.PID_P, Ki=cfg.PID_I, Kd=cfg.PID_D)
    def dec_pid_d():
        pid.Kd -= cfg.PID_D_DELTA
        logging.info("pid：d- %f" % pid.Kd)

    def inc_pid_d():
        pid.Kd += cfg.PID_D_DELTA
        logging.info("pid：d+ %f" % pid.Kd)

    def dec_pid_p():
        pid.Kp -= cfg.PID_P_DELTA
        logging.info("pid：p- %f" % pid.Kp)

    def inc_pid_p():
        pid.Kp += cfg.PID_P_DELTA
        logging.info("pid：p+ %f" % pid.Kp)

    #
    # コンピュータビジョンコントローラ
    #
    add_cv_controller(V, cfg, pid,
                      cfg.CV_CONTROLLER_MODULE,
                      cfg.CV_CONTROLLER_CLASS,
                      cfg.CV_CONTROLLER_INPUTS,
                      cfg.CV_CONTROLLER_OUTPUTS,
                      cfg.CV_CONTROLLER_CONDITION)

    recording_control = ToggleRecording(cfg.AUTO_RECORD_ON_THROTTLE, cfg.RECORD_DURING_AI)
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])


    #
    # さまざまなユーザー操作に対応するボタンを追加
    # ボタン名は設定で指定される
    # ゲームコントローラ（ジョイスティック）または Web UI のボタンを指す場合がある
    #
    # WebUI 用のプログラム可能なボタンは "web/w1" から "web/w5" までの5個
    # WebUI ボタン用のハンドラーは、run_condition にボタン名を指定したパート
    # を追加するだけで実現でき、ボタンが押されたときに実行される
    #
    have_joystick = ctr is not None and isinstance(ctr, JoystickController)

    # 録画の開始と停止を切り替えるボタン
    if cfg.TOGGLE_RECORDING_BTN:
        print(f"録画切り替えボタン: {cfg.TOGGLE_RECORDING_BTN}")
        if cfg.TOGGLE_RECORDING_BTN.startswith("web/w"):
            V.add(Lambda(lambda: recording_control.toggle_recording()), run_condition=cfg.TOGGLE_RECORDING_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.TOGGLE_RECORDING_BTN, recording_control.toggle_recording)

    # PID 定数を調整するボタン
    if cfg.DEC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"PID P を減らすボタン: {cfg.DEC_PID_P_BTN}")
        if cfg.DEC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_p()), run_condition=cfg.DEC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_P_BTN, dec_pid_p)
    if cfg.INC_PID_P_BTN and cfg.PID_P_DELTA:
        print(f"PID P を増やすボタン: {cfg.INC_PID_P_BTN}")
        if cfg.INC_PID_P_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_p()), run_condition=cfg.INC_PID_P_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_P_BTN, inc_pid_p)
    if cfg.DEC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"PID D を減らすボタン: {cfg.DEC_PID_D_BTN}")
        if cfg.DEC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: dec_pid_d()), run_condition=cfg.DEC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.DEC_PID_D_BTN, dec_pid_d)
    if cfg.INC_PID_D_BTN and cfg.PID_D_DELTA:
        print(f"PID D を増やすボタン: {cfg.INC_PID_D_BTN}")
        if cfg.INC_PID_D_BTN.startswith("web/w"):
            V.add(Lambda(lambda: inc_pid_d()), run_condition=cfg.INC_PID_D_BTN)
        elif have_joystick:
            ctr.set_button_down_trigger(cfg.INC_PID_D_BTN, inc_pid_d)

    #
    # 操舵とスロットルを変更する入力を決定
    # ユーザーモードか自動運転モードかに基づいて判断する
    #
    V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/steering', 'user/throttle',
                  'pilot/steering', 'pilot/throttle'],
          outputs=['steering', 'throttle'])


    #
    # ドライブトレインを設定
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


    #
    # データを保存する Tub を追加
    #
    inputs=['cam/image_array',
            'steering', 'throttle']

    types=['image_array',
           'float', 'float']

    #
    # データ保存パーツを作成
    #
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    meta += getattr(cfg, 'METADATA', [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if cfg.DONKEY_GYM:
        print("http://localhost:%d にアクセスすると車を操作できます。" % cfg.WEB_CONTROL_PORT)
    else:
        print("<your hostname.local>:%d にアクセスすると車を操作できます。" % cfg.WEB_CONTROL_PORT)
    if has_input_controller:
        print("コントローラーを動かして車を運転できます。")
        if isinstance(ctr, JoystickController):
            ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    #
    # Vehicle を起動
    #
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


#
# コンピュータビジョンコントローラ
#
def add_cv_controller(
        V, cfg, pid,
        module_name="donkeycar.parts.line_follower",
        class_name="LineFollower",
        inputs=['cam/image_array'],
        outputs=['pilot/steering', 'pilot/throttle', 'cv/image_array'],
        run_condition="run_pilot"):

        """Computer Vision コントローラーを Vehicle に追加する。

        指定されたモジュールとクラスからインスタンスを生成し、Vehicle
        に登録する。

        Args:
            V: ``Vehicle`` インスタンス。
            cfg: 設定オブジェクト。
            pid: 使用する PID コントローラー。
            module_name: コントローラークラスを含むモジュール名。
            class_name: コントローラークラス名。
            inputs: 入力名のリスト。
            outputs: 出力名のリスト。
            run_condition: このコントローラーを実行する条件名。
        """

        # モジュールを __import__ する
        module = __import__(module_name)

        # モジュールパスを辿ってクラスがあるモジュールを取得
        for attr in module_name.split('.')[1:]:
            module = getattr(module, attr)

        my_class = getattr(module, class_name)

        # インスタンスを生成して Vehicle に追加
        V.add(my_class(pid, cfg),
              inputs=inputs,
              outputs=outputs,
              run_condition=run_condition)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    log_level = args['--log'] or "INFO"
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('無効なログレベル: %s' % log_level)
    logging.basicConfig(level=numeric_level)

    if args['drive']:
        drive(cfg, use_joystick=args['--js'], camera_type=args['--camera'])
