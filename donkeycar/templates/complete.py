#!/usr/bin/env python3
"""
Donkey 2 車両を走行させるスクリプト。

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tubs=tubs] (--model=<model>) [--type=(linear|inferred|tensorrt_linear|tflite_linear)]

Options:
    -h --help               このヘルプを表示します。
    --js                    物理ジョイスティックを使用します。
    -f --file=<file>        1 行につき 1 つの tub ファイルへのパスを記述したテキストファイル。複数指定可能です。
    --meta=<key:value>      この走行に関するメタデータを表すキーと値の文字列。複数指定可能です。
    --myconfig=filename     使用する myconfig ファイルを指定します。
                            [default: myconfig.py]
"""
from docopt import docopt

#
# TensorFlow より後にインポートすると問題が起こるため、先に cv2 を読み込む
# 詳細: https://github.com/opencv/opencv/issues/14884#issuecomment-599852128
#
try:
    import cv2
except:
    pass


import donkeycar as dk
from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, WebFpv, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.kinematics import NormalizeSteeringAngle, UnnormalizeSteeringAngle, TwoWheelSteeringThrottle
from donkeycar.parts.kinematics import Unicycle, InverseUnicycle, UnicycleUnnormalizeAngularVelocity
from donkeycar.parts.kinematics import Bicycle, InverseBicycle, BicycleUnnormalizeAngularVelocity
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.transform import Lambda
from donkeycar.parts.pipe import Pipe
from donkeycar.utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def drive(cfg, model_path=None, use_joystick=False, model_type=None,
          camera_type='single', meta=[]):
    """車両を構築してメインループを開始する。

    各パーツは ``threaded`` フラグに応じて ``run`` または ``run_threaded``
    を呼び出され、``cfg.DRIVE_LOOP_HZ`` で指定された周期で実行される。

    Args:
        cfg: ``myconfig.py`` で定義された設定オブジェクト。
        model_path: 事前学習済みモデルのパス。
        use_joystick: 物理ジョイスティックを使用するかどうか。
        model_type: モデルの種類。``None`` の場合 ``cfg`` から決定する。
        camera_type: カメラのタイプ ``'single'`` または ``'stereo'``。
        meta: TubWriter に渡すメタデータのリスト。

    """
    logger.info(f'PID: {os.getpid()}')
    if cfg.DONKEY_GYM:
        # シミュレータが CUDA を使用するため、こちらでも CUDA を使うとリソース不足になる
        # そのため donkey_gym 使用時は CUDA を無効化する
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    # 車両を初期化
    V = dk.vehicle.Vehicle()

    # コンソールログを出力するため、最初にロギングを初期化
    if cfg.HAVE_CONSOLE_LOGGING:
        logger.setLevel(logging.getLevelName(cfg.LOGGING_LEVEL))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(cfg.LOGGING_FORMAT))
        logger.addHandler(ch)

    if cfg.HAVE_MQTT_TELEMETRY:
        from donkeycar.parts.telemetry import MqttTelemetry
        tel = MqttTelemetry(cfg)
        
    #
    # シミュレータを使用する場合の設定
    #
    add_simulator(V, cfg)


    #
    # エンコーダ、オドメトリ、姿勢推定を設定
    #
    add_odometry(V, cfg)


    #
    # メインカメラの設定
    #
    add_camera(V, cfg, camera_type)


    # LIDAR を追加
    if cfg.USE_LIDAR:
        from donkeycar.parts.lidar import RPLidar
        if cfg.LIDAR_TYPE == 'RP':
            print("RP Lidar パーツを追加します")
            lidar = RPLidar(lower_limit = cfg.LIDAR_LOWER_LIMIT, upper_limit = cfg.LIDAR_UPPER_LIMIT)
            V.add(lidar, inputs=[],outputs=['lidar/dist_array'], threaded=True)
        if cfg.LIDAR_TYPE == 'YD':
            print("YD Lidar はまだサポートされていません")

    if cfg.HAVE_TFMINI:
        from donkeycar.parts.tfmini import TFMini
        lidar = TFMini(port=cfg.TFMINI_SERIAL_PORT)
        V.add(lidar, inputs=[], outputs=['lidar/dist'], threaded=True)

    if cfg.SHOW_FPS:
        from donkeycar.parts.fps import FrequencyLogger
        V.add(FrequencyLogger(cfg.FPS_DEBUG_INTERVAL),
              outputs=["fps/current", "fps/fps_list"])

    #
    # ユーザー入力コントローラーを追加する
    # - Web コントローラーを追加
    # - 必要に応じて設定されたジョイスティックも追加
    #
    has_input_controller = hasattr(cfg, "CONTROLLER_TYPE") and cfg.CONTROLLER_TYPE != "mock"
    ctr = add_user_controller(V, cfg, use_joystick)

    #
    # 過去の学習データとの互換性を保つため 'user/steering' を 'user/angle' に変換
    #
    V.add(Pipe(), inputs=['user/steering'], outputs=['user/angle'])

    #
    # ボタン入力のマップを展開して個々のキーと値として保持
    #
    V.add(ExplodeDict(V.mem, "web/"), inputs=['web/buttons'])

    #
    # 例として、ボタンハンドラーを追加するには ``run_condition`` をボタン名に
    # 設定したパーツを追加するだけで、ボタンが押されたときに実行される
    #
    V.add(Lambda(lambda v: print(f"web/w1 がクリックされました")), inputs=["web/w1"], run_condition="web/w1")
    V.add(Lambda(lambda v: print(f"web/w2 がクリックされました")), inputs=["web/w2"], run_condition="web/w2")
    V.add(Lambda(lambda v: print(f"web/w3 がクリックされました")), inputs=["web/w3"], run_condition="web/w3")
    V.add(Lambda(lambda v: print(f"web/w4 がクリックされました")), inputs=["web/w4"], run_condition="web/w4")
    V.add(Lambda(lambda v: print(f"web/w5 がクリックされました")), inputs=["web/w5"], run_condition="web/w5")

    # このスロットルフィルターにより ESC のリバースをワンタップで解除できる
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    #
    # ユーザーモードと自動運転モードの実行条件を管理
    #
    V.add(UserPilotCondition(show_pilot_image=getattr(cfg, 'SHOW_PILOT_IMAGE', False)),
          inputs=['user/mode', "cam/image_array", "cam/image_array_trans"],
          outputs=['run_user', "run_pilot", "ui/image_array"])

    class LedConditionLogic:
        """LED の状態を決定するロジック。"""
        def __init__(self, cfg):
            """設定を受け取って初期化する。"""
            self.cfg = cfg

        def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
            """LED の点滅速度を計算する。

            Args:
                mode: 現在のモード。
                recording: 録画状態。
                recording_alert: 録画件数に応じた警告色。
                behavior_state: 行動状態。
                model_file_changed: モデルファイルが更新されたかどうか。
                track_loc: 現在のトラック位置。

            Returns:
                点滅速度。0 で消灯、-1 で点灯し続け、正数は点滅周期(秒)。
            """

            if track_loc is not None:
                led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                return -1

            if model_file_changed:
                led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                return 0.1
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if recording_alert:
                led.set_rgb(*recording_alert)
                return self.cfg.REC_COUNT_ALERT_BLINK_RATE
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if behavior_state is not None and model_type == 'behavior':
                r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                led.set_rgb(r, g, b)
                return -1 # 点灯し続ける

            if recording:
                return -1 # 点灯し続ける
            elif mode == 'user':
                return 1
            elif mode == 'local_angle':
                return 0.5
            elif mode == 'local':
                return 0.1
            return 0

    if cfg.HAVE_RGB_LED and not cfg.DONKEY_GYM:
        from donkeycar.parts.led_status import RGB_LED
        led = RGB_LED(cfg.LED_PIN_R, cfg.LED_PIN_G, cfg.LED_PIN_B, cfg.LED_INVERT)
        led.set_rgb(cfg.LED_R, cfg.LED_G, cfg.LED_B)

        V.add(LedConditionLogic(cfg), inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
              outputs=['led/blink_rate'])

        V.add(led, inputs=['led/blink_rate'])

    def get_record_alert_color(num_records):
        """記録数に対応する警告色を取得する。"

        Args:
            num_records: 現在の記録数。

        Returns:
            ``(r, g, b)`` のタプル。
        """
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        """Tub 記録数を追跡し、LED アラート用の色を出力する。"""

        def __init__(self):
            """インスタンスを初期化する。"""
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            """記録数に応じたアラート色を計算する。"""
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print(f"{num_records} 件のレコードを保存しました")

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE:
        def show_record_count_status():
            """現在の記録数を即座に表示するトリガー。"""
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1
        # pigpio_rc と MM1 は JoystickController を使用しない
        if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):
            if isinstance(ctr, JoystickController):
                # circle ボタンを流用して記録数表示を強制
                ctr.set_button_down_trigger('circle', show_record_count_status)
        else:
            
            show_record_count_status()

    #Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

    #IMU
    add_imu(V, cfg)


    # クロップ後の画像またはフルフレームを表示する FPV プレビューを使用
    if cfg.USE_FPV:
        V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)

    def load_model(kl, model_path):
        """モデルファイルを読み込む。"

        Args:
            kl: Keras ラッパー。
            model_path: モデルファイルへのパス。
        """
        start = time.time()
        print('モデルを読み込み中', model_path)
        kl.load(model_path)
        print('読み込み完了 %s 秒' % (str(time.time() - start)))

    def load_weights(kl, weights_path):
        """重みファイルのみを読み込む。"

        Args:
            kl: Keras ラッパー。
            weights_path: 重みファイルへのパス。
        """
        start = time.time()
        try:
            print('モデルの重みを読み込み中', weights_path)
            kl.model.load_weights(weights_path)
            print('読み込み完了 %s 秒' % (str(time.time() - start)))
        except Exception as e:
            print(e)
            print('ERR>> 重みの読み込みに失敗しました', weights_path)

    def load_model_json(kl, json_fnm):
        """JSON 形式のモデルを読み込み重みを設定する。"

        Args:
            kl: Keras ラッパー。
            json_fnm: JSON ファイル名。

        """
        start = time.time()
        print('モデル JSON を読み込み中', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('JSON の読み込み完了 %s 秒' % (str(time.time() - start)))
        except Exception as e:
            print(e)
            print("ERR>> モデル JSON の読み込みに失敗しました", json_fnm)

    #
    # 推論用のモデルを読み込み設定する
    #
    if model_path:
        # モデルが指定されていれば対応する Keras パーツを生成
        kl = dk.utils.get_model_by_type(model_type, cfg)

        #
        # モデル形式に合わせてリロード用のコールバックを取得
        #
        model_reload_cb = None
        if '.h5' in model_path or '.trt' in model_path or '.tflite' in \
                model_path or '.savedmodel' in model_path or '.pth':
            # 重みを含む完全なモデルを読み込む
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif '.json' in model_path:
            # 拡張子が .json の場合はモデル構造のみを読み込み、
            # 同名の .weights ファイルから重みを読み込む
            load_model_json(kl, model_path)
            weights_path = model_path.replace('.json', '.weights')
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace('.json', '.weights')
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> モデルファイルの拡張子が不明です!!")
            return

        # モデル更新を検知したら LED へ通知する
        V.add(FileWatcher(model_path, verbose=True),
              outputs=['modelfile/modified'])

        # 以下のパーツは自動運転中のみモデルをリロードし、
        # ユーザー操作を妨げないようにする
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'],
              run_condition="run_pilot")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'],
              outputs=['modelfile/reload'], run_condition="run_pilot")
        V.add(TriggeredCallback(model_path, model_reload_cb),
              inputs=["modelfile/reload"], run_condition="run_pilot")

        #
        # 推論用モデルへの入力を収集
        #
        if cfg.TRAIN_BEHAVIORS:
            bh = BehaviorPart(cfg.BEHAVIOR_LIST)
            V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
            try:
                ctr.set_button_down_trigger('L1', bh.increment_state)
            except:
                pass

            inputs = ['cam/image_array', "behavior/one_hot_state_array"]

        elif cfg.USE_LIDAR:
            inputs = ['cam/image_array', 'lidar/dist_array']

        elif cfg.HAVE_ODOM:
            inputs = ['cam/image_array', 'enc/speed']

        elif model_type == "imu":
            assert cfg.HAVE_IMU, 'Missing imu parameter in config'

            class Vectorizer:
                def run(self, *components):
                    return components

            V.add(Vectorizer, inputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                                      'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
                  outputs=['imu_array'])

            inputs = ['cam/image_array', 'imu_array']
        else:
            inputs = ['cam/image_array']

        #
        # モデルの推論結果を受け取る出力を定義
        #
        outputs = ['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        #
        # クロップや台形マスクなどの画像変換を追加し、
        # 自動運転モードの推論時にも適用されるようにする
        #
        if hasattr(cfg, 'TRANSFORMATIONS') or hasattr(cfg, 'POST_TRANSFORMATIONS'):
            from donkeycar.parts.image_transformations import ImageTransformations
            #
            # 学習時と同じ前処理・後処理をすべて追加
            #
            logger.info("推論用変換を追加します")
            V.add(ImageTransformations(cfg, 'TRANSFORMATIONS',
                                       'POST_TRANSFORMATIONS'),
                  inputs=['cam/image_array'], outputs=['cam/image_array_trans'])
            inputs = ['cam/image_array_trans'] + inputs[1:]

        V.add(kl, inputs=inputs, outputs=outputs, run_condition='run_pilot')

    #
    # ストップサインを検知したら停止する
    #
    if cfg.STOP_SIGN_DETECTOR:
        from donkeycar.parts.object_detector.stop_sign_detector \
            import StopSignDetector
        V.add(StopSignDetector(cfg.STOP_SIGN_MIN_SCORE,
                               cfg.STOP_SIGN_SHOW_BOUNDING_BOX,
                               cfg.STOP_SIGN_MAX_REVERSE_COUNT,
                               cfg.STOP_SIGN_REVERSE_THROTTLE),
              inputs=['cam/image_array', 'pilot/throttle'],
              outputs=['pilot/throttle', 'cam/image_array'])
        V.add(ThrottleFilter(), 
              inputs=['pilot/throttle'],
              outputs=['pilot/throttle'])

    #
    # レースで AI モードを開始するときに加速させるランチャー
    # ストップサイン検出を一時的に無効にし、発進後は再度有効になる
    #
    # NOTE: ランチスロットル中はパイロット速度が ``None`` になる
    #
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)
    V.add(aiLauncher,
          inputs=['user/mode', 'pilot/throttle'],
          outputs=['pilot/throttle'])

    #
    # ユーザー操作か自動運転かに応じて
    # どの入力をステアリングとスロットルに反映するか決定
    #
    V.add(DriveMode(cfg.AI_THROTTLE_MULT),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['steering', 'throttle'])


    if (cfg.CONTROLLER_TYPE != "pigpio_rc") and (cfg.CONTROLLER_TYPE != "MM1"):
        if isinstance(ctr, JoystickController):
            ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)


    # AI モード中の録画管理
    recording_control = ToggleRecording(cfg.AUTO_RECORD_ON_THROTTLE, cfg.RECORD_DURING_AI)
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])

    #
    # ドライブトレインの設定
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
    # 走行データを保存する Tub を追加
    #
    if cfg.USE_LIDAR:
        inputs = ['cam/image_array', 'lidar/dist_array', 'user/angle', 'user/throttle', 'user/mode']
        types = ['image_array', 'nparray','float', 'float', 'str']
    else:
        inputs=['cam/image_array','user/angle', 'user/throttle', 'user/mode']
        types=['image_array','float', 'float','str']

    if cfg.HAVE_ODOM:
        inputs += ['enc/speed']
        types += ['float']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_DEPTH:
        inputs += ['cam/depth_array']
        types += ['gray16_array']

    if cfg.HAVE_IMU or (cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_IMU):
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types +=['float', 'float', 'float',
           'float', 'float', 'float']

    # rbx
    if cfg.DONKEY_GYM:
        if cfg.SIM_RECORD_LOCATION:
            inputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
            types  += ['float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_GYROACCEL:
            inputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
            types  += ['float', 'float', 'float', 'float', 'float', 'float']
        if cfg.SIM_RECORD_VELOCITY:
            inputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
            types  += ['float', 'float', 'float']
        if cfg.SIM_RECORD_LIDAR:
            inputs += ['lidar/dist_array']
            types  += ['nparray']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    if cfg.HAVE_PERFMON:
        from donkeycar.parts.perfmon import PerfMonitor
        mon = PerfMonitor(cfg)
        perfmon_outputs = ['perf/cpu', 'perf/mem', 'perf/freq']
        inputs += perfmon_outputs
        types += ['float', 'float', 'float']
        V.add(mon, inputs=[], outputs=perfmon_outputs, threaded=True)

    #
    # データ保存用パーツを作成
    #
    tub_path = TubHandler(path=cfg.DATA_PATH).create_tub_path() if \
        cfg.AUTO_CREATE_NEW_TUB else cfg.DATA_PATH
    meta += getattr(cfg, 'METADATA', [])
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types, metadata=meta)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    # Telemetry（TubHandler と同じメトリクスを送信）
    if cfg.HAVE_MQTT_TELEMETRY:
        from donkeycar.parts.telemetry import MqttTelemetry
        tel = MqttTelemetry(cfg)
        telem_inputs, _ = tel.add_step_inputs(inputs, types)
        V.add(tel, inputs=telem_inputs, outputs=["tub/queue_size"], threaded=True)

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])


    if cfg.DONKEY_GYM:
        print("http://localhost:%d にアクセスすると車を操縦できます。" % cfg.WEB_CONTROL_PORT)
    else:
        print("<your hostname.local>:%d にアクセスすると車を操縦できます。" % cfg.WEB_CONTROL_PORT)
    if has_input_controller:
        print("コントローラーを操作すると車を運転できます。")
        if isinstance(ctr, JoystickController):
            ctr.set_tub(tub_writer.tub)
            ctr.print_controls()

    # 車両を走行させる
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, max_loop_count=cfg.MAX_LOOPS)


class ToggleRecording:
    def __init__(self, auto_record_on_throttle, record_in_autopilot):
        """録画状態を管理する Donkeycar のパーツ。"""
        self.auto_record_on_throttle = auto_record_on_throttle
        self.record_in_autopilot = record_in_autopilot
        self.recording_latch: bool = None
        self.toggle_latch: bool = False
        self.last_recording = None

    def set_recording(self, recording: bool):
        """次回 ``run()`` で適用する録画状態を設定する。

        Args:
            recording: ``True`` で録画、``False`` で停止。
        """
        self.recording_latch = recording

    def toggle_recording(self):
        """次回 ``run()`` 実行時に録画状態を強制的に反転させる。"""
        self.toggle_latch = True

    def run(self, mode: str, recording: bool):
        """ユーザーモードと自動運転モードに応じて録画状態を更新する。

        Args:
            mode: ``'user'``、``'local_angle'``、``'local_pilot'`` のいずれか。
            recording: 現在の録画フラグ。

        Returns:
            更新後の録画フラグ。
        """
        recording_in = recording
        if recording_in != self.last_recording:
            logging.info(f"録画状態が変わりました: {recording_in}")

        if self.toggle_latch:
            if self.auto_record_on_throttle:
                logger.info('スロットル連動録画が有効のため手動切り替えを無視します')
            else:
                recording = not self.last_recording
            self.toggle_latch = False

        if self.recording_latch is not None:
            recording = self.recording_latch
            self.recording_latch = None

        if recording and mode != 'user' and not self.record_in_autopilot:
            logging.info("自動運転中の録画は無視されました")
            recording = False

        if self.last_recording != recording:
            logging.info(f"録画設定を更新しました: {recording}")

        self.last_recording = recording

        return recording


class DriveMode:
    def __init__(self, ai_throttle_mult=1.0):
        """初期化処理。

        Args:
            ai_throttle_mult: 自動運転モード時にスロットルへ乗算する倍率。
        """
        self.ai_throttle_mult = ai_throttle_mult

    def run(self, mode,
            user_steering, user_throttle,
            pilot_steering, pilot_throttle):
        """ユーザーのモードに応じて最終的なステアリングとスロットルを決定する。

        Args:
            mode: ``'user'``、``'local_angle'``、``'local_pilot'`` のいずれか。
            user_steering: マニュアル操作時のステアリング値。
            user_throttle: マニュアル操作時のスロットル値。
            pilot_steering: 自動運転時のステアリング値。
            pilot_throttle: 自動運転時のスロットル値。

        Returns:
            ``(steering, throttle)`` のタプル。自動運転時は ``ai_throttle_mult``
            を掛けたスロットルを返す。
        """
        if mode == 'user':
            return user_steering, user_throttle
        elif mode == 'local_angle':
            return pilot_steering if pilot_steering else 0.0, user_throttle
        return (pilot_steering if pilot_steering else 0.0,
               pilot_throttle * self.ai_throttle_mult if pilot_throttle else 0.0)


class UserPilotCondition:
    def __init__(self, show_pilot_image:bool = False) -> None:
        """初期化処理。

        Args:
            show_pilot_image: ``True`` ならパイロットモードで推論画像を表示し、
                ``False`` ならユーザー画像を表示する。
        """
        self.show_pilot_image = show_pilot_image

    def run(self, mode, user_image, pilot_image):
        """実行条件と Web UI に表示する画像を決定する。

        Args:
            mode: ``'user'``、``'local_angle'``、``'local_pilot'`` のいずれか。
            user_image: 手動運転時に表示する画像。
            pilot_image: 自動運転時に表示する画像。

        Returns:
            ``(user-condition, autopilot-condition, web image)`` のタプル。
        """
        if mode == 'user':
            return True, False, user_image
        else:
            return False, True, pilot_image if self.show_pilot_image else user_image


def add_user_controller(V, cfg, use_joystick, input_image='ui/image_array'):
    """Web コントローラーおよびその他の入力デバイスを追加する。

    Args:
        V: 車両のパイプライン。呼び出し後に変更される。
        cfg: ``myconfig.py`` から読み込んだ設定。

    Returns:
        使用するコントローラー。
    """

    #
    # この Web コントローラはステアリング、スロットル、モードなどを
    # 管理するための Web サーバーを起動する
    #
    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    V.add(ctr,
          inputs=[input_image, 'tub/num_records', 'user/mode', 'recording'],
          outputs=['user/steering', 'user/throttle', 'user/mode', 'recording', 'web/buttons'],
          threaded=True)

    #
    # さらに設定されていれば物理コントローラーも追加する
    #
    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        #
        # RC コントローラー
        #
        if cfg.CONTROLLER_TYPE == "pigpio_rc":  # GPIO ピンから読み取る RC コントローラー。通常ボタンはない
            from donkeycar.parts.controller import RCReceiver
            ctr = RCReceiver(cfg)
            V.add(
                ctr,
                inputs=['user/mode', 'recording'],
                outputs=['user/steering', 'user/throttle',
                         'user/mode', 'recording'],
                threaded=False)
        else:
            #
            # `donkey createjs` コマンドで作成したカスタムゲームコントローラー設定
            #
            if cfg.CONTROLLER_TYPE == "custom":  # custom controller created with `donkey createjs` command
                from my_joystick import MyJoystickController
                ctr = MyJoystickController(
                    throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                    throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                    steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                    auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
                ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
            elif cfg.CONTROLLER_TYPE == "MM1":
                from donkeycar.parts.robohat import RoboHATController
                ctr = RoboHATController(cfg)
            elif cfg.CONTROLLER_TYPE == "mock":
                from donkeycar.parts.controller import MockController
                ctr = MockController(steering=cfg.MOCK_JOYSTICK_STEERING,
                                     throttle=cfg.MOCK_JOYSTICK_THROTTLE)
            else:
                #
                # ゲームコントローラー
                #
                from donkeycar.parts.controller import get_js_controller
                ctr = get_js_controller(cfg)
                if cfg.USE_NETWORKED_JS:
                    from donkeycar.parts.controller import JoyStickSub
                    netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                    V.add(netwkJs, threaded=True)
                    ctr.js = netwkJs
            V.add(
                ctr,
                inputs=[input_image, 'user/mode', 'recording'],
                outputs=['user/steering', 'user/throttle',
                         'user/mode', 'recording'],
                threaded=True)
    return ctr


def add_simulator(V, cfg):
    """シミュレータを追加する。

    Args:
        V: 車両のパイプライン。
        cfg: ``myconfig.py`` から読み込んだ設定。

    """
    # Donkey Gym を使用する場合、位置情報などを出力する
    # TODO: シミュレーションの出力が IMU やオドメトリ等と競合するため整理する
    if cfg.DONKEY_GYM:
        from donkeycar.parts.dgym import DonkeyGymEnv
        # rbx
        gym = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF,
                           record_location=cfg.SIM_RECORD_LOCATION, record_gyroaccel=cfg.SIM_RECORD_GYROACCEL,
                           record_velocity=cfg.SIM_RECORD_VELOCITY, record_lidar=cfg.SIM_RECORD_LIDAR,
                        #    record_distance=cfg.SIM_RECORD_DISTANCE, record_orientation=cfg.SIM_RECORD_ORIENTATION,
                           delay=cfg.SIM_ARTIFICIAL_LATENCY)
        threaded = True
        inputs = ['steering', 'throttle']
        outputs = ['cam/image_array']

        if cfg.SIM_RECORD_LOCATION:
            outputs += ['pos/pos_x', 'pos/pos_y', 'pos/pos_z', 'pos/speed', 'pos/cte']
        if cfg.SIM_RECORD_GYROACCEL:
            outputs += ['gyro/gyro_x', 'gyro/gyro_y', 'gyro/gyro_z', 'accel/accel_x', 'accel/accel_y', 'accel/accel_z']
        if cfg.SIM_RECORD_VELOCITY:
            outputs += ['vel/vel_x', 'vel/vel_y', 'vel/vel_z']
        if cfg.SIM_RECORD_LIDAR:
            outputs += ['lidar/dist_array']
        # if cfg.SIM_RECORD_DISTANCE:
        #     outputs += ['dist/left', 'dist/right']
        # if cfg.SIM_RECORD_ORIENTATION:
        #     outputs += ['roll', 'pitch', 'yaw']

        V.add(gym, inputs=inputs, outputs=outputs, threaded=threaded)


def get_camera(cfg):
    """設定に基づいたカメラパーツを取得する。

    Args:
        cfg: ``myconfig.py`` で読み込んだ設定。

    Returns:
        生成したカメラパーツ。存在しない場合は ``None``。
    """
    cam = None
    if not cfg.DONKEY_GYM:
        if cfg.CAMERA_TYPE == "PICAM":
            from donkeycar.parts.camera import PiCamera
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                           framerate=cfg.CAMERA_FRAMERATE,
                           vflip=cfg.CAMERA_VFLIP, hflip=cfg.CAMERA_HFLIP)
        elif cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam
            cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam
            cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CSIC":
            from donkeycar.parts.camera import CSICamera
            cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                            capture_width=cfg.IMAGE_W, capture_height=cfg.IMAGE_H,
                            framerate=cfg.CAMERA_FRAMERATE, gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
        elif cfg.CAMERA_TYPE == "V4L":
            from donkeycar.parts.camera import V4LCamera
            cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "IMAGE_LIST":
            from donkeycar.parts.camera import ImageListCamera
            cam = ImageListCamera(path_mask=cfg.PATH_MASK)
        elif cfg.CAMERA_TYPE == "LEOPARD":
            from donkeycar.parts.leopard_imaging import LICamera
            cam = LICamera(width=cfg.IMAGE_W, height=cfg.IMAGE_H, fps=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "MOCK":
            from donkeycar.parts.camera import MockCamera
            cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        else:
            raise Exception("未知のカメラタイプです: %s" % cfg.CAMERA_TYPE)
    return cam


def add_camera(V, cfg, camera_type):
    """カメラパーツをパイプラインへ追加する。

    Args:
        V: 車両のパイプライン。呼び出し後に変更される。
        cfg: ``myconfig.py`` から読み込んだ設定。
        camera_type: ``'single'`` または ``'stereo'``。

    """
    logger.info("cfg.CAMERA_TYPE %s"%cfg.CAMERA_TYPE)
    if camera_type == "stereo":
        if cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam

            camA = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)

        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam

            camA = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 0)
            camB = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam = 1)
        else:
            raise Exception("未対応のカメラタイプです: %s" % cfg.CAMERA_TYPE)

        V.add(camA, outputs=['cam/image_array_a'], threaded=True)
        V.add(camB, outputs=['cam/image_array_b'], threaded=True)

        from donkeycar.parts.image import StereoPair

        V.add(StereoPair(), inputs=['cam/image_array_a', 'cam/image_array_b'],
            outputs=['cam/image_array'])
        if cfg.BGR2RGB:
            from donkeycar.parts.cv import ImgBGR2RGB
            V.add(ImgBGR2RGB(), inputs=["cam/image_array_a"], outputs=["cam/image_array_a"])
            V.add(ImgBGR2RGB(), inputs=["cam/image_array_b"], outputs=["cam/image_array_b"])

    elif cfg.CAMERA_TYPE == "D435":
        from donkeycar.parts.realsense435i import RealSense435i
        cam = RealSense435i(
            enable_rgb=cfg.REALSENSE_D435_RGB,
            enable_depth=cfg.REALSENSE_D435_DEPTH,
            enable_imu=cfg.REALSENSE_D435_IMU,
            device_id=cfg.REALSENSE_D435_ID)
        V.add(cam, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array',
                       'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                       'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
              threaded=True)
    else:
        inputs = []
        outputs = ['cam/image_array']
        threaded = True
        cam = get_camera(cfg)
        if cam:
            V.add(cam, inputs=inputs, outputs=outputs, threaded=threaded)
        if cfg.BGR2RGB:
            from donkeycar.parts.cv import ImgBGR2RGB
            V.add(ImgBGR2RGB(), inputs=["cam/image_array"], outputs=["cam/image_array"])


def add_odometry(V, cfg, threaded=True):
    """オドメトリ関連パーツを追加する。

    Args:
        V: 車両のパイプライン。必要に応じて変更される。
        cfg: ``myconfig.py`` から読み込んだ設定。
        threaded: スレッド実行するかどうか。

    """
    from donkeycar.parts.pose import BicyclePose, UnicyclePose

    if cfg.HAVE_ODOM:
        poll_delay_secs = 0.01  # 姿勢推定は 100Hz で動作
        kinematics = UnicyclePose(cfg, poll_delay_secs) if cfg.HAVE_ODOM_2 else BicyclePose(cfg, poll_delay_secs)
        V.add(kinematics,
            inputs = ["throttle", "steering", None],
            outputs = ['enc/distance', 'enc/speed', 'pos/x', 'pos/y',
                       'pos/angle', 'vel/x', 'vel/y', 'vel/angle',
                       'nul/timestamp'],
            threaded = threaded)


#
# IMU セットアップ
#
def add_imu(V, cfg):
    """IMU パーツを追加する。

    Args:
        V: 車両のパイプライン。
        cfg: ``myconfig.py`` から読み込んだ設定。

    Returns:
        生成した IMU パーツ。存在しない場合は ``None``。
    """
    imu = None
    if cfg.HAVE_IMU:
        from donkeycar.parts.imu import IMU

        imu = IMU(sensor=cfg.IMU_SENSOR, addr=cfg.IMU_ADDRESS,
                  dlp_setting=cfg.IMU_DLP_CONFIG)
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)
    return imu


#
# ドライブトレインの設定
#
def add_drivetrain(V, cfg):
    """駆動系パーツを設定する。

    Args:
        V: 車両のパイプライン。
        cfg: ``myconfig.py`` から読み込んだ設定。

    """
    if (not cfg.DONKEY_GYM) and cfg.DRIVE_TRAIN_TYPE != "MOCK":
        from donkeycar.parts import actuator, pins
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle

        #
        # 差動駆動でステアリングさせるため、
        # ステアリング値に基づき左右モーターのスロットルを分配する
        #
        is_differential_drive = cfg.DRIVE_TRAIN_TYPE.startswith("DC_TWO_WHEEL")
        if is_differential_drive:
            V.add(TwoWheelSteeringThrottle(),
                  inputs=['throttle', 'steering'],
                  outputs=['left/throttle', 'right/throttle'])

        if cfg.DRIVE_TRAIN_TYPE == "PWM_STEERING_THROTTLE":
            #
            # サーボと ESC を用いた RC カー向けドライブトレイン
            # ステアリング用とスロットル用にそれぞれ PWM ピンを使用する
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PulseController

            dt = cfg.PWM_STEERING_THROTTLE
            steering_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt["PWM_STEERING_PIN"]),
                pwm_scale=dt["PWM_STEERING_SCALE"],
                pwm_inverted=dt["PWM_STEERING_INVERTED"])
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=dt["STEERING_LEFT_PWM"],
                                            right_pulse=dt["STEERING_RIGHT_PWM"])

            throttle_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt["PWM_THROTTLE_PIN"]),
                pwm_scale=dt["PWM_THROTTLE_SCALE"],
                pwm_inverted=dt['PWM_THROTTLE_INVERTED'])
            throttle = PWMThrottle(controller=throttle_controller,
                                                max_pulse=dt['THROTTLE_FORWARD_PWM'],
                                                zero_pulse=dt['THROTTLE_STOPPED_PWM'],
                                                min_pulse=dt['THROTTLE_REVERSE_PWM'])
            V.add(steering, inputs=['steering'], threaded=True)
            V.add(throttle, inputs=['throttle'], threaded=True)

        elif cfg.DRIVE_TRAIN_TYPE == "I2C_SERVO":
            #
            # このドライバーは非推奨です。代わりに ``PWM_STEERING_THROTTLE`` を使用してください
            # 将来のリリースで削除されます
            #
            from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

            steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=cfg.STEERING_LEFT_PWM,
                                            right_pulse=cfg.STEERING_RIGHT_PWM)

            throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
            throttle = PWMThrottle(controller=throttle_controller,
                                            max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                            zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                            min_pulse=cfg.THROTTLE_REVERSE_PWM)

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(throttle, inputs=['throttle'], threaded=True)

        elif cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
            dt = cfg.DC_STEER_THROTTLE
            steering = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['LEFT_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['RIGHT_DUTY_PIN']))
            throttle = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['BWD_DUTY_PIN']))

            V.add(steering, inputs=['steering'])
            V.add(throttle, inputs=['throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
            dt = cfg.DC_TWO_WHEEL
            left_motor = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['LEFT_FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['LEFT_BWD_DUTY_PIN']))
            right_motor = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['RIGHT_FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['RIGHT_BWD_DUTY_PIN']))

            V.add(left_motor, inputs=['left/throttle'])
            V.add(right_motor, inputs=['right/throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL_L298N":
            dt = cfg.DC_TWO_WHEEL_L298N
            left_motor = actuator.L298N_HBridge_3pin(
                pins.output_pin_by_id(dt['LEFT_FWD_PIN']),
                pins.output_pin_by_id(dt['LEFT_BWD_PIN']),
                pins.pwm_pin_by_id(dt['LEFT_EN_DUTY_PIN']))
            right_motor = actuator.L298N_HBridge_3pin(
                pins.output_pin_by_id(dt['RIGHT_FWD_PIN']),
                pins.output_pin_by_id(dt['RIGHT_BWD_PIN']),
                pins.pwm_pin_by_id(dt['RIGHT_EN_DUTY_PIN']))

            V.add(left_motor, inputs=['left/throttle'])
            V.add(right_motor, inputs=['right/throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_2PIN":
            #
            # 2 ピン方式の HBridge モーターとサーボを用いた駆動方式
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PulseController

            dt = cfg.SERVO_HBRIDGE_2PIN
            steering_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt['PWM_STEERING_PIN']),
                pwm_scale=dt['PWM_STEERING_SCALE'],
                pwm_inverted=dt['PWM_STEERING_INVERTED'])
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=dt['STEERING_LEFT_PWM'],
                                            right_pulse=dt['STEERING_RIGHT_PWM'])

            motor = actuator.L298N_HBridge_2pin(
                pins.pwm_pin_by_id(dt['FWD_DUTY_PIN']),
                pins.pwm_pin_by_id(dt['BWD_DUTY_PIN']))

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(motor, inputs=["throttle"])

        elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_3PIN":
            #
            # 3 ピン方式の HBridge モーターとサーボを用いた駆動方式
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PulseController

            dt = cfg.SERVO_HBRIDGE_3PIN
            steering_controller = PulseController(
                pwm_pin=pins.pwm_pin_by_id(dt['PWM_STEERING_PIN']),
                pwm_scale=dt['PWM_STEERING_SCALE'],
                pwm_inverted=dt['PWM_STEERING_INVERTED'])
            steering = PWMSteering(controller=steering_controller,
                                            left_pulse=dt['STEERING_LEFT_PWM'],
                                            right_pulse=dt['STEERING_RIGHT_PWM'])

            motor = actuator.L298N_HBridge_3pin(
                pins.output_pin_by_id(dt['FWD_PIN']),
                pins.output_pin_by_id(dt['BWD_PIN']),
                pins.pwm_pin_by_id(dt['DUTY_PIN']))

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(motor, inputs=["throttle"])

        elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
            #
            # このドライバーは非推奨です。代わりに ``SERVO_HBRIDGE_2PIN`` を使用してください
            # 将来のリリースで削除されます
            #
            from donkeycar.parts.actuator import ServoBlaster, PWMSteering
            steering_controller = ServoBlaster(cfg.STEERING_CHANNEL)  # 実際のピン番号
            # PWM パルス値は 100〜200 の範囲である必要がある
            assert(cfg.STEERING_LEFT_PWM <= 200)
            assert(cfg.STEERING_RIGHT_PWM <= 200)
            steering = PWMSteering(controller=steering_controller,
                                   left_pulse=cfg.STEERING_LEFT_PWM,
                                   right_pulse=cfg.STEERING_RIGHT_PWM)

            from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
            motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

            V.add(steering, inputs=['steering'], threaded=True)
            V.add(motor, inputs=["throttle"])

        elif cfg.DRIVE_TRAIN_TYPE == "MM1":
            from donkeycar.parts.robohat import RoboHATDriver
            V.add(RoboHATDriver(cfg), inputs=['steering', 'throttle'])

        elif cfg.DRIVE_TRAIN_TYPE == "PIGPIO_PWM":
            #
            # このドライバーは非推奨です。代わりに ``PWM_STEERING_THROTTLE`` を使用してください
            # 将来のリリースで削除されます
            #
            from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PiGPIO_PWM
            steering_controller = PiGPIO_PWM(cfg.STEERING_PWM_PIN, freq=cfg.STEERING_PWM_FREQ,
                                             inverted=cfg.STEERING_PWM_INVERTED)
            steering = PWMSteering(controller=steering_controller,
                                   left_pulse=cfg.STEERING_LEFT_PWM,
                                   right_pulse=cfg.STEERING_RIGHT_PWM)

            throttle_controller = PiGPIO_PWM(cfg.THROTTLE_PWM_PIN, freq=cfg.THROTTLE_PWM_FREQ,
                                             inverted=cfg.THROTTLE_PWM_INVERTED)
            throttle = PWMThrottle(controller=throttle_controller,
                                   max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                   zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                                   min_pulse=cfg.THROTTLE_REVERSE_PWM)
            V.add(steering, inputs=['steering'], threaded=True)
            V.add(throttle, inputs=['throttle'], threaded=True)
    
        elif cfg.DRIVE_TRAIN_TYPE == "VESC":
            from donkeycar.parts.actuator import VESC
            logger.info("ポート {} で VESC を作成中".format(cfg.VESC_SERIAL_PORT))
            vesc = VESC(cfg.VESC_SERIAL_PORT,
                          cfg.VESC_MAX_SPEED_PERCENT,
                          cfg.VESC_HAS_SENSOR,
                          cfg.VESC_START_HEARTBEAT,
                          cfg.VESC_BAUDRATE,
                          cfg.VESC_TIMEOUT,
                          cfg.VESC_STEERING_SCALE,
                          cfg.VESC_STEERING_OFFSET
                        )
            V.add(vesc, inputs=['steering', 'throttle'])


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
              model_type=model_type, camera_type=camera_type,
              meta=args['--meta'])
    elif args['train']:
        print('代わりに python train.py を使用してください。\n')
