"""車両設定を変更するためのモジュール。

このファイルはアプリケーションの ``manage.py`` スクリプトによって読み込まれ、
車両の性能を設定します。

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)
"""


import os

# パス
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

# 車両
DRIVE_LOOP_HZ = 20      # 車両ループがこの速度より速い場合に一時停止します。
MAX_LOOPS = None        # 正の整数を指定するとこの回数でループを終了できます。

# カメラ
CAMERA_TYPE = "MOCK"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # 既定は RGB=3、モノクロにする場合は 1
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
# CSIC カメラ用 - カメラが回転して取り付けられている場合は下記パラメータを変更すると表示方向が補正されます
CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => なし , 4 => 横反転, 6 => 縦反転)

# IMAGE_LIST カメラ用
# PATH_MASK = "~/mycar/data/tub_1_20-03-12/*.jpg"

#9865、必要な場合のみ上書きします（例: TX2）
PCA9685_I2C_ADDR = 0x40     # I2C アドレス。i2cdetect で確認してください
PCA9685_I2C_BUSNUM = None   # None なら自動検出。Pi ではこれで良いが他のプラットフォームではバス番号を指定

#SSD1306_128_32
USE_SSD1306_128_32 = False    # SSD_1306 OLED ディスプレイを使用するかどうか
SSD1306_128_32_I2C_BUSNUM = 1 # I2C バス番号

# ドライブトレイン
# シャシーやモーター構成を選択します。多くの場合は ``SERVO_ESC`` を使用します。
# ``DC_STEER_THROTTLE`` は HBridge PWM でステアリング用 DC モーターと駆動輪用モーターを制御します
# ``DC_TWO_WHEEL`` は HBridge PWM で左右2つの駆動モーターを制御します
# ``SERVO_HBRIDGE_PWM`` は ServoBlaster を用いて PiZero から直接ステアリングを制御し、HBridge で駆動モーターを制御します
# ``PIGPIO_PWM`` は Raspberry Pi の内部 PWM を使用します
DRIVE_TRAIN_TYPE = "MOCK" # I2C_SERVO|DC_STEER_THROTTLE|DC_TWO_WHEEL|DC_TWO_WHEEL_L298N|SERVO_HBRIDGE_PWM|PIGPIO_PWM|MM1|MOCK

# ステアリング
STEERING_CHANNEL = 1            # 9685 PWM ボードのチャンネル番号 (0-15)
STEERING_LEFT_PWM = 460         # 左いっぱいの PWM 値
STEERING_RIGHT_PWM = 290        # 右いっぱいの PWM 値

# PIGPIO_PWM 用ステアリング設定
STEERING_PWM_PIN = 13           # Broadcom 番号に基づくピン番号
STEERING_PWM_FREQ = 50          # PWM 周波数
STEERING_PWM_INVERTED = False   # PWM を反転させる必要がある場合に True

# スロットル
THROTTLE_CHANNEL = 0            # 9685 PWM ボードのチャンネル番号 (0-15)
THROTTLE_FORWARD_PWM = 500      # 最大前進スロットルの PWM 値
THROTTLE_STOPPED_PWM = 370      # 停止時の PWM 値
THROTTLE_REVERSE_PWM = 220      # 最大後退スロットルの PWM 値

# PIGPIO_PWM 用スロットル設定
THROTTLE_PWM_PIN = 18           # Broadcom 番号に基づくピン番号
THROTTLE_PWM_FREQ = 50          # PWM 周波数
THROTTLE_PWM_INVERTED = False   # PWM を反転させる必要がある場合に True

# DC_STEER_THROTTLE では 1 つのモーターをステアリング、もう 1 つを駆動用に使用します
# 以下の GPIO ピンは ``DRIVE_TRAIN_TYPE=DC_STEER_THROTTLE`` のときのみ使用します
HBRIDGE_PIN_LEFT = 18
HBRIDGE_PIN_RIGHT = 16
HBRIDGE_PIN_FWD = 15
HBRIDGE_PIN_BWD = 13

# DC_TWO_WHEEL - 左右 2 輪を駆動として使用する場合
# 以下の GPIO ピンは ``DRIVE_TRAIN_TYPE=DC_TWO_WHEEL`` のときのみ使用します
HBRIDGE_PIN_LEFT_FWD = 18
HBRIDGE_PIN_LEFT_BWD = 16
HBRIDGE_PIN_RIGHT_FWD = 15
HBRIDGE_PIN_RIGHT_BWD = 13


# 学習設定
# ``DEFAULT_MODEL_TYPE`` はトレーニング時に作成されるモデルタイプを決定します。
# さまざまなニューラルネットワーク設計から選択されます。 ``manage.py train`` や ``drive``
# コマンドに ``--type`` パラメータを渡すことで上書きできます。
DEFAULT_MODEL_TYPE = 'linear'   #(linear|categorical|tflite_linear|tensorrt_linear)
BATCH_SIZE = 128                # 勾配降下1回で使用するレコード数。GPUのメモリが不足する場合は小さくする
TRAIN_TEST_SPLIT = 0.8          # 学習に使用するデータの割合。残りは検証用
MAX_EPOCHS = 100                # データセットを何回繰り返すか
SHOW_PLOT = True                # 最終損失をポップアップ表示するか
VERBOSE_TRAIN = True            # 学習中に進捗バーを表示するか
USE_EARLY_STOP = True           # 精度が向上しなくなったら学習を停止するか
EARLY_STOP_PATIENCE = 5         # 改善が見られなくなるまで待つエポック数
MIN_DELTA = .0005               # 改善とみなす最小損失変化量
PRINT_MODEL_SUMMARY = True      # レイヤーと重みを標準出力に表示
OPTIMIZER = None                # adam, sgd, rmsprop など。None ならデフォルト
LEARNING_RATE = 0.001           # OPTIMIZER を指定したときのみ使用
LEARNING_RATE_DECAY = 0.0       # OPTIMIZER を指定したときのみ使用
SEND_BEST_MODEL_TO_PI = False   # True にすると最良モデルを自動で Pi に送信

PRUNE_CNN = False               # モデルから重みを削除して性能向上を狙います
PRUNE_PERCENT_TARGET = 75       # 目標とする剪定率
PRUNE_PERCENT_PER_ITERATION = 20 # 1 回の剪定で行う割合
PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2 # 剪定中に許容される検証損失の最大増加量
PRUNE_EVAL_PERCENT_OF_DATASET = .05  # 評価に使用するデータセットの割合

# モデル転送オプション
# モデル転送時に重みをコピーする際、いくつかのレイヤーを固定して学習中に変更しないようにできます
FREEZE_LAYERS = False               # False ならすべてのレイヤーを学習で更新
NUM_LAST_LAYERS_TO_TRAIN = 7        # レイヤーを凍結する場合、最後から何層を学習させるか

# Web コントロール
WEB_CONTROL_PORT = 8887             # Web コントローラが待ち受けるポート番号
WEB_INIT_MODE = "user"              # 起動時の制御モード。user|local_angle|local のいずれか。local を指定すると AI モードで開始

# ジョイスティック
USE_JOYSTICK_AS_DEFAULT = False      # True にすると manage.py 実行時に --js オプションなしでジョイスティックを使用
JOYSTICK_MAX_THROTTLE = 0.5         # -1～1 のスロットル値に掛けて最大スピードを制限
JOYSTICK_STEERING_SCALE = 1.0       # ステアリング感度のスケール。負数で方向反転も可能
AUTO_RECORD_ON_THROTTLE = True      # True ならスロットルがゼロ以外のとき自動で記録。False の場合は別のトリガーで切替
CONTROLLER_TYPE = 'xbox'            #(ps3|ps4|xbox|nimbus|wiiu|F710|rc3|MM1|custom) custom を選ぶと `donkey createjs` で生成した my_joystick.py を使用
USE_NETWORKED_JS = False            # ネットワーク越しのジョイスティック制御を受け付けるか
NETWORK_JS_SERVER_IP = None         # ネットワークジョイスティック制御を使う場合のサーバー IP
JOYSTICK_DEADZONE = 0.01            # 0 以外の場合、この値以下のスロットルでは記録を開始しない
JOYSTICK_THROTTLE_DIR = 1.0         # -1.0 で前後を反転、1.0 でジョイスティックの自然な前後を使用
USE_FPV = False                     # カメラ映像を FPV Web サーバーへ送信
JOYSTICK_DEVICE_FILE = "/dev/input/js0" # ジョイスティックデバイスのパス

# カテゴリカルモデルの場合、学習済みスロットルの上限を制限します
# この値は学習用 PC の ``config.py`` と ``robot.py`` で一致させることが非常に重要で、
# 一度決めたら変更しないのが理想です
MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8

# RNN または 3D モデル用
SEQUENCE_LENGTH = 3             # 時系列画像を用いるモデルの場合の枚数

# IMU
HAVE_IMU = False                # True で Mpu6050 パーツを追加しデータを記録。次の設定と併用可能
IMU_SENSOR = 'mpu6050'          # (mpu6050|mpu9250)
IMU_DLP_CONFIG = 0              # デジタルローパスフィルタ設定 (0:250Hz, 1:184Hz, 2:92Hz, 3:41Hz, 4:20Hz, 5:10Hz, 6:5Hz)

# SOMBRERO
HAVE_SOMBRERO = False           # Donkeycar ストアの Sombrero Hat を使用するとき True。Hat 上で PWM が有効になる

# ROBOHAT MM1
HAVE_ROBOHAT = False            # Robotics Masters 製 Robo HAT MM1 を使用する場合は True。RC 制御に切り替わる
MM1_STEERING_MID = 1500         # 直進できない場合はこの値を調整
MM1_MAX_FORWARD = 2000          # 最大前進スロットル。値が大きいほど速い
MM1_STOPPED_PWM = 1500
MM1_MAX_REVERSE = 1000          # 最大後退スロットル。値が小さいほど速い
MM1_SHOW_STEERING_VALUE = False
# シリアルポート
# -- Pi のデフォルト: '/dev/ttyS0'
# -- Jetson Nano: '/dev/ttyTHS1'
# -- Google Coral: '/dev/ttymxc0'
# -- Windows: 'COM3', Arduino: '/dev/ttyACM0'
# -- MacOS/Linux: 'ls /dev/tty.*' で適切なポートを確認してください
#    例:'/dev/tty.usbmodemXXXXXX' などに置き換えます
MM1_SERIAL_PORT = '/dev/ttyS0'  # MM1 のデータ送受信用シリアルポート

# 記録オプション
RECORD_DURING_AI = False        # 通常 AI モード中は記録しません。True にすると AI 用の画像とステアリングを記録しますが学習には注意
AUTO_CREATE_NEW_TUB = False     # 記録開始時に新しい tub (tub_YY_MM_DD) を作成するか、既存の data ディレクトリに追加するか

# LED
HAVE_RGB_LED = False            # 例: https://www.amazon.com/dp/B07BNRZWNF のような RGB LED を使用するか
LED_INVERT = False              # コモンアノードの場合は True。例: https://www.amazon.com/Xia-Fly-Tri-Color-Emitting-Diffused/dp/B07MYJQP8B

# LED ボードの PWM 出力ピン番号
# 物理ピン番号。参考: https://www.raspberrypi-spy.co.uk/2012/06/simple-guide-to-the-rpi-gpio-header-and-pins/
LED_PIN_R = 12
LED_PIN_G = 10
LED_PIN_B = 16

# LED の初期カラー (0-100)
LED_R = 0
LED_G = 0
LED_B = 1

# 記録数インジケーターの LED カラー
REC_COUNT_ALERT = 1000          # この回数記録すると点滅を開始
REC_COUNT_ALERT_CYC = 15        # REC_COUNT_ALERT 件ごとに点滅する周期 (1/20 秒単位)
REC_COUNT_ALERT_BLINK_RATE = 0.4 # LED の点滅速度 (秒)

# 先頭は記録数、次に (r, g, b) (0-100) のタプル。
# 記録数がその値を超えると対応する色に変化します
RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
            (3000, (5, 5, 5)),
            (5000, (5, 2, 0)),
            (10000, (0, 5, 0)),
            (15000, (0, 5, 5)),
            (20000, (0, 0, 5)), ]


# モデル再読み込み時の LED カラー (0-100)
MODEL_RELOADED_LED_R = 100
MODEL_RELOADED_LED_G = 0
MODEL_RELOADED_LED_B = 0


# ビヘイビア
# Behavioral Neural Network モデルを学習するときは振る舞いの一覧を作成し、
# ``TRAIN_BEHAVIORS`` を True に設定して ``BEHAVIOR_LED_COLORS`` で各振る舞いに色を割り当てます
TRAIN_BEHAVIORS = False
BEHAVIOR_LIST = ['Left_Lane', "Right_Lane"]
BEHAVIOR_LED_COLORS =[ (0, 10, 0), (10, 0, 0) ] #RGB タプル 各チャンネル 0-100

# ローカライザ
# ローカライザはコース上の位置を予測するニューラルネットワークです。
# まだ実験的な機能ですが、コースを ``NUM_LOCATIONS`` 個のセグメントに分割した場合に
# 現在のセグメントを推定できます
TRAIN_LOCALIZER = False
NUM_LOCATIONS = 10
BUTTON_PRESS_NEW_TUB = False # 有効にすると X ボタンを押すたびに新しい tub を作成し、走行距離ごとにデータを分けやすくします

# DonkeyGym
# Ubuntu Linux では、シミュレータを仮想ドンキーとして利用し、通常どおり ``python manage.py drive`` を実行して仮想カーを操作できます。
# これを有効にするとシミュレータのパスと環境を設定できます。
# 以下からシミュレータをダウンロードしてください: https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip
# 展開後に ``DONKEY_SIM_PATH`` を変更してください。
DONKEY_GYM = True
# レースリーグで走行する場合は "remote" を指定するか、手動でシムを起動する場合も "remote" を使用します
DONKEY_SIM_PATH = "path to sim" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" または "remote"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
GYM_CONF = { "img_h" : IMAGE_H, "img_w" : IMAGE_W, "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100 } # body style(donkey|bare|car01) body rgb 0-255
GYM_CONF["racer_name"] = "Your Name"
GYM_CONF["country"] = "Place"
GYM_CONF["bio"] = "I race robots."

SIM_HOST = "127.0.0.1"              # Virtual Race League では "trainmydonkey.com" を使用
SIM_ARTIFICIAL_LATENCY = 0          # リモートサーバー利用時の遅延を模擬するためのミリ秒単位のレイテンシ（100～400 程度が妥当）

# カメラ映像のネットワーク配信
# カメラフィードを配信する TCP サービスを作成します
PUB_CAMERA_IMAGES = False

# レース時に AI にブーストを与える設定
AI_LAUNCH_DURATION = 0.0            # AI がこの秒数だけスロットルを出力
AI_LAUNCH_THROTTLE = 0.0            # AI が出力するスロットル値
AI_LAUNCH_ENABLE_BUTTON = 'R2'      # このキーを押すとブーストが有効になる。誤作動防止のため毎回有効化が必要
AI_LAUNCH_KEEP_ENABLED = False      # False（デフォルト）の場合、毎回ボタンを押して有効化する。True の場合は "local" モードに入るたびに有効

# すべてのモデルに対して AI パイロットのスロットル出力をスケーリング
AI_THROTTLE_MULT = 1.0              # ニューラルネットワーク出力のスロットル値に掛ける倍率

# 経路追従
PATH_FILENAME = "donkey_path.pkl"   # 経路を保存するファイル名
PATH_SCALE = 5.0                    # ウェブページ上の経路表示スケール
PATH_OFFSET = (0, 0)                # 255,255 がマップの中心。原点表示をずらすオフセット
PATH_MIN_DIST = 0.3                 # この距離 (m) 移動するごとに経路ポイントを保存
PID_P = -10.0                       # PID パスフォロワーの P 値
PID_I = 0.000                       # PID パスフォロワーの I 値
PID_D = -0.2                        # PID パスフォロワーの D 値
PID_THROTTLE = 0.2                  # 経路追従中の一定スロットル値
USE_CONSTANT_THROTTLE = False       # 経路記録時のスロットル値を使うか一定値を使うか
SAVE_PATH_BTN = "cross"             # 経路保存に使うジョイスティックボタン
RESET_ORIGIN_BTN = "triangle"       # 原点に戻すジョイスティックボタン

# Intel Realsense D435/D435i 深度カメラ
REALSENSE_D435_RGB = True       # RGB 画像を取得する場合は True
REALSENSE_D435_DEPTH = True     # 深度を画像配列として取得する場合は True
REALSENSE_D435_IMU = False      # D435i のみ: IMU データを取得する場合は True
REALSENSE_D435_ID = None        # カメラのシリアル番号。1 台のみなら None で自動検出

# 一時停止標識検出
STOP_SIGN_DETECTOR = False
STOP_SIGN_MIN_SCORE = 0.2
STOP_SIGN_SHOW_BOUNDING_BOX = True
