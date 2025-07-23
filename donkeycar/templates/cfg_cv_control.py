"""`path_follow` テンプレートの設定を定義するモジュール。

`manage.py` が本ファイルを読み込み、車両の性能パラメータを設定する。
必要に応じてこのファイルで設定を上書きできるが、`update` 操作では変更されない。
"""

import os


import os

#
# ファイルパス
#
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')


#
# 車両ループ
#
DRIVE_LOOP_HZ = 20      # この速度より速い場合、車両ループは一時停止します。
MAX_LOOPS = None        # 正の整数を指定すると、車両ループはこの回数で中断します。


#
# カメラ設定
#
CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
IMAGE_W = 320
IMAGE_H = 240
IMAGE_DEPTH = 3         # 既定はRGB=3。モノクロの場合は1
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAMERA_INDEX = 0  # 'WEBCAM' や 'CVCAM' で複数カメラ接続時に使用
# CSIC カメラ用 - カメラが回転して取り付けられている場合、このパラメータを変更すると出力フレームの向きを補正できます
CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => なし, 4 => 水平反転, 6 => 垂直反転)
BGR2RGB = False  # True で BGR から RGB へ変換。opencv が必要

# IMAGE_LIST カメラ用
PATH_MASK = "~/mycar/data/tub_1_20-03-12/*.jpg"


#
# PCA9685 設定。必要な場合のみ変更 (例: TX2)
#
PCA9685_I2C_ADDR = 0x40     # I2C アドレス。i2cdetect で確認
PCA9685_I2C_BUSNUM = None   # None なら自動検出。Pi では問題ないが他プラットフォームは指定


#
# SSD1306_128_32
#
USE_SSD1306_128_32 = False    # SSD_1306 OLED ディスプレイを有効にする
SSD1306_128_32_I2C_ROTATION = 0 # 0: 正常, 1: 90度時計回り, 2: 180度反転, 3: 270度
SSD1306_RESOLUTION = 1 # 1: 128x32, 2: 128x64


#
# 計測済みロボット特性
#
AXLE_LENGTH = 0.03     # 車軸の長さ（左右タイヤ間の距離、メートル）
WHEEL_BASE = 0.1       # 前後輪間の距離（メートル）
WHEEL_RADIUS = 0.0315  # タイヤ半径（メートル）
MIN_SPEED = 0.1        # 車が停止する速度未満とみなす最小速度（m/s）
MAX_SPEED = 3.0        # 最大スピード（m/s）。スロットル1.0時の速度
MIN_THROTTLE = 0.1     # MIN_SPEED に対応するスロットル値 (0～1.0)。これ未満では停止
MAX_STEERING_ANGLE = 3.141592653589793 / 4  # 車両型ロボットの最大操舵角 (ステアリング=-1 時のタイヤ角、ラジアン)


#
# DRIVE_TRAIN_TYPE
# 使用するシャーシとモーター構成を指定します。
# 各駆動方式の詳細は https://docs.donkeycar.com/parts/actuators/ を参照してください。
# 以下から一つを選び、対応する設定セクションを更新してください。
#
# "PWM_STEERING_THROTTLE" は一般的なRCカー同様、ステアリングサーボと ESC を PWM ピン2本で制御
# "MM1" Robo HAT MM1 ボード
# "SERVO_HBRIDGE_2PIN" ステアリング用サーボと 2ピンモードの HBridge モータードライバー
# "SERVO_HBRIDGE_3PIN" ステアリング用サーボと 3ピンモードの HBridge モータードライバー
# "DC_STEER_THROTTLE" は HBridge PWM でステアリング用DCモーターと駆動用モーターを制御
# "DC_TWO_WHEEL" は2ピンモードの HBridge で左右の駆動モーターを制御
# "DC_TWO_WHEEL_L298N" は3ピンモードの HBridge(L298N) で左右の駆動モーターを制御
# "MOCK" 駆動系なし。テストリグで他機能を試すのに使用
# (非推奨) "SERVO_HBRIDGE_PWM" は PiZero の ServoBlaster でステアリングを直接PWM制御
#                                  駆動モーターには HBridge を使用
# (非推奨) "PIGPIO_PWM" は Raspberry の内部 PWM を利用
# (非推奨) "I2C_SERVO" は PCA9685 サーボコントローラでステアリングと ESC を制御 (RCカー方式)
#
DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"

#
# PWM_STEERING_THROTTLE drivetrain configuration
#
# ステアリングサーボと ESC を搭載した RCカー向け駆動系
# ステアリング用に PwmPin を、スロットル用にもう一つの PwmPin を使用
# 基本PWM周波数は60Hz想定。異なる場合は PWM_xxxx_SCALE でパルス幅を調整
#
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:40.1",   # ステアリングサーボの PWM 出力ピン
    "PWM_STEERING_SCALE": 1.0,              # PWM 周波数が60Hz以外の場合の補正用。舵角調整ではない
    "PWM_STEERING_INVERTED": False,         # ハードウェアが反転PWMを必要とする場合は True
    "PWM_THROTTLE_PIN": "PCA9685.1:40.0",   # ESC の PWM 出力ピン
    "PWM_THROTTLE_SCALE": 1.0,              # PWM 周波数が60Hz以外の場合の補正用。速度調整ではない
    "PWM_THROTTLE_INVERTED": False,         # ハードウェアが反転PWMを必要とする場合は True
    "STEERING_LEFT_PWM": 460,               # 左いっぱいの PWM 値
    "STEERING_RIGHT_PWM": 290,              # 右いっぱいの PWM 値
    "THROTTLE_FORWARD_PWM": 500,            # 最大前進スロットルの PWM 値
    "THROTTLE_STOPPED_PWM": 370,            # 停止時の PWM 値
    "THROTTLE_REVERSE_PWM": 220,            # 最大後進スロットルの PWM 値
}

#
# I2C_SERVO (PWM_STEERING_THROTTLE 推奨のため非推奨)
#
STEERING_CHANNEL = 1            #(非推奨) 9685 PWM ボードのチャンネル 0-15
STEERING_LEFT_PWM = 460         # 左いっぱいの PWM 値
STEERING_RIGHT_PWM = 290        # 右いっぱいの PWM 値
THROTTLE_CHANNEL = 0            #(非推奨) 9685 PWM ボードのチャンネル 0-15
THROTTLE_FORWARD_PWM = 500      # 最大前進スロットルの PWM 値
THROTTLE_STOPPED_PWM = 370      # 停止時の PWM 値
THROTTLE_REVERSE_PWM = 220      # 最大後進スロットルの PWM 値

#
# PIGPIO_PWM (PWM_STEERING_THROTTLE 推奨のため非推奨)
#
STEERING_PWM_PIN = 13           #(非推奨) Broadcom 番号によるピン指定
STEERING_PWM_FREQ = 50          # PWM の周波数
STEERING_PWM_INVERTED = False   # PWM を反転する必要がある場合
THROTTLE_PWM_PIN = 18           #(非推奨) Broadcom 番号によるピン指定
THROTTLE_PWM_FREQ = 50          # PWM の周波数
THROTTLE_PWM_INVERTED = False   # PWM を反転する必要がある場合

#
# SERVO_HBRIDGE_2PIN 駆動系の設定
# - ステアリングサーボと HBridge を 2 ピンモード(2 つの PWM ピン)で構成します
# - サーボは 1ms(全逆転) 〜 2ms(全前進) の PWM パルスを受け付け、1.5ms が中立です
# - モーターは前進用と後進用の 2 本の PWM ピンで制御されます
# - PWM ピンのデューティ比は 0(LOW)〜1(100% HIGH) で、モーターへの供給電力に比例します
# - 前進時は後進 PWM を 0 デューティにし、後進時は前進 PWM を 0 デューティにします
# - 両 PWM を 0(LOW) にするとモーターを切り離して惰性で停止します
# - 両 PWM を 100%(HIGH) にするとブレーキがかかります
#
# ピン指定文字列の形式:
# - RPI_GPIO : RPi/Nano のヘッダーピン出力
#   - BOARD : 物理ピン番号
#   - BCM : Broadcom GPIO 番号
#   - 例 "RPI_GPIO.BOARD.18"
# - PIGPIO : pigpio サーバ経由で出力
#   - BCM ピン番号方式のみ使用できます
#   - 例 "PIGPIO.BCM.13"
# - PCA9685 : PCA9685 のピン出力を使用
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例 "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に組み合わせ可能ですが、RPI_GPIO と PIGPIO の併用は推奨されません
#
SERVO_HBRIDGE_2PIN = {
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # 前進用デューティ比出力ピン
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 後進用デューティ比出力ピン
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",       # ステアリングサーボへの PWM ピン
    "PWM_STEERING_SCALE": 1.0,        # PWM 周波数 60Hz 以外の場合の補正。舵角調整ではない
    "PWM_STEERING_INVERTED": False,   # ハードウェアが反転 PWM を要求する場合 True
    "STEERING_LEFT_PWM": 460,         # 左いっぱいの PWM 値（`donkey calibrate` で計測）
    "STEERING_RIGHT_PWM": 290,        # 右いっぱいの PWM 値（`donkey calibrate` で計測）
}

#
# SERVO_HBRIDGE_3PIN 駆動系の設定
# - ステアリングサーボと HBridge を 3 ピンモード(2 つの TTL ピン + 1 PWM ピン)で構成します
# - サーボは 1ms(全逆転) 〜 2ms(全前進) の PWM パルスを受け付け、1.5ms が中立です
# - モーターは 3 本のピンで制御され、前進用 TTL、後進用 TTL、モーター電力用 PWM からなります
# - PWM ピンのデューティ比は 0(LOW)〜1(100% HIGH) で、モーターへの供給電力に比例します
# - 前進時は forward ピンを HIGH、backward ピンを LOW にします
# - 後進時は forward ピンを LOW、backward ピンを HIGH にします
# - 両 TTL ピンを LOW にするとモーターを切り離して惰性停止し、両方 HIGH でブレーキになります
#
# ピン指定文字列の形式:
# - RPI_GPIO : RPi/Nano のヘッダーピン出力
#   - BOARD : 物理ピン番号
#   - BCM : Broadcom GPIO 番号
#   - 例 "RPI_GPIO.BOARD.18"
# - PIGPIO : pigpio サーバ経由で出力
#   - BCM ピン番号方式のみ使用できます
#   - 例 "PIGPIO.BCM.13"
# - PCA9685 : PCA9685 のピン出力を使用
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例 "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に組み合わせ可能ですが、RPI_GPIO と PIGPIO の併用は推奨されません
#
SERVO_HBRIDGE_3PIN = {
    "FWD_PIN": "RPI_GPIO.BOARD.18",   # 前進を有効にする TTL ピン
    "BWD_PIN": "RPI_GPIO.BOARD.16",   # 後進を有効にする TTL ピン
    "DUTY_PIN": "RPI_GPIO.BOARD.35",  # モーター出力用 PWM ピン
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",   # ステアリングサーボへの PWM ピン
    "PWM_STEERING_SCALE": 1.0,        # PWM 周波数 60Hz 以外の場合の補正。舵角調整ではない
    "PWM_STEERING_INVERTED": False,   # ハードウェアが反転 PWM を要求する場合 True
    "STEERING_LEFT_PWM": 460,         # 左いっぱいの PWM 値（`donkey calibrate` で計測）
    "STEERING_RIGHT_PWM": 290,        # 右いっぱいの PWM 値（`donkey calibrate` で計測）
}

#
# DRIVETRAIN_TYPE == "SERVO_HBRIDGE_PWM" (SERVO_HBRIDGE_2PIN 推奨のため非推奨)
# - ステアリングサーボと HBridge を 2 ピンモード（PWM ピンを 2 本使用）で構成
# - ServoBlaster ライブラリを利用するが標準ではインストールされていないため、
#   使用するにはインストールが必要
# - サーボは 1 ミリ秒（全逆転）〜 2 ミリ秒（全前進）の PWM パルスを受け取り、
#   1.5 ミリ秒が中立
# - モーターは前進用と後進用の 2 本の PWM ピンで制御される
# - PWM ピンは 0（完全 LOW）〜1（完全 HIGH）のデューティ比を出力し、
#   モーターへの供給電力に比例する
# - 前進時は逆転 PWM を 0 デューティ、後進時は前進 PWM を 0 デューティにする
# - 両 PWM を 0（LOW）にするとモーターを切り離して惰性停止し、
#   両 PWM を 100%（HIGH）にするとブレーキがかかる
#
HBRIDGE_PIN_FWD = 18       # 前進用デューティ比を出力するピン
HBRIDGE_PIN_BWD = 16       # 後進用デューティ比を出力するピン
STEERING_CHANNEL = 0       # ステアリング制御用の PCA9685 チャンネル
STEERING_LEFT_PWM = 460    # 左いっぱいの PWM 値（`donkey calibrate` で計測）
STEERING_RIGHT_PWM = 290   # 右いっぱいの PWM 値（`donkey calibrate` で計測）

#
# DC_STEER_THROTTLE 駆動系: 1 つのモーターで操舵し、もう 1 つで走行します
# - L298N 型モータードライバーを 2 ピン配線で利用し、各モーターにつき前進/後進用の PWM ピンを使用します
#
# DRIVE_TRAIN_TYPE=DC_STEER_THROTTLE の GPIO ピン設定
# - RPI_GPIO : RPi/Nano のヘッダーピン出力
#   - BOARD : 物理ピン番号
#   - BCM : Broadcom GPIO 番号
#   - 例 "RPI_GPIO.BOARD.18"
# - PIGPIO : pigpio サーバ経由で出力
#   - BCM ピン番号方式のみ使用できます
#   - 例 "PIGPIO.BCM.13"
# - PCA9685 : PCA9685 のピン出力を使用
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例 "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に混在できますが、RPI_GPIO と PIGPIO の併用は推奨されません
#
DC_STEER_THROTTLE = {
    "LEFT_DUTY_PIN": "RPI_GPIO.BOARD.18",   # 左舵用デューティ比出力ピン
    "RIGHT_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 右舵用デューティ比出力ピン
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",    # 前進用デューティ比出力ピン
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",    # 後進用デューティ比出力ピン
}

#
# DC_TWO_WHEEL 駆動系のピン設定
# - L298N_HBridge_2pin ドライバーを利用した左右独立駆動です
# - 各車輪は前進・後進用の 2 本の PWM ピンで制御されます
# - PWM ピンのデューティ比は 0(LOW)〜1(100% HIGH) で、モーターへの供給電力に比例します
# - 前進時は後進 PWM を 0、後進時は前進 PWM を 0 にします
# - 両 PWM を 0(LOW) にするとモーターが切り離され惰性停止し、100%(HIGH) でブレーキとなります
#
# ピン指定文字列の形式:
# - RPI_GPIO : RPi/Nano のヘッダーピン出力
#   - BOARD : 物理ピン番号
#   - BCM : Broadcom GPIO 番号
#   - 例 "RPI_GPIO.BOARD.18"
# - PIGPIO : pigpio サーバ経由で出力
#   - BCM ピン番号方式のみ使用できます
#   - 例 "PIGPIO.BCM.13"
# - PCA9685 : PCA9685 のピン出力を使用
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例 "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に混在できますが、RPI_GPIO と PIGPIO の併用は推奨されません
#
DC_TWO_WHEEL = {
    "LEFT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # 左輪前進用デューティ比出力ピン
    "LEFT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 左輪後進用デューティ比出力ピン
    "RIGHT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.15", # 右輪前進用デューティ比出力ピン
    "RIGHT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.13", # 右輪後進用デューティ比出力ピン
}

#
# DC_TWO_WHEEL_L298N 駆動系のピン設定
# - L298N_HBridge_3pin ドライバーを使用した左右独立駆動です
# - 各車輪は 3 本のピンで制御され、前進用 TTL、後進用 TTL、モーター電力用 PWM から構成されます
# - PWM ピンのデューティ比は 0(LOW)〜1(100% HIGH) で、モーターへの供給電力に比例します
# - 前進時は forward ピンを HIGH、backward ピンを LOW にします
# - 後進時は forward ピンを LOW、backward ピンを HIGH にします
# - 両 TTL ピンを LOW にするとモーターを切り離し惰性停止、両方 HIGH でブレーキとなります
#
# DRIVE_TRAIN_TYPE=DC_TWO_WHEEL_L298N の GPIO ピン設定
# - RPI_GPIO : RPi/Nano のヘッダーピン出力
#   - BOARD : 物理ピン番号
#   - BCM : Broadcom GPIO 番号
#   - 例 "RPI_GPIO.BOARD.18"
# - PIGPIO : pigpio サーバ経由で出力
#   - BCM ピン番号方式のみ使用できます
#   - 例 "PIGPIO.BCM.13"
# - PCA9685 : PCA9685 のピン出力を使用
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例 "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に混在できますが、RPI_GPIO と PIGPIO の併用は推奨されません
#
DC_TWO_WHEEL_L298N = {
    "LEFT_FWD_PIN": "RPI_GPIO.BOARD.16",        # 左輪前進を有効にする TTL ピン
    "LEFT_BWD_PIN": "RPI_GPIO.BOARD.18",        # 左輪後進を有効にする TTL ピン
    "LEFT_EN_DUTY_PIN": "RPI_GPIO.BOARD.22",    # 左モーター速度用 PWM ピン

    "RIGHT_FWD_PIN": "RPI_GPIO.BOARD.15",       # 右輪前進を有効にする TTL ピン
    "RIGHT_BWD_PIN": "RPI_GPIO.BOARD.13",       # 右輪後進を有効にする TTL ピン
    "RIGHT_EN_DUTY_PIN": "RPI_GPIO.BOARD.11",   # 右モーター速度用 PWM ピン
}



#
# 入力コントローラ
#
#WEB CONTROL
WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # Web コントローラが待ち受けるポート番号
WEB_INIT_MODE = "user"              # 起動時のモード。user|local_angle|local のいずれか。local なら AI モードで開始

#JOYSTICK
USE_JOYSTICK_AS_DEFAULT = False      # True なら manage.py 起動時に --js オプション不要
JOYSTICK_MAX_THROTTLE = 0.5         # -1〜1 のスロットル値に掛けて最大速度を制限する係数
JOYSTICK_STEERING_SCALE = 1.0       # ステアリング感度の調整係数。負値で方向反転
AUTO_RECORD_ON_THROTTLE = False     # True ならスロットルが 0 でないとき常に録画。False なら別のトリガーで切替
CONTROLLER_TYPE = 'xbox'            #(ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom) custom は `donkey createjs` 生成の my_joystick.py を使用
USE_NETWORKED_JS = False            # ネットワーク越しのジョイスティック入力を受け付けるか
NETWORK_JS_SERVER_IP = None         # ネットワークジョイスティック使用時に接続するサーバ IP
JOYSTICK_DEADZONE = 0.01            # この値以下のスロットルでは録画を開始しない
JOYSTICK_THROTTLE_DIR = -1.0         # -1.0 で前後を反転、1.0 で自然な方向
USE_FPV = False                     # カメラ映像を FPV Web サーバへ送信
JOYSTICK_DEVICE_FILE = "/dev/input/js0" # ジョイスティックのデバイスファイル


#SOMBRERO
HAVE_SOMBRERO = False           # Donkeycar ストアの Sombrero Hat 使用時は True にして PWM を有効化

#PIGPIO RC コントロール
STEERING_RC_GPIO = 26
THROTTLE_RC_GPIO = 20
DATA_WIPER_RC_GPIO = 19
PIGPIO_STEERING_MID = 1500         # 直進できない場合に調整する値
PIGPIO_MAX_FORWARD = 2000          # 最大前進スロットル値。大きいほど速い
PIGPIO_STOPPED_PWM = 1500
PIGPIO_MAX_REVERSE = 1000          # 最大後進スロットル値。小さいほど速い
PIGPIO_SHOW_STEERING_VALUE = False
PIGPIO_INVERT = False
PIGPIO_JITTER = 0.025   # この値未満では信号がないものとみなす


# ROBOHAT MM1 コントローラ
MM1_STEERING_MID = 1500         # 直進できない場合に調整する値
MM1_MAX_FORWARD = 2000          # 最大前進スロットル値。大きいほど速い
MM1_STOPPED_PWM = 1500
MM1_MAX_REVERSE = 1000          # 最大後進スロットル値。小さいほど速い
MM1_SHOW_STEERING_VALUE = False
# シリアルポート
# -- Pi の既定: '/dev/ttyS0'
# -- Jetson Nano: '/dev/ttyTHS1'
# -- Google Coral: '/dev/ttymxc0'
# -- Windows: 'COM3', Arduino: '/dev/ttyACM0'
# -- MacOS/Linux: 適切なポートを `ls /dev/tty.*` で確認して置き換える
#  例: '/dev/tty.usbmodemXXXXXX'
MM1_SERIAL_PORT = '/dev/ttyS0'  # MM1 とのデータ送受信用シリアルポート


#
# LOGGING
#
HAVE_CONSOLE_LOGGING = True
LOGGING_LEVEL = 'INFO'          # Python のログレベル: 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
LOGGING_FORMAT = '%(message)s'  # Python のログフォーマット - https://docs.python.org/3/library/logging.html#formatter-objects


#
# MQTT TELEMETRY
#
HAVE_MQTT_TELEMETRY = False
TELEMETRY_DONKEY_NAME = 'my_robot1234'
TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
TELEMETRY_MQTT_JSON_ENABLE = False
TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
TELEMETRY_MQTT_BROKER_PORT = 1883
TELEMETRY_PUBLISH_PERIOD = 1
TELEMETRY_LOGGING_ENABLE = True
TELEMETRY_LOGGING_LEVEL = 'INFO' # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
TELEMETRY_LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
TELEMETRY_DEFAULT_TYPES = 'float,float'


#
# PERFORMANCE MONITOR
#
HAVE_PERFMON = False


#
# RECORD OPTIONS
#
RECORD_DURING_AI = False        # 通常 AI モードでは録画しない。True にすると AI の画像とステアリングを記録するが、学習には使用しないこと
AUTO_CREATE_NEW_TUB = False     # True なら録画開始時に新しい tub (tub_YY_MM_DD) ディレクトリを作成。False なら既存ディレクトリに追加


#
# LED
#
# RGB LED を使う場合は True にする (例: https://www.amazon.com/dp/B07BNRZWNF)
HAVE_RGB_LED = False
# 共通アノードの LED を使う場合は True (例: https://www.amazon.com/Xia-Fly-Tri-Color-Emitting-Diffused/dp/B07MYJQP8B)
LED_INVERT = False

#PWM 出力用 LED ボードのピン番号
# これは物理ピン番号。参考: https://www.raspberrypi-spy.co.uk/2012/06/simple-guide-to-the-rpi-gpio-header-and-pins/
LED_PIN_R = 12
LED_PIN_G = 10
LED_PIN_B = 16

#LED status color, 0-100
LED_R = 0
LED_G = 0
LED_B = 1

#LED Color for record count indicator
REC_COUNT_ALERT = 1000          # この件数を超えたら点滅で通知
REC_COUNT_ALERT_CYC = 15        # REC_COUNT_ALERT 件ごとに点滅するサイクル数(1/20秒単位)
REC_COUNT_ALERT_BLINK_RATE = 0.4 # LED 点滅のオンオフ周期(秒)

# 最初の数値がレコード数、2 番目のタプルが色 (r, g, b) (0-100)
# レコード数がその値を超えると指定した色を使用
RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
            (3000, (5, 5, 5)),
            (5000, (5, 2, 0)),
            (10000, (0, 5, 0)),
            (15000, (0, 5, 5)),
            (20000, (0, 0, 5)), ]

#LED status color, 0-100, for model reloaded alert
MODEL_RELOADED_LED_R = 100
MODEL_RELOADED_LED_G = 0
MODEL_RELOADED_LED_B = 0


#
# DonkeyGym
#
# Ubuntu Linux 環境のみ、シミュレータを仮想ドンキーとして使用し
# 通常の `python manage.py drive` コマンドで仮想車を操作できます。
# そのための設定で、シミュレータ実行ファイルのパスや環境を指定します。
# https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip からバイナリを取得し、展開後 DONKEY_SIM_PATH を修正してください。
DONKEY_GYM = False
DONKEY_SIM_PATH = "シミュレータのパス" # 仮想レースリーグで走らせる場合は "remote" を指定。手動でシムを起動する場合も "remote" を使用
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
GYM_CONF = { "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # body style(donkey|bare|car01) body rgb 0-255
GYM_CONF["racer_name"] = "あなたの名前"
GYM_CONF["country"] = "場所"
GYM_CONF["bio"] = "私はロボットレースをします。"

SIM_HOST = "127.0.0.1"              # 仮想レースリーグでは "trainmydonkey.com" を使用
SIM_ARTIFICIAL_LATENCY = 0          # 操作遅延のエミュレート用ミリ秒単位遅延。リモートサーバ利用時は 100～400 が適当

# シミュレータから情報を保存 (pln)
SIM_RECORD_LOCATION = False
SIM_RECORD_GYROACCEL= False
SIM_RECORD_VELOCITY = False
SIM_RECORD_LIDAR = False

# TCP ソケット経由でカメラ画像を配信
# カメラ映像を公開する TCP サービスを作成する際に使用
PUB_CAMERA_IMAGES = False


#
# AI Overrides
#
# 起動時に AI を優先してスロットルを出力するランチモード設定
AI_LAUNCH_DURATION = 0.0            # AI がスロットルを出力する時間(秒)
AI_LAUNCH_THROTTLE = 0.0            # AI が出力するスロットル値
AI_LAUNCH_ENABLE_BUTTON = 'R2'      # このキーを押すとブーストが有効になる。誤動作防止のため毎回有効化する必要あり
AI_LAUNCH_KEEP_ENABLED = False      # False の場合、AI_LAUNCH_ENABLE_BUTTON を都度押す必要がある。True だと local モードに入るたび自動で有効

# スロットル出力スケール: すべてのモデルで AI パイロットのスロットル値に乗算
AI_THROTTLE_MULT = 1.0              # すべての NN モデルのスロットル値に掛ける倍率


#
# Intel Realsense D435/D435i 深度認識カメラ
#
REALSENSE_D435_RGB = True       # RGB 画像を取得する場合は True
REALSENSE_D435_DEPTH = False    # 深度画像を取得する場合は True
REALSENSE_D435_IMU = False      # D435i の場合に IMU データを取得するなら True
REALSENSE_D435_ID = None        # カメラのシリアル番号。1 台のみなら None で自動検出


#
# 一時停止標識検出
#
STOP_SIGN_DETECTOR = False
STOP_SIGN_MIN_SCORE = 0.2
STOP_SIGN_SHOW_BOUNDING_BOX = True
STOP_SIGN_MAX_REVERSE_COUNT = 10    # 停止標識検出時に何回バックするか。0 でバック無効
STOP_SIGN_REVERSE_THROTTLE = -0.5     # 停止標識検出時にバックする際のスロットル

#
# FPS 計測
#
SHOW_FPS = False
FPS_DEBUG_INTERVAL = 10    # 周波数情報を表示する間隔(秒)

#
# コンピュータビジョン用テンプレート
#
# どのパーツを自動操縦に使うか設定する。必要に応じて自作の autopilot に変更
CV_CONTROLLER_MODULE = "donkeycar.parts.line_follower"
CV_CONTROLLER_CLASS = "LineFollower"
CV_CONTROLLER_INPUTS = ['cam/image_array']
CV_CONTROLLER_OUTPUTS = ['pilot/steering', 'pilot/throttle', 'cv/image_array']
CV_CONTROLLER_CONDITION = "run_pilot"

# LineFollower - line color and detection area
SCAN_Y = 100          # 上端から走査を開始する画素数
SCAN_HEIGHT = 20      # 走査範囲の高さ(ピクセル)
COLOR_THRESHOLD_LOW  = (0, 50, 50)    # HSV での濃い黄色 (opencv HSV 色相は 0..179、彩度と明度は 0..255)
COLOR_THRESHOLD_HIGH = (50, 255, 255) # HSV での明るい黄色 (opencv HSV 色相は 0..179、彩度と明度は 0..255)

# LineFollower - 期待するライン位置と検出閾値
TARGET_PIXEL = None   # None でなければ黄色ラインの想定水平位置(ピクセル)
                      # None の場合、起動時にライン位置を検出する。
                      # そのため起動前に車を適切な位置へ置いておく必要がある。
                      # あるいは IMAGE_W / 2 を設定して中央ラインに追従させる
TARGET_THRESHOLD = 10 # 車両が向くべき TARGET_PIXEL からの許容ピクセル数
                      # この幅を超えたらステアリングを変更; これによりアルゴリズムの過剰反応を抑制
                      # ライン上または近くにあるときに車が過敏になりすぎるのを防ぐ
CONFIDENCE_THRESHOLD = 0.0015   # サンプル範囲内で黄色と判定されるピクセルの割合
                                # サンプル範囲の高さは SCAN_HEIGHT ピクセルで、総サンプル数は IMAGE_W x SCAN_HEIGHT
                                # 全ピクセルを黄色と判定させたい場合、閾値は SCAN_HEIGHT / (IMAGE_W x SCAN_HEIGHT) または (1 / IMAGE_W)
                                # サンプル領域の半分を黄色とみなしたいなら (1 / IMAGE_W) / 2
                                # コンソールに `No line detected` が頻出する場合は
                                # 閾値を下げてください

# LineFollower - ストレートで加速しカーブで減速するスロットルステップコントローラー
THROTTLE_MAX = 0.3    # コントローラが出力する最大スロットル値
THROTTLE_MIN = 0.15   # コントローラが出力する最小スロットル値
THROTTLE_INITIAL = THROTTLE_MIN  # 初期スロットル値
THROTTLE_STEP = 0.05  # ラインを外れた際のスロットル変化量

# これら 3 つの PID 定数は走行特性を大きく左右する。調整する際はまず他を 0 にして
# Kp、次に Kd、最後に Ki の順で設定する
PID_P = -0.01         # PID 制御の比例項
PID_I = 0.000         # PID 制御の積分項
PID_D = -0.0001       # PID 制御の微分項

PID_P_DELTA = 0.005   # P 値を増減させる量
PID_D_DELTA = 0.00005 # D 値を増減させる量

OVERLAY_IMAGE = True  # True で Web UI のカメラ画像に CV オーバーレイを表示
                      # 注意: 保存されるデータには影響しない


#
# ボタンにパスフォロー機能を割り当てる
# ゲームパッドのボタン、または Web UI のボタン ('web/w1'～'web/w5') を使用可能
# None にするとゲームコントローラのデフォルトを使用
# 注: クロスボタンは緊急停止用に予約済み
#
TOGGLE_RECORDING_BTN = "option" # 録画モードを切り替えるボタン
INC_PID_D_BTN = None            # PID 'D' 値を PID_D_DELTA だけ変更するボタン
DEC_PID_D_BTN = None            # PID 'D' 値を -PID_D_DELTA だけ変更するボタン
INC_PID_P_BTN = "R2"            # PID 'P' 値を PID_P_DELTA だけ変更するボタン
DEC_PID_P_BTN = "L2"            # PID 'P' 値を -PID_P_DELTA だけ変更するボタン

