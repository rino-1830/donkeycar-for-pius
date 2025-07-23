"""パスフォロー: 'path_follow' テンプレートの設定

このファイルは車のアプリケーションの ``manage.py`` スクリプトによって読み込まれ、
車の性能を変更します。

必要に応じて、このファイルで設定値を上書きできます。
``update`` 操作を実行してもこのファイルは変更されません。
"""

import os


import os

#
# FILE PATHS
#
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')


#
# 車両ループ
#
DRIVE_LOOP_HZ = 20      # この速度より速い場合、車両ループは一時停止します
MAX_LOOPS = None        # 正の整数を指定すると、車両ループはその回数で終了します


#
# カメラ設定
#
CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # デフォルトはRGB=3、モノクロなら1
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAMERA_INDEX = 0  # 'WEBCAM' と 'CVCAM' で複数カメラが接続されている場合に使用
# CSIC カメラ用 - カメラを回転して取り付けた場合、以下の値を変更するとフレームの向きを補正できます
CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => 変更なし, 4 => 水平反転, 6 => 垂直反転)
BGR2RGB = False  # BRG 形式から RGB 形式へ変換する場合は True。opencv が必要

# IMAGE_LIST カメラ用
PATH_MASK = "~/mycar/data/tub_1_20-03-12/*.jpg"


#
# PCA9685。必要な場合のみオーバーライドします（例: TX2）
#
PCA9685_I2C_ADDR = 0x40     # I2C アドレス。i2cdetect で確認
PCA9685_I2C_BUSNUM = None   # None なら自動検出。Pi では問題ないが、他のプラットフォームではバス番号を指定


#
# SSD1306_128_32
#
USE_SSD1306_128_32 = False    # SSD_1306 OLED ディスプレイを有効にする
SSD1306_128_32_I2C_ROTATION = 0 # 0: 上向き、1: 時計回り90度、2: 180度回転、3: 270度
SSD1306_RESOLUTION = 1 # 1 = 128x32、2 = 128x64


#
# 計測されたロボットの特性
#
AXLE_LENGTH = 0.03     # 車軸長さ。左右の車輪間の距離 (m)
WHEEL_BASE = 0.1       # 前輪と後輪の距離 (m)
WHEEL_RADIUS = 0.0315  # 車輪半径 (m)
MIN_SPEED = 0.1        # 最低速度 (m/s)。これ未満では停止する
MAX_SPEED = 3.0        # 最高速度 (m/s)。スロットル1.0時の速度
MIN_THROTTLE = 0.1     # MIN_SPEED に対応するスロットル値 (0〜1.0)。これ未満では停止
MAX_STEERING_ANGLE = 3.141592653589793 / 4  # 自動車型ロボットの最大操舵角 (ラジアン)。steering==-1 に対応


#
# DRIVE_TRAIN_TYPE
# ここで車体およびモーター構成を指定します。
# 詳細は Actuators ドキュメント https://docs.donkeycar.com/parts/actuators/ を参照してください
# 各駆動方式の詳細な説明は https://docs.donkeycar.com/parts/actuators/ を参照してください。
# 以下から一つ選び、関連する設定を更新します:
#
# "PWM_STEERING_THROTTLE" 標準的なRCカーのようにステアリングサーボとESCを2本のPWMピンで制御
# "MM1" Robo HAT MM1 ボード
# "SERVO_HBRIDGE_2PIN" ステアリングはサーボ、モーターは2ピン方式のHBridge
# "SERVO_HBRIDGE_3PIN" ステアリングはサーボ、モーターは3ピン方式のHBridge
# "DC_STEER_THROTTLE" 1つのステアリングDCモーターと1つの駆動輪モーターをHBridgeで制御
# "DC_TWO_WHEEL" 左右2つの駆動モーターを2ピンHBridgeで制御
# "DC_TWO_WHEEL_L298N" 左右2つの駆動モーターを3ピンHBridgeで制御
# "MOCK" 駆動装置なし。テストリグで他の機能を試すために使用
# (非推奨) "SERVO_HBRIDGE_PWM" PiZero から ServoBlaster でPWMを直接出力してステアリング制御、HBridgeで駆動
# (非推奨) "PIGPIO_PWM" Raspberry の内部PWMを使用
# (非推奨) "I2C_SERVO" PCA9685 サーボコントローラでステアリングサーボとESCを制御
#
DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"

#
# PWM_STEERING_THROTTLE ドライブトレイン設定
#
# ステアリングサーボとESCを備えたRCカー向けのドライブトレインです。
# ステアリングは ``PwmPin``、スロットルは2つ目の ``PwmPin`` を使用します。
# 基本PWM周波数は60Hz想定。PWM_xxxx_SCALEで非標準周波数に対応します
#
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:40.1",   # ステアリングサーボ用 PWM 出力ピン
    "PWM_STEERING_SCALE": 1.0,              # 60Hz 以外の PWM 周波数を補正。操舵範囲調整には使わない
    "PWM_STEERING_INVERTED": False,         # ハードウェアが反転 PWM を要求する場合 True
    "PWM_THROTTLE_PIN": "PCA9685.1:40.0",   # ESC 用の PWM 出力ピン
    "PWM_THROTTLE_SCALE": 1.0,              # 60Hz以外のPWM周波数を補正するための値。速度調整には使わない
    "PWM_THROTTLE_INVERTED": False,         # ハードウェアが反転PWMを要求する場合は True
    "STEERING_LEFT_PWM": 460,               # 左いっぱいのステアリング PWM 値
    "STEERING_RIGHT_PWM": 290,              # 右いっぱいのステアリング PWM 値
    "THROTTLE_FORWARD_PWM": 500,            # 前進最大スロットルの PWM 値
    "THROTTLE_STOPPED_PWM": 370,            # 停止時の PWM 値
    "THROTTLE_REVERSE_PWM": 220,            # 後退最大スロットルの PWM 値
}

#
# I2C_SERVO（PWM_STEERING_THROTTLE 推奨のため非推奨）
#
STEERING_CHANNEL = 1            #（非推奨）9685 PWM ボードのチャンネル 0〜15
STEERING_LEFT_PWM = 460         # 左いっぱいのステアリング PWM 値
STEERING_RIGHT_PWM = 290        # 右いっぱいのステアリング PWM 値
THROTTLE_CHANNEL = 0            #（非推奨）9685 PWM ボードのチャンネル 0〜15
THROTTLE_FORWARD_PWM = 500      # 前進最大スロットルの PWM 値
THROTTLE_STOPPED_PWM = 370      # 停止時の PWM 値
THROTTLE_REVERSE_PWM = 220      # 後退最大スロットルの PWM 値

#
# PIGPIO_PWM（PWM_STEERING_THROTTLE 推奨のため非推奨）
#
STEERING_PWM_PIN = 13           #（非推奨）Broadcom 番号で指定するピン
STEERING_PWM_FREQ = 50          # PWM 周波数
STEERING_PWM_INVERTED = False   # PWM を反転する必要がある場合 True
THROTTLE_PWM_PIN = 18           #（非推奨）Broadcom 番号で指定するピン
THROTTLE_PWM_FREQ = 50          # PWM 周波数
THROTTLE_PWM_INVERTED = False   # PWM を反転する必要がある場合 True

#
# SERVO_HBRIDGE_2PIN ドライブトレイン設定
# - ステアリングサーボと2ピン方式のHBridgeを使用
# - サーボは1ms(全後退)〜2ms(全前進)のPWMパルスを取り、1.5msが中立
# - モーターは2本のPWMピンで制御し、前進用と後退用を持つ
# - PWMピンは0(LOW)〜1(100% HIGH)のデューティ比で出力し、供給電力に比例
# - 前進では逆転PWMが0、後退では前進PWMが0
# - 両方のPWMを0にするとモーターを切り離して惰性走行
# - 両方とも100%にするとブレーキ
#
# ピン指定文字列の形式:
# - RPI_GPIO を使う場合: RPi/Nano のヘッダピン出力
#   - BOARD 番号方式
#   - BCM (Broadcom) 番号方式
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO を使う場合: pigpio サーバー経由で RPi のヘッダピン出力
#   - BCM 番号方式のみ
#   - 例: "PIGPIO.BCM.13"
# - PCA9685 を使う場合: PCA9685 ピン出力
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に組み合わせ可能だが、RPI_GPIO と PIGPIO の併用は推奨しない
#
SERVO_HBRIDGE_2PIN = {
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # 前進用 PWM 出力ピン
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 後退用 PWM 出力ピン
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",       # ステアリングサーボ用 PWM ピン
    "PWM_STEERING_SCALE": 1.0,        # 60Hz以外のPWM周波数を補正。操舵範囲調整には使わない
    "PWM_STEERING_INVERTED": False,   # ハードウェアが反転PWMを要求する場合 True
    "STEERING_LEFT_PWM": 460,         # 左いっぱいの PWM 値 (`donkey calibrate` で計測)
    "STEERING_RIGHT_PWM": 290,        # 右いっぱいの PWM 値 (`donkey calibrate` で計測)
}

#
# SERVO_HBRIDGE_3PIN ドライブトレイン設定
# - ステアリングサーボと3ピン方式のHBridgeを使用
# - サーボは1ms(全後退)〜2ms(全前進)のPWMで、1.5msが中立
# - モーターは3本のピンで制御し、前進用TTL、後退用TTL、PWM出力を使います
# - PWMピンは0〜1のデューティ比でモーターへの電力を調整します
# - 前進時はforwardピンがHIGH、backwardピンがLOW
# - 後退時はforwardピンがLOW、backwardピンがHIGH
# - 両ピンLOWでモーターを切り離し惰性走行、両ピンHIGHでブレーキ
#
# ピン指定文字列の形式:
# - RPI_GPIO: RPi/Nanoのヘッダピン出力
#   - BOARD番号方式
#   - BCM番号方式
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO: pigpioサーバー経由のヘッダ出力(BCM番号のみ)
#   - 例: "PIGPIO.BCM.13"
# - PCA9685: PCA9685ピン出力
#   - I2Cチャンネルとアドレスをコロン区切りで指定
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685は自由に組み合わせ可能だが、RPI_GPIOとPIGPIOの併用は非推奨
#
SERVO_HBRIDGE_3PIN = {
    "FWD_PIN": "RPI_GPIO.BOARD.18",   # HIGH でモーターを前進させる TTL ピン
    "BWD_PIN": "RPI_GPIO.BOARD.16",   # HIGH でモーターを後退させる TTL ピン
    "DUTY_PIN": "RPI_GPIO.BOARD.35",  # モーター速度を決める PWM 出力ピン
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",   # ステアリングサーボ用 PWM 出力ピン
    "PWM_STEERING_SCALE": 1.0,        # 60Hz 以外の PWM 周波数を補正。操舵範囲調整には利用しない
    "PWM_STEERING_INVERTED": False,   # ハードウェアが反転 PWM を要求する場合 True
    "STEERING_LEFT_PWM": 460,         # 左いっぱいの PWM 値（`donkey calibrate` で計測）
    "STEERING_RIGHT_PWM": 290,        # 右いっぱいの PWM 値（`donkey calibrate` で計測）
}

#
# DRIVETRAIN_TYPE == "SERVO_HBRIDGE_PWM" （SERVO_HBRIDGE_2PIN を推奨）
# - ステアリングサーボと 2 ピンモードの HBridge を設定
# - デフォルトでは ServoBlaster ライブラリがインストールされていないため、
#   利用するには別途インストールが必要
# - サーボは 1ms（全後退）〜2ms（全前進）の標準 PWM パルスを受け取り、
#   1.5ms が中立位置
# - モーターは前進用と後退用の 2 本の PWM ピンで制御
# - PWM ピンは 0（完全 LOW）〜1（100% HIGH）のデューティ比を出力し、
#   モーターへ供給する電力に比例
# - 前進時は後退用 PWM を 0、後退時は前進用 PWM を 0 にする
# - 両方の PWM を 0 にするとモーターを切り離して惰性走行
# - 両方の PWM を 100% にするとブレーキが掛かる
#
HBRIDGE_PIN_FWD = 18       # モーター前進用 PWM 出力ピン
HBRIDGE_PIN_BWD = 16       # モーター後退用 PWM 出力ピン
STEERING_CHANNEL = 0       # ステアリング制御に使用する PCA9685 のチャンネル
STEERING_LEFT_PWM = 460    # 左いっぱいの PWM 値（`donkey calibrate` で計測）
STEERING_RIGHT_PWM = 290   # 右いっぱいの PWM 値（`donkey calibrate` で計測）

#
# DC_STEER_THROTTLE ドライブトレイン（1 つのモーターで走行、もう 1 つでステアリング）
# - L298N 型モータードライバを 2 ピン配線で使用し、各モーターを PWM 2 本で制御
#   1 本は前進（または右）用、もう 1 本は後退（または左）用
#
# DRIVE_TRAIN_TYPE=DC_STEER_THROTTLE 用 GPIO ピン設定
# - RPI_GPIO: RPi/Nano のヘッダピンを直接制御
#   - BOARD 番号方式
#   - BCM(Broadcom) 番号方式
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO: pigpio サーバー経由でヘッダピンを制御
#   - BCM 番号方式のみ
#   - 例: "PIGPIO.BCM.13"
# - PCA9685: PCA9685 ピン出力
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 を自由に組み合わせ可能だが、RPI_GPIO と PIGPIO の併用は推奨しない
#
DC_STEER_THROTTLE = {
    "LEFT_DUTY_PIN": "RPI_GPIO.BOARD.18",   # 左旋回用 PWM 出力ピン
    "RIGHT_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 右旋回用 PWM 出力ピン
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",    # 前進用 PWM 出力ピン
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",    # 後退用 PWM 出力ピン
}

#
# DC_TWO_WHEEL ドライブトレイン用ピン設定
# - L298N_HBridge_2pin ドライバを使用
# - 左右 2 つのモーターによる差動駆動
# - 各モーターは前進用 PWM と後退用 PWM の 2 本で制御
# - PWM ピンは 0（LOW）〜1（HIGH）のデューティ比を出力し、モーターへの電力に比例
# - 前進時は後退 PWM を 0、後退時は前進 PWM を 0 に設定
# - 両 PWM を 0 にするとモーターを切り離して惰性走行
# - 両 PWM を 100% にするとブレーキ
#
# ピン指定文字列の形式:
# - RPI_GPIO: RPi/Nano のヘッダピン出力
#   - BOARD 番号方式
#   - BCM 番号方式
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO: pigpio サーバー経由でヘッダ出力（BCM 番号のみ）
#   - 例: "PIGPIO.BCM.13"
# - PCA9685: PCA9685 ピン出力
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に組み合わせられるが、RPI_GPIO と PIGPIO の併用は非推奨
#
DC_TWO_WHEEL = {
    "LEFT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # 左モーター前進用 PWM 出力ピン
    "LEFT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 左モーター後退用 PWM 出力ピン
    "RIGHT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.15", # 右モーター前進用 PWM 出力ピン
    "RIGHT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.13", # 右モーター後退用 PWM 出力ピン
}

#
# DC_TWO_WHEEL_L298N ドライブトレイン用ピン設定
# - L298N_HBridge_3pin ドライバを使用
# - 左右の車輪を差動駆動します
# - 各車輪は 3 本のピンで制御され、
#   前進用 TTL 出力、後退用 TTL 出力、
#   そして PWM 出力でモーターを駆動します
# - PWM ピンは 0 (完全に LOW) から 1 (100% HIGH) のデューティ比を出力し、
#   モーターに供給する電力に比例します
# - 前進時は forward ピンが HIGH、backward ピンが LOW
# - 後退時は forward ピンが LOW、backward ピンが HIGH
# - 両方のピンが LOW の場合、モーターを切り離して惰性走行します
#   両方のピンを HIGH にするとブレーキがかかります。
#
# DRIVE_TRAIN_TYPE=DC_TWO_WHEEL_L298N 用の GPIO 設定
# - RPI_GPIO を使う場合
#   - BOARD 番号方式を利用
#   - BCM(Broadcom) 番号方式を利用
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO を使う場合
#   - BCM 番号方式のみ対応
#   - 例: "PIGPIO.BCM.13"
# - PCA9685 を使う場合
#   - コロン区切りで I2C チャンネルとアドレスを指定
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO と PIGPIO、PCA9685 は自由に組み合わせ可能だが
#   RPI_GPIO と PIGPIO の併用は推奨されません。
#
DC_TWO_WHEEL_L298N = {
    "LEFT_FWD_PIN": "RPI_GPIO.BOARD.16",        # 左モーター前進用 TTL 出力ピン
    "LEFT_BWD_PIN": "RPI_GPIO.BOARD.18",        # 左モーター後退用 TTL 出力ピン
    "LEFT_EN_DUTY_PIN": "RPI_GPIO.BOARD.22",    # 左モーター速度制御 PWM ピン

    "RIGHT_FWD_PIN": "RPI_GPIO.BOARD.15",       # 右モーター前進用 TTL 出力ピン
    "RIGHT_BWD_PIN": "RPI_GPIO.BOARD.13",       # 右モーター後退用 TTL 出力ピン
    "RIGHT_EN_DUTY_PIN": "RPI_GPIO.BOARD.11",   # 右モーター速度制御 PWM ピン
}


#
# ODOMETRY（オドメトリ）
#
HAVE_ODOM = False               # オドメータ／エンコーダを搭載しているか
HAVE_ODOM_2 = False             # 差動駆動ロボットのように2つ目のエンコーダがあるか
                                # この場合、1つ目のエンコーダは左輪、
                                # 2つ目は右輪エンコーダとして扱います
ENCODER_TYPE = 'GPIO'           # エンコーダの種類 (GPIO|arduino)
                                # - "GPIO" はシングルチャンネルエンコーダを
                                #   RPi/Jetson の GPIO ヘッダーに直接接続する方式
                                #   ODOM_PIN にボード番号で指定します
                                # - "arduino" はシリアル接続されたマイコンを利用
                                #   ODOM_SERIAL に接続先ポートを指定します
                                #   arduino/encoder/encoder.ino を参照すると
                                #   連続送信とオンデマンド送信の例があります
ENCODER_PPR = 20                # エンコーダ軸1回転当たりのパルス数
ENCODER_DEBOUNCE_NS = 0         # 次のパルスを取り込むまで待つナノ秒数
                                # ノイズで割り込みが多発する場合の除去に使用
                                # 必要ならオシロスコープ等で計測するか
                                # 値を変えて試してみてください
FORWARD_ONLY = 1
FORWARD_REVERSE = 2
FORWARD_REVERSE_STOP = 3
TACHOMETER_MODE = FORWARD_REVERSE  # FORWARD_ONLY, FORWARD_REVERSE または FORWARD_REVERSE_STOP
                                # デュアルチャネルエンコーダでは FORWARD_ONLY が常に正しいモードです。
                                # シングルチャネルエンコーダでは用途に応じてモードを選択します。
                                # - FORWARD_ONLY は常に tick を増やし続けます。常に前進していると仮定します
                                #     常に正のスロットルを想定しており、広いサーキットでのレースに最適です。
                                #   車が常にスロットルを開けて走行し、後退や停止を考慮しない状況に向きます。
                                # - FORWARD_REVERSE はスロットル値から前進か後退かを判断し
                                #   スロットル値に応じて tick を増減します。スロットルが 0 の場合は
                                #   最後に非ゼロだったスロットル値に基づき増減します。これにより惰性走行を表現します。
                                #   このモードは、スロットルが
                                #   0 でも車が惰性で進む状況で有効です。例えばレースで減速のために惰性走行するが
                                #   完全には停止しない場合など。
                                # - FORWARD_REVERSE_STOP はスロットル値から前進・後退・停止を判断します。
                                #   このモードは低速で方向転換を行うロボットに適しており、
                                #   SLAM を行う際にはゆっくり探索し、後退が必要になる場合があります。
MM_PER_TICK = WHEEL_RADIUS * 2 * 3.141592653589793 * 1000 / ENCODER_PPR           # エンコーダ1tickで進む距離(mm)。車を1m転がし総tick数を1000で割って求めます
ODOM_SERIAL = '/dev/ttyACM0'    # ENCODER_TYPE='arduino' のとき接続するシリアルポート
ODOM_SERIAL_BAUDRATE = 115200   # シリアルポートで使用するボーレート
ODOM_PIN = 13                   # ENCODER_TYPE=GPIO の場合に使う GPIO 番号(ボードモード)
ODOM_PIN_2 = 14                 # 差動駆動で2つ目のエンコーダを使う場合の GPIO
ODOM_SMOOTHING = 1              # 速度計算に用いるオドメータ読み取り回数
ODOM_DEBUG = False              # 実行中に速度と距離を出力するか


#
# LIDAR（ライダー）
#
USE_LIDAR = False
LIDAR_TYPE = 'RP'  # (RP) ※YD LiDAR は未実装
LIDAR_LOWER_LIMIT = 90 # 記録する角度範囲の下限。車体で遮られる部分や後方を除外するときに使用。RP A1M8 LiDAR では "0" がモーター方向
LIDAR_UPPER_LIMIT = 270


# IMU 用設定
HAVE_IMU = False                # True にすると Mpu6050 パーツが追加されデータを記録
IMU_SENSOR = 'mpu6050'          # (mpu6050|mpu9250)
IMU_ADDRESS = 0x68              # AD0 ピンを High にした場合は 0x69、そうでなければ 0x68
IMU_DLP_CONFIG = 0              # デジタルローパスフィルタ設定 (0:250Hz, 1:184Hz, 2:92Hz, 3:41Hz, 4:20Hz, 5:10Hz, 6:5Hz)


#
# 入力コントローラ
#
#WEB コントロール
WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # Web コントローラが待ち受けるポート
WEB_INIT_MODE = "user"              # 起動時の制御モード (user|local_angle|local)。local を指定すると AI モード開始

#ジョイスティック
USE_JOYSTICK_AS_DEFAULT = False      # True にすると manage.py 起動時に --js オプションなしでジョイスティックを使用
JOYSTICK_MAX_THROTTLE = 0.5         # -1〜1 のスロットル値に乗算して最高速度を制限
JOYSTICK_STEERING_SCALE = 1.0       # ステアリング感度調整。負値で方向反転
AUTO_RECORD_ON_THROTTLE = False     # True ならスロットルが 0 でない間自動記録。False なら別のトリガで開始
CONTROLLER_TYPE = 'xbox'            # (ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom) custom は `donkey createjs` で作成した my_joystick.py を使用
USE_NETWORKED_JS = False            # ネットワーク越しのジョイスティック入力を受け付けるか
NETWORK_JS_SERVER_IP = None         # ネットワークジョイスティックのサーバー IP
JOYSTICK_DEADZONE = 0.01            # 0 以外ならこのスロットル値を超えると記録を開始
JOYSTICK_THROTTLE_DIR = -1.0         # -1.0 で前後を反転、1.0 でジョイスティックの向きをそのまま使用
USE_FPV = False                     # カメラ画像を FPV ウェブサーバーへ送信
JOYSTICK_DEVICE_FILE = "/dev/input/js0" # ジョイスティックデバイスのパス


#SOMBRERO
HAVE_SOMBRERO = False           # Donkeycar ストアの Sombrero Hat を使用する場合 True。PWM が有効になります

#PIGPIO RC control
STEERING_RC_GPIO = 26
THROTTLE_RC_GPIO = 20
DATA_WIPER_RC_GPIO = 19
PIGPIO_STEERING_MID = 1500         # 直進しない場合はこの値を調整
PIGPIO_MAX_FORWARD = 2000          # 前進最大スロットル。値が大きいほど速い
PIGPIO_STOPPED_PWM = 1500
PIGPIO_MAX_REVERSE = 1000          # 後退最大スロットル。値が小さいほど速い
PIGPIO_SHOW_STEERING_VALUE = False
PIGPIO_INVERT = False
PIGPIO_JITTER = 0.025   # この値未満では信号として扱わないしきい値


# ROBOHAT MM1 コントローラ
MM1_STEERING_MID = 1500         # 直進しない場合はこの値を調整
MM1_MAX_FORWARD = 2000          # 前進最大スロットル。値が大きいほど速い
MM1_STOPPED_PWM = 1500
MM1_MAX_REVERSE = 1000          # 後退最大スロットル。値が小さいほど速い
MM1_SHOW_STEERING_VALUE = False
# シリアルポート
# -- Pi のデフォルト: '/dev/ttyS0'
# -- Jetson Nano: '/dev/ttyTHS1'
# -- Google Coral: '/dev/ttymxc0'
# -- Windows: 'COM3', Arduino: '/dev/ttyACM0'
# -- MacOS/Linux: 'ls /dev/tty.*' で適切なポートを確認してください
#  例:'/dev/tty.usbmodemXXXXXX' に置き換えてください
MM1_SERIAL_PORT = '/dev/ttyS0'  # MM1 との通信に使用するシリアルポート


#
# ロギング
#
HAVE_CONSOLE_LOGGING = True
LOGGING_LEVEL = 'INFO'          # Python のロギングレベル ('NOTSET'|'DEBUG'|'INFO'|'WARNING'|'ERROR'|'FATAL'|'CRITICAL')
LOGGING_FORMAT = '%(message)s'  # Python のロギングフォーマット - https://docs.python.org/3/library/logging.html#formatter-objects


#
# MQTT テレメトリ
#
HAVE_MQTT_TELEMETRY = False
TELEMETRY_DONKEY_NAME = 'my_robot1234'
TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
TELEMETRY_MQTT_JSON_ENABLE = False
TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
TELEMETRY_MQTT_BROKER_PORT = 1883
TELEMETRY_PUBLISH_PERIOD = 1
TELEMETRY_LOGGING_ENABLE = True
TELEMETRY_LOGGING_LEVEL = 'INFO' # Python のロギングレベル ('NOTSET'|'DEBUG'|'INFO'|'WARNING'|'ERROR'|'FATAL'|'CRITICAL')
TELEMETRY_LOGGING_FORMAT = '%(message)s'  # Python のロギングフォーマット - https://docs.python.org/3/library/logging.html#formatter-objects
TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
TELEMETRY_DEFAULT_TYPES = 'float,float'


#
# パフォーマンスモニタ
#
HAVE_PERFMON = False


#
# 記録オプション
#
RECORD_DURING_AI = False        # 通常AIモード中は記録しません。True にすると AI 用に画像とステアリングを記録しますが、学習には注意してください。
AUTO_CREATE_NEW_TUB = False     # 録画時に新しい tub (tub_YY_MM_DD) ディレクトリを作成するか、既存 data ディレクトリへ追記するか


#
# LED
#
HAVE_RGB_LED = False            # RGB LED を持っている場合は True
LED_INVERT = False              # 共通アノードの場合は True。例: https://www.amazon.com/Xia-Fly-Tri-Color-Emitting-Diffused/dp/B07MYJQP8B

# PWM 出力に使用する LED ピン番号
# 物理ピン番号。参考: https://www.raspberrypi-spy.co.uk/2012/06/simple-guide-to-the-rpi-gpio-header-and-pins/
LED_PIN_R = 12
LED_PIN_G = 10
LED_PIN_B = 16

# LED ステータス色 (0-100)
LED_R = 0
LED_G = 0
LED_B = 1

# 記録数に応じた LED の色
REC_COUNT_ALERT = 1000          # 何件記録したら点滅させるか
REC_COUNT_ALERT_CYC = 15        # REC_COUNT_ALERT 件ごとに1/20秒単位で何回点滅させるか
REC_COUNT_ALERT_BLINK_RATE = 0.4 # LED を点滅させる速さ(秒)

# 最初の値が記録数、次が色 (r,g,b 0-100)
# 記録数がこの値を超えると指定した色に変化
RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
            (3000, (5, 5, 5)),
            (5000, (5, 2, 0)),
            (10000, (0, 5, 0)),
            (15000, (0, 5, 5)),
            (20000, (0, 0, 5)), ]

# LED ステータス色 (0-100), for model reloaded alert
MODEL_RELOADED_LED_R = 100
MODEL_RELOADED_LED_G = 0
MODEL_RELOADED_LED_B = 0


#
# DonkeyGym
#
# Ubuntu Linux ではシミュレータを仮想ドンキーとして利用でき、
# 通常の 'python manage.py drive' コマンドで仮想カーを操作できます。
# これを有効にするとシミュレータのパスと環境を設定します。
# シミュレータのバイナリは https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip から入手できます
# ダウンロードしたら展開し、DONKEY_SIM_PATH を変更してください。
DONKEY_GYM = False
DONKEY_SIM_PATH = "path to sim" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" when racing on virtual-race-league use "remote", or user "remote" when you want to start the sim manually first.
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
GYM_CONF = { "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # body style(donkey|bare|car01) body rgb 0-255
GYM_CONF["racer_name"] = "Your Name"
GYM_CONF["country"] = "Place"
GYM_CONF["bio"] = "I race robots."

SIM_HOST = "127.0.0.1"              # バーチャルレースリーグで走行する場合はホスト "trainmydonkey.com" を使用
SIM_ARTIFICIAL_LATENCY = 0          # リモートサーバ利用時の遅延をエミュレートするためのミリ秒単位遅延。100〜400ms 程度が妥当

# シミュレータからの情報保存 (pln)
SIM_RECORD_LOCATION = False
SIM_RECORD_GYROACCEL= False
SIM_RECORD_VELOCITY = False
SIM_RECORD_LIDAR = False

# カメラを TCP ソケットで配信する
# カメラ配信用の TCP サービスを作成する場合に使用
PUB_CAMERA_IMAGES = False


#
# AI オーバーライド
#
# 起動時に AI を上書きするモード。ユーザ操作から自動操縦へ移行
AI_LAUNCH_DURATION = 0.0            # AI がこの秒数だけスロットルを出力
AI_LAUNCH_THROTTLE = 0.0            # AI が出力するスロットル値
AI_LAUNCH_ENABLE_BUTTON = 'R2'      # このキーを押すとブーストが有効になる。誤操作を防ぐため毎回有効化が必要
AI_LAUNCH_KEEP_ENABLED = False      # False の場合、毎回 AI_LAUNCH_ENABLE_BUTTON を押す必要があり安全です。True では毎回自動で有効になります

# スロットルスケーリング: AI パイロットのスロットル出力を全モデルで倍率調整
AI_THROTTLE_MULT = 1.0              # この倍率が NN モデルからのスロットル値すべてに掛けられる


#
# Intel Realsense D43530FBD435i 深度カメラ
#
REALSENSE_D435_RGB = True       # True で RGB 画像を取得
REALSENSE_D435_DEPTH = True     # True で深度を画像配列として取得
REALSENSE_D435_IMU = False      # True で IMU データを取得 (D435i のみ)
REALSENSE_D435_ID = None        # カメラのシリアル番号。1台のみなら None で自動検出


#
# 一時停止標識検出
#
STOP_SIGN_DETECTOR = False
STOP_SIGN_MIN_SCORE = 0.2
STOP_SIGN_SHOW_BOUNDING_BOX = True
STOP_SIGN_MAX_REVERSE_COUNT = 10    # 停止標識検出時に何回バックするか。0でバックしない
STOP_SIGN_REVERSE_THROTTLE = -0.5     # 停止標識検出時にバックする際のスロットル


#
# FPS カウンタ
#
SHOW_FPS = False
FPS_DEBUG_INTERVAL = 10    # 周期情報をシェルに出力する間隔(秒)

#
# トラッキングカメラ
#
HAVE_T265 = False       # True で Intel Realsense T265 を位置情報源として使用

#
# GPS
#
HAVE_GPS = False            # True で GPS 位置を読み取る
GPS_SERIAL = '/dev/ttyUSB0' # シリアルデバイスのパス。例 '/dev/ttyAMA1' や '/dev/ttyUSB0'
GPS_SERIAL_BAUDRATE = 115200
GPS_NMEA_PATH = None        # GPS を記録するファイル例: "nmea.csv"
                            # これを設定するとウェイポイント記録時に
                            # 対応する NMEA センテンスも保存され
                            # そのタイムスタンプと共にこのファイルへ保存されます。次に
                            # 自動運転モードでパスを読み込むと
                            # 記録した NMEA センテンスが再生されます。
                            # これはデバッグや PID 調整用であり、
                            # 車を走らせ続ける必要がありません。
GPS_DEBUG = False  # True で UTM 座標を記録 (大量にログが出ます)

#
# パスフォロー
#
PATH_FILENAME = "donkey_path.csv"   # 走行パスを x,y の CSV 形式で保存するファイル名
PATH_DEBUG = True                   # True で x,y 位置をログに記録
PATH_SCALE = 10.0                   # Web ページでパスを表示する際の縮尺
PATH_OFFSET = (255, 255)            # 255,255 がマップの中心。原点表示位置をずらすオフセット
PATH_MIN_DIST = 0.2                 # この距離(m) 進むごとにパス点を保存
PATH_SEARCH_LENGTH = None           # 最寄り点を探すポイント数。None なら全体を検索
PATH_LOOK_AHEAD = 1                 # CTE 計算に含める最寄り点から先のポイント数
PATH_LOOK_BEHIND = 1                # CTE 計算に含める最寄り点から後ろのポイント数   
PID_P = -0.5                        # PID パスフォロー用比例項
PID_I = 0.000                       # PID パスフォロー用積分項
PID_D = -0.3                        # PID パスフォロー用微分項
PID_THROTTLE = 0.50                 # パスフォロー時の一定スロットル値
USE_CONSTANT_THROTTLE = False       # 記録時のスロットルを使うか、一定値を使うか
PID_D_DELTA = 0.25                  # D を増減する際の変化量
PID_P_DELTA = 0.25                  # P を増減する際の変化量

#
# パスフォロー機能をボタンに割り当てる
# ゲームパッドのボタンまたは Web UI ('web/w1'〜'web/w5') を利用可能
# None の場合はゲームコントローラのデフォルトを使用
# 十字ボタンは非常停止に予約されています
#
SAVE_PATH_BTN = "circle"        # パスを保存するボタン
LOAD_PATH_BTN = "x"             # パスを読み込むボタン
RESET_ORIGIN_BTN = "square"     # 車を原点に戻すボタン
ERASE_PATH_BTN = "triangle"     # パスを削除するボタン
TOGGLE_RECORDING_BTN = "option" # 記録モードを切り替えるボタン
INC_PID_D_BTN = None            # PID 'D' を PID_D_DELTA 分増減するボタン
DEC_PID_D_BTN = None            # PID 'D' を -PID_D_DELTA 分変更するボタン
INC_PID_P_BTN = "R2"            # PID 'P' を PID_P_DELTA 分増減するボタン
DEC_PID_P_BTN = "L2"            # PID 'P' を -PID_P_DELTA 分変更するボタン

# Intel Realsense T265 トラッキングカメラ
REALSENSE_T265_ID = None # カメラのシリアル番号。1台のみなら None で自動検出
WHEEL_ODOM_CALIB = "calibration_odometry.json"

