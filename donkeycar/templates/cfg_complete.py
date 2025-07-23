"""車両設定ファイル.

このファイルは ``manage.py`` スクリプトによって読み込まれ、車両の挙動を変更
します。

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)
"""


import os

# パス設定
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

# 車両パラメータ
DRIVE_LOOP_HZ = 20      # ループがこの速度より速い場合は一時停止
MAX_LOOPS = None        # 正の整数を指定すると、その回数でループを終了

# カメラ設定
CAMERA_TYPE = "PICAM"   # 利用可能: PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # デフォルトRGB=3。モノクロの場合は1
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAMERA_INDEX = 0  # WEBCAMやCVCAMで複数接続時に使用する番号
# CSICカメラ用 - カメラが回転して取り付けられている場合、下記を変更して向きを補正
CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0: なし, 4:水平反転, 6:垂直反転)
BGR2RGB = False  # TrueでBGRをRGBへ変換。opencvが必要
SHOW_PILOT_IMAGE = False  # 自動運転時に推論へ使用する画像を表示

# IMAGE_LISTカメラ用設定
# PATH_MASK = "~/mycar/data/tub_1_20-03-12/*.jpg"

#9865、必要な場合のみ上書き。例: TX2
PCA9685_I2C_ADDR = 0x40     # I2Cアドレス。i2cdetectで確認
PCA9685_I2C_BUSNUM = None   # Noneなら自動検出。Pi以外ではバス番号を指定

#SSD1306_128_32
USE_SSD1306_128_32 = False    # SSD_1306 OLEDディスプレイを有効化
SSD1306_128_32_I2C_ROTATION = 0 # 0: 正常 1:90度右回転 2:180度反転 3:270度回転
SSD1306_RESOLUTION = 1 # 1: 128x32 2: 128x64

#
# DRIVE_TRAIN_TYPE
# 使用するシャーシとモーター構成を指定します。
# 詳細は Actuators ドキュメント https://docs.donkeycar.com/parts/actuators/ を参照してください。
# いずれかを選び、対応する設定を変更します。
#
# "PWM_STEERING_THROTTLE" : ステアリングサーボとESCを2つのPWMで制御
# "MM1" : Robo HAT MM1 ボード
# "SERVO_HBRIDGE_2PIN" : ステアリング用サーボ + 2ピンHBridgeモータドライバ
# "SERVO_HBRIDGE_3PIN" : ステアリング用サーボ + 3ピンHBridgeモータドライバ
# "DC_STEER_THROTTLE" : DCモーターでステアリングし、もう1つのモーターで駆動
# "DC_TWO_WHEEL" : 左右2つのモーターを2ピンHBridgeで制御
# "DC_TWO_WHEEL_L298N" : L298N HBridge(3ピン)で2つのモーターを制御
# "MOCK" : 駆動系なし。テスト用
# "VESC" : VESCモータコントローラを使用
# （非推奨）"SERVO_HBRIDGE_PWM" : PiZeroのServoBlasterでステアリング、HBridgeで駆動
# （非推奨）"PIGPIO_PWM" : Raspberryの内部PWMを使用
# （非推奨）"I2C_SERVO" : PCA9685でステアリングサーボとESCを制御
#
DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"

#
# PWM_STEERING_THROTTLE
#
# ステアリングサーボと ESC を備えた RC カー向けの駆動方式。
# ステアリング用の PwmPin とスロットル用の PwmPin を利用します。
# 基本 PWM 周波数は 60Hz を想定しており、異なる周波数の場合は PWM_xxxx_SCALE で補正します。
#
PWM_STEERING_THROTTLE = {
    "PWM_STEERING_PIN": "PCA9685.1:40.1",   # ステアリングサーボ用 PWM ピン
    "PWM_STEERING_SCALE": 1.0,              # PWM 周波数が 60Hz 以外のときの補正値
    "PWM_STEERING_INVERTED": False,         # ハードウェアが反転パルスを要求する場合 True
    "PWM_THROTTLE_PIN": "PCA9685.1:40.0",   # ESC 用 PWM ピン
    "PWM_THROTTLE_SCALE": 1.0,              # PWM 周波数が 60Hz 以外のときの補正値
    "PWM_THROTTLE_INVERTED": False,         # ハードウェアが反転パルスを要求する場合 True
    "STEERING_LEFT_PWM": 460,               # 左いっぱいのステアリング PWM 値
    "STEERING_RIGHT_PWM": 290,              # 右いっぱいのステアリング PWM 値
    "THROTTLE_FORWARD_PWM": 500,            # 前進最大 PWM 値
    "THROTTLE_STOPPED_PWM": 370,            # 停止時の PWM 値
    "THROTTLE_REVERSE_PWM": 220,            # 後退最大 PWM 値
}

#
# I2C_SERVO (PWM_STEERING_THROTTLE を推奨)
#
STEERING_CHANNEL = 1            #（非推奨）9685 ボードのチャンネル 0-15
STEERING_LEFT_PWM = 460         # 左いっぱいのステアリング PWM 値
STEERING_RIGHT_PWM = 290        # 右いっぱいのステアリング PWM 値
THROTTLE_CHANNEL = 0            #（非推奨）9685 ボードのチャンネル 0-15
THROTTLE_FORWARD_PWM = 500      # 前進最大 PWM 値
THROTTLE_STOPPED_PWM = 370      # 停止時の PWM 値
THROTTLE_REVERSE_PWM = 220      # 後退最大 PWM 値

#
# PIGPIO_PWM (PWM_STEERING_THROTTLE を推奨)
#
STEERING_PWM_PIN = 13           #（非推奨）Broadcom 番号で指定するピン
STEERING_PWM_FREQ = 50          # PWM 周波数
STEERING_PWM_INVERTED = False   # PWM を反転する必要がある場合 True
THROTTLE_PWM_PIN = 18           #（非推奨）Broadcom 番号で指定するピン
THROTTLE_PWM_FREQ = 50          # PWM 周波数
THROTTLE_PWM_INVERTED = False   # PWM を反転する必要がある場合 True

#
# SERVO_HBRIDGE_2PIN
# - ステアリングサーボと HBridge を 2 ピン PWM で制御します
# - サーボは 1ms(全左)～2ms(全右) のパルスを受け取り、1.5ms が中立です
# - モーターは 2 つの PWM ピンで制御され、
#   一方が前進、もう一方が後退(リバース)用です。
# - PWM ピンは 0 (完全 LOW) から 1 (100% HIGH) のデューティ比を出力し、
#   値が大きいほどモーターへ伝える電力が大きくなります。
# - 前進時は後退側 PWM を 0 に、
#   後退時は前進側 PWM を 0 にします。
# - 両方の PWM を 0 (LOW) にするとモーターが切り離され、慣性で停止します。
# - 両方の PWM を 100% (HIGH) にするとブレーキがかかります。
#
# ピン指定文字列の形式:
# - RPI_GPIO を使うと Raspberry Pi/Nano のヘッダピンへ出力できます
#   - BOARD: 物理ピン番号方式
#   - BCM: Broadcom GPIO 番号方式
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO を使うと pigpio サーバを介して RPi のヘッダピンへ出力します
#   - BCM 番号方式のみ使用可能
#   - 例: "PIGPIO.BCM.13"
# - PCA9685 を使うと PCA9685 のピンへ出力します
#   - I2C チャンネルとアドレスをコロン区切りで指定
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO, PIGPIO, PCA9685 は自由に混在できますが、
#   RPI_GPIO と PIGPIO の併用は推奨されません。
#
SERVO_HBRIDGE_2PIN = {
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # provides forward duty cycle to motor
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # provides reverse duty cycle to motor
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",       # provides servo pulse to steering servo
    "PWM_STEERING_SCALE": 1.0,        # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
    "PWM_STEERING_INVERTED": False,   # True if hardware requires an inverted PWM pulse
    "STEERING_LEFT_PWM": 460,         # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
    "STEERING_RIGHT_PWM": 290,        # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
}

#
# SERVO_HBRIDGE_3PIN
# - ステアリングサーボと HBridge を 3 ピン構成(2 TTL + 1 PWM)で制御します
# - サーボは 1ms～2ms のパルスを受け取り、1.5ms が中立です
# - モーターは 3 本のピンで制御し、
#   1 本は前進用 TTL、1 本は後退用 TTL、残り 1 本は PWM です
# - PWM ピンは 0～1 のデューティ比を出力し、値が大きいほど供給電力が増えます
# - 前進時は前進ピンを HIGH、後退ピンを LOW にします
# - 後退時は前進ピンを LOW、後退ピンを HIGH にします
# - 前進・後退の両ピンを LOW にするとモーターが切り離され、慣性で停止します
# - 両ピンを HIGH にするとブレーキがかかります
#
# ピン指定文字列の形式:
# - RPI_GPIO を使うと Raspberry Pi/Nano のヘッダピンへ出力できます
#   - BOARD: 物理ピン番号方式
#   - BCM: Broadcom GPIO 番号方式
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO を使うと pigpio サーバを介してヘッダピンへ出力します
#   - BCM 番号方式のみ使用可能
#   - 例: "PIGPIO.BCM.13"
# - PCA9685 を使うと PCA9685 のピンへ出力します
#   - I2C チャンネルとアドレスをコロン区切りで指定
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO, PIGPIO, PCA9685 は自由に混在できますが、
#   RPI_GPIO と PIGPIO の併用は推奨されません。
#
SERVO_HBRIDGE_3PIN = {
    "FWD_PIN": "RPI_GPIO.BOARD.18",   # モーター前進を有効にする TTL ピン
    "BWD_PIN": "RPI_GPIO.BOARD.16",   # モーター後退を有効にする TTL ピン
    "DUTY_PIN": "RPI_GPIO.BOARD.35",  # モーターへデューティ比を出力
    "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",   # ステアリングサーボへのパルス出力
    "PWM_STEERING_SCALE": 1.0,        # PWM 周波数が 60Hz 以外の場合の補正値
    "PWM_STEERING_INVERTED": False,   # ハードウェアが反転パルスを要求する場合 True
    "STEERING_LEFT_PWM": 460,         # 左最大舵角の PWM 値 (`donkey calibrate` で計測)
    "STEERING_RIGHT_PWM": 290,        # 右最大舵角の PWM 値 (`donkey calibrate` で計測)
}

#
# DRIVETRAIN_TYPE == "SERVO_HBRIDGE_PWM" (現在は SERVO_HBRIDGE_2PIN を推奨)
# - ステアリングサーボと HBridge を 2 ピン PWM で制御します
# - ServoBlaster ライブラリが必要ですが、デフォルトではインストールされていません
# - サーボは 1ms～2ms のパルスを受け取り、1.5ms が中立です
# - モーターは 2 本の PWM ピンで制御し、1 本が前進、もう 1 本が後退用です
# - PWM ピンは 0～1 のデューティ比を出力し、値が大きいほど電力が増えます
# - 前進時は後退側 PWM を 0 に、後退時は前進側 PWM を 0 にします
# - 両 PWM を 0 にするとモーターが切り離され、慣性で停止します
# - 両 PWM を 1 にするとブレーキがかかります
#
HBRIDGE_PIN_FWD = 18       # 前進デューティ比出力ピン
HBRIDGE_PIN_BWD = 16       # 後退デューティ比出力ピン
STEERING_CHANNEL = 0       # ステアリング制御用 PCA9685 チャンネル
STEERING_LEFT_PWM = 460    # 左最大舵角の PWM 値 (`donkey calibrate` で計測)
STEERING_RIGHT_PWM = 290   # 右最大舵角の PWM 値 (`donkey calibrate` で計測)

# VESC コントローラを使用する場合の設定。主に VESC_SERIAL_PORT と VESC_MAX_SPEED_PERCENT を変更します
 VESC_MAX_SPEED_PERCENT =.2  # 実際の速度に対する最大速度割合
 VESC_SERIAL_PORT= "/dev/ttyACM0" # シリアル通信に使用するデバイス。ls /dev/tty* で確認可能
 VESC_HAS_SENSOR= True # ブラシレスモーターがホールセンサーを使用しているか
 VESC_START_HEARTBEAT= True # コマンド維持のためのハートビートスレッドを自動開始するか
 VESC_BAUDRATE= 115200 # シリアル通信のボーレート。通常変更不要
 VESC_TIMEOUT= 0.05 # シリアル通信のタイムアウト
 VESC_STEERING_SCALE= 0.5 # VESC は 0〜1 のステア入力。ジョイスティックは -1〜1 なので範囲を調整
 VESC_STEERING_OFFSET = 0.5 # 上記調整と合わせてステア入力を 0〜1 に移動

#
# DC_STEER_THROTTLE は一つのモーターでステアリング、もう一つで走行する方式
# - 2 ピン配線の L298N タイプモータードライバを使用し、各モーターを 2 本の PWM ピンで制御します
#   一方が前進(右)用、もう一方が後退(左)用です
# 
# DRIVE_TRAIN_TYPE=DC_STEER_THROTTLE 用の GPIO ピン設定
# - RPI_GPIO を使うと Pi/Nano のヘッダピンに出力できます
#   - BOARD: 物理ピン番号
#   - BCM: Broadcom GPIO 番号
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO を使うと pigpio サーバ経由でヘッダピンへ出力します
#   - BCM 番号のみ使用
#   - 例: "PIGPIO.BCM.13"
# - PCA9685 を使う場合は I2C チャンネルとアドレスをコロンで区切って指定します
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は混在可能ですが、RPI_GPIO と PIGPIO の混在は推奨されません
#
DC_STEER_THROTTLE = {
    "LEFT_DUTY_PIN": "RPI_GPIO.BOARD.18",   # 左旋回用 PWM ピン
    "RIGHT_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 右旋回用 PWM ピン
    "FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",    # 前進用 PWM ピン
    "BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",    # 後退用 PWM ピン
}

#
# DC_TWO_WHEEL のピン設定
# - L298N_HBridge_2pin ドライバを利用します
# - 左右 2 輪の差動駆動方式です
# - 各ホイールは 2 本の PWM ピンで制御され、
#   前進用と後退用に分かれます
# - PWM ピンは 0 (完全 LOW) から 1 (100% HIGH) のデューティ比を出力し、
#   値が大きいほどモーターへの電力が増えます
# - 前進時は後退 PWM を 0 に、後退時は前進 PWM を 0 にします
# - 両 PWM を 0 にするとモーターが切り離され、慣性で停止します
# - 両 PWM を 100% にするとブレーキとなります
#
# ピン指定文字列の形式:
# - RPI_GPIO を使用すると Raspberry Pi/Nano のヘッダピンに出力できます
#   - BOARD を用いると物理ピン番号方式になります
#   - BCM を用いると Broadcom GPIO 番号方式になります
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO を使用すると pigpio サーバ経由でヘッダピンへ出力します
#   - BCM 番号方式のみ使用可能
#   - 例: "PIGPIO.BCM.13"
# - PCA9685 を使用すると PCA9685 のピンへ出力します
#   - I2C チャンネルとアドレスをコロン区切りで指定します
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に組み合わせられますが、
#   RPI_GPIO と PIGPIO の併用は推奨されません。
#
DC_TWO_WHEEL = {
    "LEFT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # 左輪前進用 PWM ピン
    "LEFT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # 左輪後退用 PWM ピン
    "RIGHT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.15", # 右輪前進用 PWM ピン
    "RIGHT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.13", # 右輪後退用 PWM ピン
}

#
# DC_TWO_WHEEL_L298N のピン設定
# - L298N_HBridge_3pin ドライバを利用します
# - 左右 2 輪の差動駆動方式です
# - 各ホイールは 3 本のピンで制御されます
#   前進用 TTL、後退用 TTL、そして PWM の 1 本です
# - PWM ピンは 0 (完全 LOW) から 1 (100% HIGH) のデューティ比を出力し、
#   値が大きいほど供給電力が増えます
# - 前進時は前進ピンを HIGH、後退ピンを LOW にします
# - 後退時は前進ピンを LOW、後退ピンを HIGH にします
# - 両 TTL ピンを LOW にするとモーターが切り離され、慣性で停止します
# - 両 TTL ピンを HIGH にするとブレーキとなります
#
# DRIVE_TRAIN_TYPE=DC_TWO_WHEEL_L298N 用の GPIO ピン設定
# - RPI_GPIO を使用すると RPi/Nano のヘッダピンに出力できます
#   - BOARD を用いると物理ピン番号方式になります
#   - BCM を用いると Broadcom GPIO 番号方式になります
#   - 例: "RPI_GPIO.BOARD.18"
# - PIGPIO を使用すると pigpio サーバ経由でヘッダピンに出力します
#   - BCM 番号方式のみ使用可能
#   - 例: "PIGPIO.BCM.13"
# - PCA9685 を使用すると PCA9685 のピンに出力します
#   - I2C チャンネルとアドレスをコロン区切りで指定します
#   - 例: "PCA9685.1:40.13"
# - RPI_GPIO、PIGPIO、PCA9685 は自由に組み合わせられますが、
#   RPI_GPIO と PIGPIO の併用は推奨されません。
#
DC_TWO_WHEEL_L298N = {
    "LEFT_FWD_PIN": "RPI_GPIO.BOARD.16",        # 左輪前進を有効にする TTL ピン
    "LEFT_BWD_PIN": "RPI_GPIO.BOARD.18",        # 左輪後退を有効にする TTL ピン
    "LEFT_EN_DUTY_PIN": "RPI_GPIO.BOARD.22",    # 左モーター速度制御用 PWM ピン

    "RIGHT_FWD_PIN": "RPI_GPIO.BOARD.15",       # 右輪前進を有効にする TTL ピン
    "RIGHT_BWD_PIN": "RPI_GPIO.BOARD.13",       # 右輪後退を有効にする TTL ピン
    "RIGHT_EN_DUTY_PIN": "RPI_GPIO.BOARD.11",   # 右モーター速度制御用 PWM ピン
}

#ODOMETRY
HAVE_ODOM = False                   # オドメーター／エンコーダを使用するか
ENCODER_TYPE = 'GPIO'            # エンコーダの種類 GPIO|Arduino|Astar
MM_PER_TICK = 12.7625               # 1ティックあたりの移動距離(mm)。車を1m動かし、総ティック数を1000で割る
ODOM_PIN = 13                        # GPIO 使用時の入力ピン番号
ODOM_DEBUG = False                  # 速度や距離をログ出力するか

# #LIDAR
USE_LIDAR = False
LIDAR_TYPE = 'RP' #(RP|YD)
LIDAR_LOWER_LIMIT = 90 # 記録する角度範囲の下限。車体で遮られる領域や背面を除外するのに利用。RP A1M8 Lidar では "0" がモーター方向
LIDAR_UPPER_LIMIT = 270

# TFMINI
HAVE_TFMINI = False
TFMINI_SERIAL_PORT = "/dev/serial0" # tfmini のシリアルポート。配線もしくは USB シリアルアダプタを使用

#TRAINING
# デフォルトで使用する AI フレームワーク (tensorflow|pytorch)
DEFAULT_AI_FRAMEWORK = 'tensorflow'

# DEFAULT_MODEL_TYPE は学習時に生成するモデルを決定します。
# コマンドライン引数 --type を指定することで上書きできます。
# tensorflow 用モデル: (linear|categorical|tflite_linear|tensorrt_linear)
# pytorch 用モデル: (resnet18)
DEFAULT_MODEL_TYPE = 'linear'
BATCH_SIZE = 128                # 勾配降下の1ステップで使用するレコード数。GPU メモリが不足する場合は小さくする
TRAIN_TEST_SPLIT = 0.8          # 学習に使うデータの割合。残りは検証用
MAX_EPOCHS = 100                # データセットを繰り返す回数
SHOW_PLOT = True                # 学習後に損失のグラフを表示するか
VERBOSE_TRAIN = True            # 学習中に進捗バーを表示するか
USE_EARLY_STOP = True           # フィットが改善しないと判断したら学習を停止するか
EARLY_STOP_PATIENCE = 5         # 改善が見られなくなるまで待つエポック数
MIN_DELTA = .0005               # 改善とみなす最小損失変化量
PRINT_MODEL_SUMMARY = True      # ネットワーク構造と重みを標準出力に表示
OPTIMIZER = None                # adam, sgd, rmsprop など。None ならデフォルト
LEARNING_RATE = 0.001           # OPTIMIZER 指定時のみ使用
LEARNING_RATE_DECAY = 0.0       # OPTIMIZER 指定時のみ使用
SEND_BEST_MODEL_TO_PI = False   # True にすると学習中に最良モデルを自動で Pi へ送信
CREATE_TF_LITE = True           # 学習時に自動で tflite モデルを生成
CREATE_TENSOR_RT = False        # 学習時に自動で tensorrt モデルを生成

PRUNE_CNN = False               # モデルの重みを削減して性能向上を図るか
PRUNE_PERCENT_TARGET = 75       # 目標とする剪定率
PRUNE_PERCENT_PER_ITERATION = 20 # 1 回の剪定で除去する割合
PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2 # 剪定中に許容する最大バリデーション損失増加量
PRUNE_EVAL_PERCENT_OF_DATASET = .05  # 剪定評価に使用するデータセットの割合

#
# Augmentations と Transformations
#
# - Augmentation は学習時のみランダムに適用される画像変換で、データの多様性を高めます。
#   利用可能な Augmentation:
#   - BRIGHTNESS  - 画像の明るさを変更。 [albumentations](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast) 参照
#   - BLUR        - 画像をぼかす。 [albumentations](https://albumentations.ai/docs/api_reference/augmentations/blur/transforms/#albumentations.augmentations.blur.transforms.Blur) 参照
#
# - Transformation は学習時・推論時の双方で必ず適用され、設定順に処理されます。利用可能な変換例:
#   - 画像にマスクを適用:
#     - 'CROP'      - 画像の四辺を矩形マスク
#     - 'TRAPEZE'   - 台形マスクを適用
#   - 画像を強調:
#     - 'CANNY'     - Canny エッジ検出
#     - 'BLUR'      - 画像をぼかす
#   - 画像サイズ変更
#     - 'RESIZE'    - 指定した幅高さへリサイズ
#     - 'SCALE'     - 指定倍率でリサイズ
#   - 色空間変換
#     - 'RGB2BGR'   - RGB から BGR
#     - 'BGR2RGB'   - BGR から RGB
#     - 'RGB2HSV'   - RGB から HSV
#     - 'HSV2RGB'   - HSV から RGB
#     - 'BGR2HSV'   - BGR から HSV
#     - 'HSV2BGR'   - HSV から BGR
#     - 'RGB2GRAY'  - RGB からグレースケール
#     - 'BGR2GRAY'  - BGR からグレースケール
#     - 'HSV2GRAY'  - HSV からグレースケール
#     - 'GRAY2RGB'  - グレースケールから RGB
#     - 'GRAY2BGR'  - グレースケールから BGR
#
# 独自の変換クラスを作成してパイプラインに組み込むこともできます。
# - `CUSTOM_CROP` のように `CUSTOM` で始まるラベルを使用し、TRANSFORMATIONS または POST_TRANSFORMATIONS に追加します。
#   例: `POST_TRANSFORMATIONS = ['CUSTOM_CROP']`
# - カスタム変換を実装したモジュールとクラスを設定します。
#   - モジュール設定はラベル + `_MODULE` とし、値にはクラスを含む Python ファイルの絶対パスを指定します。
#     例: `CUSTOM_CROP_MODULE = "/home/pi/mycar/my_custom_transformer.py"`
#   - クラス設定はラベル + `_CLASS` とし、クラス名を指定します。例: `CUSTOM_CROP_CLASS = "CustomCropTransformer"`
# - コンストラクタには Config オブジェクトが渡されるので、必要な設定項目を myconfig.py に定義して読み込みます。
#   可能ならラベルを接頭辞にして他の設定と衝突しないようにします。例えばカスタムクロップでは以下のように境界値を設定できます。
#   ```
#   CUSTOM_CROP_TOP = 45    # rows to ignore on the top of the image
#   CUSTOM_CROP_BOTTOM = 5  # rows ignore on the bottom of the image
#   CUSTOM_CROP_RIGHT = 10  # pixels to ignore on the right of the image
#   CUSTOM_CROP_LEFT = 10   # pixels to ignore on the left of the image
#   ```
# - カスタムクラスには画像を受け取り画像を返す `run` メソッドが必要です。
#   ここで独自の変換処理を実装します。
# - 例えばクロップ後にぼかしを掛けるクラスは次のようになります。
#   ```
#   from donkeycar.parts.cv import ImgCropMask, ImgSimpleBlur
#
#   class CustomCropTransformer:
#       def __init__(self, config) -> None:
#           self.top = config.CUSTOM_CROP_TOP
#           self.bottom = config.CUSTOM_CROP_BOTTOM
#           self.left = config.CUSTOM_CROP_LEFT
#           self.right = config.CUSTOM_CROP_RIGHT
#           self.crop = ImgCropMask(self.left, self.top, self.right, self.bottom)
#           self.blur = ImgSimpleBlur()
#
#       def run(self, image):
#           image = self.crop.run(image)
#           return self.blur.run(image)
#   ```
#
AUGMENTATIONS = []         # 学習時のみ画像に適用してデータの多様性を増やす変換
TRANSFORMATIONS = []       # 学習前に実施する変換。AUGMENTATIONS は変換後の画像に適用される
POST_TRANSFORMATIONS = []  # 学習用 Augmentation 適用後の画像に施す追加変換

# 明るさやぼかしを行う場合は AUGMENTATIONS に 'MULTIPLY' や 'BLUR' を追加
AUG_BRIGHTNESS_RANGE = 0.2  # [-0.2, 0.2] と解釈される
AUG_BLUR_RANGE = (0, 3)

# "CROP" Transformation
# 画像の四辺を矩形でマスクします。
# 値を大きくしすぎると stride が負になりモデルが無効になるので注意してください。
# # # # # # # # # # # # #
# xxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxx #
# xx                 xx # top
# xx                 xx #
# xx                 xx #
# xxxxxxxxxxxxxxxxxxxxx # bottom
# xxxxxxxxxxxxxxxxxxxxx #
# # # # # # # # # # # # #
ROI_CROP_TOP = 45               # 画像上端で無視する行数
ROI_CROP_BOTTOM = 0             # 画像下端で無視する行数
ROI_CROP_RIGHT = 0              # 画像右端で無視する列数
ROI_CROP_LEFT = 0               # 画像左端で無視する列数

# "TRAPEZE" 変換
# 台形で画像の境界をマスクします。
# # # # # # # # # # # # # #
# xxxxxxxxxxxxxxxxxxxxxxx #
# xxxx ul     ur xxxxxxxx # min_y
# xxx             xxxxxxx #
# xx               xxxxxx #
# x                 xxxxx #
# ll                lr xx # max_y
# # # # # # # # # # # # # #
ROI_TRAPEZE_LL = 0
ROI_TRAPEZE_LR = 160
ROI_TRAPEZE_UL = 20
ROI_TRAPEZE_UR = 140
ROI_TRAPEZE_MIN_Y = 60
ROI_TRAPEZE_MAX_Y = 120

# "CANNY" エッジ検出変換
CANNY_LOW_THRESHOLD = 60    # エッジ検出の下限閾値
CANNY_HIGH_THRESHOLD = 110  # エッジ検出の上限閾値
CANNY_APERTURE = 3          # オペレータのサイズ(奇数)。choices=[3, 5, 7]

# "BLUR" 変換 (Augmentation のぼかしとは別)
BLUR_KERNEL = 5        # ぼかしカーネルの横幅(px)
BLUR_KERNEL_Y = None   # 縦幅(px)。None なら正方形
BLUR_GAUSSIAN = True   # True ならガウシアン、False なら単純ブラー

# "RESIZE" 変換
RESIZE_WIDTH = 160     # リサイズ後の幅(px)
RESIZE_HEIGHT = 120    # リサイズ後の高さ(px)

# "SCALE" 変換
SCALE_WIDTH = 1.0      # 横方向の倍率
SCALE_HEIGHT = None    # 縦方向の倍率。None ならアスペクト比維持

# モデル転移時のオプション
# 既存重みをコピーする際に、一部の層を固定して学習させないようにするか
FREEZE_LAYERS = False               # False なら全層を学習対象とする
NUM_LAST_LAYERS_TO_TRAIN = 7        # 固定する場合、末尾から何層を学習させるか

# WEB コントローラ
WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # Web コントローラが待ち受けるポート
WEB_INIT_MODE = "user"              # 起動時の制御モード user|local_angle|local

# ジョイスティック
USE_JOYSTICK_AS_DEFAULT = False      # True なら manage.py 起動時に --js を省略してもジョイスティックを使用
JOYSTICK_MAX_THROTTLE = 0.5         # -1〜1 のスロットル値に乗算して最大速度を制限
JOYSTICK_STEERING_SCALE = 1.0       # ステア感度のスケール。負にすると方向反転
AUTO_RECORD_ON_THROTTLE = True      # スロットルがゼロ以外のとき自動で記録開始
CONTROLLER_TYPE = 'xbox'            # (ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom)
USE_NETWORKED_JS = False            # ネットワーク越しにジョイスティック入力を受け付けるか
NETWORK_JS_SERVER_IP = None         # ネットワークジョイスティック使用時のサーバ IP
JOYSTICK_DEADZONE = 0.01            # この値以下のスロットルでは記録しない
JOYSTICK_THROTTLE_DIR = -1.0         # -1.0 で前後反転、1.0 で通常方向
USE_FPV = False                     # カメラ画像を FPV サーバへ送信するか
JOYSTICK_DEVICE_FILE = "/dev/input/js0" # ジョイスティックのデバイスファイル

# カテゴリカルモデル用: 学習したスロットルの上限値を制限
# 学習 PC とロボット双方の config.py で同じ値にする必要がある
MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8

# RNN または 3D モデル用
SEQUENCE_LENGTH = 3             # 時系列で使用する画像枚数

# IMU
HAVE_IMU = False                # True にすると Mpu6050 パーツを追加して記録
IMU_SENSOR = 'mpu6050'          # (mpu6050|mpu9250)
IMU_ADDRESS = 0x68              # AD0 ピンが High の場合 0x69、そうでなければ 0x68
IMU_DLP_CONFIG = 0              # デジタルローパスフィルタ設定 (0:250Hz, 1:184Hz, 2:92Hz, 3:41Hz, 4:20Hz, 5:10Hz, 6:5Hz)

# SOMBRERO
HAVE_SOMBRERO = False           # Donkeycar ストアの Sombrero Hat 使用時は True

# PIGPIO RC 制御
STEERING_RC_GPIO = 26
THROTTLE_RC_GPIO = 20
DATA_WIPER_RC_GPIO = 19
PIGPIO_STEERING_MID = 1500         # 直進しない場合はこの値を調整
PIGPIO_MAX_FORWARD = 2000          # 前進最大スロットル。大きいほど速い
PIGPIO_STOPPED_PWM = 1500
PIGPIO_MAX_REVERSE = 1000          # 後退最大スロットル。小さいほど速い
PIGPIO_SHOW_STEERING_VALUE = False
PIGPIO_INVERT = False
PIGPIO_JITTER = 0.025   # この値以下は無信号とみなす



# ROBOHAT MM1
MM1_STEERING_MID = 1500         # 直進できない場合に調整する値
MM1_MAX_FORWARD = 2000          # 前進最大スロットル。大きいほど速い
MM1_STOPPED_PWM = 1500
MM1_MAX_REVERSE = 1000          # 後退最大スロットル。小さいほど速い
MM1_SHOW_STEERING_VALUE = False
# シリアルポート
# -- Pi のデフォルト: '/dev/ttyS0'
# -- Jetson Nano: '/dev/ttyTHS1'
# -- Google Coral: '/dev/ttymxc0'
# -- Windows: 'COM3', Arduino: '/dev/ttyACM0'
# -- Mac/Linux は `ls /dev/tty.*` で適切なポートを確認して置き換える
MM1_SERIAL_PORT = '/dev/ttyS0'  # MM1 通信用シリアルポート

#LOGGING
HAVE_CONSOLE_LOGGING = True
LOGGING_LEVEL = 'INFO'          # Python のログレベル: 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
LOGGING_FORMAT = '%(message)s'  # Python のログフォーマット

# TELEMETRY
HAVE_MQTT_TELEMETRY = False
TELEMETRY_DONKEY_NAME = 'my_robot1234'
TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
TELEMETRY_MQTT_JSON_ENABLE = False
TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
TELEMETRY_MQTT_BROKER_PORT = 1883
TELEMETRY_PUBLISH_PERIOD = 1
TELEMETRY_LOGGING_ENABLE = True
TELEMETRY_LOGGING_LEVEL = 'INFO' # Python のログレベル
TELEMETRY_LOGGING_FORMAT = '%(message)s'  # Python のログフォーマット
TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
TELEMETRY_DEFAULT_TYPES = 'float,float'

# パフォーマンスモニタ
HAVE_PERFMON = False

# 録画オプション
RECORD_DURING_AI = False        # AI モード中も記録する場合は True
AUTO_CREATE_NEW_TUB = False     # 新しい tub_YY_MM_DD ディレクトリを自動作成

#LED
HAVE_RGB_LED = False            # 例: https://www.amazon.com/dp/B07BNRZWNF のような RGB LED を使用するか
LED_INVERT = False              # コモンアノード LED の場合 True

# LED 用 PWM ピン番号 (物理ピン)
# 参考: https://www.raspberrypi-spy.co.uk/2012/06/simple-guide-to-the-rpi-gpio-header-and-pins/
LED_PIN_R = 12
LED_PIN_G = 10
LED_PIN_B = 16

# LED の基本色 (0-100)
LED_R = 0
LED_G = 0
LED_B = 1

# 記録数インジケータ用 LED 色
REC_COUNT_ALERT = 1000          # この件数ごとに点滅開始
REC_COUNT_ALERT_CYC = 15        # 点滅サイクル数(1/20秒単位)
REC_COUNT_ALERT_BLINK_RATE = 0.4 # 点滅速度(秒)

# 先頭は記録数、次が (r, g, b) 0-100
# 記録数が閾値を超えると指定色を使用
RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
            (3000, (5, 5, 5)),
            (5000, (5, 2, 0)),
            (10000, (0, 5, 0)),
            (15000, (0, 5, 5)),
            (20000, (0, 0, 5)), ]


# モデル再読み込み時の LED 色 (0-100)
MODEL_RELOADED_LED_R = 100
MODEL_RELOADED_LED_G = 0
MODEL_RELOADED_LED_B = 0


# BEHAVIORS
# Behavioral Neural Network モデル学習時に使用する挙動リスト
# TRAIN_BEHAVIORS を True にし、BEHAVIOR_LED_COLORS で各挙動の色を指定
TRAIN_BEHAVIORS = False
BEHAVIOR_LIST = ['Left_Lane', "Right_Lane"]
BEHAVIOR_LED_COLORS = [(0, 10, 0), (10, 0, 0)]  #RGB tuples 0-100 per chanel

# Localizer
# コース上の位置を推定するニューラルネットワーク。実験的機能
# コースを NUM_LOCATIONS の区間に分割し、その区間を予測する
TRAIN_LOCALIZER = False
NUM_LOCATIONS = 10
BUTTON_PRESS_NEW_TUB = False # True にすると X ボタンを押すたび新しい tub を作成

# DonkeyGym
# Ubuntu Linux 上のみ利用可能なシミュレータ。manage.py drive コマンドで仮想車両を操作できる
# 上記リポジトリからシミュレータをダウンロードし、DONKEY_SIM_PATH を適宜変更する
DONKEY_GYM = False
DONKEY_SIM_PATH = "path to sim" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" when racing on virtual-race-league use "remote", or user "remote" when you want to start the sim manually first.
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
GYM_CONF = { "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # body style(donkey|bare|car01) body rgb 0-255
GYM_CONF["racer_name"] = "Your Name"
GYM_CONF["country"] = "Place"
GYM_CONF["bio"] = "I race robots."

SIM_HOST = "127.0.0.1"              # バーチャルレースでは "trainmydonkey.com" を使用
SIM_ARTIFICIAL_LATENCY = 0          # 遠隔操作時の遅延を模擬するミリ秒単位の値

# Save info from Simulator (pln)
SIM_RECORD_LOCATION = False
SIM_RECORD_GYROACCEL= False
SIM_RECORD_VELOCITY = False
SIM_RECORD_LIDAR = False

# カメラ画像をネットワーク配信
# TCP サービスを作成して画像を配信する
PUB_CAMERA_IMAGES = False

# レース時に AI を加速させたい場合の設定
AI_LAUNCH_DURATION = 0.0            # 指定時間だけ AI がスロットルを出力
AI_LAUNCH_THROTTLE = 0.0            # 加速時のスロットル値
AI_LAUNCH_ENABLE_BUTTON = 'R2'      # ブースト有効化に使うボタン
AI_LAUNCH_KEEP_ENABLED = False      # False なら毎回ボタンを押して有効化

# すべてのモデルの出力スロットルをスケーリング
AI_THROTTLE_MULT = 1.0              # 乗算係数

# パスフォロー
PATH_FILENAME = "donkey_path.pkl"   # パスを保存するファイル名
PATH_SCALE = 5.0                    # Web 表示時の拡大率
PATH_OFFSET = (0, 0)                # 255,255 が地図の中心。原点表示のオフセット
PATH_MIN_DIST = 0.3                 # この距離(m)移動ごとにパス点を保存
PID_P = -10.0                       # PID の比例項
PID_I = 0.000                       # PID の積分項
PID_D = -0.2                        # PID の微分項
PID_THROTTLE = 0.2                  # パス走行時の一定スロットル値
USE_CONSTANT_THROTTLE = False       # 録画時に取得したスロットルを使うか一定値を使うか
SAVE_PATH_BTN = "cross"             # パス保存ボタン
RESET_ORIGIN_BTN = "triangle"       # 原点リセットボタン

# Intel Realsense D435/D435i
REALSENSE_D435_RGB = True       # RGB 画像を取得する
REALSENSE_D435_DEPTH = True     # 深度画像を取得する
REALSENSE_D435_IMU = False      # D435i の IMU データを取得する
REALSENSE_D435_ID = None        # カメラのシリアル番号。1 台のみなら None

# 一時停止標識検出
STOP_SIGN_DETECTOR = False
STOP_SIGN_MIN_SCORE = 0.2
STOP_SIGN_SHOW_BOUNDING_BOX = True
STOP_SIGN_MAX_REVERSE_COUNT = 10    # 停止標識検出時の後退回数。0 で無効
STOP_SIGN_REVERSE_THROTTLE = -0.5     # 後退時のスロットル値

# FPS カウンタ
SHOW_FPS = False
FPS_DEBUG_INTERVAL = 10    # 何秒ごとにフレームレートを表示するか
