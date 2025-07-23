"""車の設定を定義するモジュール。

このファイルは `manage.py` スクリプトによって読み込まれ、車の性能を変更します。

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)
"""


import os

#パス
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#車両
DRIVE_LOOP_HZ = 20
MAX_LOOPS = None

#カメラ
CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|MOCK|IMAGE_LIST)
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # デフォルトは RGB=3。白黒画像の場合は 1
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
# CSICカメラが回転した状態で取り付けられている場合、下記のパラメータを変更すると出力フレームの向きが修正される
CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => none , 4 => Flip horizontally, 6 => Flip vertically)

#9865 は必要な場合のみ上書きする、例: TX2
PCA9685_I2C_ADDR = 0x40
PCA9685_I2C_BUSNUM = None

#ステアリング
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 460
STEERING_RIGHT_PWM = 290

#スロットル
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 500
THROTTLE_STOPPED_PWM = 370
THROTTLE_REVERSE_PWM = 220

DRIVE_TRAIN_TYPE = "SERVO_ESC" # SERVO_ESC|DC_STEER_THROTTLE|DC_TWO_WHEEL|SERVO_HBRIDGE_PWM|PIGPIO_PWM|MM1|MOCK のいずれか

# #LIDAR（ライダー）
USE_LIDAR = False
LIDAR_TYPE = 'RP' #(RP|YD)
LIDAR_LOWER_LIMIT = 44 # 記録する角度。車体に遮られる領域や後方を見る範囲を除外するのに使用する。RP A1M8 Lidar では「0」がモーター方向になる点に注意
LIDAR_UPPER_LIMIT = 136

# #RCコントロール
USE_RC = False
STEERING_RC_GPIO = 26
THROTTLE_RC_GPIO = 20
DATA_WIPER_RC_GPIO = 19

#ログ設定
HAVE_CONSOLE_LOGGING = True
LOGGING_LEVEL = 'INFO'          # Python のログレベル: 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
LOGGING_FORMAT = '%(message)s'  # Python のログフォーマット - https://docs.python.org/3/library/logging.html#formatter-objects


#学習設定
DEFAULT_AI_FRAMEWORK = 'tensorflow'  # 使用するAIフレームワークのデフォルト (tensorflow|pytorch)
DEFAULT_MODEL_TYPE = 'linear' #(linear|categorical|rnn|imu|behavior|3d|localizer|latent) のいずれか
CREATE_TF_LITE = True  # トレーニング時に自動で tflite モデルを生成
CREATE_TENSOR_RT = False  # トレーニング時に自動で TensorRT モデルを生成
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8
MAX_EPOCHS = 100
SHOW_PLOT = True
VERBOSE_TRAIN = True
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 5
MIN_DELTA = .0005
PRINT_MODEL_SUMMARY = True      # 層や重みを標準出力に表示
OPTIMIZER = None                # adam、sgd、rmsprop など。None の場合はデフォルト
LEARNING_RATE = 0.001           # OPTIMIZER を指定した場合のみ使用
LEARNING_RATE_DECAY = 0.0       # OPTIMIZER を指定した場合のみ使用
PRUNE_CNN = False
PRUNE_PERCENT_TARGET = 75 # 目標とする剪定率
PRUNE_PERCENT_PER_ITERATION = 20 # 1 回の剪定で行う割合
PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2 # 剪定中に許容される最大バリデーション損失増加量
PRUNE_EVAL_PERCENT_OF_DATASET = .05  # モデル評価に使用するデータセットの割合

#モデル転移オプション
FREEZE_LAYERS = False
NUM_LAST_LAYERS_TO_TRAIN = 7

#カテゴリカルモデルの場合、学習済みスロットルの上限を制限する
#この値は学習用PCのconfig.pyとrobot.pyで一致させることが非常に重要
#一度設定したらできれば変更しない方がよい
MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8

#RNN または 3D
SEQUENCE_LENGTH = 3

# 画像増強と変換
AUGMENTATIONS = []
TRANSFORMATIONS = []
# 明るさとぼかしの設定。'BRIGHTNESS' と 'BLUR' を
# AUGMENTATIONS に指定する
AUG_BRIGHTNESS_RANGE = 0.2  # 値 0.2 は [-0.2, 0.2] と解釈される
AUG_BLUR_RANGE = (0, 3)
# 切り取るピクセル数。TRANSFORMATIONS に 'CROP' を設定している必要がある
ROI_CROP_TOP = 45
ROI_CROP_BOTTOM = 0
ROI_CROP_RIGHT = 0
ROI_CROP_LEFT = 0
# 台形領域の詳細は augmentations.py を参照。'TRAPEZE' を
# TRANSFORMATIONS に設定している必要がある
ROI_TRAPEZE_LL = 0
ROI_TRAPEZE_LR = 160
ROI_TRAPEZE_UL = 20
ROI_TRAPEZE_UR = 140
ROI_TRAPEZE_MIN_Y = 60
ROI_TRAPEZE_MAX_Y = 120



#SOMBRERO ボード
HAVE_SOMBRERO = False

#記録オプション
RECORD_DURING_AI = False
AUTO_CREATE_NEW_TUB = False     #録画時に新しいディレクトリ(tub_YY_MM_DD)を自動作成するか。False の場合は既存データに追記

#ジョイスティック
USE_JOYSTICK_AS_DEFAULT = False     #manage.py 起動時に True にすると --js オプションなしでジョイスティックを使用
JOYSTICK_MAX_THROTTLE = 0.5         #この係数を -1〜1 のスロットル値に掛けて最高速度を制限する。コントローラーを落とした場合や速度を抑えたい場合に便利
JOYSTICK_STEERING_SCALE = 1.0       #ステアリング感度を下げたい場合に使用。ステアリング値 -1〜1 に掛ける。負値で向きを反転可能
AUTO_RECORD_ON_THROTTLE = True      #True の場合、スロットルが 0 でないとき自動で録画。False の場合は別のトリガーで録画を切り替える（通常はジョイスティックのサークルボタン）
CONTROLLER_TYPE='ps3'               #(ps3|ps4|xbox|nimbus|wiiu|F710|rc3|MM1|custom) custom を選ぶと `donkey createjs` コマンドで作成した my_joystick.py が実行される
USE_NETWORKED_JS = False            #ネットワーク経由でジョイスティック入力を受け付けるか
NETWORK_JS_SERVER_IP = "192.168.0.1"#ネットワークジョイスティック使用時に入力を提供するサーバーの IP
JOYSTICK_DEADZONE = 0.0             # 0 以外の場合、記録開始をトリガーする最小スロットル値
JOYSTICK_THROTTLE_DIR = -1.0        # -1.0 で前後を反転、1.0 でジョイスティックの自然な方向を使用
USE_FPV = False                     # カメラ画像を FPV Web サーバーへ送信
JOYSTICK_DEVICE_FILE = "/dev/input/js0" # ジョイスティックへアクセスする Unix デバイスファイル

#WEBコントロール
WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # Web コントローラが待ち受けるポート番号
WEB_INIT_MODE = "user"              # 起動時の制御モード。user|local_angle|local のいずれか。local を指定するとAIモードで開始

#走行設定
AI_THROTTLE_MULT = 1.0              # NN モデルの出力するスロットル値に掛ける倍率


#DonkeyGym 設定
#Ubuntu Linux 環境のみ、シミュレータを仮想ドンキーとして利用でき、
#通常の `python manage.py drive` コマンドで仮想車両を操作できる。
#これを有効にし、シミュレータのパスと環境を設定する。
#シミュレータのバイナリは https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip から取得し、
#展開後に DONKEY_SIM_PATH を変更すること。
DONKEY_GYM = False
DONKEY_SIM_PATH = "path to sim" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" 仮想レースリーグで走る際は "remote" を使用。シムを手動で起動したい場合も "remote"
DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
GYM_CONF = { "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # ボディスタイル(donkey|bare|car01) とボディカラー(0-255)
GYM_CONF["racer_name"] = "Your Name"
GYM_CONF["country"] = "Place"
GYM_CONF["bio"] = "I race robots."

SIM_HOST = "127.0.0.1"              # virtual-race-league で走るときは "trainmydonkey.com" を使用
SIM_ARTIFICIAL_LATENCY = 0          # 制御の遅延をミリ秒単位で指定。リモートサーバ利用時の遅延を模擬するのに有用。100〜400ms 程度が妥当
