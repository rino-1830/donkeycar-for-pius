""" 
車の設定

このファイルはあなたの車アプリケーションの manage.py スクリプトによって読み込まれ、車の性能を変更します。

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
MAX_LOOPS = 100000

#カメラ
IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3         # デフォルトはRGB=3。モノクロの場合は1を指定

#9865、必要な場合のみ上書き。例: TX2
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

#学習
DEFAULT_MODEL_TYPE = 'linear' #(linear|categorical|rnn|imu|behavior|3d|localizer|latent) のいずれか
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8
MAX_EPOCHS = 100
SHOW_PLOT = True
VERBOSE_TRAIN = True
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 5
MIN_DELTA = .0005
PRINT_MODEL_SUMMARY = True      # レイヤーと重みを標準出力に表示する
OPTIMIZER = None                # adam、sgd、rmsprop など。None ならデフォルトを使用
LEARNING_RATE = 0.001           # OPTIMIZER を指定した場合のみ使用
LEARNING_RATE_DECAY = 0.0       # OPTIMIZER を指定した場合のみ使用
PRUNE_CNN = False
PRUNE_PERCENT_TARGET = 75 # 目標とするプルーニング割合
PRUNE_PERCENT_PER_ITERATION = 20 # 各イテレーションで行うプルーニングの割合
PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2 # プルーニング中に許容される最大の検証損失増加
PRUNE_EVAL_PERCENT_OF_DATASET = .05  # モデル評価に使用するデータセットの割合

#モデル転移オプション
FREEZE_LAYERS = False
NUM_LAST_LAYERS_TO_TRAIN = 7

#カテゴリカルモデルでは学習されたスロットルの上限を制限する
#この値は学習用PCのconfig.pyとrobot.pyで一致させることが非常に重要
#一度設定したら理想的には変更しないこと
MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.8

#RNN または 3D
SEQUENCE_LENGTH = 3

#SOMBRERO
HAVE_SOMBRERO = False

#記録オプション
RECORD_DURING_AI = False
