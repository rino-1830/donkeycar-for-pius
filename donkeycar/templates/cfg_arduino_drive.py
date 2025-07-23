"""車両設定

このファイルは車両アプリケーションの ``manage.py`` スクリプトによって読み込まれ、
車の性能を変更します。

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
DRIVE_LOOP_HZ = 20
MAX_LOOPS = None

# ステアリング
STEERING_ARDUINO_PIN = 6
STEERING_ARDUINO_LEFT_PWM = 120
STEERING_ARDUINO_RIGHT_PWM = 40

# スロットル
THROTTLE_ARDUINO_PIN = 5
THROTTLE_ARDUINO_FORWARD_PWM = 105
THROTTLE_ARDUINO_STOPPED_PWM = 90
THROTTLE_ARDUINO_REVERSE_PWM = 75

# ジョイスティック
USE_JOYSTICK_AS_DEFAULT = False     # manage.py を起動するとき、True ならジョイスティックを使うのに --js オプションが不要になります
JOYSTICK_MAX_THROTTLE = 0.8         # このスカラーは -1 ～ 1 のスロットル値に掛け合わせて最大スロットルを制限します。コントローラを落としてしまった場合や全速を必要としないときに役立ちます。
JOYSTICK_STEERING_SCALE = 1.0       # 反応を鈍くしたい場合があります。このスカラーは -1 ～ 1 のステアリング値に掛けます。負の値にすると方向が反転します。
AUTO_RECORD_ON_THROTTLE = True      # True の場合、スロットルが 0 でないとき常に記録します。False の場合、別のトリガーで手動で記録を切り替える必要があります。通常はジョイスティックのサークルボタンです。
CONTROLLER_TYPE='F710'               # (ps3|ps4|xbox|nimbus|wiiu|F710|rc3)
USE_NETWORKED_JS = False            # ネットワーク経由でリモートジョイスティック操作を受け付けるかどうか
NETWORK_JS_SERVER_IP = "192.168.0.1"# ネットワークジョイスティック操作を受け取るとき、この情報を配信する IP
JOYSTICK_DEADZONE = 0.0             # 非ゼロの場合、記録を開始する最小スロットル値
JOYSTICK_THROTTLE_DIR = -1.0        # 前後を反転するには -1.0、ジョイスティックの自然な前後を使うには 1.0 を指定
USE_FPV = False                     # カメラデータを FPV Web サーバーへ送信する
