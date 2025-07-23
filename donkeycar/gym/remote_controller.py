"""remote_controller モジュール.

作者: Tawn Kramer
日付: 2019-01-24
説明: ネットワーク越しに Donkey ロボットを制御する
"""

import time

from donkeycar.parts.network import MQTTValueSub, MQTTValuePub
from donkeycar.parts.image import JpgToImgArr

class DonkeyRemoteContoller:
    """遠隔地の Donkey ロボットを操作するためのコントローラー。"""
    def __init__(self, donkey_name, mqtt_broker, sensor_size=(120, 160, 3)):
        """インスタンスを初期化する。

        Args:
            donkey_name: Donkey ロボットの名前。
            mqtt_broker: MQTT ブローカーのホスト名。
            sensor_size: カメラ画像のサイズ ``(H, W, C)``。
        """

        self.camera_sub = MQTTValueSub("donkey/%s/camera" % donkey_name, broker=mqtt_broker)
        self.controller_pub = MQTTValuePub("donkey/%s/controls" % donkey_name, broker=mqtt_broker)
        self.jpgToImg = JpgToImgArr()
        self.sensor_size = sensor_size

    def get_sensor_size(self):
        """センサーサイズを取得する。"""

        return self.sensor_size

    def wait_until_connected(self):
        """接続完了まで待機する。"""

        pass

    def take_action(self, action):
        """アクションを送信する。"""

        self.controller_pub.run(action)

    def quit(self):
        """接続を終了する。"""

        self.camera_sub.shutdown()
        self.controller_pub.shutdown()

    def get_original_image(self):
        """最新の画像を取得する。"""

        return self.img

    def observe(self):
        """センサーから画像を取得して返す。"""

        jpg = self.camera_sub.run()
        self.img = self.jpgToImg.run(jpg)
        return self.img


    
