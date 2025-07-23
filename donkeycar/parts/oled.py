# Adafruit ssd1306 ライブラリが必要: pip install adafruit-circuitpython-ssd1306

"""OLED 表示にテキストを描画するためのユーティリティ群。"""

import subprocess
import time
from board import SCL, SDA
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306


class OLEDDisplay(object):
    """OLED ディスプレイにテキストを描画するクラス。"""
    def __init__(self, rotation=0, resolution=1):
        """インスタンスを初期化する。

        Args:
            rotation (int): ディスプレイの回転角度。
            resolution (int): 解像度種別。2 を指定すると高さ 64、それ以外は 32。
        """
        # プレースホルダー
        self._EMPTY = ''
        # 表示できるテキスト行数
        self._SLOT_COUNT = 4
        self.slots = [self._EMPTY] * self._SLOT_COUNT
        self.display = None
        self.rotation = rotation
        if resolution == 2:
            self.height = 64
        else:
            self.height = 32

    def init_display(self):
        """OLED ディスプレイを初期化する。"""
        if self.display is None:
            # I2C インターフェースを作成する。
            i2c = busio.I2C(SCL, SDA)
            # SSD1306 OLED クラスを生成する。
            # 最初の 2 つの引数はピクセルの幅と高さで、使用するディスプレイに合わせて変更する。
            self.display = adafruit_ssd1306.SSD1306_I2C(128, self.height, i2c)
            self.display.rotation = self.rotation


            self.display.fill(0)
            self.display.show()

            # 描画用の空画像を作成する。
            # 1 ビットカラー用にモード '1' で画像を作成すること。
            self.width = self.display.width
            self.image = Image.new("1", (self.width, self.height))

            # 画像に描画するためのオブジェクトを取得する。
            self.draw = ImageDraw.Draw(self.image)

            # 画像を消去するため黒色で塗りつぶす。
            self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)
            # フォントを読み込む
            self.font = ImageFont.load_default()
            self.clear_display()

    def clear_display(self):
        """ディスプレイをクリアする。"""
        if self.draw is not None:
            self.draw.rectangle((0, 0, self.width, self.height), outline=0, fill=0)

    def update_slot(self, index, text):
        """表示スロットを更新する。"""
        if index < len(self.slots):
            self.slots[index] = text

    def clear_slot(self, index):
        """指定したスロットを空にする。"""
        if index < len(self.slots):
            self.slots[index] = self._EMPTY

    def update(self):
        """スロットの内容をディスプレイに表示する。"""
        x = 0
        top = -2
        self.clear_display()
        for i in range(self._SLOT_COUNT):
            text = self.slots[i]
            if len(text) > 0:
                self.draw.text((x, top), text, font=self.font, fill=255)
                top += 8

        # 更新
        self.display.rotation = self.rotation
        self.display.image(self.image)
        self.display.show()


class OLEDPart(object):
    """OLED ディスプレイにステータスを表示するパート。"""
    def __init__(self, rotation, resolution, auto_record_on_throttle=False):
        """パートを初期化する。

        Args:
            rotation (int): ディスプレイの回転角度。
            resolution (int): 解像度種別。
            auto_record_on_throttle (bool, optional): スロットル操作で自動記録するかどうか。
        """
        self.oled = OLEDDisplay(rotation, resolution)
        self.oled.init_display()
        self.on = False
        if auto_record_on_throttle:
            self.recording = '自動'
        else:
            self.recording = 'いいえ'
        self.num_records = 0
        self.user_mode = None
        eth0 = OLEDPart.get_ip_address('eth0')
        wlan0 = OLEDPart.get_ip_address('wlan0')
        if eth0 is not None:
            self.eth0 = 'eth0:%s' % (eth0)
        else:
            self.eth0 = None
        if wlan0 is not None:
            self.wlan0 = 'wlan0:%s' % (wlan0)
        else:
            self.wlan0 = None

    def run(self):
        """パートを有効化する。"""
        if not self.on:
            self.on = True

    def run_threaded(self, recording, num_records, user_mode):
        """スレッド実行時に呼び出され、表示内容を更新する。

        Args:
            recording (bool): 録画中かどうか。
            num_records (int): 記録件数。
            user_mode (str): ユーザーモード。
        """
        if num_records is not None and num_records > 0:
            self.num_records = num_records

        if recording:
            self.recording = 'はい (記録数 = %s)' % (self.num_records)
        else:
            self.recording = 'いいえ (記録数 = %s)' % (self.num_records)

        self.user_mode = 'ユーザーモード (%s)' % (user_mode)

    def update_slots(self):
        """スロットの内容を最新の状態にする。"""
        updates = [self.eth0, self.wlan0, self.recording, self.user_mode]
        index = 0
        # スロットを更新
        for update in updates:
            if update is not None:
                self.oled.update_slot(index, update)
                index += 1

        # ディスプレイを更新
        self.oled.update()

    def update(self):
        """スレッドを自走させてディスプレイを更新する。"""
        self.on = True
        # 単独のスレッドループを実行
        while self.on:
            self.update_slots()

    def shutdown(self):
        """ディスプレイをクリアして停止する。"""
        self.oled.clear_display()
        self.on = False

    # https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/jetbot/utils/utils.py

    @classmethod
    def get_ip_address(cls, interface):
        """指定インターフェースの IP アドレスを取得する。

        Args:
            interface (str): インターフェース名。

        Returns:
            str | None: 取得できない場合は ``None``。
        """
        if OLEDPart.get_network_interface_state(interface) == 'down':
            return None
        cmd = "ifconfig %s | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'" % interface
        return subprocess.check_output(cmd, shell=True).decode('ascii')[:-1]

    @classmethod
    def get_network_interface_state(cls, interface):
        """インターフェースの状態を取得する。

        Args:
            interface (str): インターフェース名。

        Returns:
            str: ``up`` や ``down`` などの状態文字列。
        """
        return subprocess.check_output('cat /sys/class/net/%s/operstate' % interface, shell=True).decode('ascii')[:-1]
