"""エンコーダーとオドメトリ部品。

donkeycar.parts.Tachometer と donkeycar.parts.Odometer を使用するため本モジュールは非推奨。
"""

from datetime import datetime
from donkeycar.utilities.deprecated import deprecated
import re
import time


# Arduino クラスは、外部マイコンで読み取られるクアドラチャエンコーダーやモーターエンコーダー用です。
# Arduino や Teensy などから USB シリアル経由で RaspberryPi または Nano にシリアルデータを送信します。
# マイコンはこのスケッチで書き込んでください（Arduino IDE を使用）：https://github.com/zlite/donkeycar/tree/master/donkeycar/parts/encoder/encoder
# Donkeycar の tests フォルダーにある `test_encoder.py` を使って、正しいピンに接続されているか確認するか、使用するピンに合わせて編集してください。

# 以下の `mm_per_tick` は車両ごとに調整が必要です。1 メートルを測って車を転がし、1.0 になるよう調整してください。
# このクラスは 10Hz でオドメーターをサンプリングし、直近 10 回の読み取りの移動平均から速度を計算します。

@deprecated("donkeycar.parts.tachometer.Tachometer(SerialEncoder) を推奨")
class ArduinoEncoder(object):
    """Arduinoから送られるエンコーダー値を読み取るクラス。

    Attributes:
        ser (serial.Serial): エンコーダーが接続されたシリアルポート。
        ticks (int): 現在のティック数。
        lasttick (int): 最後に受信したティック数。
        debug (bool): デバッグ出力を行うかどうか。
        on (bool): スレッド実行フラグ。
        mm_per_tick (float): ティック当たりの移動距離[mm]。
    """
    def __init__(self, mm_per_tick=0.0000599, debug=False):
        """インスタンスを初期化する。

        Args:
            mm_per_tick (float, optional): ティック当たりの移動距離[mm]。デフォルトは ``0.0000599``。
            debug (bool, optional): デバッグ出力を行うかどうか。デフォルトは ``False``。
        """
        import serial
        import serial.tools.list_ports
        from donkeycar.parts.pigpio_enc import OdomDist
        for item in serial.tools.list_ports.comports():
            print(item)  # 接続されているシリアルポートを一覧表示する
        self.ser = serial.Serial('/dev/ttyACM0', 115200, 8, 'N', 1, timeout=0.1)
        # オドメーターの値を初期化する
        self.ser.write(str.encode('r'))  # エンコーダーをゼロにリスタート
        self.ticks = 0
        self.lasttick = 0
        self.debug = debug
        self.on = True
        self.mm_per_tick = mm_per_tick

    def update(self):
        """シリアルポートからデータを取得し速度と距離を計算する。"""

        while self.on:
            input = ''
            while (self.ser.in_waiting > 0):   # シリアルポートを読み取り、データがあるか確認する
                buffer = self.ser.readline()
                input = buffer.decode('ascii')
            self.ser.write(str.encode('p'))  # 'p' 文字を送信して次の読み取りをリクエストする
            if input != '':
                temp = input.strip()  # 空白を取り除く
                if (temp.isnumeric()):
                    self.ticks = int(temp)
                    self.lasttick = self.ticks
            else: self.ticks = self.lasttick
            self.speed, self.distance = self.OdomDist(self.ticks, self.mm_per_tick)

    def run_threaded(self):
        """スレッドモードで速度を返す。"""

        self.speed
        return self.speed


    def shutdown(self):
        """スレッドを停止してリソースを解放する。"""

        print('Arduinoエンコーダーを停止しています')
        self.on = False
        time.sleep(.5)


@deprecated("未使用のため非推奨")
class AStarSpeed:
    """Pololu A-Starから速度と加速度を読み取るクラス。

    Attributes:
        speed (float): 現在の速度[m/s]。
        linaccel (dict | None): 加速度情報。
        sensor (TeensyRCin): データ取得に用いるセンサー。
        on (bool): スレッド実行フラグ。
    """
    def __init__(self):
        """インスタンスを初期化する。"""

        from donkeycar.parts.teensy import TeensyRCin
        self.speed = 0
        self.linaccel = None
        self.sensor = TeensyRCin(0)
        self.on = True

    def update(self):
        """センサーから受信したデータを解析し速度を更新する。"""

        encoder_pattern = re.compile('^E ([-0-9]+)( ([-0-9]+))?( ([-0-9]+))?$')
        linaccel_pattern = re.compile('^L ([-.0-9]+) ([-.0-9]+) ([-.0-9]+) ([-0-9]+)$')

        while self.on:
            start = datetime.now()

            l = self.sensor.astar_readline()
            while l:
                m = encoder_pattern.match(l.decode('utf-8'))

                if m:
                    value = int(m.group(1))
                    # rospy.loginfo("%s: Receiver E got %d" % (self.node_name, value))  # デバッグ用ログ
                    # 速度
                    # 1回転あたり40ティック
                    # 外周0.377m
                    # 0.1秒ごと
                    if len(m.group(3)) > 0:
                        period = 0.001 * int(m.group(3))
                    else:
                        period = 0.1

                    self.speed = 0.377 * (float(value) / 40) / period   # 単位はm/s
                else:
                    m = linaccel_pattern.match(l.decode('utf-8'))

                    if m:
                        la = { 'x': float(m.group(1)), 'y': float(m.group(2)), 'z': float(m.group(3)) }

                        self.linaccel = la
                        print("mw 加速度= " + str(self.linaccel))

                l = self.sensor.astar_readline()

            stop = datetime.now()
            s = 0.1 - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

    def run_threaded(self):
        """現在の速度を返す。"""

        return self.speed # , self.linaccel（未使用）

    def shutdown(self):
        """スレッドを停止してリソースを解放する。"""

        self.on = False
        print('AStarSpeed を停止しています')
        time.sleep(.5)


@deprecated("donkeycar.parts.tachometer.Tachometer(GpioEncoder) を推奨")
class RotaryEncoder():
    """GPIO入力を利用したロータリーエンコーダー計測クラス。

    Attributes:
        pi (pigpio.pi): pigpio のインスタンス。
        pin (int): エンコーダー信号を受け取るGPIOピン番号。
        m_per_tick (float): ティックあたりの移動距離[m]。
        poll_delay (float): ループの待機時間[s]。
        meters (float): 走行距離[m]。
        last_time (float): 前回計測時刻。
        meters_per_second (float): 現在の速度[m/s]。
        counter (int): ティックカウンター。
        on (bool): スレッド実行フラグ。
        debug (bool): デバッグ出力を行うかどうか。
        top_speed (float): 最高速度[m/s]。
    """
    def __init__(self, mm_per_tick=0.306096, pin=13, poll_delay=0.0166, debug=False):
        """インスタンスを初期化する。"""

        import pigpio
        self.pi = pigpio.pi()
        self.pin = pin
        self.pi.set_mode(self.pin, pigpio.INPUT)
        self.pi.set_pull_up_down(self.pin, pigpio.PUD_DOWN)
        self.cb = self.pi.callback(self.pin, pigpio.FALLING_EDGE, self._cb)


        # オドメーターの値を初期化する
        self.m_per_tick = mm_per_tick / 1000.0
        self.poll_delay = poll_delay
        self.meters = 0
        self.last_time = time.time()
        self.meters_per_second = 0
        self.counter = 0
        self.on = True
        self.debug = debug
        self.top_speed = 0
        self.prev_dist = 0.

    def _cb(self, pin, level, tick):
        """割り込みコールバックでティックを増加させる。"""

        self.counter += 1

    def update(self):
        """エンコーダー値を読み取り速度と距離を計算する。"""

        while(self.on):
                
            # ティックを保存してカウンターをリセットする
            ticks = self.counter
            self.counter = 0
            
            # 最後の時間間隔を保存してタイマーをリセットする
            start_time = self.last_time
            end_time = time.time()
            self.last_time = end_time
            
            # 経過時間と移動距離を計算する
            seconds = end_time - start_time
            distance = ticks * self.m_per_tick
            velocity = distance / seconds
            
            # オドメーターの値を更新する
            self.meters += distance
            self.meters_per_second = velocity
            if(self.meters_per_second > self.top_speed):
                self.top_speed = self.meters_per_second

            # デバッグ用のコンソール出力
            if(self.debug):
                print('秒数:', seconds)
                print('距離:', distance)
                print('速度:', velocity)

                print('距離(m):', round(self.meters, 4))
                print('速度(m/s):', self.meters_per_second)

            time.sleep(self.poll_delay)

    def run_threaded(self, throttle):
        """スレッドモードで速度を返す。"""

        self.prev_dist = self.meters
        return self.meters_per_second

    def shutdown(self):
        """スレッドを停止し pigpio を終了する。"""

        self.on = False
        print('ロータリーエンコーダーを停止しています')
        print("\t走行距離: {} メートル".format(round(self.meters, 4)))
        print("\t最高速度: {} メートル/秒".format(round(self.top_speed, 4)))
        if self.cb != None:
            self.cb.cancel()
            self.cb = None
        self.pi.stop()