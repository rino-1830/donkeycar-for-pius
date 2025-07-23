"""
actuators.py
モーターやサーボを制御するクラス群。
これらのクラスはドライブループで使用される前にミクサークラスでラップされる。
"""

from abc import ABC, abstractmethod
import time
import logging
from typing import Tuple

import donkeycar as dk
from donkeycar import utils
from donkeycar.utils import clamp

logger = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO
except ImportError as e:
    logger.warn(f"RPi.GPIO をインポートできませんでした: {e}")
    globals()["GPIO"] = None

from donkeycar.parts.pins import OutputPin, PwmPin, PinState
from donkeycar.utilities.deprecated import deprecated

logger = logging.getLogger(__name__)


#
# pwm/duty-cycle/pulse
# - 標準的なRCサーボのパルス幅は1ミリ秒(フルリバース)から2ミリ秒(フルフォワード)までで、
#   1.5ミリ秒が中立(停止)となる。
# - これらのパルスは通常50Hz(20ミリ秒周期)で送信される。
# - つまり標準の50Hzでは1msのパルスが5%のデューティ比、2msが10%のデューティ比となる。
# - 重要なのはパルスの長さであり、必ず1ms〜2msの範囲に収める必要がある。
# - そのため異なる周波数を用いる場合は、1ms〜2msのパルスとなるようデューティ比を調整する。
# - 例えば60Hzを使用するなら、1msのパルスは0.05 * 60 / 50 = 0.06(6%)のデューティ比となる。
# - PCA9685のデフォルト周波数は60Hzで、設定値も60Hz・12ビット値を前提としている。
#   12ビット値を使うのはPCA9685の解像度が12ビットだからで、1msは0.06 * 4096 ≒ 246、
#   中立の0.09は0.09 * 4096 ≒ 367、全開の0.12は0.12 * 4096 ≒ 492となる。
# - これらは基礎理解のための一般論で、最終的なデューティ比やパルス長はハードウェアや
#   走行方針(速度を抑えるため低めのスロットル値を用いるなど)によって決定される。
#

def duty_cycle(pulse_ms:float, frequency_hz:float) -> float:
    """パルス長と周波数からデューティ比を計算する。

    Args:
        pulse_ms: 目的のパルス幅[ms]
        frequency_hz: PWM 周波数[Hz]

    Returns:
        0〜1のデューティ比
    """
    ms_per_cycle = 1000 / frequency_hz
    duty = pulse_ms / ms_per_cycle
    return duty


def pulse_ms(pulse_bits:int) -> float:
    """12ビット値からパルス幅をミリ秒で算出する。

    Donkeycar のスロットルおよびステアリング PWM 値は PCA9685 の
    12ビットパルス値を基準としており、0 が 0% のデューティ比、4095 が 100%
    を意味する。

    Args:
        pulse_bits: 0〜4095 の 12 ビット整数

    Returns:
        パルス幅[ms]
    """
    if pulse_bits < 0 or pulse_bits > 4095:
        raise ValueError("pulse_bits は 0〜4095 の範囲でなければなりません (12bit 整数)")
    return pulse_bits / 4095


class PulseController:
    """指定した ``PwmPin`` を利用してサーボ用 PWM パルスを出力するコントローラ。

    pins.py に各種ピンプロバイダーの実装がある。
    """

    def __init__(self, pwm_pin:PwmPin, pwm_scale:float = 1.0, pwm_inverted:bool = False) -> None:
        """コンストラクタ。

        Args:
            pwm_pin: パルスを出力する ``PwmPin``
            pwm_scale: 非標準周波数を補正するための12ビット値スケール
            pwm_inverted: True ならデューティ比を反転する
        """
        self.pwm_pin = pwm_pin
        self.scale = pwm_scale
        self.inverted = pwm_inverted
        self.started = pwm_pin.state() != PinState.NOT_STARTED

    def set_pulse(self, pulse:int) -> None:
        """12ビット整数(0〜4095)でパルス幅を設定する。

        Args:
            pulse: 0〜4095 のパルス値
        """
        if pulse < 0 or pulse > 4095:
            logging.error("pulse は 0〜4095 の範囲でなければなりません")
            pulse = clamp(pulse, 0, 4095)

        if not self.started:
            self.pwm_pin.start()
            self.started = True
        if self.inverted:
            pulse = 4095 - pulse
        self.pwm_pin.duty_cycle(int(pulse * self.scale) / 4095)

    def run(self, pulse:int) -> None:
        """12ビット整数(0〜4095)でパルス幅を設定する。``set_pulse`` のエイリアス。

        Args:
            pulse: 0〜4095 のパルス値
        """
        self.set_pulse(pulse)


@deprecated("PulseController を推奨します。将来のリリースで削除されます")
class PCA9685:
    """PCA9685 ボードを用いた PWM モーターコントローラ。

    多くのRCカーで使用される。
    """
    def __init__(self, channel, address=0x40, frequency=60, busnum=None, init_delay=0.1):

        self.default_freq = 60
        self.pwm_scale = frequency / self.default_freq

        import Adafruit_PCA9685
        # PCA9685 をデフォルトアドレス(0x40)で初期化
        if busnum is not None:
            from Adafruit_GPIO import I2C
            # get_bus 関数を自分の実装に置き換える
            def get_bus():
                return busnum
            I2C.get_default_bus = get_bus
        self.pwm = Adafruit_PCA9685.PCA9685(address=address)
        self.pwm.set_pwm_freq(frequency)
        self.channel = channel
        time.sleep(init_delay) # "Tamiya TBLE-02" は少し跳ねることがあるため待機

    def set_high(self):
        self.pwm.set_pwm(self.channel, 4096, 0)

    def set_low(self):
        self.pwm.set_pwm(self.channel, 0, 4096)

    def set_duty_cycle(self, duty_cycle):
        if duty_cycle < 0 or duty_cycle > 1:
            logging.error("duty_cycle は 0〜1 の範囲でなければなりません")
            duty_cycle = clamp(duty_cycle, 0, 1)
            
        if duty_cycle == 1:
            self.set_high()
        elif duty_cycle == 0:
            self.set_low()
        else:
            # デューティ比は 12 ビット値の割合
            pulse = int(4096 * duty_cycle)
            try:
                self.pwm.set_pwm(self.channel, 0, pulse)
            except:
                self.pwm.set_pwm(self.channel, 0, pulse)

    def set_pulse(self, pulse):
        try:
            self.pwm.set_pwm(self.channel, 0, int(pulse * self.pwm_scale))
        except:
            self.pwm.set_pwm(self.channel, 0, int(pulse * self.pwm_scale))

    def run(self, pulse):
        self.set_pulse(pulse)


class VESC:
    """pyvesc を使用した VESC モーターコントローラ。

    主に電動スケートボードで使用される。

    Args:
        serial_port: VESC と通信するポート。Linux では ``/dev/ttyACM1`` など
        has_sensor: pyvesc のデフォルト値
        start_heartbeat: ハートビートを開始するかどうか。失われた場合に速度を止める
        baudrate: VESC との通信速度
        timeout: 接続確立を諦めるまでの時間
        percent: モーターに設定するデューティ比の最大割合

    このクラスは pyvesc ライブラリを用いて VESC と通信し、サーボ角度(0〜1)と
    デューティ比(車速)をスロットル値に基づいて設定する。

    pip で ``pyvesc`` をインストールすると速度設定のみ可能なファイルが生成され、
    サーボ角を設定できない。そのため以下のコマンドでインストールすることを推奨する:
    ``pip install git+https://github.com/LiamBindle/PyVESC.git@master``
    """
    def __init__(self, serial_port, percent=.2, has_sensor=False, start_heartbeat=True, baudrate=115200, timeout=0.05, steering_scale = 1.0, steering_offset = 0.0 ):
        
        try:
            import pyvesc
        except Exception as err:
            print("\n\n\n\n", err, "\n")
            print("pyvesc をインポートしてサーボ位置も設定できるようにするには以下のコマンドを実行してください:")
            print("pip install git+https://github.com/LiamBindle/PyVESC.git@master")
            print("\n\n\n")
            time.sleep(1)
            raise
        
        assert percent <= 1 and percent >= -1,'\n\nMAX_VESC_SPEED にはパーセンテージのみ使用できます(推奨値は約0.2)。負値を指定するとモーターの回転方向が反転します'
        self.steering_scale = steering_scale
        self.steering_offset = steering_offset
        self.percent = percent
        
        try:
            self.v = pyvesc.VESC(serial_port, has_sensor, start_heartbeat, baudrate, timeout)
        except Exception as err:
            print("\n\n\n\n", err)
            print("\n\n権限エラーを解決するには次のコマンドを試してください:")
            print("sudo chmod a+rw {}".format(serial_port), "\n\n\n\n")
            time.sleep(1)
            raise
        
    def run(self, angle, throttle):
        self.v.set_servo((angle * self.steering_scale) + self.steering_offset)
        self.v.set_duty_cycle(throttle*self.percent)


@deprecated("PulseController を推奨します。将来のリリースで削除されます")
class PiGPIO_PWM():
    """pigpio モジュールとデーモンを使用して Raspberry Pi の GPIO ピンからハードウェア PWM を発生させる。

    追加ハードウェアなしで PCA9685 の代替として利用できる。

    インストールと設定:
        sudo apt update && sudo apt install pigpio python3-pigpio
        sudo systemctl start pigpiod

    パルス幅の範囲は PCA9685 とは異なり、12K〜170K まで変化する。
    ステアリング信号を反転させる回路を使う場合は ``inverted`` を True に設定する。
    設定ファイル等から受け取るパルス値のデフォルト倍率は 100。
    """

    def __init__(self, pin, pgio=None, freq=75, inverted=False):
        import pigpio
        self.pin = pin
        self.pgio = pgio or pigpio.pi()
        self.freq = freq
        self.inverted = inverted
        self.pgio.set_mode(self.pin, pigpio.OUTPUT)
        self.dead_zone = 37000

    def __del__(self):
        self.pgio.stop()

    def set_pulse(self, pulse):
        # 出力値を計算
        self.output = pulse * 200
        if self.output > 0:
            self.pgio.hardware_PWM(self.pin, self.freq, int(self.output if self.inverted == False else 1e6 - self.output))


    def run(self, pulse):
        self.set_pulse(pulse)


class PWMSteering:
    """PWM パルスコントローラのラッパーで、角度を PWM パルスに変換する。"""
    LEFT_ANGLE = -1
    RIGHT_ANGLE = 1

    def __init__(self, controller, left_pulse, right_pulse):

        if controller is None:
            raise ValueError("PWMSteering には set_pulse メソッドを持つコントローラが必要です")
        set_pulse = getattr(controller, "set_pulse", None)
        if set_pulse is None or not callable(set_pulse):
            raise ValueError("controller は set_pulse メソッドを持っていなければなりません")
        if not utils.is_number_type(left_pulse):
            raise ValueError("left_pulse は数値でなければなりません")
        if not utils.is_number_type(right_pulse):
            raise ValueError("right_pulse は数値でなければなりません")

        self.controller = controller
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse
        self.pulse = dk.utils.map_range(0, self.LEFT_ANGLE, self.RIGHT_ANGLE,
                                        self.left_pulse, self.right_pulse)
        self.running = True
        logger.info('PWM ステアリングを作成しました')

    def update(self):
        while self.running:
            self.controller.set_pulse(self.pulse)

    def run_threaded(self, angle):
        # 絶対角度を車両が実際に操舵できる角度に変換する
        angle = utils.clamp(angle, self.LEFT_ANGLE, self.RIGHT_ANGLE)
        self.pulse = dk.utils.map_range(angle,
                                        self.LEFT_ANGLE, self.RIGHT_ANGLE,
                                        self.left_pulse, self.right_pulse)

    def run(self, angle):
        self.run_threaded(angle)
        self.controller.set_pulse(self.pulse)

    def shutdown(self):
        # ハンドルをまっすぐに戻す
        self.pulse = 0
        time.sleep(0.3)
        self.running = False


class PWMThrottle:
    """PWM パルスコントローラのラッパーで、-1〜1 のスロットル値を PWM パルスに変換する。"""
    MIN_THROTTLE = -1
    MAX_THROTTLE = 1

    def __init__(self, controller, max_pulse, min_pulse, zero_pulse):

        if controller is None:
            raise ValueError("PWMThrottle には set_pulse メソッドを持つコントローラが必要です")
        set_pulse = getattr(controller, "set_pulse", None)
        if set_pulse is None or not callable(set_pulse):
            raise ValueError("controller は set_pulse メソッドを持っていなければなりません")

        self.controller = controller
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse
        self.pulse = zero_pulse

        # ゼロパルスを送信して ESC をキャリブレーション
        logger.info("ESC を初期化します")
        self.controller.set_pulse(self.max_pulse)
        time.sleep(0.01)
        self.controller.set_pulse(self.min_pulse)
        time.sleep(0.01)
        self.controller.set_pulse(self.zero_pulse)
        time.sleep(1)
        self.running = True
        logger.info('PWM スロットルを作成しました')

    def update(self):
        while self.running:
            self.controller.set_pulse(self.pulse)

    def run_threaded(self, throttle):
        throttle = utils.clamp(throttle, self.MIN_THROTTLE, self.MAX_THROTTLE)
        if throttle > 0:
            self.pulse = dk.utils.map_range(throttle, 0, self.MAX_THROTTLE,
                                            self.zero_pulse, self.max_pulse)
        else:
            self.pulse = dk.utils.map_range(throttle, self.MIN_THROTTLE, 0,
                                            self.min_pulse, self.zero_pulse)

    def run(self, throttle):
        self.run_threaded(throttle)
        self.controller.set_pulse(self.pulse)

    def shutdown(self):
        # 車両を停止する
        self.run(0)
        self.running = False


#
# これは冗長と思われる。もし本当に PCA9685 をエミュレートするなら既存のコードを使えば良いはず。
# - どのテンプレートでも使用されていない
# - ドキュメント等でも説明されていない
# - PCA9685 をエミュレートしているなら既存コードが利用できるはず
# - pins.py に Firmata ドライバを実装する予定があり、Teensy でも Firmata が動作するためサポート可能
#   参考: https://www.pjrc.com/teensy/td_libs_Firmata.html
#
@deprecated("JHat はフレームワークで未サポート・未ドキュメントです。将来のリリースで削除されます")
class JHat:
    '''
    Teensy を用いて PCA9685 をエミュレートする PWM モータコントローラ。
    '''
    def __init__(self, channel, address=0x40, frequency=60, busnum=None):
        logger.info("Hat を起動します")
        import Adafruit_PCA9685
        LED0_OFF_L = 0x08
        # PCA9685 をデフォルトアドレス (0x40) で初期化
        if busnum is not None:
            from Adafruit_GPIO import I2C
            # get_bus 関数を独自実装に差し替える
            def get_bus():
                return busnum
            I2C.get_default_bus = get_bus
        self.pwm = Adafruit_PCA9685.PCA9685(address=address)
        self.pwm.set_pwm_freq(frequency)
        self.channel = channel
        self.register = LED0_OFF_L+4*channel

        # 割り込みをより効率的に使うため独自の write を登録
        self.pwm.set_pwm = self.set_pwm
        
    def set_pulse(self, pulse):
        self.set_pwm(self.channel, 0, pulse) 

    def set_pwm(self, channel, on, off):
        # 単一の PWM チャンネルを設定
        self.pwm._device.writeList(self.register, [off & 0xFF, off >> 8])
        
    def run(self, pulse):
        self.set_pulse(pulse)


#
# JHatReader は削除予定です
# - どのテンプレートにも組み込まれていません
# - ドキュメントなどでも説明されていません
# RC 受信入力を読み取る方法のようなので、本来は controllers.py に統合すべきです。
# その対応が取れれば残しますが、そうでなければ削除する予定です。
#
@deprecated("JHatReader はフレームワークで未サポート・未ドキュメントです。将来のリリースで削除される可能性があります")
class JHatReader:
    """Teensy から RC コントロール値を読み取る。"""
    def __init__(self, channel, address=0x40, frequency=60, busnum=None):
        import Adafruit_PCA9685
        self.pwm = Adafruit_PCA9685.PCA9685(address=address)
        self.pwm.set_pwm_freq(frequency)
        self.register = 0 # I2C 読み出しではアドレスを指定しない
        self.steering = 0
        self.throttle = 0
        self.running = True
        # リセット信号を送る
        self.pwm._device.writeRaw8(0x06)

    def read_pwm(self):
        """I2C バス経由で Teensy に読み出し要求を送り、直近の RC 入力による PWM 値を取得する。"""
        h1 = self.pwm._device.readU8(self.register)
        # ヘッダの最初のバイトは 100 でなければならず、そうでない場合は読み取り位置がずれている
        while h1 != 100:
            logger.debug("ヘッダ開始まで読み飛ばします")
            h1 = self.pwm._device.readU8(self.register)
        
        h2 = self.pwm._device.readU8(self.register)
        # h2 は現在無視している

        val_a = self.pwm._device.readU8(self.register)
        val_b = self.pwm._device.readU8(self.register)
        self.steering = (val_b << 8) + val_a
        
        val_c = self.pwm._device.readU8(self.register)
        val_d = self.pwm._device.readU8(self.register)
        self.throttle = (val_d << 8) + val_c

        # 値を -1 から 1 の範囲へスケーリングする
        self.steering = (self.steering - 1500.0) / 500.0  + 0.158
        self.throttle = (self.throttle - 1500.0) / 500.0  + 0.136

    def update(self):
        while(self.running):
            self.read_pwm()
        
    def run_threaded(self):
        return self.steering, self.throttle

    def shutdown(self):
        self.running = False
        time.sleep(0.1)


#
# Adafruit_DCMotor_Hat のサポートは削除予定
# - どのテンプレートにも組み込まれていません
# - ドキュメントなどにも説明がありません
# 対応策としては DRIVE_TRAIN_TYPE = "DC_TWO_WHEEL_ADAFRUIT" を追加し
# complete.py へ統合する方法が考えられます
#
@deprecated("この機能はフレームワークで未サポート・未ドキュメントです。将来のリリースで削除される可能性があります")
class Adafruit_DCMotor_Hat:
    """Adafruit 製 DC モータコントローラ。

    差動駆動車の各モータに使用する。
    """
    def __init__(self, motor_num):
        from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
        import atexit
        
        self.FORWARD = Adafruit_MotorHAT.FORWARD
        self.BACKWARD = Adafruit_MotorHAT.BACKWARD
        self.mh = Adafruit_MotorHAT(addr=0x60) 
        
        self.motor = self.mh.getMotor(motor_num)
        self.motor_num = motor_num
        
        atexit.register(self.turn_off_motors)
        self.speed = 0
        self.throttle = 0
    
        
    def run(self, speed):
        """速度を更新する。1 が前進全開、-1 が全後退。"""
        if speed > 1 or speed < -1:
            raise ValueError("speed は 1(前進)から -1(後退) の範囲でなければなりません")
        
        self.speed = speed
        self.throttle = int(dk.utils.map_range(abs(speed), -1, 1, -255, 255))
        
        if speed > 0:            
            self.motor.run(self.FORWARD)
        else:
            self.motor.run(self.BACKWARD)
            
        self.motor.setSpeed(self.throttle)
        

    def shutdown(self):
        self.mh.getMotor(self.motor_num).run(Adafruit_MotorHAT.RELEASE)


#
# Maestro Servo Controller のサポートは削除予定
# - どのテンプレートにも統合されていない
# - ドキュメントでも説明されていない
# - 追加の AStar マイコンが必要とされるようだが対応ファームウェアが無い
# これらが解決できれば pins.py のピン提供 API に Maestro 対応を追加し、
# 汎用モータドライバの TTL/PWM ソースとして利用できる
# AStar コントローラが必須でないなら pins.py へ統合してこのクラスを削除する方が簡単
#
@deprecated("この機能はフレームワークで未サポート・未ドキュメントです。将来のリリースで削除される可能性があります")
class Maestro:
    '''
    Pololu 製 Maestro サーボコントローラ。

    MaestroControlCenter を使用して speed と acceleration を 0 に設定する。
    詳細: https://www.pololu.com/docs/0J40/all
    '''
    import threading

    maestro_device = None
    astar_device = None
    maestro_lock = threading.Lock()
    astar_lock = threading.Lock()

    def __init__(self, channel, frequency = 60):
        import serial

        if Maestro.maestro_device == None:
            Maestro.maestro_device = serial.Serial('/dev/ttyACM0', 115200)

        self.channel = channel
        self.frequency = frequency
        self.lturn = False
        self.rturn = False
        self.headlights = False
        self.brakelights = False

        if Maestro.astar_device == None:
            Maestro.astar_device = serial.Serial('/dev/ttyACM2', 115200, timeout= 0.01)

    def set_pulse(self, pulse):
        # Adafruit の値からパルス幅を再計算
        w = pulse * (1 / (self.frequency * 4096)) # 秒単位
        w *= 1000 * 1000  # マイクロ秒単位
        w *= 4  # Maestro が要求する 1/4 マイクロ秒単位
        w = int(w)

        with Maestro.maestro_lock:
            Maestro.maestro_device.write(bytearray([ 0x84,
                                                     self.channel,
                                                     (w & 0x7F),
                                                     ((w >> 7) & 0x7F)]))

    def set_turn_left(self, v):
        if self.lturn != v:
            self.lturn = v
            b = bytearray('L' if v else 'l', 'ascii')
            with Maestro.astar_lock:
                Maestro.astar_device.write(b)

    def set_turn_right(self, v):
        if self.rturn != v:
            self.rturn = v
            b = bytearray('R' if v else 'r', 'ascii')
            with Maestro.astar_lock:
                Maestro.astar_device.write(b)

    def set_headlight(self, v):
        if self.headlights != v:
            self.headlights = v
            b = bytearray('H' if v else 'h', 'ascii')
            with Maestro.astar_lock:
                Maestro.astar_device.write(b)

    def set_brake(self, v):
        if self.brakelights != v:
            self.brakelights = v
            b = bytearray('B' if v else 'b', 'ascii')
            with Maestro.astar_lock:
                Maestro.astar_device.write(b)

    def readline(self):
        ret = None
        with Maestro.astar_lock:
            # 次のような形式の行を想定している
            # E n nnn n
            if Maestro.astar_device.inWaiting() > 8:
                ret = Maestro.astar_device.readline()

        if ret is not None:
            ret = ret.rstrip()

        return ret


#
# Teensy サポートは削除検討中
# - どのテンプレートにも組み込まれていない
# - ドキュメントも存在しない
# - Teensy にファームウェアを書き込む必要があると思われるが参照がない
# これらが解決すれば pins.py のピン提供 API に統合して TTL/PWM ソースとして利用可能になる
# もう一つの方法として Firmata プロトコル (Arduino スケッチの一種) を実装する案がある
# 詳細は ArduinoFirmata を参照
#
@deprecated("この機能はフレームワークで未サポート・未ドキュメントです。将来のリリースで削除される可能性があります")
class Teensy:
    '''
    Teensy を用いたサーボコントローラ。
    '''
    import threading

    teensy_device = None
    astar_device = None
    teensy_lock = threading.Lock()
    astar_lock = threading.Lock()

    def __init__(self, channel, frequency = 60):
        import serial

        if Teensy.teensy_device == None:
            Teensy.teensy_device = serial.Serial('/dev/teensy', 115200, timeout = 0.01)

        self.channel = channel
        self.frequency = frequency
        self.lturn = False
        self.rturn = False
        self.headlights = False
        self.brakelights = False

        if Teensy.astar_device == None:
            Teensy.astar_device = serial.Serial('/dev/astar', 115200, timeout = 0.01)

    def set_pulse(self, pulse):
        # Adafruit の値からパルス幅を再計算
        w = pulse * (1 / (self.frequency * 4096)) # 秒単位
        w *= 1000 * 1000  # マイクロ秒単位

        with Teensy.teensy_lock:
            Teensy.teensy_device.write(("%c %.1f\n" % (self.channel, w)).encode('ascii'))

    def set_turn_left(self, v):
        if self.lturn != v:
            self.lturn = v
            b = bytearray('L' if v else 'l', 'ascii')
            with Teensy.astar_lock:
                Teensy.astar_device.write(b)

    def set_turn_right(self, v):
        if self.rturn != v:
            self.rturn = v
            b = bytearray('R' if v else 'r', 'ascii')
            with Teensy.astar_lock:
                Teensy.astar_device.write(b)

    def set_headlight(self, v):
        if self.headlights != v:
            self.headlights = v
            b = bytearray('H' if v else 'h', 'ascii')
            with Teensy.astar_lock:
                Teensy.astar_device.write(b)

    def set_brake(self, v):
        if self.brakelights != v:
            self.brakelights = v
            b = bytearray('B' if v else 'b', 'ascii')
            with Teensy.astar_lock:
                Teensy.astar_device.write(b)

    def teensy_readline(self):
        ret = None
        with Teensy.teensy_lock:
            # 次のような形式の行を想定している
            # E n nnn n
            if Teensy.teensy_device.inWaiting() > 8:
                ret = Teensy.teensy_device.readline()

        if ret != None:
            ret = ret.rstrip()

        return ret

    def astar_readline(self):
        ret = None
        with Teensy.astar_lock:
            # 次のような形式の行を想定している
            # E n nnn n
            if Teensy.astar_device.inWaiting() > 8:
                ret = Teensy.astar_device.readline()

        if ret != None:
            ret = ret.rstrip()

        return ret


class MockController(object):
    def __init__(self):
        pass

    def run(self, pulse):
        pass

    def shutdown(self):
        pass


class L298N_HBridge_3pin(object):
    """L298N H ブリッジを使用したモータ制御(3ピン)。

    DRIVETRAIN_TYPE= ``DC_TWO_WHEEL_L298N`` を選択すると利用される。
    2 本の ``OutputPin`` で方向を選択し、``PwmPin`` で出力を制御する。
    ピンプロバイダーの実装は pins.py を参照。

    3 ピンモードでの L298N 配線例は
    https://www.electronicshub.org/raspberry-pi-l298n-interface-tutorial-control-dc-motor-l298n-raspberry-pi/
    を参照。TB6612FNG など L298N を模倣したドライバにも当てはまる。
    """

    def __init__(self, pin_forward:OutputPin, pin_backward:OutputPin, pwm_pin:PwmPin, zero_throttle:float=0, max_duty=0.9):
        """コンストラクタ。

        Args:
            pin_forward: HIGH 時にモータが時計回りに回転し、``pwm_pin`` のデューティ比が適用される ``OutputPin``
            pin_backward: HIGH 時にモータが反時計回りに回転し、``pwm_pin`` のデューティ比が適用される ``OutputPin``
            pwm_pin: 0〜1 のデューティ比を取る ``PwmPin``
            zero_throttle: この値以下は 0 とみなす
            max_duty: モータに送信する最大デューティ比

        Note:
            ``pin_forward`` と ``pin_backward`` が両方 LOW の場合、モータは接続解除状態となり慣性で停止する。
            両方 HIGH の場合は強制停止となり、ブレーキとして利用できる。
        """
        self.pin_forward = pin_forward
        self.pin_backward = pin_backward
        self.pwm_pin = pwm_pin
        self.zero_throttle = zero_throttle
        self.throttle = 0
        self.max_duty = max_duty
        self.pin_forward.start(PinState.LOW)
        self.pin_backward.start(PinState.LOW)
        self.pwm_pin.start(0)

    def run(self, throttle:float) -> None:
        """モータ速度を更新する。

        Args:
            throttle: -1〜1 のスロットル値。1 が前進全開、-1 が全後退。
        """
        if throttle is None:
            logger.warn("TwoWheelSteeringThrottle の throttle が None です")
            return
        if throttle > 1 or throttle < -1:
            logger.warn( f"TwoWheelSteeringThrottle の throttle が {throttle} ですが、1(前進)から -1(後退) の範囲でなければなりません")
            throttle = clamp(throttle, -1, 1)
        
        self.speed = throttle
        self.throttle = dk.utils.map_range_float(throttle, -1, 1, -self.max_duty, self.max_duty)
        if self.throttle > self.zero_throttle:
            self.pwm_pin.duty_cycle(self.throttle)
            self.pin_backward.output(PinState.LOW)
            self.pin_forward.output(PinState.HIGH)
        elif self.throttle < -self.zero_throttle:
            self.pwm_pin.duty_cycle(-self.throttle)
            self.pin_forward.output(PinState.LOW)
            self.pin_backward.output(PinState.HIGH)
        else:
            self.pwm_pin.duty_cycle(0)
            self.pin_forward.output(PinState.LOW)
            self.pin_backward.output(PinState.LOW)

    def shutdown(self):
        self.pwm_pin.stop()
        self.pin_forward.stop()
        self.pin_backward.stop()


class TwoWheelSteeringThrottle(object):
    """
    Modify individual differential drive wheel throttles
    in order to implemeht steering.
    """

    def run(self, throttle:float, steering:float) -> Tuple[float, float]:
        """
        :param throttle:float throttle value in range -1 to 1,
                        where 1 is full forward and -1 is full backwards.
        :param steering:float steering value in range -1 to 1,
                        where -1 is full left and 1 is full right.
        :return: tuple of left motor and right motor throttle values in range -1 to 1
                 where 1 is full forward and -1 is full backwards.
        """
        if throttle is None:
            logger.warn("TwoWheelSteeringThrottle の throttle が None です")
            return
        if steering is None:
            logger.warn("TwoWheelSteeringThrottle の steering が None です")
            return
        if throttle > 1 or throttle < -1:
            logger.warn( f"TwoWheelSteeringThrottle の throttle が {throttle} ですが、1(前進)から -1(後退) の範囲でなければなりません")
            throttle = clamp(throttle, -1, 1)
        if steering > 1 or steering < -1:
            logger.warn( f"TwoWheelSteeringThrottle の steering が {steering} ですが、1(右)から -1(左) の範囲でなければなりません")
            steering = clamp(steering, -1, 1)

        left_motor_speed = throttle
        right_motor_speed = throttle

        if steering < 0:
            left_motor_speed *= (1.0 - (-steering))
        elif steering > 0:
            right_motor_speed *= (1.0 - steering)

        return left_motor_speed, right_motor_speed

    def shutdown(self) -> None:
        pass


class L298N_HBridge_2pin(object):
    """
    """ミニ L298N H ブリッジを 2 本の ``PwmPin`` で制御するモータドライバ。

    DRIVETRAIN_TYPE= ``DC_TWO_WHEEL`` を選択すると利用される。
    ピンプロバイダーの実装は pins.py を参照。

    配線例は
    https://www.instructables.com/Tutorial-for-Dual-Channel-DC-Motor-Driver-Board-PW/
    を参照。L9110S/HG7881 などのモータドライバでも使用でき、
    その配線例は
    https://electropeak.com/learn/interfacing-l9110s-dual-channel-h-bridge-motor-driver-module-with-arduino/
    を参照。
    """
    """

    def __init__(self, pin_forward:PwmPin, pin_backward:PwmPin, zero_throttle:float=0, max_duty = 0.9):
        """コンストラクタ。

        Args:
            pin_forward: デューティ比 0〜1 を受け取り、0 で停止、1 で全開となる ``PwmPin``。正転用
            pin_backward: デューティ比 0〜1 を受け取り、逆転用
            zero_throttle: この値以下は 0 とみなす
            max_duty: モータへ送信する最大デューティ比

        Note:
            ``pin_forward`` と ``pin_backward`` がともに 0 の場合、モータは接続解除状態となる。
            両方 1 の場合は強制停止し、ブレーキとして利用できる。
            ``max_duty`` は 0〜1 で、多くの場合 0.9 が良いとされる。
        """
        self.pin_forward = pin_forward
        self.pin_backward = pin_backward
        self.zero_throttle = zero_throttle
        self.max_duty = max_duty

        self.throttle=0
        self.speed=0
        
        self.pin_forward.start(0)
        self.pin_backward.start(0)

    def run(self, throttle:float) -> None:
        """モータ速度を更新する。

        Args:
            throttle: -1〜1 のスロットル値。1 が前進全開、-1 が全後退。
        """
        if throttle is None:
            logger.warn("TwoWheelSteeringThrottle の throttle が None です")
            return
        if throttle > 1 or throttle < -1:
            logger.warn( f"TwoWheelSteeringThrottle の throttle が {throttle} ですが、1(前進)から -1(後退) の範囲でなければなりません")
            throttle = clamp(throttle, -1, 1)

        self.speed = throttle
        self.throttle = dk.utils.map_range_float(throttle, -1, 1, -self.max_duty, self.max_duty)
        
        if self.throttle > self.zero_throttle:
            self.pin_backward.duty_cycle(0)
            self.pin_forward.duty_cycle(self.throttle)
        elif self.throttle < -self.zero_throttle:
            self.pin_forward.duty_cycle(0)
            self.pin_backward.duty_cycle(-self.throttle)
        else:
            self.pin_forward.duty_cycle(0)
            self.pin_backward.duty_cycle(0)

    def shutdown(self):
        self.pin_forward.stop()
        self.pin_backward.stop()

    
#
# これは pins.py と PulseController に置き換えられる予定。
# GPIO ピンは RPi.GPIO または PIGPIO で設定できるため冗長である。
#
@deprecated("将来のリリースでは PulseController が推奨され、本クラスは削除されます")
class RPi_GPIO_Servo(object):
    """Raspberry Pi の GPIO ピンからサーボを制御する。"""
    def __init__(self, pin, pin_scheme=None, freq=50, min=5.0, max=7.8):
        self.pin = pin
        if pin_scheme is None:
            pin_scheme = GPIO.BCM
        GPIO.setmode(pin_scheme)
        GPIO.setup(self.pin, GPIO.OUT)

        self.throttle = 0
        self.pwm = GPIO.PWM(self.pin, freq)
        self.pwm.start(0)
        self.min = min
        self.max = max

    def run(self, pulse):
        """1 で全開、-1 で全後退となるようモータ速度を更新する。"""
        # 90 程度が良い最大値とされている
        self.throttle = dk.map_frange(pulse, -1.0, 1.0, self.min, self.max)
        # logger.debug(pulse, self.throttle)
        self.pwm.ChangeDutyCycle(self.throttle)

    def shutdown(self):
        self.pwm.stop()
        GPIO.cleanup()


#
# これは pins.py へ置き換えられる予定。GPIO ピンは RPi.GPIO または PIGPIO で設定可能なため ServoBlaster は冗長
#
@deprecated("将来のリリースでは PulseController が推奨され、本クラスは削除されます")
class ServoBlaster(object):
    """Raspberry Pi の GPIO ピンからサーボを制御するクラス。

    ユーザー空間サービスを用いて DMA 制御ブロック経由で効率的に PWM を生成する。
    インストール方法:
        https://github.com/richardghirst/PiBits/tree/master/ServoBlaster
        cd PiBits/ServoBlaster/user
        make
        sudo ./servod
    上記でデーモンが起動し ``/dev/servoblaster`` が作成される。

    コマンドラインからのテスト例:
        echo P1-16=120 > /dev/servoblaster
    これにより物理ピン16へ 1200us の PWM パルスが送られる。

    起動時に自動開始させたい場合は ``sudo make install`` を実行する。
    """
    def __init__(self, pin):
        self.pin = pin
        self.servoblaster = open('/dev/servoblaster', 'w')
        self.min = min
        self.max = max

    def set_pulse(self, pulse):
        s = 'P1-%d=%d\n' % (self.pin, pulse)
        self.servoblaster.write(s)
        self.servoblaster.flush()

    def run(self, pulse):
        self.set_pulse(pulse)

    def shutdown(self):
        self.run((self.max + self.min) / 2)
        self.servoblaster.close()

#
# TODO: ArduinoFirmata のサポートを pin プロバイダーに統合し、このコードを削除して
#       PulseController を使えるようにする
#
# Arduino/マイコンの PWM サポート。
# Firmata は汎用マイコンを遠隔設定するための仕様で、ここでは Arduino 実装を利用する。
#
# セットアップ方法は https://docs.donkeycar.com/parts/actuators/#arduino を参照
# Firmata プロトコル https://github.com/firmata/protocol
# Arduino 実装 https://github.com/firmata/arduino
#
# NOTE: 1ms〜2ms のサーボパルスを高解像度で扱える汎用 InputPin/OutputPin/PwmPin を
#       作成するには、Arduino Firmata のサンプルを基に独自のスケッチを作成する
#       必要があるかもしれない。デフォルトの analogWrite は解像度が低く周波数制御も
#       不十分でサーボには向かない。そのため Servo.h ライブラリを用い、ピンをアナ
#       ログ出力設定した際に Servo インスタンスを動的に追加して writeMicroseconds
#       で出力し、値をパルスのオン時間[µs]として解釈する。詳細は上記 Arduino Firmata
#       リポジトリの各種サンプルを参照。
#
@deprecated("将来のリリースでは Arduino サポートが pins.py に追加され、本クラスは削除されます")
class ArduinoFirmata:
    """Arduino ボードを用いた PWM コントローラ。

    Latte Panda のように Arduino を内蔵したボードで特に有用。
    Arduino 側には Standard Firmata スケッチを読み込む必要がある。
    詳細は ``docs/parts/actuators.md`` を参照。
    """

    def __init__(self, servo_pin = 6, esc_pin = 5):
        from pymata_aio.pymata3 import PyMata3
        self.board = PyMata3()
        self.board.sleep(0.015)
        self.servo_pin = servo_pin
        self.esc_pin = esc_pin
        self.board.servo_config(servo_pin)
        self.board.servo_config(esc_pin)

    def set_pulse(self, pin, angle):
        try:
            self.board.analog_write(pin, int(angle))
        except:
            self.board.analog_write(pin, int(angle))

    def set_servo_pulse(self, angle):
        self.set_pulse(self.servo_pin, int(angle))

    def set_esc_pulse(self, angle):
        self.set_pulse(self.esc_pin, int(angle))


@deprecated("将来のリリースでは Arduino の PWM サポートが pins.py に追加され、本クラスは削除されます")
class ArdPWMSteering:
    """Arduino Firmata コントローラのラッパーで、角度を PWM パルスへ変換する。"""
    LEFT_ANGLE = -1
    RIGHT_ANGLE = 1

    def __init__(self,
                 controller=None,
                 left_pulse=60,
                 right_pulse=120):

        self.controller = controller
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse
        self.pulse = dk.utils.map_range(0, self.LEFT_ANGLE, self.RIGHT_ANGLE,
                                        self.left_pulse, self.right_pulse)
        self.running = True
        logger.info('Arduino PWM ステアリングを作成しました')

    def run(self, angle):
        # 絶対角度を車両が実現できる角度に変換
        self.pulse = dk.utils.map_range(angle,
                                        self.LEFT_ANGLE, self.RIGHT_ANGLE,
                                        self.left_pulse, self.right_pulse)
        self.controller.set_servo_pulse(self.pulse)

    def shutdown(self):
        # ハンドルをまっすぐに戻す
        self.pulse = dk.utils.map_range(0, self.LEFT_ANGLE, self.RIGHT_ANGLE,
                                        self.left_pulse, self.right_pulse)
        time.sleep(0.3)
        self.running = False


@deprecated("将来のリリースでは Arduino の PWM サポートが pins.py に追加され、本クラスは削除されます")
class ArdPWMThrottle:

    """Arduino Firmata コントローラのラッパーで、-1〜1 のスロットル値を PWM パルスに変換する。"""
    MIN_THROTTLE = -1
    MAX_THROTTLE = 1

    def __init__(self,
                 controller=None,
                 max_pulse=105,
                 min_pulse=75,
                 zero_pulse=90):

        self.controller = controller
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse
        self.pulse = zero_pulse

        # ゼロパルスを送信して ESC をキャリブレーション
        logger.info("ESC を初期化します")
        self.controller.set_esc_pulse(self.max_pulse)
        time.sleep(0.01)
        self.controller.set_esc_pulse(self.min_pulse)
        time.sleep(0.01)
        self.controller.set_esc_pulse(self.zero_pulse)
        time.sleep(1)
        self.running = True
        logger.info('Arduino PWM スロットルを作成しました')

    def run(self, throttle):
        if throttle > 0:
            self.pulse = dk.utils.map_range(throttle, 0, self.MAX_THROTTLE,
                                            self.zero_pulse, self.max_pulse)
        else:
            self.pulse = dk.utils.map_range(throttle, self.MIN_THROTTLE, 0,
                                            self.min_pulse, self.zero_pulse)
        self.controller.set_esc_pulse(self.pulse)

    def shutdown(self):
        # 車両を停止する
        self.run(0)
        self.running = False
