"""pins.py - TTL と PWM ピンの高レベル抽象化。

TTL や PWM ピンを利用するドライバーを、異なるライブラリや技術でも再利用できるようにする。

抽象クラス `InputPin`、`OutputPin`、`PwmPin` は、ピンの開始・使用・後始末のインターフェースを提供する。
ファクトリー関数 `input_pin_by_id()`、`output_pin_by_id()`、`pwm_pin_by_id()` は、
ピン提供元や属性を表す文字列 ID からピンを生成する。
`Rpi.GPIO` ライブラリと `PCA9685` の実装が用意されている。

ピン ID を使うことで、異なる提供元・番号方式・設定を 1 つの文字列で指定できる。

Rpi.GPIO ライブラリの ``GPIO.BOARD`` 番号方式でピン番号 13 を使用する例::
 pin = input_pin_by_id("RPI_GPIO.BOARD.13")

Rpi.GPIO ライブラリの ``GPIO.BCM`` 方式で GPIO 番号 33 を使用する例::
 pin = output_pin_by_id("RPI_GPIO.BCM.33")

バス 0、アドレス ``0x40`` の PCA9685 でチャネル 7 を使用する例::
 pin = pwm_pin_by_id("PCA9685.0:40.7")
"""
from abc import ABC, abstractmethod
from typing import Any, Callable
import logging


logger = logging.getLogger(__name__)


class PinState:
    LOW: int = 0
    HIGH: int = 1
    NOT_STARTED: int = -1


class PinEdge:
    RISING: int = 1
    FALLING: int = 2
    BOTH: int = 3


class PinPull:
    PULL_NONE: int = 1
    PULL_UP: int = 2
    PULL_DOWN: int = 3


class PinProvider:
    RPI_GPIO = "RPI_GPIO"
    PCA9685 = "PCA9685"
    PIGPIO = "PIGPIO"


class PinScheme:
    BOARD = "BOARD"  # 基板番号方式
    BCM = "BCM"      # Broadcom GPIO 番号方式


#
# #### 入出力および PWM ピンの基本インターフェース
# #### 実装はこれらの抽象クラスを継承する
#

class InputPin(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def start(self, on_input=None, edge: int = PinEdge.RISING) -> None:
        """ピンを入力モードで開始する。

        Args:
            on_input: 立ち上がり/立ち下がり検出時に呼び出す無引数関数。無視する場合は ``None``。
            edge: ``on_input`` を呼ぶエッジ種別。既定は :class:`PinEdge.RISING`。

        Raises:
            RuntimeError: すでに開始されている場合。

        その他の状態は :func:`state` を呼び、``PinState.NOT_STARTED`` かどうかで確認できる。
        """
        pass  # サブクラスで実装すること

    @abstractmethod
    def stop(self) -> None:
        """ピンを停止し ``PinState.NOT_STARTED`` に戻す。"""
        pass  # サブクラスで実装すること

    @abstractmethod
    def state(self) -> int:
        """最後に読み取った入力状態を返す。

        ピンを再度読み取るわけではなく、:meth:`input` が返した最後の値を返す。
        ピンが開始されていない、または停止している場合は ``PinState.NOT_STARTED`` を返す。
        """
        return PinState.NOT_STARTED  # サブクラスで実装すること

    @abstractmethod
    def input(self) -> int:
        """ピンの入力状態を読み取る。

        Returns:
            int: ``PinState.LOW`` または ``PinState.HIGH``。開始していない場合は
            ``PinState.NOT_STARTED`` を返す。
        """
        return PinState.NOT_STARTED  # サブクラスで実装すること


class OutputPin(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def start(self, state: int = PinState.LOW) -> None:
        """出力モードでピンを開始し、初期状態を設定する。

        Args:
            state: 開始時の :class:`PinState`。

        Raises:
            RuntimeError: すでに開始されている場合。

        :func:`state` を呼び ``PinState.NOT_STARTED`` かどうかで開始済みか確認できる。
        """
        pass  # サブクラスで実装すること

    @abstractmethod
    def stop(self) -> None:
        """ピンを停止して ``PinState.NOT_STARTED`` に戻す。"""
        pass  # サブクラスで実装すること

    @abstractmethod
    def state(self) -> int:
        """最後に設定した出力状態を返す。

        ピンを再度読み取ることはせず、:meth:`output` で設定した最後の値を返す。
        ピンが開始されていない、または停止している場合 ``PinState.NOT_STARTED`` を返す。

        Returns:
            int: 現在の出力状態、または未開始なら ``PinState.NOT_STARTED``。
        """
        return PinState.NOT_STARTED  # サブクラスで実装すること

    @abstractmethod
    def output(self, state: int) -> None:
        """ピンの出力状態を設定する。

        Args:
            state: ``PinState.LOW`` か ``PinState.HIGH``。

        Raises:
            RuntimeError: ピンが開始されていない場合。
        """
        pass  # サブクラスで実装すること


class PwmPin(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def start(self, duty: float = 0) -> None:
        """出力モードでピンを開始し、指定したデューティ比で動作させる。

        Args:
            duty: 0〜1 の範囲で指定するデューティ比。

        Raises:
            RuntimeError: すでに開始されている場合。

        開始済みかどうかは :func:`state` を呼び ``PinState.NOT_STARTED`` か確認する。
        """
        pass  # サブクラスで実装すること

    @abstractmethod
    def stop(self) -> None:
        """ピンを停止し ``PinState.NOT_STARTED`` に戻す。"""
        pass  # サブクラスで実装すること

    @abstractmethod
    def state(self) -> float:
        """最後に設定した出力 duty を返す。

        ピンを再度読み取らず、:meth:`duty_cycle` で設定した値をそのまま返す。
        ピンが開始されていない、または停止している場合は ``PinState.NOT_STARTED`` を返す。

        Returns:
            float: 現在の duty。未開始なら ``PinState.NOT_STARTED``。
        """
        return PinState.NOT_STARTED  # サブクラスで実装すること

    @abstractmethod
    def duty_cycle(self, duty: float) -> None:
        """出力 duty を設定する。

        Args:
            duty: 0〜1 の範囲の duty (0%〜100%)。

        Raises:
            RuntimeError: ピンが開始されていない場合。
        """
        pass  # サブクラスで実装すること


#
# ####### ファクトリーメソッド
#

#
# ピン ID を使うことで、異なるピンプロバイダーや番号体系、設定を
# 1 つの文字列で選択できる。
#
# Rpi.GPIO ライブラリの GPIO.BOARD 番号方式でピン番号 13 を使う例
# "RPI_GPIO.BOARD.13"
#
# Rpi.GPIO ライブラリの GPIO.BCM 方式で GPIO 番号 33 を使う例
# "RPI_GPIO.BCM.33"
#
# バス 0、アドレス 0x40 の PCA9685 でチャネル 7 を使う例
# "PCA9685.0:40.7"
#
def output_pin_by_id(pin_id: str, frequency_hz: int = 60) -> OutputPin:
    """ピン ID から TTL 出力ピンを取得する。

    Args:
        pin_id: ピンを指定する文字列 ID。
        frequency_hz: PCA9685 用の周波数 (Hz)。

    Returns:
        OutputPin: 構築された出力ピン。
    """
    parts = pin_id.split(".")
    if parts[0] == PinProvider.PCA9685:
        pin_provider = parts[0]
        i2c_bus, i2c_address = parts[1].split(":")
        i2c_bus = int(i2c_bus)
        i2c_address = int(i2c_address, base=16)
        frequency_hz = int(frequency_hz)
        pin_number = int(parts[2])
        return output_pin(pin_provider, pin_number, i2c_bus=i2c_bus, i2c_address=i2c_address, frequency_hz=frequency_hz)

    if parts[0] == PinProvider.RPI_GPIO:
        pin_provider = parts[0]
        pin_scheme = parts[1]
        pin_number = int(parts[2])
        return output_pin(pin_provider, pin_number, pin_scheme=pin_scheme)

    if parts[0] == PinProvider.PIGPIO:
        pin_provider = parts[0]
        if PinScheme.BCM != parts[1]:
            raise ValueError("PIGPIO ではピン方式は BCM でなければなりません")
        pin_number = int(parts[2])
        return output_pin(pin_provider, pin_number, pin_scheme=PinScheme.BCM)

    raise ValueError(f"不明なピンプロバイダー {parts[0]}")


def pwm_pin_by_id(pin_id: str, frequency_hz: int = 60) -> PwmPin:
    """ピン ID から PWM 出力ピンを取得する。

    Args:
        pin_id: ピンを指定する文字列 ID。
        frequency_hz: デューティ周波数 (Hz)。

    Returns:
        PwmPin: 構築された PWM ピン。
    """
    parts = pin_id.split(".")
    if parts[0] == PinProvider.PCA9685:
        pin_provider = parts[0]
        i2c_bus, i2c_address = parts[1].split(":")
        i2c_bus = int(i2c_bus)
        i2c_address = int(i2c_address, base=16)
        pin_number = int(parts[2])
        return pwm_pin(pin_provider, pin_number, i2c_bus=i2c_bus, i2c_address=i2c_address, frequency_hz=frequency_hz)

    if parts[0] == PinProvider.RPI_GPIO:
        pin_provider = parts[0]
        pin_scheme = parts[1]
        pin_number = int(parts[2])
        return pwm_pin(pin_provider, pin_number, pin_scheme=pin_scheme, frequency_hz=frequency_hz)

    if parts[0] == PinProvider.PIGPIO:
        pin_provider = parts[0]
        if PinScheme.BCM != parts[1]:
            raise ValueError("PIGPIO ではピン方式は BCM でなければなりません")
        pin_number = int(parts[2])
        return pwm_pin(pin_provider, pin_number, pin_scheme=PinScheme.BCM, frequency_hz=frequency_hz)

    raise ValueError(f"不明なピンプロバイダー {parts[0]}")


def input_pin_by_id(pin_id: str, pull: int = PinPull.PULL_NONE) -> InputPin:
    """ピン ID から TTL 入力ピンを取得する。"""
    parts = pin_id.split(".")
    if parts[0] == PinProvider.PCA9685:
        raise RuntimeError("PinProvider.PCA9685 は InputPin を実装していません")

    if parts[0] == PinProvider.RPI_GPIO:
        pin_provider = parts[0]
        pin_scheme = parts[1]
        pin_number = int(parts[2])
        return input_pin(pin_provider, pin_number, pin_scheme=pin_scheme, pull=pull)

    if parts[0] == PinProvider.PIGPIO:
        pin_provider = parts[0]
        if PinScheme.BCM != parts[1]:
            raise ValueError("Pin scheme must be BCM for PIGPIO")
        pin_number = int(parts[2])
        return input_pin(pin_provider, pin_number, pin_scheme=PinScheme.BCM, pull=pull)

    raise ValueError(f"不明なピンプロバイダー {parts[0]}")


def input_pin(
        pin_provider: str,
        pin_number: int,
        pin_scheme: str = PinScheme.BOARD,
        pull: int = PinPull.PULL_NONE) -> InputPin:
    """指定されたピンプロバイダーから :class:`InputPin` を構築する。

    PCA9685 は入力ピンを提供できない点に注意。

    Args:
        pin_provider: :class:`PinProvider` の文字列。
        pin_number: 0 から始まるピン番号。
        pin_scheme: :class:`PinScheme` の文字列。
        pull: :class:`PinPull` の値。

    Returns:
        InputPin: 構築された入力ピン。

    Raises:
        RuntimeError: 無効な ``pin_provider`` の場合。
    """
    if pin_provider == PinProvider.RPI_GPIO:
        return InputPinGpio(pin_number, pin_scheme, pull)
    if pin_provider == PinProvider.PCA9685:
        raise RuntimeError("PinProvider.PCA9685 は InputPin を実装していません")
    if pin_provider == PinProvider.PIGPIO:
        if pin_scheme != PinScheme.BCM:
            raise ValueError("PIGPIO では PinScheme.BCM を使用する必要があります")
        return InputPinPigpio(pin_number, pull)
    raise RuntimeError(f"未知の PinProvider ({pin_provider})")


def output_pin(
        pin_provider: str,
        pin_number: int,
        pin_scheme: str = PinScheme.BOARD,
        i2c_bus: int = 0,
        i2c_address: int = 40,
        frequency_hz: int = 60) -> OutputPin:
    """指定されたプロバイダーから :class:`OutputPin` を構築する。

    PCA9685 は入力ピンを提供できない点に注意。

    Args:
        pin_provider: :class:`PinProvider` の文字列。
        pin_number: 0 から始まるピン番号。
        pin_scheme: :class:`PinScheme` の文字列。
        i2c_bus: I2C デバイスのバス番号。
        i2c_address: I2C デバイスのアドレス。
        frequency_hz: PCA9685 用の周波数 (Hz)。

    Returns:
        InputPin: 構築された出力ピン。

    Raises:
        RuntimeError: ``pin_provider`` が無効な場合。
    """
    if pin_provider == PinProvider.RPI_GPIO:
        return OutputPinGpio(pin_number, pin_scheme)
    if pin_provider == PinProvider.PCA9685:
        return OutputPinPCA9685(pin_number, pca9685(i2c_bus, i2c_address, frequency_hz))
    if pin_provider == PinProvider.PIGPIO:
        if pin_scheme != PinScheme.BCM:
            raise ValueError("PIGPIO では PinScheme.BCM を使用する必要があります")
        return OutputPinPigpio(pin_number)
    raise RuntimeError(f"未知の PinProvider ({pin_provider})")


def pwm_pin(
        pin_provider: str,
        pin_number: int,
        pin_scheme: str = PinScheme.BOARD,
        frequency_hz: int = 60,
        i2c_bus: int = 0,
        i2c_address: int = 40) -> PwmPin:
    """指定されたプロバイダーから :class:`PwmPin` を構築する。

    Args:
        pin_provider: :class:`PinProvider` の文字列。
        pin_number: 0 から始まるピン番号。
        pin_scheme: :class:`PinScheme` の文字列。
        frequency_hz: デューティ周波数 (Hz)。
        i2c_bus: I2C デバイスのバス番号。
        i2c_address: I2C デバイスのアドレス。

    Returns:
        PwmPin: 構築された PWM ピン。

    Raises:
        RuntimeError: ``pin_provider`` が無効な場合。
    """
    if pin_provider == PinProvider.RPI_GPIO:
        return PwmPinGpio(pin_number, pin_scheme, frequency_hz)
    if pin_provider == PinProvider.PCA9685:
        return PwmPinPCA9685(pin_number, pca9685(i2c_bus, i2c_address, frequency_hz))
    if pin_provider == PinProvider.PIGPIO:
        if pin_scheme != PinScheme.BCM:
            raise ValueError("PIGPIO では PinScheme.BCM を使用する必要があります")
        return PwmPinPigpio(pin_number, frequency_hz)
    raise RuntimeError(f"未知の PinProvider ({pin_provider})")


#
# ----- RPi.GPIO/Jetson.GPIO 実装 -----
#
try:
    import RPi.GPIO as GPIO
    # 抽象 API を GPIO 定数に変換するためのテーブル
    gpio_pin_edge = [None, GPIO.RISING, GPIO.FALLING, GPIO.BOTH]
    gpio_pin_pull = [None, GPIO.PUD_OFF, GPIO.PUD_DOWN, GPIO.PUD_UP]
    gpio_pin_scheme = {PinScheme.BOARD: GPIO.BOARD, PinScheme.BCM: GPIO.BCM}
except ImportError:
    logger.warn("RPi.GPIO をインポートできませんでした。")
    globals()["GPIO"] = None


def gpio_fn(pin_scheme:int, fn:Callable[[], Any]):
    """GPIO 関数呼び出し前にピン方式を設定するための便利関数。

    RPi.GPIO は実行中に 1 つの方式しか設定できない。既に異なる方式が設定され
    ている場合は ``RuntimeError`` を送出して誤った出力を防止する。

    Args:
        pin_scheme: ``GPIO.BOARD`` か ``GPIO.BCM``。
        fn: ピン方式設定後に呼び出す無引数関数。

    Returns:
        Any: ``fn`` の戻り値。

    Raises:
        RuntimeError: 既に別の方式が設定されている場合。
    """
    prev_scheme = GPIO.getmode()
    if prev_scheme is None:
        GPIO.setmode(pin_scheme)
    elif prev_scheme != pin_scheme:
        raise RuntimeError(f"Attempt to change GPIO pin scheme from ({prev_scheme}) to ({pin_scheme})"
                           " after it has been set.  All RPi.GPIO pins must use the same pin scheme.")
    val = fn()
    return val


class InputPinGpio(InputPin):
    def __init__(self, pin_number: int, pin_scheme: str, pull: int = PinPull.PULL_NONE) -> None:
        """RPi.GPIO/Jetson.GPIO を用いた TTL 入力ピン。"""
        self.pin_number = pin_number
        self.pin_scheme = gpio_pin_scheme[pin_scheme]
        self.pin_scheme_str = pin_scheme
        self.pull = pull
        self.on_input = None
        self._state = PinState.NOT_STARTED
        super().__init__()

    def _callback(self, pin_number):
        if self.on_input is not None:
            self.on_input()

    def start(self, on_input=None, edge=PinEdge.RISING) -> None:
        """入力ピンを開始し、必要ならコールバックを登録する。

        Args:
            on_input: エッジ検出時に呼び出す無引数関数。無視する場合 ``None``。
            edge: ``on_input`` を呼び出すエッジ種別。既定は :class:`PinEdge.RISING`。

        Raises:
            RuntimeError: 既に開始されている場合。
        """
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError(f"InputPinGpio({self.pin_number}) は既に開始されています")

        # ピンが使用中でないことを確認してから開始する
        gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
        gpio_fn(self.pin_scheme, lambda: GPIO.setup(self.pin_number, GPIO.IN, pull_up_down=gpio_pin_pull[self.pull]))
        if on_input is not None:
            self.on_input = on_input
            gpio_fn(
                self.pin_scheme,
                lambda: GPIO.add_event_detect(self.pin_number, gpio_pin_edge[edge], callback=self._callback))
        self.input()  # 初回状態を取得
        logger.info(f"InputPin 'RPI_GPIO.{self.pin_scheme_str}.{self.pin_number}' を開始しました")

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            self.on_input = None
            gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
            self._state = PinState.NOT_STARTED
            logger.info(f"InputPin 'RPI_GPIO.{self.pin_scheme_str}.{self.pin_number}' を停止しました")

    def state(self) -> int:
        return self._state

    def input(self) -> int:
        self._state = gpio_fn(self.pin_scheme, lambda: GPIO.input(self.pin_number))
        return self._state


class OutputPinGpio(OutputPin):
    """Rpi.GPIO/Jetson.GPIO を用いた TTL 出力ピン。"""
    def __init__(self, pin_number: int, pin_scheme: str) -> None:
        self.pin_number = pin_number
        self.pin_scheme_str = pin_scheme
        self.pin_scheme = gpio_pin_scheme[pin_scheme]
        self._state = PinState.NOT_STARTED

    def start(self, state: int = PinState.LOW) -> None:
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError(f"OutputPinGpio({self.pin_number}) は既に開始されています")
        # ピンを一旦停止して使用中でないことを確認してから開始する
        gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
        gpio_fn(self.pin_scheme, lambda: GPIO.setup(self.pin_number, GPIO.OUT))
        self.output(state)
        logger.info(f"OutputPin 'RPI_GPIO.{self.pin_scheme_str}.{self.pin_number}' を開始しました")

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
            self._state = PinState.NOT_STARTED
            logger.info(f"OutputPin 'RPI_GPIO.{self.pin_scheme_str}.{self.pin_number}' を停止しました")

    def state(self) -> int:
        return self._state

    def output(self, state: int) -> None:
        gpio_fn(self.pin_scheme, lambda: GPIO.output(self.pin_number, state))
        self._state = state


class PwmPinGpio(PwmPin):
    """Rpi.GPIO/Jetson.GPIO を用いた PWM 出力ピン。"""
    def __init__(self, pin_number: int, pin_scheme: str, frequency_hz: float = 50) -> None:
        self.pin_number = pin_number
        self.pin_scheme_str = pin_scheme
        self.pin_scheme = gpio_pin_scheme[pin_scheme]
        self.frequency = int(frequency_hz)
        self.pwm = None
        self._state = PinState.NOT_STARTED

    def start(self, duty: float = 0) -> None:
        if self.pwm is not None:
            raise RuntimeError("PwmPinGpio は既に開始されています")
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle は 0〜1 の範囲でなければなりません")

        # ピンが使用中でないことを確認してから開始する
        gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
        gpio_fn(self.pin_scheme, lambda: GPIO.setup(self.pin_number, GPIO.OUT))
        self.pwm = gpio_fn(self.pin_scheme, lambda: GPIO.PWM(self.pin_number, self.frequency))
        self.pwm.start(duty * 100)  # duty は 0〜100 の範囲で指定
        self._state = duty
        logger.info(f"PwmPin 'RPI_GPIO.{self.pin_scheme_str}.{self.pin_number}' を開始しました")

    def stop(self) -> None:
        if self.pwm is not None:
            self.pwm.stop()
            gpio_fn(self.pin_scheme, lambda: GPIO.cleanup(self.pin_number))
            logger.info(f"PwmPin 'RPI_GPIO.{self.pin_scheme_str}.{self.pin_number}' を停止しました")
        self._state = PinState.NOT_STARTED


    def state(self) -> float:
        return self._state

    def duty_cycle(self, duty: float) -> None:
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle は 0〜1 の範囲でなければなりません")
        self.pwm.ChangeDutyCycle(duty * 100)  # duty は 0〜100 の範囲
        self._state = duty


#
# ----- PCA9685 実装 -----
#
class PCA9685:
    '''
    Pin controller using PCA9685 boards.
    This is used for most RC Cars.  This
    driver can output ttl HIGH or LOW or
    produce a duty cycle at the given frequency.
    '''
    def __init__(self, busnum: int, address: int, frequency: int):

        import Adafruit_PCA9685
        if busnum is not None:
            from Adafruit_GPIO import I2C

            # monkey-patch I2C driver to use our bus number
            def get_bus():
                return busnum

            I2C.get_default_bus = get_bus
        self.pwm = Adafruit_PCA9685.PCA9685(address=address)
        self.pwm.set_pwm_freq(frequency)
        self._frequency = frequency

    def get_frequency(self):
        return self._frequency

    def set_high(self, channel: int):
        self.pwm.set_pwm(channel, 4096, 0)

    def set_low(self, channel: int):
        self.pwm.set_pwm(channel, 0, 4096)

    def set_duty_cycle(self, channel: int, duty_cycle: float):
        if duty_cycle < 0 or duty_cycle > 1:
            raise ValueError("duty_cycle は 0〜1 の範囲でなければなりません")
        if duty_cycle == 1:
            self.set_high(channel)
        elif duty_cycle == 0:
            self.set_low(channel)
        else:
            # デューティ比を 12bit スケールに変換
            pulse = int(4096 * duty_cycle)
            try:
                self.pwm.set_pwm(channel, 0, pulse)
            except Exception as e:
                logger.error(f'PCA9685 チャンネル {channel} でエラー: {str(e)}')


#
# PCA9685 のシングルトンを保持するマップ
# キーは "busnum:address"
#
_pca9685 = {}


def pca9685(busnum: int, address: int, frequency: int = 60):
    """PCA9685 ドライバーを取得するファクトリー関数。

    同じバス番号とアドレスの組み合わせではシングルトンを再利用する。
    異なる周波数で要求された場合は ``ValueError`` を送出する。

    Args:
        busnum: PCA9685 の I2C バス番号。
        address: PCA9685 の I2C アドレス。
        frequency: デューティ周波数 (Hz)。

    Raises:
        ValueError: 既存のコントローラーと周波数が異なる場合。

    Returns:
        PCA9685: 構築または再利用されたドライバー。
    """
    key = str(busnum) + ":" + hex(address)
    pca = _pca9685.get(key)
    if pca is None:
        pca = PCA9685(busnum, address, frequency)
    if pca.get_frequency() != frequency:
        raise ValueError(
            f"Frequency {frequency} conflicts with pca9685 at {key} "
            f"with frequency {pca.pwm.get_pwm_freq()}")
    return pca


class OutputPinPCA9685(ABC):
    """PCA9685 を使用した TTL 出力ピン。"""
    def __init__(self, pin_number: int, pca9685: PCA9685) -> None:
        self.pin_number = pin_number
        self.pca9685 = pca9685
        self._state = PinState.NOT_STARTED

    def start(self, state: int = PinState.LOW) -> None:
        """出力モードでピンを開始する。

        Args:
            state: 開始時の :class:`PinState`。

        Raises:
            RuntimeError: 既に開始されている場合。
        """
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError(f"pin({self.pin_number}) は既に開始されています")
        self._state = 0  # 初回出力を可能にするための暫定値
        self.output(state)

    def stop(self) -> None:
        """
        Stop the pin and return it to PinState.NOT_STARTED
        """
        if self.state() != PinState.NOT_STARTED:
            self.output(PinState.LOW)
            self._state = PinState.NOT_STARTED

    def state(self) -> int:
        """最後に設定した出力状態を返す。

        ピンが開始されていない、または停止している場合は
        ``PinState.NOT_STARTED`` を返す。

        Returns:
            int: 現在の出力状態。
        """
        return self._state

    def output(self, state: int) -> None:
        """出力状態を書き込む。

        Args:
            state: ``PinState.LOW`` または ``PinState.HIGH``。
        """
        if self.state() == PinState.NOT_STARTED:
            raise RuntimeError(f"pin({self.pin_number}) は開始されていません")
        if state == PinState.HIGH:
            self.pca9685.set_high(self.pin_number)
        else:
            self.pca9685.set_low(self.pin_number)
        self._state = state


class PwmPinPCA9685(PwmPin):
    """PCA9685 を使用した PWM 出力ピン。"""
    def __init__(self, pin_number: int, pca9685: PCA9685) -> None:
        self.pin_number = pin_number
        self.pca9685 = pca9685
        self._state = PinState.NOT_STARTED

    def start(self, duty: float = 0) -> None:
        """指定した duty でピンを開始する。

        Args:
            duty: 0〜1 の範囲で指定する duty。

        Raises:
            RuntimeError: 既に開始されている場合。
        """
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError(f"pin({self.pin_number}) は既に開始されています")
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle は 0〜1 の範囲でなければなりません")
        self._state = 0  # 初回の duty_cycle 設定を可能にするための暫定値
        self.duty_cycle(duty)
        self._state = duty

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            self.duty_cycle(0)
            self._state = PinState.NOT_STARTED

    def state(self) -> float:
        """最後に設定した duty を返す。

        Returns:
            float: duty (0〜1) または ``PinState.NOT_STARTED``。
        """
        return self._state

    def duty_cycle(self, duty: float) -> None:
        """出力 duty を設定する。

        Args:
            duty: 0〜1 の範囲の duty。

        Raises:
            RuntimeError: 開始されていない場合。
        """
        if self.state() == PinState.NOT_STARTED:
            raise RuntimeError(f"pin({self.pin_number}) は開始されていません")
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle は 0〜1 の範囲でなければなりません")
        self.pca9685.set_duty_cycle(self.pin_number, duty)
        self._state = duty


#
# ----- PIGPIO 実装 -----
#

# pigpio は任意インストール
try:
    import pigpio
    pigpio_pin_edge = [None, pigpio.RISING_EDGE, pigpio.FALLING_EDGE, pigpio.EITHER_EDGE]
    pigpio_pin_pull = [None, pigpio.PUD_OFF, pigpio.PUD_DOWN, pigpio.PUD_UP]
except ImportError:
    logger.warn("pigpio をインポートできませんでした。")
    globals()["pigpio"] = None


class InputPinPigpio(InputPin):
    def __init__(self, pin_number: int, pull: int = PinPull.PULL_NONE, pgpio=None) -> None:
        """PiGPIO ライブラリを利用した TTL 入力ピン。"""
        self.pgpio = pgpio
        self.pin_number = pin_number
        self.pull = pigpio_pin_pull[pull]
        self.on_input = None
        self._state = PinState.NOT_STARTED

    def __del__(self):
        self.stop()

    def _callback(self, gpio, level, tock):
        if self.on_input is not None:
            self.on_input()

    def start(self, on_input=None, edge=PinEdge.RISING) -> None:
        """入力ピンを開始し、必要ならコールバックを設定する。

        Args:
            on_input: エッジ検出時に呼び出される無引数関数。無視する場合 ``None``。
            edge: ``on_input`` を起動するエッジ種類。既定は :class:`PinEdge.RISING`。

        Raises:
            RuntimeError: 既に開始されている場合。
        """
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError(f"InputPinPigpio({self.pin_number}) は既に開始されています")

        self.pgpio = self.pgpio or pigpio.pi()
        self.pgpio.set_mode(self.pin_number, pigpio.INPUT)
        self.pgpio.set_pull_up_down(self.pin_number, self.pull)

        if on_input is not None:
            self.on_input = on_input
            self.pgpio.callback(self.pin_number, pigpio_pin_edge[edge], self._callback)
        self._state = self.pgpio.read(self.pin_number)  # 初期状態を読み取る

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            self.pgpio.stop()
            self.pgpio = None
            self.on_input = None
            self._state = PinState.NOT_STARTED

    def state(self) -> int:
        """最後に ``input()`` が返した値を返す。

        入力を再度読み取らず、保存した値をそのまま返す。

        Returns:
            int: ``PinState.LOW``/``PinState.HIGH`` または ``PinState.NOT_STARTED``。
        """
        return self._state

    def input(self) -> int:
        """入力ピンの状態を読み取る。

        Returns:
            int: ``PinState.LOW``/``PinState.HIGH`` もしくは ``PinState.NOT_STARTED``。
        """
        if self.state() != PinState.NOT_STARTED:
            self._state = self.pgpio.read(self.pin_number)
        return self._state


class OutputPinPigpio(OutputPin):
    """pigpio ライブラリを利用した TTL 出力ピン。"""
    def __init__(self, pin_number: int, pgpio=None) -> None:
        self.pgpio = pgpio
        self.pin_number = pin_number
        self._state = PinState.NOT_STARTED

    def start(self, state: int = PinState.LOW) -> None:
        """出力モードでピンを開始する。

        Args:
            state: 開始時の :class:`PinState`。

        Raises:
            RuntimeError: 既に開始されている場合。
        """
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError("OutputPin は既に開始されています")

        self.pgpio = self.pgpio or pigpio.pi()
        self.pgpio.set_mode(self.pin_number, pigpio.OUTPUT)
        self.pgpio.write(self.pin_number, state)  # 初期状態を設定
        self._state = state

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            self.pgpio.write(self.pin_number, PinState.LOW)
            self.pgpio.stop()
            self.pgpio = None
            self._state = PinState.NOT_STARTED

    def state(self) -> int:
        """最後に設定した出力状態を返す。

        Returns:
            int: ``PinState.LOW``/``PinState.HIGH`` または ``PinState.NOT_STARTED``。
        """
        return self._state

    def output(self, state: int) -> None:
        """出力状態を書き込む。

        Args:
            state: ``PinState.LOW`` または ``PinState.HIGH``。
        """
        if self.state() != PinState.NOT_STARTED:
            self.pgpio.write(self.pin_number, state)
            self._state = state


class PwmPinPigpio(PwmPin):
    """pigpio ライブラリを利用した PWM 出力ピン。"""
    def __init__(self, pin_number: int, frequency_hz: float = 50, pgpio=None) -> None:
        self.pgpio = pgpio
        self.pin_number: int = pin_number
        self.frequency: int = int(frequency_hz)
        self._state: int = PinState.NOT_STARTED

    def start(self, duty: float = 0) -> None:
        """指定したデューティ比で PWM 出力を開始する。

        Args:
            duty: 0〜1 の範囲で指定するデューティ比。

        Raises:
            RuntimeError: 既に開始されている場合。
        """
        if self.state() != PinState.NOT_STARTED:
            raise RuntimeError(f"InputPinPigpio({self.pin_number}) は既に開始されています")
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle は 0〜1 の範囲でなければなりません")
        self.pgpio = self.pgpio or pigpio.pi()
        self.pgpio.set_mode(self.pin_number, pigpio.OUTPUT)
        self.pgpio.set_PWM_frequency(self.pin_number, self.frequency)
        self.pgpio.set_PWM_range(self.pin_number, 4095)  # 12bit、PCA9685 と同じ
        self.pgpio.set_PWM_dutycycle(self.pin_number, int(duty * 4095))  # 初期状態を設定
        self._state = duty

    def stop(self) -> None:
        if self.state() != PinState.NOT_STARTED:
            self.pgpio.write(self.pin_number, PinState.LOW)
            self.pgpio.stop()
            self.pgpio = None
        self._state = PinState.NOT_STARTED

    def state(self) -> float:
        """
        This returns the last set duty cycle.
        :return: duty cycle in range 0 to 1 OR PinState.NOT_STARTED in not started
        """
        return self._state

    def duty_cycle(self, duty: float) -> None:
        """
        Write a duty cycle to the output pin
        :param duty: duty cycle in range 0 to 1
        :except: RuntimeError if not started
        """
        if duty < 0 or duty > 1:
            raise ValueError("duty_cycle は 0〜1 の範囲でなければなりません")
        if self.state() != PinState.NOT_STARTED:
            self.pgpio.set_PWM_dutycycle(self.pin_number, int(duty * 4095))
            self._state = duty


if __name__ == '__main__':
    import argparse
    import sys
    import time

    #
    # RPi ボードピン 33 (BCM.13 相当) で 50% duty を 10 秒間出力する例
    # python pins.py --pwm-pin=RPI_GPIO.BOARD.33 --duty=0.5 --time=10
    #
    # RPi ボードピン 35 (BCM.19 相当) を 10 秒間入力する例
    # python pins.py --in-pin=RPI_GPIO.BOARD.35 --time=10
    #
    # RPi ボードピン 33 を 50% duty で出力し、ピン 35 を割り込み付き入力で読む例
    # python pins.py --pwm-pin=RPI_GPIO.BOARD.33 --duty=0.5 --in-pin=RPI_GPIO.BOARD.35 -int=rising --time=10
    #
    # RPi ボードピン 33 を出力、ピン 35 を割り込み付き入力で読む例
    # python pins.py --out-pin=RPI_GPIO.BOARD.33 --duty=0.5 --in-pin=RPI_GPIO.BOARD.35 -int=rising --time=10
    #
    #
    # 引数を解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pwm-pin", type=str, default=None,
                        help="PWM ピン ID。例: 'PCA9685:1:60.13' または 'RPI_GPIO.BCM.13'")
    parser.add_argument("-hz", "--hertz", type=int, default=60,
                        help="PWM 信号の周波数(Hz)。既定は 60Hz")
    parser.add_argument("-d", "--duty", type=float, default=0.5,
                        help="デューティ比 (0〜1)。既定は 0.5")

    parser.add_argument("-o", "--out-pin", type=str, default=None,
                        help="TTL 出力ピン ID。例: 'PCA9685:1:60.13'、'RPI_GPIO.BOARD.35'、'RPI_GPIO.BCM.13'")

    parser.add_argument("-i", "--in-pin", type=str, default=None,
                        help="TTL 入力ピン ID。例: 'RPI_GPIO.BOARD.35' や 'RPI_GPIO.BCM.19'")
    parser.add_argument("-pu", "--pull", type=str, choices=['up', 'down', 'none'], default='none',
                        help="入力ピンのプルアップ設定。'up'、'down'、'none' のいずれか")
    parser.add_argument("-int", "--interrupt", type=str, choices=['falling', 'rising', 'both', 'none'], default='none',
                        help="割り込みを使用するエッジ種別: 'falling'、'rising'、'both' のいずれか")
    parser.add_argument("-tm", "--time", type=float, default=1, help="テスト実行時間(秒)")
    parser.add_argument("-db", "--debug", action='store_true', help="デバッグ出力を表示")
    parser.add_argument("-th", "--threaded", action='store_true', help="スレッドモードで実行")

    # Read arguments from command line
    args = parser.parse_args()

    help = []
    if args.hertz < 1:
        help.append("-hz/--hertz は 1 以上で指定してください")

    if args.duty < 0 or args.duty > 1:
        help.append("-d/--duty は 0〜1 の範囲で指定してください")

    if args.pwm_pin is None and args.out_pin is None and args.in_pin is None:
        help.append("-o/--out-pin、-p/--pwm-pin、-i/--in-pin のいずれかを指定してください")

    if args.pwm_pin is not None and args.out_pin is not None:
        help.append("-o/--out-pin と -p/--pwm-pin は同時に指定できません")

    if args.time < 1:
        help.append("-tm/--time は 0 より大きい値を指定してください")

    if len(help) > 0:
        parser.print_help()
        for h in help:
            print("  " + h)
        sys.exit(1)

    pin_pull = {
        'up': PinPull.PULL_UP,
        'down': PinPull.PULL_DOWN,
        'none': PinPull.PULL_NONE
    }
    pin_edge = {
        'none': None,
        'falling': PinEdge.FALLING,
        'rising': PinEdge.RISING,
        'both': PinEdge.BOTH
    }

    def on_input():
        state = ttl_in_pin.input()
        if state == PinState.HIGH:
            print("+", ttl_in_pin.pin_number, time.time()*1000)
        elif state == PinState.LOW:
            print("-", ttl_in_pin.pin_number, time.time()*1000)

    pwm_out_pin: PwmPin = None
    ttl_out_pin: OutputPin = None
    ttl_in_pin: InputPin = None
    try:
        #
        # 適切な種類のピンを構築
        #
        if args.in_pin is not None:
            ttl_in_pin = input_pin_by_id(args.in_pin, pin_pull[args.pull])
            if args.interrupt != 'none':
                ttl_in_pin.start(on_input=on_input, edge=pin_edge[args.interrupt])
            else:
                ttl_in_pin.start()

        if args.pwm_pin is not None:
            pwm_out_pin = pwm_pin_by_id(args.pwm_pin, args.hertz)
            pwm_out_pin.start(args.duty)

        if args.out_pin is not None:
            ttl_out_pin = output_pin_by_id(args.out_pin, args.hertz)
            ttl_out_pin.start(PinState.LOW)

        start_time = time.time()
        end_time = start_time + args.time
        while start_time < end_time:
            if ttl_out_pin is not None:
                if args.duty > 0:
                    ttl_out_pin.output(PinState.HIGH)
                    time.sleep(1 / args.hertz * args.duty)
                if args.duty < 1:
                    ttl_out_pin.output(PinState.LOW)
                    time.sleep(1 / args.hertz * (1 - args.duty))
            else:
                # バックグラウンドスレッドに処理を譲る
                sleep_time = 1/args.hertz - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0)  # 他スレッドへ処理を譲る
            start_time = time.time()

    except KeyboardInterrupt:
        print('早期終了します。')
    except Exception as e:
        print(e)
        exit(1)
    finally:
        if pwm_out_pin is not None:
            pwm_out_pin.stop()
        if ttl_out_pin is not None:
            ttl_out_pin.stop()
        if ttl_in_pin is not None:
            ttl_in_pin.stop()
