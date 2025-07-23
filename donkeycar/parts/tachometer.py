from abc import (ABC, abstractmethod)
import logging
import threading
import time
from typing import Tuple

from donkeycar.utils import is_number_type
from donkeycar.parts.serial_port import SerialPort
from donkeycar.parts.pins import InputPin, PinEdge


logger = logging.getLogger("donkeycar.parts.tachometer")


def sign(value) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


class EncoderMode:
    FORWARD_ONLY = 1  # 符号付きのカウントを単純に加算
    FORWARD_REVERSE = 2  # スロットルが負のときは減算
    FORWARD_REVERSE_STOP = 3  # スロットルがゼロのときは無視


class AbstractEncoder(ABC):
    """エンコーダのインターフェース。

    このクラスを継承し、:py:meth:`start_ticks`, :py:meth:`stop_ticks`,
    :py:meth:`poll_ticks` を実装して新しいエンコーダクラスを作成する。
    :class:`Tachometer` のコンストラクタはエンコーダを受け取る。
    """
    @abstractmethod
    def start_ticks(self):
        """エンコーダを初期化する。"""
        pass

    @abstractmethod
    def stop_ticks(self):
        """エンコーダのリソースを解放する。"""
        pass

    @abstractmethod
    def poll_ticks(self, direction: int):
        """エンコーダのカウントを更新する。

        Args:
            direction (int): 前進なら1、後退なら-1、停止なら0。

        エンコーダから新しい値を取得する。
        """
        pass

    @abstractmethod
    def get_ticks(self, encoder_index: int = 0) -> int:
        """直近に取得したエンコーダカウントを返す。

        Args:
            encoder_index (int): 0始まりのエンコーダ番号。

        Returns:
            int: 最新のカウント値。

        このメソッドは :py:meth:`poll_ticks` で取得した値を返すだけで、
        新しい値は取得しない。
        """
        return 0


class SerialEncoder(AbstractEncoder):
    """シリアルポート越しにカウントを取得するエンコーダ。

    反対側には通常、エンコーダを読むマイコンがあり、単一チャネルエンコーダ
    （カウントが増えるだけ）またはクアドラチャエンコーダ（増減する）のいずれかを
    使用する。

    クアドラチャエンコーダは前進・後退・停止を検出できるため、デフォルトの
    ``FORWARD_ONLY`` モードを使えばカウント変化が正しく合算される。

    単一チャネルエンコーダは進行方向を判別できないため、符号付きスロットル値を
    方向として利用し、カウントを増減させる。ゼロスロットルで惰性走行する車両には
    ``FORWARD_BACKWARD`` を、すぐに停止する車両には ``FORWARD_BACKWARD_STOP`` を
    選ぶとよい。

    このクラスは ``r/p/c`` プロトコルを実装したマイコンとシリアル接続していることを
    想定する。

    **コマンド**（各行末に ``\n``）:

    - ``r`` : 位置をゼロにリセット
    - ``p`` : 直ちに位置を送信
    - ``c`` : 連続モード開始/停止。整数が続く場合は読み取り間隔(ms)、
      何も続かない場合は停止

    マイコンは ``ticks,time`` を ``;`` で区切った形式で1行ずつ送信する。
    例: ``{ticks},{milliseconds};{ticks},{milliseconds}\n``

    ``donkeycar/arduino/encoder/encoder.ino`` にこのプロトコルを用いた例がある。
    他のマイコンやピン配置、単一チャネルエンコーダを使う場合は適宜変更すること。
    """
    def __init__(self, serial_port:SerialPort=None, debug=False):
        if serial_port is None:
            raise ValueError("serial_port は SerialPort インスタンスでなければなりません")
        self.ser = serial_port
        self.ticks = [0,0]
        self.lasttick = [0,0]
        self.lock = threading.Lock()
        self.buffered_ticks = []
        self.running = False
    
    def start_ticks(self):
        self.ser.start()
        self.ser.writeln('r')  # エンコーダをゼロにリセット
        # if self.poll_delay_secs > 0:
        #     # 非ゼロ間隔が指定された場合は連続モード開始
        #     self.ser.writeln("c" + str(int(self.poll_delay_secs * 1000)))
        self.ser.writeln('p')  # 最初のカウントを要求
        self.running = True

    def stop_ticks(self):
        self.running = False
        if self.ser is not None:
            self.ser.stop()

    #
    # TODO: 深刻なバグ; 各エンコーダごとに独立した方向を持つべき。
    #       poll_ticks でエンコーダ番号を受け取るか、チャンネル毎に方向を保持するかのどちらかにする。
    #       左右の車輪を逆方向に回すことは現状ないため差し迫った問題ではない。
    #
    def poll_ticks(self, direction: int):
        """エンコーダのカウントを読み取る。

        Args:
            direction (int): 1は前進、-1は後退、0は停止。

        Returns:
            None

        この呼び出しでエンコーダから最新の値を取得する。
        """
        #
        # 受信待ちのデータがあればシリアルポートから読み込み、
        # 全ての行を読み取って最新の行だけを使用する。
        #
        input = ''
        while (self.running and (self.ser.buffered() > 0) and (input == "")):
            _, input = self.ser.readln()

        #
        # "p" コマンドを送信して次の読み取りを予約する
        #
        self.ser.writeln('p')  

        #
        # データがあればカウントを更新し、なければ前回値を使用する。
        # 複数エンコーダのデータはセミコロン区切り、
        # 1エンコーダの値はカンマ区切りで "ticks,time" の形式となる。
        # 例: "ticks,ticksMs;ticks,ticksMs"
        #
        if input != '':
            try:
                #
                # エンコーダ毎の出力を分割
                # "ticks,time;ticks,time" -> ["ticks,time", "ticks,time"]
                #
                values = [s.strip() for s in input.split(';')]

                #
                # 各エンコーダの tick/time ペアを分割
                # ["ticks,time", "ticks,time"] -> [["ticks", "time"], ["ticks", "time"]]
                #
                values = [v.split(',') for v in values]
                for i in range(len(values)):
                    total_ticks = int((values[i][0]).strip())

                    delta_ticks = 0
                    if i < len(self.lasttick):
                        delta_ticks = total_ticks - self.lasttick[i]

                    #
                    # スレッドセーフなバッファに保存する。
                    # マイコンが返すチャンネル数に合わせてバッファを拡張し、
                    # 設定との同期を不要にする。
                    #
                    if i < len(self.lasttick):
                        self.lasttick[i] = total_ticks
                    else:
                        self.lasttick.append(total_ticks)

                    if i < len(self.buffered_ticks):
                        self.buffered_ticks[i] += delta_ticks * direction
                    else:
                        self.buffered_ticks.append(delta_ticks * direction)
            except ValueError:
                logger.error("シリアルからのエンコーダ値の解析に失敗")

        #
        # ロックを取得できた場合は共有値を更新してバッファをクリアし、
        # 取得できなければ次回までスキップする。
        #
        if self.lock.acquire(blocking=False):
            try:
                for i in range(len(self.buffered_ticks)):
                    #
                    # マイコンが返すチャンネル数に合わせて配列を拡張する
                    #
                    if i < len(self.ticks):
                        self.ticks[i] += self.buffered_ticks[i]
                    else:
                        self.ticks.append(self.buffered_ticks[i])
                    self.buffered_ticks[i] = 0
            finally:
                self.lock.release()

    def get_ticks(self, encoder_index:int=0) -> int:
        """
        Get last polled encoder ticks
        encoder_index: zero based index of encoder.
        return: Most recently polled encoder ticks

        This will return the same value as the
        most recent call to poll_ticks().  It 
        will not request new values from the encoder.
        It will not block.
        """
        with self.lock:
            return self.ticks[encoder_index] if encoder_index < len(self.ticks) else 0


class EncoderChannel(AbstractEncoder):
    """SerialEncoder をラップし、2 番目のチャンネルを独立したエンコーダとして扱う。

    親 ``SerialEncoder`` を先にポーリングしてからこのチャンネルの
    :py:meth:`get_ticks` を呼ぶ必要があるため、必ず親 ``SerialEncoder`` の後に
    追加すること。
    """
    def __init__(self, encoder:SerialEncoder, channel:int) -> None:
        self.encoder = encoder
        self.channel = channel
        super().__init__()

    def start_ticks(self):
        if not self.encoder.running:
            self.encoder.start_ticks()

    def stop_ticks(self):
        if self.encoder.running:
            self.encoder.stop_ticks()

    def poll_ticks(self, direction:int):
        self.encoder.poll_ticks(direction)

    def get_ticks(self, encoder_index:int=0) -> int:
        return self.encoder.get_ticks(encoder_index=self.channel)


class GpioEncoder(AbstractEncoder):
    """InputPin で読み取る単一チャネルのエンコーダ。

    Args:
        gpio_pin (InputPin): 信号を受け取るピン。
        debounce_ns (int): 次の信号を受け付けるまでの遅延時間(ナノ秒)。
        debug (bool): デバッグログを出力するかどうか。
    """
    def __init__(self, gpio_pin: InputPin, debounce_ns:int=0, debug=False):
        # gpio_pin の検証
        if gpio_pin is None:
            raise ValueError('エンコーダ入力ピンは有効な InputPin でなければなりません')

        self.debug = debug
        self.counter = 0
        self._cb_counter = 0
        self.direction = 0
        self.pin = gpio_pin
        self.debounce_ns:int = debounce_ns
        self.debounce_time:int = 0
        if self.debounce_ns > 0:
            logger.warn("GpioEncoder: debounce_ns は無視されます")
        self.lock = threading.Lock()

    def _cb(self):
        """GPIO 割り込みで呼び出されるコールバック。"""
        #
        # ロック待ちでブロックしないよう内部カウンターを更新し、
        # ロックが取れたときにその値を公開用カウンターへ反映する。
        #
        self._cb_counter += 1
        if self.lock.acquire(blocking=False):
            try:
                self.counter += self._cb_counter * self.direction
                self._cb_counter = 0
            finally:
                self.lock.release()
            
    def start_ticks(self):
        # GPIO ピンの設定
        self.pin.start(on_input=lambda: self._cb(), edge=PinEdge.RISING)
        logger.info(
            f'GpioEncoder: 入力ピン "RPI_GPIO.{self.pin.pin_scheme_str}.{self.pin.pin_number}" を開始しました。'
        )

    def poll_ticks(self, direction: int):
        """エンコーダのカウントを読み取る。

        Args:
            direction (int): 1は前進、-1は後退、0は停止。
        """
        with self.lock:
            self.direction = direction

    def stop_ticks(self):
        self.pin.stop()
        logger.info(
            f'GpioEncoder: 入力ピン "RPI_GPIO.{self.pin.pin_scheme_str}.{self.pin.pin_number}" を停止しました。'
        )

    def get_ticks(self, encoder_index:int=0) -> int:
        """
        Get last polled encoder ticks
        encoder_index: zero based index of encoder.
        return: Most recently polled encoder ticks

        This will return the same value as the
        most recent call to poll_ticks().  It 
        will not request new values from the encoder.
        This will not block.
        """
        with self.lock:
            return self.counter if encoder_index == 0 else 0


class MockEncoder(AbstractEncoder):
    """
    A mock encoder that turns throttle values into ticks.
    It generates ENCODER_PPR ticks per second at full throttle.
    The run() method must be called at the same rate as the
    tachometer calls the poll() method.
    """
    def __init__(self, ticks_per_second: float):
        self.ticks_per_second = ticks_per_second
        self.throttle = 0
        self.ticks = 0
        self.remainder_ticks = 0
        self.timestamp = None
        self.running = False

    def run(self, throttle:float, timestamp: int = None):
        """モックエンコーダを更新する。"""
        if timestamp is None:
            timestamp = time.time()

        # poll() が None を渡した場合は最後に渡された値を利用する
        if throttle is not None:
            self.throttle = throttle

    def start_ticks(self):
        self.running = True

    def stop_ticks(self):
        self.running = False

    def poll_ticks(self, direction: int):
        timestamp = time.time()
        last_time = self.timestamp if self.timestamp is not None else timestamp
        self.timestamp = timestamp

        if self.running:
            delta_time = timestamp - last_time
            delta_ticks = abs(self.throttle) * direction * self.ticks_per_second * delta_time + self.remainder_ticks
            delta_int_ticks = int(delta_ticks)
            self.ticks += delta_int_ticks
            self.remainder_ticks = delta_ticks - delta_int_ticks

    def get_ticks(self, encoder_index: int = 0) -> int:
        return self.ticks


class Tachometer:
    """エンコーダのカウントを回転数に変換するクラス。

    スロットル入力に応じて進行方向を変更することもできる。
    """

    def __init__(self,
                 encoder:AbstractEncoder,
                 ticks_per_revolution:float=1,
                 direction_mode=EncoderMode.FORWARD_ONLY,
                 poll_delay_secs:float=0.01,
                 debug=False):
        """Tachometer を初期化する。

        Args:
            encoder (AbstractEncoder): 使用するエンコーダインスタンス。
            ticks_per_revolution (float): ホイール1回転あたりのエンコーダカウント。
                生のカウントが欲しい場合は ``1`` を設定する。
            direction_mode (EncoderMode): スロットルによる回転方向の扱い方。
            poll_delay_secs (float): エンコーダをポーリングする間隔(秒)。
            debug (bool): デバッグ出力を有効にするか。
        """

        if encoder is None:
            raise ValueError("encoder は AbstractEncoder のインスタンスでなければなりません")
        self.encoder = encoder
        self.running:bool = False
        self.ticks_per_revolution:float = ticks_per_revolution
        self.direction_mode = direction_mode
        self.ticks:int = 0
        self.direction:int = 1  # default to forward ticks
        self.timestamp:float = 0
        self.throttle = 0.0
        self.debug = debug
        self.poll_delay_secs = poll_delay_secs
        self.encoder.start_ticks()
        self.running = True

    # TODO: poll() にスロットルを渡さずに済むようリファクタリングする
    def poll(self, throttle, timestamp):
        """エンコーダをポーリングして値を更新する。

        Args:
            throttle (float): 正なら前進、負なら後退、0 は停止。
            timestamp (int, optional): この読み取りに適用する時刻。省略時は現在時刻。
        """

        if self.running:
            # タイムスタンプが指定されていない場合は現在時刻を使用
            if timestamp is None:
                timestamp = time.time()

            # 方向モードに応じて向きを設定
            if throttle is not None:
                if EncoderMode.FORWARD_REVERSE == self.direction_mode:
                    # スロットルがゼロの場合は惰性走行として方向を維持
                    if throttle != 0:
                        self.direction = sign(throttle)
                elif EncoderMode.FORWARD_REVERSE_STOP == self.direction_mode:
                    self.direction = sign(throttle)

            lastTicks = self.ticks
            self.timestamp = timestamp
            self.encoder.poll_ticks(self.direction)
            self.ticks = self.encoder.get_ticks()
            if self.debug and self.ticks != lastTicks:
                logger.info(
                    "タコメーター: t = {}, r = {}, ts = {}".format(
                        self.ticks,
                        self.ticks / self.ticks_per_revolution,
                        timestamp,
                    )
                )

    def update(self):
        while(self.running):
            self.poll(self.throttle, None)
            time.sleep(self.poll_delay_secs)  # 他のスレッドへCPU時間を譲る

    def run_threaded(self, throttle:float=0.0, timestamp:float=None) -> Tuple[float, float]:
        """スレッド実行用のインターフェース。

        Args:
            throttle (float): 正なら前進、負なら後退、0 は停止。
            timestamp (float, optional): 読み取りに適用する時刻。省略時は現在時刻。

        Returns:
            Tuple[float, float]: (累積回転数, タイムスタンプ)
        """
        if self.running:
            thisTimestamp = self.timestamp
            thisRevolutions = self.ticks / self.ticks_per_revolution

            # update throttle for next poll()
            if throttle is not None:
                self.throttle = throttle
            self.timestamp = timestamp if timestamp is not None else time.time()

            # return (revolutions, timestamp)
            return thisRevolutions, thisTimestamp
        return 0, self.timestamp

    def run(self, throttle:float=1.0, timestamp:float=None) -> Tuple[float, float]:
        """ポーリングして結果を返すシンプルな実行ループ。

        Args:
            throttle (float): 方向決定に利用するスロットル値。
            timestamp (float, optional): 更新に使用する時刻。省略時は現在時刻。
        """
        if self.running:
            # update throttle for next poll()
            self.throttle = throttle if throttle is not None else 0
            self.timestamp = timestamp if timestamp is not None else time.time()
            self.poll(throttle, timestamp)

            # return (revolutions, timestamp)
            return self.ticks / self.ticks_per_revolution, self.timestamp
        return 0, self.timestamp

    def shutdown(self):
        self.running = False
        self.encoder.stop_ticks()


class InverseTachometer:
    """シミュレータ用: 距離から回転数を計算する。"""
    def __init__(self, meters_per_revolution:float):
        self.meters_per_revolution = meters_per_revolution
        self.revolutions = 0.0
        self.timestamp = time.time()

    def run(self, distance:float, timestamp=None):
        return self.run_threaded(distance, timestamp)

    def run_threaded(self, distance:float, timestamp=None):
        # タイムスタンプが指定されていればそれを使用
        if timestamp is None:
            timestamp = time.time()
        if is_number_type(distance):
            self.timestamp = timestamp
            self.revolutions = distance / self.meters_per_revolution
        else:
            logger.error("distance は float でなければなりません")
        return self.revolutions, self.timestamp

# TODO: throttle から tick を生成する MockThrottleEncoder を作成する
# TODO: 距離から tick を生成する MockInverseEncoder を作成する
# TODO: これらを使えばモック駆動系やシミュレータ用の姿勢推定パイプラインを
#       構築できるはず。スロットルや距離をどう渡すかが課題なので、
#       いっそパーツ化すべきかもしれない。


if __name__ == "__main__":
    import argparse
    from threading import Thread
    import sys
    import time
    from donkeycar.parts.pins import input_pin_by_id

    # 引数を解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rate", type=float, default=20,
                        help="1秒あたりの読み取り回数")
    parser.add_argument("-n", "--number", type=int, default=40,
                        help="収集する読み取り数")
    parser.add_argument("-ppr", "--pulses-per-revolution", type=int, required=True,
                        help="出力軸1回転あたりのパルス数")
    parser.add_argument("-d", "--direction-mode", type=int, default=1,
                        help="1=FORWARD_ONLY, 2=FORWARD_REVERSE, 3=FORWARD_REVERSE_STOP")
    parser.add_argument("-s", "--serial-port", type=str, default=None,
                        help="開くシリアルポート (例: '/dev/ttyACM0')")
    parser.add_argument("-b", "--baud-rate", type=int, default=115200,
                        help="シリアルポートのボーレート")
    parser.add_argument("-e", "--encoder-index", type=int, default=0,
                        help="複数エンコーダがある場合のインデックス (0始まり)")
    parser.add_argument("-p", "--pin", type=str, default=None,
                        help="エンコーダ入力ピンの指定 (例: 'RPI_GPIO.BCM.22')")  # noqa
    parser.add_argument("-dbc", "--debounce-ns", type=int, default=100,
                        help="GPIOピン読み取り時のデバウンス遅延(ns)")  # noqa
    parser.add_argument("-db", "--debug", action='store_true', help="デバッグ出力を表示")
    parser.add_argument("-t", "--threaded", action='store_true', help="スレッドモードで実行")

    # コマンドライン引数を読み込む
    args = parser.parse_args()
    
    help = []
    if args.rate < 1:
        help.append("-r/--rate: 1以上でなければなりません")
        
    if args.number < 1:
        help.append("-n/--number: 1以上でなければなりません")
        
    if args.direction_mode < 1 and args.direction_mode > 3:
        help.append("-d/--direction-mode は 1, 2, 3 のいずれかでなければなりません")

    if args.pulses_per_revolution <= 0:
        help.append("-ppr/--pulses-per-revolution は 0 より大きい値を指定してください")
        
    if args.serial_port is None and args.pin is None:
        help.append("-s/--serial_port または -p/--pin のどちらかを指定してください")

    if args.serial_port is not None and args.pin is not None:
        help.append("-s/--serial_port と -p/--pin はどちらか一方のみ指定してください")

    if args.serial_port is not None and len(args.serial_port) == 0:
        help.append("-s/--serial-port を指定する場合は空文字列にしないでください")
      
    if args.baud_rate <= 0:
        help.append("-b/--baud-rate は 0 より大きい値を指定してください")
        
    if args.pin is not None and args.pin == "":
        help.append("-p/--pin を指定する場合は空にしないでください")

    if args.debounce_ns < 0:
        help.append("-dbc/--debounce-ns は 0 以上を指定してください")
                
    if len(help) > 0:
        parser.print_help()
        for h in help:
            print("  " + h)
        sys.exit(1)
        
    update_thread = None
    serial_port = None
    tachometer = None
    
    try:
        scan_count = 0
        seconds_per_scan = 1.0 / args.rate
        scan_time = time.time() + seconds_per_scan

        #
        # construct a tachometer part of the correct type
        #
        if args.serial_port is not None:
            serial_port = SerialPort(args.serial_port, args.baud_rate)
            tachometer = Tachometer(
                encoder=EncoderChannel(SerialEncoder(serial_port=serial_port, debug=args.debug), args.encoder_index),
                ticks_per_revolution=args.pulses_per_revolution, 
                direction_mode=args.direction_mode, 
                poll_delay_secs=1/(args.rate*2), 
                debug=args.debug)
        if args.pin is not None:
            tachometer = Tachometer(
                encoder=GpioEncoder(gpio_pin=input_pin_by_id(args.pin), debounce_ns=args.debounce_ns, debug=args.debug),
                ticks_per_revolution=args.pulses_per_revolution, 
                direction_mode=args.direction_mode,
                debug=args.debug)
        
        #
        # スレッド部分を開始し、プロット表示用スレッドを起動
        #
        if args.threaded:
            update_thread = Thread(target=tachometer.update, args=())
            update_thread.start()
        
        while scan_count < args.number:
            start_time = time.time()

            # 計測を出力
            scan_count += 1

            # 最新の計測を取得して表示
            if args.threaded:
                measurements = tachometer.run_threaded()
            else:
                measurements = tachometer.run()

            print(measurements)
                                    
            # バックグラウンドスレッドに処理時間を与える
            sleep_time = seconds_per_scan - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                time.sleep(0)  # 他のスレッドへ時間を譲る

    except KeyboardInterrupt:
        print('途中で停止します。')
    except Exception as e:
        print(e)
        exit(1)
    finally:
        if tachometer is not None:
            tachometer.shutdown()
        if update_thread is not None:
            update_thread.join()  # wait for thread to end
