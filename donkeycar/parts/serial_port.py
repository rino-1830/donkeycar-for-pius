import logging
from typing import Tuple
import serial
import serial.tools.list_ports
import threading
import time
import donkeycar.utilities.dk_platform as dk_platform


logger = logging.getLogger(__name__)


class SerialPort:
    """シリアルポートの接続、読み取り、書き込みを行うラッパー。

    生の pyserial API の代わりに使用してください。例外を捕捉してバイト列と
    文字列の相互変換を自動で行い、テスト時にモックしやすいよう抽象化します。
    """
    def __init__(
        self,
        port: str = '/dev/ttyACM0',
        baudrate: int = 115200,
        bits: int = 8,
        parity: str = 'N',
        stop_bits: int = 1,
        charset: str = 'ascii',
        timeout: float = 0.1,
    ):
        """インスタンスを初期化する。

        Args:
            port: 接続するシリアルポート。
            baudrate: ボーレート。
            bits: データビット数。
            parity: パリティビット。
            stop_bits: ストップビット数。
            charset: 文字列変換に使用する文字セット。
            timeout: タイムアウト秒数。
        """
        self.port = port
        self.baudrate = baudrate
        self.bits = bits
        self.parity = parity
        self.stop_bits = stop_bits
        self.charset = charset
        self.timeout = timeout
        self.ser = None

    def start(self):
        """シリアルポートを開く。

        Returns:
            SerialPort: 自身のインスタンス。
        """
        for item in serial.tools.list_ports.comports():
            logger.info(item)  # すべてのシリアルポートを表示
        self.ser = serial.Serial(
            self.port,
            self.baudrate,
            self.bits,
            self.parity,
            self.stop_bits,
            timeout=self.timeout,
        )
        logger.debug("シリアルポートを開きました " + self.ser.name)
        return self

    def stop(self):
        """シリアルポートを閉じる。

        Returns:
            SerialPort: 自身のインスタンス。
        """
        if self.ser is not None:
            sp = self.ser
            self.ser = None
            sp.close()
        return self

    def buffered(self) -> int:
        """バッファされている文字数を返す。

        Returns:
            int: バッファにある文字数。
        """
        if self.ser is None or not self.ser.is_open:
            return 0

        # macOS では ``ser.in_waiting`` が常に 0 を返すため、バッファがあるものと見なす
        if dk_platform.is_mac():
            return 1

        try:
            return self.ser.in_waiting
        except serial.serialutil.SerialException:
            return 0

    def clear(self):
        """シリアルの読み取りバッファをクリアする。

        Returns:
            SerialPort: 自身のインスタンス。
        """
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.reset_input_buffer()
        except serial.serialutil.SerialException:
            pass
        return self

    def readBytes(self, count:int=0) -> Tuple[bool, bytes]:
        """バッファにデータがあればバイト列を読み取る。

        Args:
            count: 読み取るバイト数。

        Returns:
            Tuple[bool, bytes]: ``count`` バイトが読めたかどうかと、読み取った
            バイト列。必要数がない場合は空のバイト列を返す。
        """
        if self.ser is None or not self.ser.is_open:
            return (False, b'')

        try:
            input = ''
            waiting = self.buffered() >= count
            if waiting:   # シリアルポートを読み取り、データがあるか確認する
                input = self.ser.read(count)
            return (waiting, input)
        except (serial.serialutil.SerialException, TypeError):
            logger.warn("シリアルポートからのバイト読み取りに失敗しました")
            return (False, b'')

    def read(self, count:int=0) -> Tuple[bool, str]:
        """バッファにデータがあれば文字列として読み取る。

        Args:
            count: 読み取るバイト数。

        Returns:
            Tuple[bool, str]: ``count`` バイトを取得できたかどうかと、
            読み取った文字列。必要数がない場合は空文字列を返す。
        """
        ok, bytestring = self.readBytes(count)
        try:
            return (ok, bytestring.decode(self.charset))
        except UnicodeDecodeError:
            # 初回の読み取りでは枠組みが壊れたデータが含まれることがある
            return (False, "")

    def readln(self) -> Tuple[bool, str]:
        """バッファにデータがあれば1行読み取る。

        このメソッドは行末が読み取れるまでブロックし、行末文字も結果に含めます。

        Returns:
            Tuple[bool, str]: 行を取得できたかどうかと、取得した行。読めなかった
            場合は空文字列を返します。
        """
        if self.ser is None or not self.ser.is_open:
            return (False, "")

        try:
            input = ''
            waiting = self.buffered() > 0
            if waiting:   # シリアルポートを読み取り、データがあるか確認する
                buffer = self.ser.readline()
                input = buffer.decode(self.charset)
            return (waiting, input)
        except (serial.serialutil.SerialException, TypeError):
            logger.warn("シリアルポートから行の読み取りに失敗しました")
            return (False, "")
        except UnicodeDecodeError:
            # 初回の読み取りでは枠組みが壊れたデータが含まれることがある
            logger.warn("シリアルポートの行をUnicodeデコードできませんでした")
            return (False, "")

    def writeBytes(self, value:bytes):
        """バイト列をシリアルポートへ書き込む。"""
        if self.ser is not None and self.ser.is_open:
            try:
                self.ser.write(value)
            except (serial.serialutil.SerialException, TypeError):
                logger.warn("シリアルポートへ書き込めません")

    def write(self, value:str):
        """文字列をシリアルポートへ書き込む。"""
        self.writeBytes(value.encode())

    def writeln(self, value:str):
        """文字列を改行付きで書き込む。"""
        self.write(value + '\n')


class SerialLineReader:
    """シリアルポートから行を読み取る Donkeycar パーツ。"""
    def __init__(self, serial: SerialPort, max_lines: int = 0, debug: bool = False):
        """インスタンスを初期化する。

        Args:
            serial: 使用する ``SerialPort`` インスタンス。
            max_lines: 1 回の読み取りサイクルで返す最大行数。0 は無制限。
            debug: デバッグログを有効にするかどうか。
        """
        self.serial = serial
        self.max_lines = max_lines  # 1 回の読み取りサイクルで返す最大行数
        self.debug = debug
        self.lines = []
        self.lock = threading.Lock()
        self.running = True
        self._open()
        self.clear()

    def _open(self):
        """排他的にシリアルポートを開き、バッファをクリアする。"""
        with self.lock:
            self.serial.start().clear()

    def _close(self):
        """排他的にシリアルポートを閉じる。"""
        with self.lock:
            self.serial.stop()

    def clear(self):
        """行バッファとシリアルポートの入力バッファをクリアする。"""
        with self.lock:
            self.lines = []
            self.serial.clear()

    def _readline(self) -> str:
        """スレッドセーフに1行読み取る。

        Returns:
            str: 行を読み取れた場合はその内容、なければ ``None``。
        """
        if self.lock.acquire(blocking=False):
            try:
                # TODO: Macintoshでは ``Serial.in_waiting`` が常に 0 を返す
                if dk_platform.is_mac() or (self.serial.buffered() > 0):
                    success, buffer = self.serial.readln()
                    if success:
                        return buffer
            finally:
                self.lock.release()
        return None

    def run(self):
        """スレッドを使用しない場合に行を読み取る。

        Returns:
            list[tuple[float, str]]: 読み取った行のタイムスタンプ付きリスト。
        """
        if self.running:
            # 非スレッドモードでは ``max_lines`` 分だけ読み込んで返す
            lines = []
            line = self._readline()
            while line is not None:
                lines.append((time.time(), line))
                line = None
                if (
                    self.max_lines is None
                    or self.max_lines == 0
                    or len(lines) < self.max_lines
                ):
                    line = self._readline()
            return lines
        return []

    def run_threaded(self):
        """スレッド実行時に蓄積された行を返す。"""
        if not self.running:
            return []

        # 蓄積された読み取り結果を返す
        with self.lock:
            lines = self.lines
            self.lines = []
            return lines


    def update(self):
        """シリアルポートを開いて無限ループで読み取りを続ける。

        NOTE: このメソッドは非スレッド実行の ``run()`` とは互換性がない。
        """
        buffered_lines = []  # ローカル読み取りバッファ
        while self.running:
            line = self._readline()
            if line:
                buffered_lines.append((time.time(), line))
            if buffered_lines:
                # ``self.positions`` へのアクセスをスレッドセーフに行う
                # ブロックはせず、書き込めない場合は ``buffered_lines`` に残す
                # 書き込める場合は ``self.positions`` へ移動してバッファをクリアする
                if self.lock.acquire(blocking=False):
                    try:
                        self.lines += buffered_lines
                        buffered_lines = []
                    finally:
                        self.lock.release()
            time.sleep(0)  # 他のスレッドに実行時間を譲る

    def shutdown(self):
        """ループを停止してシリアルポートを閉じる。"""
        self.running = False
        self._close()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--serial",
        type=str,
        required=True,
        help="シリアルポートのアドレス。例: '/dev/tty.usbmodem1411'",
    )
    parser.add_argument(
        "-b",
        "--baudrate",
        type=int,
        default=9600,
        help="シリアルポートのボーレート。",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=0.5,
        help="シリアルポートのタイムアウト(秒)。",
    )
    parser.add_argument(
        "-sp",
        "--samples",
        type=int,
        default=5,
        help="1 回の読み取りサイクルで処理する行数。0 で無制限。",
    )
    parser.add_argument(
        "-th",
        "--threaded",
        action='store_true',
        help="スレッドモードで実行する。",
    )
    parser.add_argument(
        "-db",
        "--debug",
        action='store_true',
        help="詳細ログを有効にする",
    )
    args = parser.parse_args()

    if args.samples < 0:
        print("読み取りサイクルごとのサンプル数は 0 または正の数で指定してください")
        parser.print_help()
        sys.exit(0)

    if args.timeout <= 0:
        print("タイムアウトは 0 より大きい値を指定してください")
        parser.print_help()
        sys.exit(0)

    update_thread = None
    reader = None

    try:
        serial_port = SerialPort(args.serial, baudrate=args.baudrate, timeout=args.timeout)
        line_reader = SerialLineReader(serial_port, max_lines=args.samples, debug=args.debug)

        # スレッド部分を起動し、プロット表示用のウィンドウも開く
        if args.threaded:
            update_thread = threading.Thread(target=line_reader.update, args=())
            update_thread.start()


        def read_lines():
            return line_reader.run_threaded() if args.threaded else line_reader.run()

        while line_reader.running:
            readings = read_lines()
            if readings:
                # 読み取った行を表示するだけ
                for line in readings:
                    print(line)
    finally:
        if line_reader:
            line_reader.shutdown()
        if update_thread is not None:
            update_thread.join()  # スレッドの終了を待つ

