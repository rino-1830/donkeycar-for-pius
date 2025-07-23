"""循環バッファ実装モジュール。"""

class CircularBuffer:
    """固定容量の循環バッファ。

    キュー、スタック、配列のいずれとしても利用可能。
    """
    def __init__(self, capacity:int, defaultValue=None) -> None:
        if capacity <= 0:
            raise ValueError("capacity は 0 より大きくなければなりません")
        self.defaultValue = defaultValue
        self.buffer:list = [None] * capacity
        self.capacity:int = capacity
        self.count:int = 0
        self.tailIndex:int = 0

    def head(self):
        """先頭要素を非破壊で取得する。

        Returns:
            list が空でなければ最新に追加された要素、空であればコンストラクタで
            指定したデフォルト値。
        """
        if self.count > 0:
            return self.buffer[(self.tailIndex + self.count - 1) % self.capacity]
        return self.defaultValue

    def tail(self):
        """末尾要素を非破壊で取得する。

        Returns:
            list が空でなければ最も古い要素、空であればコンストラクタで指定した
            デフォルト値。
        """
        if self.count > 0:
            return self.buffer[self.tailIndex]
        return self.defaultValue

    def enqueue(self, value):
        """値を先頭に追加する。

        バッファが満杯の場合は末尾の値を破棄して空きを作る。
        """
        if self.count < self.capacity:
            self.count += 1
        else:
            # 末尾の要素を捨てる
            self.tailIndex = (self.tailIndex + 1) % self.capacity

        # 先頭に値を書き込む
        self.buffer[(self.tailIndex + self.count - 1) % self.capacity] = value    

    def dequeue(self):
        """末尾の値を取り出して返す。

        Returns:
            list が空でなければ末尾の値、空であればデフォルト値。
        """
        theValue = self.tail()
        if self.count > 0:
            self.count -= 1
            self.tailIndex = (self.tailIndex + 1) % self.capacity
        return theValue

    def push(self, value):
        """値を先頭に追加する。

        バッファが満杯の場合は ``IndexError`` を送出する。
        """
        if self.count >= self.capacity:
            raise IndexError("満杯のバッファへ push しようとしました")

        self.enqueue(value)

    def pop(self):
        """先頭の値を取り出して返す。

        Returns:
            list が空でなければ先頭の値、空であればデフォルト値。
        """
        theValue = self.head()
        if self.count > 0:
            self.count -= 1
        return theValue

    def append(self, value):
        """値を末尾（インデックス ``count-1``）に追加する。

        バッファが満杯の場合は ``IndexError`` を送出する。
        """
        if self.count >= self.capacity:
            raise IndexError("満杯のバッファへ append しようとしました")

        # 末尾に空きを作って値を入れる
        self.count += 1
        self.tailIndex = (self.tailIndex - 1) % self.capacity
        self.buffer[self.tailIndex] = value


    def get(self, i:int):
        """指定インデックスの値を取得する。

        先頭を 0、末尾を ``count-1`` とする。

        Args:
            i: 取得するインデックス。

        Returns:
            指定インデックスの値。範囲外の場合はデフォルト値。
        """
        if (i >= 0) and (i < self.count):
            return self.buffer[(self.tailIndex + (self.count + i - 1)) % self.capacity]

        return self.defaultValue

    def set(self, i:int, value):
        """指定インデックスに値を設定する。

        先頭を 0、末尾を ``count-1`` とする。

        Args:
            i: 設定するインデックス。
            value: 設定する値。

        Raises:
            IndexError: インデックスが範囲外の場合。
        """
        if (i >= 0) and (i < self.count):
            self.buffer[(self.tailIndex + (self.count + i - 1)) % self.capacity] = value
            return
        raise IndexError("バッファのインデックスが範囲外です")

    def truncateTo(self, count):
        """リストを指定数まで切り詰める。

        与えられた数が ``count()`` 以上なら変更せず、
        小さい場合は末尾から要素を削除して合わせる。キューの容量は変わらない。

        例:
            先頭以外をすべて削除するには ``truncateTo(1)`` を呼び出す。

        Args:
            count: 残す要素数の上限。

        Raises:
            ValueError: ``count`` が範囲外の場合。
        """
        if count < 0 or count > self.capacity:
            raise ValueError("count が範囲外です")
        self.count = count
        self.tailIndex = (self.tailIndex + count) % self.capacity

