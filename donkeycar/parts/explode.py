"""辞書の引数を個別の名前付き引数に展開するパーツ。"""


class ExplodeDict:
    """辞書を個々の名前付き出力に展開するパーツ。"""

    def __init__(self, memory, output_prefix=""):
        """キーと値を分割して出力メモリに書き込む。

        Args:
            memory (dict): 出力先となるメモリ。
            output_prefix (str, optional): 出力時にキーの前に付加する文字列。
                デフォルトは空文字。
        """
        self.memory = memory
        self.prefix = output_prefix

    def run(self, key_values):
        """辞書を展開してメモリに保存する。

        Args:
            key_values (dict): 出力するキーと値の辞書。

        Returns:
            None: 返り値は常に ``None``。
        """
        if type(key_values) is dict:
            for key, value in key_values.items():
                self.memory[self.prefix + key] = value
        return None
