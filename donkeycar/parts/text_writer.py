"""テキストファイルへのログ機能を提供するモジュール."""

import logging
import os

logger = logging.getLogger(__name__)


class TextLogger:
    """テキストファイルへデータを記録するクラス.

    ``row_to_line()`` で変換された「行」は 1 行の文字列として保存される。
    基本実装では単なるテキストとして扱うが、サブクラスは
    ``row_to_line()`` と ``line_to_row()`` をオーバーライドすることで、
    タプルや配列などの構造化データを CSV のように保存できる。
    """
    def __init__(self, file_path: str, append: bool = False, allow_empty_file: bool = False,
                 allow_empty_line: bool = True):
        """インスタンスを生成する.

        Args:
            file_path: 保存先ファイルのパス.
            append: 追記モードで書き込むかどうか.
            allow_empty_file: 空ファイルの保存を許可するかどうか.
            allow_empty_line: 空行の保存を許可するかどうか.
        """
        self.file_path = file_path
        self.append = append
        self.allow_empty_file = allow_empty_file
        self.allow_empty_line = allow_empty_line
        self.rows = []

    def run(self, recording, rows):
        """新しい行をバッファへ追加する.

        Args:
            recording: 録画中かどうかを示すフラグ.
            rows: 追加する行のリスト.

        Returns:
            現在保持している行のリスト.
        """
        if recording and len is not None and len(rows) > 0:
            self.rows += rows
        return self.rows

    def length(self):
        """保持している行数を返す."""
        return len(self.rows)

    def is_empty(self):
        """データが空かどうかを判定する."""
        return 0 == self.length()

    def is_loaded(self):
        """データが読み込まれているかどうかを返す."""
        return not self.is_empty()

    def get(self, row_index:int):
        """インデックスで指定した行を取得する.

        Args:
            row_index: 取得する行のインデックス.

        Returns:
            行オブジェクト。存在しない場合は ``None``。
        """
        return self.rows[row_index] if (row_index >= 0) and (row_index < self.length()) else None

    def reset(self):
        """保持している行をすべて削除する."""
        self.rows = []
        return True

    def row_to_line(self, row):
        """行オブジェクトを文字列へ変換する.

        Args:
            row: 変換する行オブジェクト.

        Returns:
            変換後の文字列。行が ``None`` の場合は ``None`` を返す。
        """
        if row is not None:
            line = str(row)
            if self.allow_empty_line or len(line) > 0:
                return line
        return None

    def line_to_row(self, line: str):
        """文字列を行オブジェクトに変換する.

        Args:
            line: 変換する文字列.

        Returns:
            行オブジェクト。変換できない場合は ``None``。
        """
        if isinstance(line, str):
            line = line.rstrip('\n')
            if self.allow_empty_line or len(line) > 0:
                return line
        return None

    def save(self):
        """現在の行リストをファイルへ保存する.

        Returns:
            保存に成功した場合は ``True``、それ以外は ``False``。
        """
        if self.is_loaded() or self.allow_empty_file:
            with open(self.file_path, "a" if self.append else "w") as fp:
                for row in self.rows:
                    line = self.row_to_line(row)
                    if line is not None:
                        fp.write(self.row_to_line(row))
                        fp.write("\n")
            return True
        return False

    def load(self):
        """ファイルから行リストを読み込む.

        Returns:
            読み込みに成功した場合は ``True``、それ以外は ``False``。
        """
        if os.path.exists(self.file_path):
            rows = []
            with open(self.file_path, "r") as file:
                for line in file:
                    row = self.line_to_row(line)
                    if row is not None:
                        rows.append(row)
            if rows or self.allow_empty_file:
                self.rows = rows
                return True
        return False


class CsvLogger(TextLogger):
    """イテラブルをカンマ区切りのテキストへ記録するロガー.

    区切り文字は ``separator`` で変更可能。
    """
    def __init__(self, file_path: str, append: bool = False, allow_empty_file: bool = False,
                 allow_empty_line: bool = True, separator: str = ",", field_count: int = None,
                 trim: bool = True):
        """インスタンスを生成する.

        Args:
            file_path: 保存先ファイルのパス.
            append: 追記モードで書き込むかどうか.
            allow_empty_file: 空ファイルの保存を許可するかどうか.
            allow_empty_line: 空行の保存を許可するかどうか.
            separator: フィールドを区切る文字.
            field_count: 期待するフィールド数。 ``None`` の場合は検証しない。
            trim: 各フィールドを ``strip()`` でトリムするかどうか.
        """
        super().__init__(file_path, append, allow_empty_file, allow_empty_line)
        self.separator = separator
        self.field_count = field_count
        self.trim = trim

    def row_to_line(self, row):
        """行オブジェクトを区切り文字で連結した文字列に変換する.

        Args:
            row: 変換する行オブジェクト（イテラブル）。

        Returns:
            変換後の文字列。 ``None`` の場合は ``None`` を返す。
        """
        if row is not None:
            line = self.separator.join([str(field) for field in row])
            if self.allow_empty_line or len(line) > 0:
                return line
        return None

    def line_to_row(self, line:str):
        """文字列を行オブジェクトに変換する.

        Args:
            line: 読み込む文字列.

        Returns:
            行オブジェクト。変換できない場合は ``None``。
        """
        row = None
        if isinstance(line, str):
            row = line.rstrip('\n').split(self.separator)
            field_count = len(row)
            if self.field_count is None or field_count == self.field_count:
                if self.trim:
                    row = [field.strip() for field in row]
            else:
                row = None
                logger.debug(f"CsvLogger: フィールド数が {field_count} の行を破棄します")
        else:
            logging.error("CsvLogger: line_to_row には文字列を渡す必要があります")
        return row
