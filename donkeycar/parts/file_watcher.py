import os

class FileWatcher(object):
    """特定のファイルを監視し、変更時に通知するクラス。"""

    def __init__(self, filename, verbose=False):
        """インスタンスを初期化する。

        Args:
            filename (str): 監視対象のファイルパス。
            verbose (bool, optional): 変更を標準出力へ表示するかどうか。
                デフォルトは ``False``。
        """
        self.modified_time = os.path.getmtime(filename)
        self.filename = filename
        self.verbose = verbose

    def run(self):
        """ファイルの変更を確認する。

        Returns:
            bool: ファイルの変更が検知された場合 ``True``。変更が完了した
            ことを示すものではない点に注意。
        """
        m_time = os.path.getmtime(self.filename)

        if m_time != self.modified_time:
            self.modified_time = m_time
            if self.verbose:
                print(self.filename, "が変更されました。")
            return True
            
        return False


