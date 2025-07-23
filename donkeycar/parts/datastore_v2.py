import json
import mmap
import os
import time
import logging
from pathlib import Path

"""datastore_v2 モジュール
===========================

改良されたカタログ形式のデータストアを提供する。
"""

logger = logging.getLogger(__name__)


NEWLINE = '\n'
NEWLINE_STRIP = '\r\n'


class Seekable(object):
    """改行区切りのレコードを扱うシーク可能なファイルリーダー兼ライター。

    行長のインデックスを保持することで任意の行へのシークを ``O(1)`` で実行できる。
    """

    def __init__(self, file, read_only=False, line_lengths=list()):
        self.line_lengths = list()
        self.cumulative_lengths = list()
        self.method = 'r' if read_only else 'a+'
        self.file = open(file, self.method, newline=NEWLINE)
        # ファイルが読み取り専用の場合はメモリマッピングで性能を向上させる。
        if self.method == 'r':
            self.file = mmap.mmap(self.file.fileno(), length=0,
                                  access=mmap.ACCESS_READ)
        self.total_length = 0
        if len(line_lengths) <= 0:
            self._read_contents()
        else:
            self.line_lengths.extend(line_lengths)
            for line_length in self.line_lengths:
                self.total_length += line_length
                self.cumulative_lengths.append(self.total_length)

    def _read_contents(self):
        self.line_lengths.clear()
        self.cumulative_lengths.clear()
        self.total_length = 0
        self.file.seek(0)
        contents = self.file.readline()
        while len(contents) > 0:
            line_length = len(contents)
            self.line_lengths.append(line_length)
            self.total_length += line_length
            self.cumulative_lengths.append(self.total_length)
            contents = self.file.readline()
        self.seek_end_of_file()

    def __enter__(self):
        return self

    def writeline(self, contents):
        if self.method == 'r':
            raise RuntimeError(f'Seekable {self.file} は読み取り専用です。')

        has_newline = contents[-1] == NEWLINE
        if has_newline:
            line = contents
        else:
            line = f'{contents}{NEWLINE}'

        offset = len(line)
        self.total_length += offset
        self.line_lengths.append(offset)
        self.cumulative_lengths.append(self.total_length)
        self.file.write(line)
        self.file.flush()

    def _line_start_offset(self, line_number):
        return self._offset_until(line_number - 1)

    def _line_end_offset(self, line_number):
        return self._offset_until(line_number)

    def _offset_until(self, line_index):
        end_index = line_index - 1
        return self.cumulative_lengths[end_index] \
            if 0 <= end_index < len(self.cumulative_lengths) else 0

    def readline(self):
        contents = self.file.readline()
        # Seekable がメモリマップされたファイルの場合、readline() は `bytes` を返す
        if isinstance(contents, bytes):
            contents = contents.decode(encoding='utf-8')
        return contents.rstrip(NEWLINE_STRIP)

    def seek_line_start(self, line_number):
        self.file.seek(self._line_start_offset(line_number))

    def seek_end_of_file(self):
        self.file.seek(self.total_length)

    def truncate_until_end(self, line_number):
        self.line_lengths = self.line_lengths[:line_number]
        self.cumulative_lengths = self.cumulative_lengths[:line_number]
        self.total_length = self.cumulative_lengths[-1] \
            if len(self.cumulative_lengths) > 0 else 0
        self.seek_end_of_file()
        self.file.truncate()
    
    def read_from(self, line_number):
        current_offset = self.file.tell()
        self.seek_line_start(line_number)
        lines = list()
        contents = self.readline()
        while len(contents) > 0:
            lines.append(contents)
            contents = self.readline()
        
        self.file.seek(current_offset)
        return lines
    
    def update_line(self, line_number, contents):
        lines = self.read_from(line_number)
        length = len(lines)
        self.truncate_until_end(line_number - 1)
        self.writeline(contents)
        if length > 1:
            for line in lines[1:]:
                self.writeline(line)

    def lines(self):
        return len(self.line_lengths)

    def has_content(self):
        return self.lines() > 0

    def close(self):
        self.file.close()

    def __exit__(self, type, value, traceback):
        self.close()


class Catalog(object):
    """    改行区切りのレコードを保持するファイル。

    [ json object record ] \n
    [ json object record ] \n
    """
    def __init__(self, path, read_only=False, start_index=0):
        self.path = Path(os.path.expanduser(path))
        self.manifest = CatalogMetadata(self.path,
                                        read_only=read_only,
                                        start_index=start_index)
        self.seekable = Seekable(self.path.as_posix(),
                                 line_lengths=self.manifest.line_lengths(),
                                 read_only=read_only)

    def _exit_handler(self):
        self.close()

    def write_record(self, record):
        # レコードを追加しマニフェストを更新する
        contents = json.dumps(record, allow_nan=False, sort_keys=True)
        self.seekable.writeline(contents)
        line_lengths = self.seekable.line_lengths
        self.manifest.update_line_lengths(line_lengths)

    def close(self):
        self.manifest.close()
        self.seekable.close()


class CatalogMetadata(object):
    """カタログのメタデータを管理するクラス。"""
    def __init__(self, catalog_path, read_only=False, start_index=0):
        path = Path(catalog_path)
        manifest_name = f'{path.stem}.catalog_manifest'
        self.manifest_path = Path(os.path.join(path.parent.as_posix(),
                                               manifest_name))
        self.seekeable = Seekable(self.manifest_path, read_only=read_only)
        has_contents = False
        if os.path.exists(self.manifest_path) and self.seekeable.has_content():
            self.seekeable.seek_line_start(1)
            contents = self.seekeable.readline()
            if contents:
                self.contents = json.loads(contents)
                has_contents = True

        if not has_contents:
            # 新しいカタログメタデータのエントリ
            self.contents = dict()
            self.contents['path'] = self.manifest_path.name
            created_at = time.time()
            self.contents['created_at'] = created_at
            self.contents['start_index'] = start_index
            self.contents['line_lengths'] = list()
            self._update()

    def update_line_lengths(self, new_lengths):
        self.contents['line_lengths'] = new_lengths
        self._update()

    def line_lengths(self):
        return self.contents['line_lengths']

    def start_index(self):
        return self.contents['start_index']

    def _update(self):
        contents = json.dumps(self.contents, allow_nan=False, sort_keys=True)
        self.seekeable.truncate_until_end(0)
        self.seekeable.writeline(contents)

    def close(self):
        self.seekeable.close()


class Manifest(object):
    """複数のカタログを管理するマニフェスト。
    
    [ json array of inputs ]\n
    [ json array of types ]\n
    [ json object with user metadata ]\n
    [ json object with manifest metadata ]\n
    [ json object with catalog metadata ]\n
    """

    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_len=1000, read_only=False):
        self.base_path = Path(os.path.expanduser(base_path)).absolute()
        self.manifest_path = Path(os.path.join(self.base_path, 'manifest.json'))
        self.inputs = inputs
        self.types = types
        self._read_metadata(metadata)
        self.manifest_metadata = dict()
        self.max_len = max_len
        self.read_only = read_only
        self.current_catalog = None
        self.current_index = 0
        self.catalog_paths = list()
        self.catalog_metadata = dict()
        self.deleted_indexes = set()
        self._updated_session = False
        has_catalogs = False

        if self.manifest_path.exists():
            self.seekeable = Seekable(self.manifest_path, read_only=self.read_only)
            if self.seekeable.has_content():
                self._read_contents()
            has_catalogs = len(self.catalog_paths) > 0

        else:
            created_at = time.time()
            self.manifest_metadata['created_at'] = created_at
            if not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)
                print(f'新しいデータストアを作成しました: {self.base_path.as_posix()}')
            self.seekeable = Seekable(self.manifest_path, read_only=self.read_only)

        if not has_catalogs:
            self._write_contents()
            self._add_catalog()
        else:
            last_known_catalog = os.path.join(self.base_path,
                                              self.catalog_paths[-1])
            print(f'カタログ {last_known_catalog} を使用します')
            self.current_catalog = Catalog(last_known_catalog,
                                           read_only=self.read_only,
                                           start_index=self.current_index)
        # 各レコードに追加される新しい session_id を生成する
        # Tub.write_record() が呼ばれた際に使用される
        self.session_id = self.create_new_session()

    def write_record(self, record):
        new_catalog = self.current_index > 0 \
                      and (self.current_index % self.max_len) == 0
        if new_catalog:
            self._add_catalog()

        self.current_catalog.write_record(record)
        self.current_index += 1
        # 最後のインデックスを記録するためメタデータを更新
        self._update_catalog_metadata(update=True)
        # このメソッドが一度でも呼ばれたら session_id 更新フラグを立てる
        # セッション終了時に session_id のメタデータが更新される
        if not self._updated_session:
            self._updated_session = True

    def delete_records(self, record_indexes):
        # 実際にはレコードを削除せず、削除済みとしてマークする
        if isinstance(record_indexes, int):
            record_indexes = {record_indexes}
        self.deleted_indexes.update(record_indexes)
        self._update_catalog_metadata(update=True)

    def restore_records(self, record_indexes):
        # 実際にはレコードを削除せず、削除マークを解除する
        if isinstance(record_indexes, int):
            record_indexes = {record_indexes}
        self.deleted_indexes.difference_update(record_indexes)
        self._update_catalog_metadata(update=True)

    def _add_catalog(self):
        current_length = len(self.catalog_paths)
        catalog_name = f'catalog_{current_length}.catalog'
        catalog_path = os.path.join(self.base_path, catalog_name)
        current_catalog = self.current_catalog
        self.current_catalog = Catalog(catalog_path,
                                       start_index=self.current_index,
                                       read_only=self.read_only)
        # 相対パスを保存
        self.catalog_paths.append(catalog_name)
        self._update_catalog_metadata(update=True)
        if current_catalog:
            current_catalog.close()

    def _read_metadata(self, metadata=[]):
        self.metadata = dict()
        for kv in metadata:
            kvs = kv.split(":")
            if len(kvs) == 2:
                self.metadata[kvs[0]] = kvs[1]
            else:
                logger.error(
                    f'Metadata は key:value 形式のキー・バリューの組で指定する必要があります。{kv} を無視します')

    def _read_contents(self):
        self.seekeable.seek_line_start(1)
        self.inputs = json.loads(self.seekeable.readline())
        self.types = json.loads(self.seekeable.readline())
        self.metadata = json.loads(self.seekeable.readline())
        self.manifest_metadata = json.loads(self.seekeable.readline())
        # カタログのメタデータ
        catalog_metadata = json.loads(self.seekeable.readline())
        self.catalog_paths = catalog_metadata['paths']
        self.current_index = catalog_metadata['current_index']
        self.max_len = catalog_metadata['max_len']
        self.deleted_indexes = set(catalog_metadata['deleted_indexes'])

    def _write_contents(self):
        self.seekeable.truncate_until_end(0)
        self.seekeable.writeline(json.dumps(self.inputs))
        self.seekeable.writeline(json.dumps(self.types))
        self.seekeable.writeline(json.dumps(self.metadata))
        self.seekeable.writeline(json.dumps(self.manifest_metadata))
        self._update_catalog_metadata(update=False)

    def _update_catalog_metadata(self, update=True):
        if update:
            self.seekeable.truncate_until_end(4)
        # カタログのメタデータ
        catalog_metadata = dict()
        catalog_metadata['paths'] = self.catalog_paths
        catalog_metadata['current_index'] = self.current_index
        catalog_metadata['max_len'] = self.max_len
        catalog_metadata['deleted_indexes'] = list(self.deleted_indexes)
        self.catalog_metadata = catalog_metadata
        self.seekeable.writeline(json.dumps(catalog_metadata))

    def create_new_session(self):
        """新しいセッション ID を生成してメタデータへ追加する。

        Returns:
            str: 生成されたセッション ID。
        """
        sessions = self.manifest_metadata.get('sessions', {})
        last_id = -1
        if sessions:
            last_id = sessions['last_id']
        else:
            sessions['all_full_ids'] = []
        this_id = last_id + 1
        date = time.strftime('%y-%m-%d')
        this_full_id = date + '_' + str(this_id)
        sessions['last_id'] = this_id
        sessions['last_full_id'] = this_full_id
        sessions['all_full_ids'].append(this_full_id)
        self.manifest_metadata['sessions'] = sessions
        return this_full_id

    def close(self):
        """カタログ・マニフェスト・manifest.json のファイルを閉じる。

        レコードが存在した場合は更新された session_id を manifest.json に書き戻す。
        """
        # レコードがあった場合は session_id の辞書を更新してメタデータへ書き込む
        # レコードが無い場合は session_id 情報を変更しない
        if self._updated_session:
            self.seekeable.update_line(4, json.dumps(self.manifest_metadata))
        self.current_catalog.close()
        self.seekeable.close()

    def __iter__(self):
        return ManifestIterator(self)

    def __len__(self):
        # current_index は既に次のインデックスを指している
        return self.current_index - len(self.deleted_indexes)


class ManifestIterator(object):
    """Manifest 用のイテレータ。

    ``__next__()`` が呼ばれたときにカタログエントリを遅延して返す。
    """
    def __init__(self, manifest):
        self.manifest = manifest
        self.has_catalogs = len(self.manifest.catalog_paths) > 0
        self.current_index = 0
        self.current_catalog_index = 0
        self.current_catalog = None

    def __next__(self):
        while True:
            if not self.has_catalogs:
                raise StopIteration('カタログがありません')

            if self.current_catalog_index >= len(self.manifest.catalog_paths):
                raise StopIteration('これ以上のカタログはありません')

            if self.current_catalog is None:
                current_catalog_path = os.path.join(
                    self.manifest.base_path,
                    self.manifest.catalog_paths[self.current_catalog_index])
                self.current_catalog = Catalog(current_catalog_path,
                                               read_only=self.manifest.read_only)
                self.current_catalog.seekable.seek_line_start(1)

            contents = self.current_catalog.seekable.readline()
            if contents is not None and len(contents) > 0:
                # 先に進める準備ができたときに current_index を確認する
                # 下層のイテレータを進めるための処理
                current_index = self.current_index
                self.current_index += 1
                if current_index in self.manifest.deleted_indexes:
                    # 削除マークされているインデックスはスキップ
                    continue
                else:
                    try:
                        record = json.loads(contents)
                        return record
                    except Exception:
                        print(f'インデックス {current_index} のレコードを無視します')
                        continue
            else:
                self.current_catalog = None
                self.current_catalog_index += 1

    next = __next__

    def __len__(self):
        return self.manifest.__len__()
