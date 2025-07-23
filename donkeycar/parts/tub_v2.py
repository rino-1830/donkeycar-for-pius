"""Tub v2 モジュール。

センサー データの保存および操作を行うクラスを提供する。
"""

import atexit
import os
import time
from datetime import datetime
import json

import numpy as np
from PIL import Image

from donkeycar.parts.datastore_v2 import Manifest, ManifestIterator


class Tub(object):
    """センサー データをキーと値の形式で保存するデータストア。

    str、int、float、image_array、image、array の各データ型を扱う。
    """

    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000, read_only=False):
        """コンストラクタ。

        Args:
            base_path: Tub を保存するベースディレクトリ。
            inputs: 入力項目名のリスト。
            types: 各入力のデータ型。
            metadata: 追加メタデータのリスト。
            max_catalog_len: マニフェストの最大長。
            read_only: 読み込み専用モードかどうか。
        """
        self.base_path = base_path
        self.images_base_path = os.path.join(self.base_path, Tub.images())
        self.inputs = inputs
        self.types = types
        self.metadata = metadata
        self.manifest = Manifest(base_path, inputs=inputs, types=types,
                                 metadata=metadata, max_len=max_catalog_len,
                                 read_only=read_only)
        self.input_types = dict(zip(self.inputs, self.types))
        # 必要に応じて images フォルダを作成する
        if not os.path.exists(self.images_base_path):
            os.makedirs(self.images_base_path, exist_ok=True)

    def write_record(self, record=None):
        """画像を含むさまざまなデータ型を処理できる。"""
        contents = dict()
        for key, value in record.items():
            if value is None:
                continue
            elif key not in self.input_types:
                continue
            else:
                input_type = self.input_types[key]
                if input_type == 'float':
                    # np.float() 型を適切に処理する
                    contents[key] = float(value)
                elif input_type == 'str':
                    contents[key] = value
                elif input_type == 'int':
                    contents[key] = int(value)
                elif input_type == 'boolean':
                    contents[key] = bool(value)
                elif input_type == 'nparray':
                    contents[key] = value.tolist()
                elif input_type == 'list' or input_type == 'vector':
                    contents[key] = list(value)
                elif input_type == 'image_array':
                    # 画像配列を処理する
                    image = Image.fromarray(np.uint8(value))
                    name = Tub._image_file_name(self.manifest.current_index, key)
                    image_path = os.path.join(self.images_base_path, name)
                    image.save(image_path)
                    contents[key] = name
                elif input_type == 'gray16_array':
                    # np.uint16 を 16bit PNG として保存する
                    image = Image.fromarray(np.uint16(value))
                    name = Tub._image_file_name(self.manifest.current_index, key, ext='.png')
                    image_path = os.path.join(self.images_base_path, name)
                    image.save(image_path)
                    contents[key]=name

        # プライベートなプロパティ
        contents['_timestamp_ms'] = int(round(time.time() * 1000))
        contents['_index'] = self.manifest.current_index
        contents['_session_id'] = self.manifest.session_id

        self.manifest.write_record(contents)

    def delete_records(self, record_indexes):
        self.manifest.delete_records(record_indexes)

    def delete_last_n_records(self, n):
        # 削除されていないインデックスを順序付きで作成する
        all_alive_indexes = sorted(set(range(self.manifest.current_index))
                                   - self.manifest.deleted_indexes)
        to_delete_indexes = all_alive_indexes[-n:]
        self.manifest.delete_records(to_delete_indexes)

    def restore_records(self, record_indexes):
        self.manifest.restore_records(record_indexes)

    def close(self):
        self.manifest.close()

    def __iter__(self):
        return ManifestIterator(self.manifest)

    def __len__(self):
        return self.manifest.__len__()

    @classmethod
    def images(cls):
        return 'images'

    @classmethod
    def _image_file_name(cls, index, key, extension='.jpg'):
        key_prefix = key.replace('/', '_')
        name = '_'.join([str(index), key_prefix, extension])
        # ポータビリティ確保のため相対パスを返す
        return name


class TubWriter(object):
    """データストアへレコードを書き込む Donkey パーツ。"""
    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000):
        self.tub = Tub(base_path, inputs, types, metadata, max_catalog_len)

    def run(self, *args):
        assert len(self.tub.inputs) == len(args), \
            f'入力は {len(self.tub.inputs)} 個必要ですが、{len(args)} 個渡されました'
        record = dict(zip(self.tub.inputs, args))
        self.tub.write_record(record)
        return self.tub.manifest.current_index

    def __iter__(self):
        return self.tub.__iter__()

    def close(self):
        self.tub.close()

    def shutdown(self):
        self.close()


class TubWiper:
    """Tub の末尾から複数のレコードを削除する Donkey パーツ。

    録画中に不正なデータを取り除くために使う。車両ループで呼び出される
    ため、削除は連続したアクティベーション中 1 回のみ実行される。
    再度実行するには入力トリガーを一度解除する必要があり、さもないと複
    数回実行される可能性がある。
    """
    def __init__(self, tub, num_records=20):
        """コンストラクタ。

        Args:
            tub: 操作対象の tub。
            num_records: 削除するレコード数。
        """
        self._tub = tub
        self._num_records = num_records
        self._active_loop = False

    def run(self, is_delete):
        """車両ループ内で呼び出されるメソッド。

        トリガーが ``False`` から ``True`` に変わったときにのみレコードを
        削除する。

        Args:
            is_delete: 呼び出し側が削除をトリガーしたかどうか。
        """
        # 入力が真でデバウンスされている場合のみ実行する
        if is_delete:
            if not self._active_loop:
                # 実行コマンド
                self._tub.delete_last_n_records(self._num_records)
                # ループカウンタを増やす
                self._active_loop = True
        else:
            # トリガーが解除されたらアクティブループをリセットする
            self._active_loop = False