#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# すべての機能が移行された時点で削除予定のアーカイブコード。
"""データストアを提供するモジュール。

センサー値や画像を保存・読み込みするためのクラス群を含む。
"""
import datetime
import glob
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image


class Tub(object):
    """キーと値の形式でセンサー情報を保存するデータストア。

    `str`、`int`、`float`、`image_array`、`image`、`array` 型に対応する。

    例:
        >>> path = '~/mydonkey/test_tub'
        >>> inputs = ['user/speed', 'cam/image']
        >>> types = ['float', 'image']
        >>> t = Tub(path=path, inputs=inputs, types=types)
    """

    def __init__(self, path, inputs=None, types=None, user_meta=[]):

        self.path = os.path.expanduser(path)
        # print('tub のパス:', self.path)
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.exclude_path = os.path.join(self.path, "exclude.json")
        self.df = None

        exists = os.path.exists(self.path)

        if exists:
            # ログとメタ情報を読み込む
            # print("Tub は存在します: {}".format(self.path))
            try:
                with open(self.meta_path, 'r') as f:
                    self.meta = json.load(f)
            except FileNotFoundError:
                self.meta = {'inputs': [], 'types': []}

            try:
                with open(self.exclude_path,'r') as f:
                    excl = json.load(f) # リストとして保存されている
                    self.exclude = set(excl)
            except FileNotFoundError:
                self.exclude = set()

            try:
                self.current_ix = self.get_last_ix() + 1
            except ValueError:
                self.current_ix = 0

            if 'start' in self.meta:
                self.start_time = self.meta['start']
            else:
                self.start_time = time.time()
                self.meta['start'] = self.start_time

        elif not exists and inputs:
            print('Tubが存在しません。新しいTubを作成します...')
            self.start_time = time.time()
            # ログを作成してメタ情報を保存
            os.makedirs(self.path)
            self.meta = {'inputs': inputs, 'types': types, 'start': self.start_time}
            for kv in user_meta:
                kvs = kv.split(":")
                if len(kvs) == 2:
                    self.meta[kvs[0]] = kvs[1]
                # それ以外は例外？メッセージを表示？
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f)
            self.current_ix = 0
            self.exclude = set()
            print('新しいTubを作成しました: {}'.format(self.path))
        else:
            msg = "指定されたTubのパスが存在せず、新しいTubを作成するためのメタ情報" \
                  "(inputs と types) が与えられていません。Tub のパスを確認するか、" \
                  "メタ情報を指定してください。"

            raise AttributeError(msg)

    def get_last_ix(self):
        index = self.get_index()           
        return max(index)

    def update_df(self):
        import pandas as pd
        df = pd.DataFrame([self.get_json_record(i)
                           for i in self.get_index(shuffled=False)])
        self.df = df

    def get_df(self):
        if self.df is None:
            self.update_df()
        return self.df

    def get_index(self, shuffled=True):
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6] == 'record']

        def get_file_ix(file_name):
            try:
                name = file_name.split('.')[0]
                num = int(name.split('_')[1])
            except:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]
        
        if shuffled:
            random.shuffle(nums)
        else:
            nums = sorted(nums)
            
        return nums 

    @property
    def inputs(self):
        return list(self.meta['inputs'])

    @property
    def types(self):
        return list(self.meta['types'])

    def get_input_type(self, key):
        input_types = dict(zip(self.inputs, self.types))
        return input_types.get(key)

    def write_json_record(self, json_data):
        path = self.get_json_record_path(self.current_ix)
        try:
            with open(path, 'w') as fp:
                json.dump(json_data, fp)

        except TypeError:
            print('レコード処理で問題が発生しました:', json_data)
        except FileNotFoundError:
            raise
        except:
            print('予期せぬエラー:', sys.exc_info()[0])
            raise

    def get_num_records(self):
        import glob
        files = glob.glob(os.path.join(self.path, 'record_*.json'))
        return len(files)

    def make_record_paths_absolute(self, record_dict):
        # パスを絶対パスに変換
        d = {}
        for k, v in record_dict.items():
            if type(v) == str: # ファイル名
                if '.' in v:
                    v = os.path.join(self.path, v)
            d[k] = v

        return d

    def check(self, fix=False):
        """すべてのレコードを読み込めるか確認する。

        Args:
            fix (bool): ``True`` の場合、問題のあるレコードを削除する。
        """
        print('Tubをチェック中:%s.' % self.path)
        print('%d 件のレコードが見つかりました。' % self.get_num_records())
        problems = False
        for ix in self.get_index(shuffled=False):
            try:
                self.get_record(ix)
            except:
                problems = True
                if fix == False:
                    print('レコードに問題があります:', self.path, ix)
                else:
                    print('問題のあるレコードを削除します:', self.path, ix)
                    self.remove_record(ix)
        if not problems:
            print('問題は見つかりませんでした。')

    def remove_record(self, ix):
        """指定したレコードに関連するデータを削除する。"""
        record = self.get_json_record_path(ix)
        os.unlink(record)

    def put_record(self, data):
        """画像などCSVに保存できない値を保存する。

        返り値として、保存した値への参照を含むレコードを返す。
        """
        json_data = {}
        self.current_ix += 1
        
        for key, val in data.items():
            typ = self.get_input_type(key)

            if (val is not None) and (typ == 'float'):
                # json が扱えない numpy.float32 などの場合に備える
                json_data[key] = float(val)

            elif typ in ['str', 'float', 'int', 'boolean', 'vector']:
                json_data[key] = val

            elif typ is 'image':
                path = self.make_file_path(key)
                val.save(path)
                json_data[key]=path

            elif typ == 'image_array':
                img = Image.fromarray(np.uint8(val))
                name = self.make_file_name(key, ext='.jpg')
                img.save(os.path.join(self.path, name))
                json_data[key]=name

            elif typ == 'gray16_array':
                # np.uint16 を 16bit の PNG として保存する
                img = Image.fromarray(np.uint16(val))
                name = self.make_file_name(key, ext='.png')
                img.save(os.path.join(self.path, name))
                json_data[key]=name
                
            elif typ == 'nparray':
                # numpy 配列を Python オブジェクトに変換して JSON で扱えるようにする
                json_data[key] = val.tolist()

            else:
                msg = 'Tubはこの型 {} の扱い方がわかりません。'.format(typ)
                raise TypeError(msg)

        json_data['milliseconds'] = int((time.time() - self.start_time) * 1000)

        self.write_json_record(json_data)
        return self.current_ix

    def erase_last_n_records(self, num_erase):
        """指定数のレコードを削除し、現在のインデックスを巻き戻す。"""
        last_erase = self.current_ix
        first_erase = last_erase - num_erase
        self.current_ix = first_erase - 1
        if self.current_ix < 0:
            self.current_ix = 0

        for i in range(first_erase, last_erase):
            if i < 0:
                continue
            self.erase_record(i)

    def erase_record(self, i):
        json_path = self.get_json_record_path(i)
        if os.path.exists(json_path):
            os.unlink(json_path)
        img_filename = '%d_cam-image_array_.jpg' % i
        img_path = os.path.join(self.path, img_filename)
        if os.path.exists(img_path):
            os.unlink(img_path)

    def get_json_record_path(self, ix):
        return os.path.join(self.path, 'record_' + str(ix) + '.json')

    def get_json_record(self, ix):
        path = self.get_json_record_path(ix)
        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError:
            raise Exception('不正なレコード: %d。`python manage.py check --fix` を実行してください' % ix)
        except FileNotFoundError:
            raise
        except:
            print('予期せぬエラー:', sys.exc_info()[0])
            raise

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict

    def get_record(self, ix):
        json_data = self.get_json_record(ix)
        data = self.read_record(json_data)
        return data

    def read_record(self, record_dict):
        data = {}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)
            # 別ファイルとして保存されたオブジェクトを読み込む
            if typ == 'image_array':
                img = Image.open((val))
                val = np.array(img)
            data[key] = val
        return data

    def gather_records(self):
        ri = lambda fnm: int(os.path.basename(fnm).split('_')[1].split('.')[0])
        record_paths = glob.glob(os.path.join(self.path, 'record_*.json'))
        if len(self.exclude) > 0:
            record_paths = [f for f in record_paths if ri(f) not in self.exclude]
        record_paths.sort(key=ri)
        return record_paths

    def make_file_name(self, key, ext='.png', ix=None):
        this_ix = ix
        if this_ix is None:
            this_ix = self.current_ix
        name = '_'.join([str(this_ix), key, ext])
        name = name.replace('/', '-')
        return name

    def delete(self):
        """このTubのフォルダーおよびファイルを削除する。"""
        import shutil
        shutil.rmtree(self.path)

    def shutdown(self):
        pass

    def excluded(self, index):
        return index in self.exclude

    def exclude_index(self, index):
        self.exclude.add(index)

    def include_index(self, index):
        try:
            self.exclude.remove(index)
        except:
            pass

    def write_exclude(self):
        if 0 == len(self.exclude):
            # 除外セットが空なら空のファイルを残さないようにする
            if os.path.exists(self.exclude_path):
                os.unlink(self.exclude_path)
        else:
            with open(self.exclude_path, 'w') as f:
                json.dump(list(self.exclude), f)


class TubWriter(Tub):
    def __init__(self, *args, **kwargs):
        super(TubWriter, self).__init__(*args, **kwargs)

    def run(self, *args):
        """Donkeyパーツとして値を受け取り保存する。"""
        assert len(self.inputs) == len(args)
        record = dict(zip(self.inputs, args))
        self.put_record(record)
        return self.current_ix


class TubReader(Tub):
    def __init__(self, path, *args, **kwargs):
        super(TubReader, self).__init__(*args, **kwargs)

    def run(self, *args):
        """Donkeyパーツとして指定されたキーの値を順に取得する。"""
        record_dict = self.get_record()
        record = [record_dict[key] for key in args]
        return record


class TubHandler:
    """複数のTubを作成・管理するヘルパークラス。"""

    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def get_tub_list(self, path):
        folders = next(os.walk(path))[1]
        return folders

    def next_tub_number(self, path):
        def get_tub_num(tub_name):
            try:
                num = int(tub_name.split('_')[1])
            except:
                num = 0
            return num

        folders = self.get_tub_list(path)
        numbers = [get_tub_num(x) for x in folders]
        next_number = max(numbers+[0]) + 1
        return next_number

    def create_tub_path(self):
        tub_num = self.next_tub_number(self.path)
        date = datetime.datetime.now().strftime('%y-%m-%d')
        name = '_'.join(['tub', str(tub_num), date])
        tub_path = os.path.join(self.path, name)
        return tub_path

    def new_tub_writer(self, inputs, types, user_meta=[]):
        tub_path = self.create_tub_path()
        tw = TubWriter(path=tub_path, inputs=inputs, types=types, user_meta=user_meta)
        return tw


class TubImageStacker(Tub):
    """直近3レコードの画像を重ねて1枚の画像として扱うTub。

    単純なフィードフォワード型NNでも動きを学習できるように、過去2枚と現在の
    1枚を3チャンネルに配置して保存する。ImageFIFOパーツを使用している場合は
    このクラスは不要だが、推論時には同じ方式で画像を供給する必要がある。
    """
    
    def rgb2gray(self, rgb):
        """RGB画像をグレースケール1チャンネルに変換する。"""
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def stack3Images(self, img_a, img_b, img_c):
        """3枚のRGB画像をグレースケール化し3チャンネルに配置する。"""
        width, height, _ = img_a.shape

        gray_a = self.rgb2gray(img_a)
        gray_b = self.rgb2gray(img_b)
        gray_c = self.rgb2gray(img_c)
        
        img_arr = np.zeros([width, height, 3], dtype=np.dtype('B'))

        img_arr[...,0] = np.reshape(gray_a, (width, height))
        img_arr[...,1] = np.reshape(gray_b, (width, height))
        img_arr[...,2] = np.reshape(gray_c, (width, height))

        return img_arr

    def get_record(self, ix):
        """現在のレコードと直前2レコードを取得し画像を合成する。"""
        data = super(TubImageStacker, self).get_record(ix)

        if ix > 1:
            data_ch1 = super(TubImageStacker, self).get_record(ix - 1)
            data_ch0 = super(TubImageStacker, self).get_record(ix - 2)

            json_data = self.get_json_record(ix)
            for key, val in json_data.items():
                typ = self.get_input_type(key)

                # 別ファイルとして保存されたオブジェクトを読み込む
                if typ == 'image':
                    val = self.stack3Images(data_ch0[key], data_ch1[key], data[key])
                    data[key] = val
                elif typ == 'image_array':
                    img = self.stack3Images(data_ch0[key], data_ch1[key], data[key])
                    val = np.array(img)

        return data


class TubTimeStacker(TubImageStacker):
    """時間方向にレコードを積み重ねて学習するためのTub。

    ネットワークに先を見越した判断を学習させるため、現在からのフレーム
    オフセットを指定して複数のレコードをまとめて返す。
    """

    def __init__(self, frame_list, *args, **kwargs):
        """フレームオフセットのリストを受け取り画像を時系列で積む。"""
        super(TubTimeStacker, self).__init__(*args, **kwargs)
        self.frame_list = frame_list
  
    def get_record(self, ix):
        """複数のレコードを時系列でまとめたデータを返す。"""
        data = {}
        for i, iOffset in enumerate(self.frame_list):
            iRec = ix + iOffset
            
            try:
                json_data = self.get_json_record(iRec)
            except FileNotFoundError:
                pass
            except:
                pass

            for key, val in json_data.items():
                typ = self.get_input_type(key)

                # 最初の画像だけを別ファイルから読み込む
                if typ == 'image' and i == 0:
                    val = Image.open(os.path.join(self.path, val))
                    data[key] = val                    
                elif typ == 'image_array' and i == 0:
                    d = super(TubTimeStacker, self).get_record(ix)
                    data[key] = d[key]
                else:
                    '''
                    キーに ``_offset`` を付与して
                    例えば ``user/angle`` が ``user/angle_0`` になる
                    '''
                    new_key = key + "_" + str(iOffset)
                    data[new_key] = val
        return data


class TubGroup(Tub):
    """複数のTubをまとめて扱うためのクラス。"""

    def __init__(self, tub_paths):
        import pandas as pd

        tub_paths = self.resolve_tub_paths(tub_paths)
        print('TubGroup: tubのパス:', tub_paths)
        tubs = [Tub(path) for path in tub_paths]
        self.input_types = {}

        record_count = 0
        for t in tubs:
            t.update_df()
            record_count += len(t.df)
            self.input_types.update(dict(zip(t.inputs, t.types)))

        print('tubを結合しています。合計 {} 件のレコード。完了まで {} 分ほどかかる可能性があります.'
              .format(record_count, int(record_count / 300000)))

        self.meta = {'inputs': list(self.input_types.keys()),
                     'types': list(self.input_types.values())}

        self.df = pd.concat([t.df for t in tubs], axis=0, join='inner')

    def find_tub_paths(self, path):
        matches = []
        path = os.path.expanduser(path)
        for file in glob.glob(path):
            if os.path.isdir(file):
                matches.append(os.path.join(os.path.abspath(file)))
        return matches

    def resolve_tub_paths(self, path_list):
        path_list = path_list.split(",")
        resolved_paths = []
        for path in path_list:
            paths = self.find_tub_paths(path)
            resolved_paths += paths
        return resolved_paths
