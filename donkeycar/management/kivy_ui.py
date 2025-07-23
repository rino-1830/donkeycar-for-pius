"""Kivy を用いた DonkeyCar 管理 UI モジュール."""

import json
import re
import time
from copy import copy, deepcopy
from datetime import datetime
from functools import partial
from subprocess import Popen, PIPE, STDOUT
from threading import Thread
from collections import namedtuple
from kivy.logger import Logger

import io
import os
import atexit
import yaml
from PIL import Image as PilImage
import pandas as pd
import numpy as np
import plotly.express as px
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.properties import NumericProperty, ObjectProperty, StringProperty, \
    ListProperty, BooleanProperty
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import SpinnerOption, Spinner

from donkeycar import load_config
from donkeycar.parts.image_transformations import ImageTransformations
from donkeycar.parts.tub_v2 import Tub
from donkeycar.pipeline.augmentations import ImageAugmentation
from donkeycar.pipeline.database import PilotDatabase
from donkeycar.pipeline.types import TubRecord
from donkeycar.utils import get_model_by_type
from donkeycar.pipeline.training import train

Logger.propagate = False

Builder.load_file(os.path.join(os.path.dirname(__file__), 'ui.kv'))
Window.clearcolor = (0.2, 0.2, 0.2, 1)
LABEL_SPINNER_TEXT = '追加/削除'

# プログレスバーに表示するフィールドの情報を保持するデータ構造。
# フィールド名、設定ファイル内の最大値の識別子、中央寄せかどうかを含む。
FieldProperty = namedtuple('FieldProperty',
                           ['field', 'max_value_id', 'centered'])


def get_norm_value(value, cfg, field_property, normalised=True):
    """値を正規化または逆正規化して返す。

    Args:
        value (float): 変換対象の値。
        cfg: 設定オブジェクト。
        field_property (FieldProperty): フィールドの設定。
        normalised (bool): ``True`` なら値を正規化、``False`` なら元に戻す。

    Returns:
        float: 変換後の値。
    """
    max_val_key = field_property.max_value_id
    max_value = getattr(cfg, max_val_key, 1.0)
    out_val = value / max_value if normalised else value * max_value
    return out_val


def tub_screen():
    """現在実行中のアプリから ``TubScreen`` を取得する。"""
    return App.get_running_app().tub_screen if App.get_running_app() else None


def pilot_screen():
    """現在実行中のアプリから ``PilotScreen`` を取得する。"""
    return App.get_running_app().pilot_screen if App.get_running_app() else None


def train_screen():
    """現在実行中のアプリから ``TrainScreen`` を取得する。"""
    return App.get_running_app().train_screen if App.get_running_app() else None


def car_screen():
    """現在実行中のアプリから ``CarScreen`` を取得する。"""
    return App.get_running_app().car_screen if App.get_running_app() else None


def recursive_update(target, source):
    """辞書を再帰的に更新する。

    Args:
        target (dict): 更新対象の辞書。
        source (dict): 変更を含む辞書。

    Returns:
        bool: 両方が辞書なら ``True``、それ以外は ``False``。
    """
    if isinstance(target, dict) and isinstance(source, dict):
        for k, v in source.items():
            v_t = target.get(k)
            if not recursive_update(v_t, v):
                target[k] = v
        return True
    else:
        return False


def decompose(field):
    """'gyroscope_1' のようなベクトル名を ``('gyroscope', 1)`` に分解する。"""
    field_split = field.split('_')
    if len(field_split) > 1 and field_split[-1].isdigit():
        return '_'.join(field_split[:-1]), int(field_split[-1])
    return field, None


class RcFileHandler:
    """フィールドの表示設定や最後に開いたディレクトリなどを保存する設定ファイルを扱うクラス。"""

    # これらのエントリーはすべての Tub に存在するため、設定ファイルには不要
    known_entries = [
        FieldProperty('user/angle', '', centered=True),
        FieldProperty('user/throttle', '', centered=False),
        FieldProperty('pilot/angle', '', centered=True),
        FieldProperty('pilot/throttle', '', centered=False),
    ]

    def __init__(self, file_path='~/.donkeyrc'):
        """インスタンスを初期化する。

        Args:
            file_path (str): 設定ファイルのパス。
        """
        self.file_path = os.path.expanduser(file_path)
        self.data = self.create_data()
        recursive_update(self.data, self.read_file())
        self.field_properties = self.create_field_properties()

        def exit_hook():
            self.write_file()
        # プログラム終了時に自動で設定を保存する
        atexit.register(exit_hook)

    def create_field_properties(self):
        """既知のフィールド設定とファイルの設定を統合する。"""
        field_properties = {entry.field: entry for entry in self.known_entries}
        field_list = self.data.get('field_mapping')
        if field_list is None:
            field_list = {}
        for entry in field_list:
            assert isinstance(entry, dict), \
                'field_mapping の各エントリーには辞書が必要です'
            field_property = FieldProperty(**entry)
            field_properties[field_property.field] = field_property
        return field_properties

    def create_data(self):
        """初期データを生成する。"""
        data = dict()
        data['user_pilot_map'] = {'user/throttle': 'pilot/throttle',
                                  'user/angle': 'pilot/angle'}
        return data

    def read_file(self):
        """設定ファイルを読み込む。"""
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                Logger.info(f'Donkeyrc: 設定ファイル {self.file_path} を読み込みました')
                return data
        else:
            Logger.warn(f'Donkeyrc: 設定ファイル {self.file_path} が存在しません')
            return {}

    def write_file(self):
        """現在の設定をファイルへ書き出す。"""
        if os.path.exists(self.file_path):
            Logger.info(f'Donkeyrc: 設定ファイル {self.file_path} を更新しました')
        with open(self.file_path, mode='w') as f:
            self.data['time_stamp'] = datetime.now()
            data = yaml.dump(self.data, f)
            return data


rc_handler = RcFileHandler()


class MySpinnerOption(SpinnerOption):
    """ Customization for Spinner """
    pass


class MySpinner(Spinner):
    """ Customization of Spinner drop down menu """
    def __init__(self, **kwargs):
        super().__init__(option_cls=MySpinnerOption, **kwargs)


class FileChooserPopup(Popup):
    """ファイル選択用のポップアップウィンドウ。"""
    load = ObjectProperty()
    root_path = StringProperty()
    filters = ListProperty()


class FileChooserBase:
    """ファイル選択ウィジェットの基底クラス。"""
    file_path = StringProperty("ファイルが選択されていません")
    popup = ObjectProperty(None)
    root_path = os.path.expanduser('~')
    title = StringProperty(None)
    filters = ListProperty()

    def open_popup(self):
        self.popup = FileChooserPopup(load=self.load, root_path=self.root_path,
                                      title=self.title, filters=self.filters)
        self.popup.open()

    def load(self, selection):
        """選択したファイルを ``file_path`` に設定し処理を実行する。"""
        self.file_path = str(selection[0])
        self.popup.dismiss()
        self.load_action()

    def load_action(self):
        """``file_path`` 更新時に実行される仮想メソッド。"""
        pass


class ConfigManager(BoxLayout, FileChooserBase):
    """車両ディレクトリから設定ファイルを読み込むクラス。"""
    config = ObjectProperty(None)
    file_path = StringProperty(rc_handler.data.get('car_dir', ''))

    def load_action(self):
        """ Load the config from the file path"""
        if self.file_path:
            try:
                path = os.path.join(self.file_path, 'config.py')
                self.config = load_config(path)
                # 読み込みに成功したら設定に保存
                rc_handler.data['car_dir'] = self.file_path
            except FileNotFoundError:
                Logger.error(f'Config: Directory {self.file_path} has no '
                             f'config.py')
            except Exception as e:
                Logger.error(f'Config: {e}')


class TubLoader(BoxLayout, FileChooserBase):
    """Tub を読み込み直す際に他のウィジェットも更新するクラス。"""
    file_path = StringProperty(rc_handler.data.get('last_tub', ''))
    tub = ObjectProperty(None)
    len = NumericProperty(1)
    records = None

    def load_action(self):
        """``file_path`` を基に Tub を読み込む。"""
        if self.update_tub():
            # 更新に成功したらアプリ設定へ保存する
            rc_handler.data['last_tub'] = self.file_path

    def update_tub(self, event=None):
        if not self.file_path:
            return False
        # まだ設定が読み込まれていなければ戻る
        cfg = tub_screen().ids.config_manager.config
        if not cfg:
            return False
        # 少なくとも tub パスに manifest.json があるか確認する
        if not os.path.exists(os.path.join(self.file_path, 'manifest.json')):
            tub_screen().status(f'パス {self.file_path} は有効な tub ではありません')
            return False
        try:
            if self.tub:
                self.tub.close()
            self.tub = Tub(self.file_path)
        except Exception as e:
            tub_screen().status(f'チューブの読み込みに失敗しました: {str(e)}')
            return False
        # Tub 画面でフィルタが設定されているか確認する
        # expression = tub_screen().ids.tub_filter.filter_expression
        train_filter = getattr(cfg, 'TRAIN_FILTER', None)

        # フィルタを適用するための関数を定義
        def select(underlying):
            if not train_filter:
                return True
            else:
                try:
                    record = TubRecord(cfg, self.tub.base_path, underlying)
                    res = train_filter(record)
                    return res
                except KeyError as err:
                    Logger.error(f'Filter: {err}')
                    return True

        self.records = [TubRecord(cfg, self.tub.base_path, record)
                        for record in self.tub if select(record)]
        self.len = len(self.records)
        if self.len > 0:
            tub_screen().index = 0
            tub_screen().ids.data_plot.update_dataframe_from_tub()
            msg = f'{self.file_path} を読み込みました ({self.len} 件のレコード)'
        else:
            msg = f'{self.file_path} にはレコードがありません'
        tub_screen().status(msg)
        return True


class LabelBar(BoxLayout):
    """ Widget that combines a label with a progress bar. This is used to
        display the record fields in the data panel."""
    field = StringProperty()
    field_property = ObjectProperty()
    config = ObjectProperty()
    msg = ''

    def update(self, record):
        """現在のレコードが更新されるたびに呼び出される。"""
        if not record:
            return
        field, index = decompose(self.field)
        if field in record.underlying:
            val = record.underlying[field]
            if index is not None:
                val = val[index]
            # フィールド設定が存在する場合はバーを更新
            if self.field_property:
                norm_value = get_norm_value(val, self.config,
                                            self.field_property)
                new_bar_val = (norm_value + 1) * 50 if \
                    self.field_property.centered else norm_value * 100
                self.ids.bar.value = new_bar_val
            self.ids.field_label.text = self.field
            if isinstance(val, float) or isinstance(val, np.float32):
                text = f'{val:+07.3f}'
            elif isinstance(val, int):
                text = f'{val:10}'
            else:
                text = str(val)
            self.ids.value_label.text = text
        else:
            Logger.error(f'Record: Bad record {record.underlying["_index"]} - '
                         f'missing field {self.field}')


class DataPanel(BoxLayout):
    """ Data panel widget that contains the label/bar widgets and the drop
        down menu to select/deselect fields."""
    record = ObjectProperty()
    # パイロット比較画面では角度とスロットル（または速度）のみを表示する二重モードを使用
    dual_mode = BooleanProperty(False)
    auto_text = StringProperty(LABEL_SPINNER_TEXT)
    throttle_field = StringProperty('user/throttle')
    link = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = {}
        self.screen = ObjectProperty()

    def add_remove(self):
        """ドロップダウンの選択に応じて ``LabelBar`` を追加または削除する。"""
        field = self.ids.data_spinner.text
        if field is LABEL_SPINNER_TEXT:
            return
        if field in self.labels and not self.dual_mode:
            self.remove_widget(self.labels[field])
            del(self.labels[field])
            self.screen.status(f'{field} を削除します')
        else:
            # 二重モードでは2番目の項目を新しいものと入れ替える
            if self.dual_mode and len(self.labels) == 2:
                k, v = list(self.labels.items())[-1]
                self.remove_widget(v)
                del(self.labels[k])
            field_property = rc_handler.field_properties.get(decompose(field)[0])
            cfg = tub_screen().ids.config_manager.config
            lb = LabelBar(field=field, field_property=field_property, config=cfg)
            self.labels[field] = lb
            self.add_widget(lb)
            lb.update(self.record)
            if len(self.labels) == 2:
                self.throttle_field = field
            self.screen.status(f'{field} を追加します')
        if self.screen.name == 'tub':
            self.screen.ids.data_plot.plot_from_current_bars()
        self.ids.data_spinner.text = LABEL_SPINNER_TEXT
        self.auto_text = field

    def on_record(self, obj, record):
        """``self.record`` が更新されるたびに呼び出される Kivy のフック。"""
        for v in self.labels.values():
            v.update(record)

    def clear(self):
        for v in self.labels.values():
            self.remove_widget(v)
        self.labels.clear()


class FullImage(Image):
    """ Widget to display an image that fills the space. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core_image = None

    def update(self, record):
        """レコードが更新されるたびに画像を再描画する。"""
        try:
            img_arr = self.get_image(record)
            pil_image = PilImage.fromarray(img_arr)
            bytes_io = io.BytesIO()
            pil_image.save(bytes_io, format='png')
            bytes_io.seek(0)
            self.core_image = CoreImage(bytes_io, ext='png')
            self.texture = self.core_image.texture
        except KeyError as e:
            Logger.error(f'Record: Missing key: {e}')
        except Exception as e:
            Logger.error(f'Record: Bad record: {e}')

    def get_image(self, record):
        return record.image()


class ControlPanel(BoxLayout):
    """コントロールパネルの操作をまとめたクラス。"""
    screen = ObjectProperty()
    speed = NumericProperty(1.0)
    record_display = StringProperty()
    clock = None
    fwd = None

    def start(self, fwd=True, continuous=False):
        """レコードを順送り／逆送りする。

        Args:
            fwd (bool): 前進するか後退するか。
            continuous (bool): ``True`` なら連続再生。"""
        # このウィジェットは2つの画面で使われるため、設定は tub 画面の
        # ConfigManager を参照する
        cfg = tub_screen().ids.config_manager.config
        hz = cfg.DRIVE_LOOP_HZ if cfg else 20
        time.sleep(0.1)
        call = partial(self.step, fwd, continuous)
        if continuous:
            self.fwd = fwd
            s = float(self.speed) * hz
            cycle_time = 1.0 / s
        else:
            cycle_time = 0.08
        self.clock = Clock.schedule_interval(call, cycle_time)

    def step(self, fwd=True, continuous=False, *largs):
        """1 ステップ進めてインデックスを範囲内に保つ。"""
        if self.screen.index is None:
            self.screen.status("Tub が読み込まれていません")
            return
        new_index = self.screen.index + (1 if fwd else -1)
        if new_index >= tub_screen().ids.tub_loader.len:
            new_index = 0
        elif new_index < 0:
            new_index = tub_screen().ids.tub_loader.len - 1
        self.screen.index = new_index
        msg = f'Donkey を{"連続実行" if continuous else "1 ステップ"} ' \
              f'{"前進" if fwd else "後退"}'
        if not continuous:
            msg += f' - {"→" if fwd else "←"} キーでも操作できます'
        else:
            msg += ' - <space> で実行/停止を切り替えます'
        self.screen.status(msg)

    def stop(self):
        if self.clock:
            self.clock.cancel()
            self.clock = None

    def restart(self):
        if self.clock:
            self.stop()
            self.start(self.fwd, True)

    def update_speed(self, up=True):
        """コントローラーの速度設定を変更する。"""
        values = self.ids.control_spinner.values
        idx = values.index(self.ids.control_spinner.text)
        if up and idx < len(values) - 1:
            self.ids.control_spinner.text = values[idx + 1]
        elif not up and idx > 0:
            self.ids.control_spinner.text = values[idx - 1]

    def set_button_status(self, disabled=True):
        """全てのボタンの有効／無効を切り替える。"""
        self.ids.run_bwd.disabled = self.ids.run_fwd.disabled = \
            self.ids.step_fwd.disabled = self.ids.step_bwd.disabled = disabled

    def on_keyboard(self, key, scancode):
        """キー入力を処理する。"""
        if key == ' ':
            if self.clock and self.clock.is_triggered:
                self.stop()
                self.set_button_status(disabled=False)
                self.screen.status('Donkey を停止しました')
            else:
                self.start(continuous=True)
                self.set_button_status(disabled=True)
        elif scancode == 79:
            self.step(fwd=True)
        elif scancode == 80:
            self.step(fwd=False)
        elif scancode == 45:
            self.update_speed(up=False)
        elif scancode == 46:
            self.update_speed(up=True)


class PaddedBoxLayout(BoxLayout):
    pass


class TubEditor(PaddedBoxLayout):
    """Tub の削除・復元や再読込を行う編集ウィジェット。"""
    lr = ListProperty([0, 0])

    def set_lr(self, is_l=True):
        """現在のレコード番号を左または右の範囲に設定する。"""
        if not tub_screen().current_record:
            return
        self.lr[0 if is_l else 1] = tub_screen().current_record.underlying['_index']

    def del_lr(self, is_del):
        """指定した範囲のレコードを削除または復元する。"""
        tub = tub_screen().ids.tub_loader.tub
        if self.lr[1] >= self.lr[0]:
            selected = list(range(*self.lr))
        else:
            last_id = tub.manifest.current_index
            selected = list(range(self.lr[0], last_id))
            selected += list(range(self.lr[1]))
        tub.delete_records(selected) if is_del else tub.restore_records(selected)


class TubFilter(PaddedBoxLayout):
    """Tub のレコードをフィルタリングするウィジェット。"""
    filter_expression = StringProperty(None)
    record_filter = StringProperty(rc_handler.data.get('record_filter', ''))

    def update_filter(self):
        filter_text = self.ids.record_filter.text
        config = tub_screen().ids.config_manager.config
        # 空文字列ならフィルターを解除
        if filter_text == '':
            self.record_filter = ''
            self.filter_expression = ''
            rc_handler.data['record_filter'] = self.record_filter
            if hasattr(config, 'TRAIN_FILTER'):
                delattr(config, 'TRAIN_FILTER')
            tub_screen().status('フィルターをクリアしました')
            return
        filter_expression = self.create_filter_string(filter_text)
        try:
            record = tub_screen().current_record
            filter_func_text = f"""def filter_func(record): 
                                       return {filter_expression}       
                                """
            # ここで関数 'filter_func' を生成する
            ldict = {}
            exec(filter_func_text, globals(), ldict)
            filter_func = ldict['filter_func']
            res = filter_func(record)
            status = f'現在のレコードに対するフィルター結果: {res}'
            if isinstance(res, bool):
                self.record_filter = filter_text
                self.filter_expression = filter_expression
                rc_handler.data['record_filter'] = self.record_filter
                setattr(config, 'TRAIN_FILTER', filter_func)
            else:
                status += ' - 真偽値を返さない式は適用できません'
            status += ' - 効果を見るには <Reload tub> を押してください'
            tub_screen().status(status)
        except Exception as e:
            tub_screen().status(f'フィルター処理でエラー: {e}')

    @staticmethod
    def create_filter_string(filter_text, record_name='record'):
        """フィルター式内のフィールド名を ``record.underlying`` 形式へ変換する。

        Args:
            filter_text (str): 例 ``'user/throttle > 0.1'`` のような文字列。
            record_name (str): レコード変数の名称。

        Returns:
            str: 変換された文字列。
        """
        for field in tub_screen().current_record.underlying.keys():
            field_list = filter_text.split(field)
            if len(field_list) > 1:
                filter_text = f'{record_name}.underlying["{field}"]'\
                    .join(field_list)
        return filter_text


class DataPlot(PaddedBoxLayout):
    """matplotlib のインタラクティブグラフを表示するパネル。"""
    df = ObjectProperty(force_dispatch=True, allownone=True)

    def plot_from_current_bars(self, in_app=True):
        """選択されたバーからグラフを描画する。"""
        tub = tub_screen().ids.tub_loader.tub
        field_map = dict(zip(tub.manifest.inputs, tub.manifest.types))
        # 選択されているフィールドがなければすべてのフィールドを使用
        all_cols = tub_screen().ids.data_panel.labels.keys() or self.df.columns
        cols = [c for c in all_cols if decompose(c)[0] in field_map
                and field_map[decompose(c)[0]] not in ('image_array', 'str')]

        df = self.df[cols]
        if df is None:
            return
        # ミリ秒のタイムスタンプは値が大きすぎるためプロットしない
        df = df.drop(labels=['_timestamp_ms'], axis=1, errors='ignore')

        if in_app:
            tub_screen().ids.graph.df = df
        else:
            fig = px.line(df, x=df.index, y=df.columns, title=tub.base_path)
            fig.update_xaxes(rangeslider=dict(visible=True))
            fig.show()

    def unravel_vectors(self):
        """ベクトルやリスト型の項目を展開する。"""
        manifest = tub_screen().ids.tub_loader.tub.manifest
        for k, v in zip(manifest.inputs, manifest.types):
            if v == 'vector' or v == 'list':
                dim = len(tub_screen().current_record.underlying[k])
                df_keys = [k + f'_{i}' for i in range(dim)]
                self.df[df_keys] = pd.DataFrame(self.df[k].tolist(),
                                                index=self.df.index)
                self.df.drop(k, axis=1, inplace=True)

    def update_dataframe_from_tub(self):
        """Tub の再読み込み時に DataFrame を作成し UI を更新する。"""
        generator = (t.underlying for t in tub_screen().ids.tub_loader.records)
        self.df = pd.DataFrame(generator).dropna()
        to_drop = {'cam/image_array'}
        self.df.drop(labels=to_drop, axis=1, errors='ignore', inplace=True)
        self.df.set_index('_index', inplace=True)
        self.unravel_vectors()
        tub_screen().ids.data_panel.ids.data_spinner.values = self.df.columns
        self.plot_from_current_bars()


class TabBar(BoxLayout):
    """タブボタンの有効・無効を制御する。"""
    manager = ObjectProperty(None)

    def disable_only(self, bar_name):
        """指定したバーのみ有効にする。"""
        this_button_name = bar_name + '_btn'
        for button_name, button in self.ids.items():
            button.disabled = button_name == this_button_name


class TubScreen(Screen):
    """ First screen of the app managing the tub data. """
    index = NumericProperty(None, force_dispatch=True)
    current_record = ObjectProperty(None)
    keys_enabled = BooleanProperty(True)

    def initialise(self, e):
        self.ids.config_manager.load_action()
        self.ids.tub_loader.update_tub()

    def on_index(self, obj, index):
        """``self.index`` が変化した際に呼ばれる。"""
        if index >= 0:
            self.current_record = self.ids.tub_loader.records[index]
            self.ids.slider.value = index

    def on_current_record(self, obj, record):
        """``self.current_record`` が変化した際に呼ばれる。"""
        self.ids.img.update(record)
        i = record.underlying['_index']
        self.ids.control_panel.record_display = f"Record {i:06}"

    def status(self, msg):
        self.ids.status.text = msg

    def on_keyboard(self, instance, keycode, scancode, key, modifiers):
        if self.keys_enabled:
            self.ids.control_panel.on_keyboard(key, scancode)


class PilotLoader(BoxLayout, FileChooserBase):
    """パイロットモデルのロードを管理するクラス。"""
    num = StringProperty()
    model_type = StringProperty()
    pilot = ObjectProperty(None)
    filters = ['*.h5', '*.tflite', '*.savedmodel', '*.trt']

    def load_action(self):
        if self.file_path and self.pilot:
            try:
                self.pilot.load(os.path.join(self.file_path))
                rc_handler.data['pilot_' + self.num] = self.file_path
                rc_handler.data['model_type_' + self.num] = self.model_type
                self.ids.pilot_spinner.text = self.model_type
                Logger.info(f'Pilot: Successfully loaded {self.file_path}')
            except FileNotFoundError:
                Logger.error(f'Pilot: Model {self.file_path} not found')
            except Exception as e:
                Logger.error(f'Failed loading {self.file_path}: {e}')

    def on_model_type(self, obj, model_type):
        """``self.model_type`` が変更された際に呼び出される。"""
        if self.model_type and self.model_type != 'Model type':
            cfg = tub_screen().ids.config_manager.config
            if cfg:
                self.pilot = get_model_by_type(self.model_type, cfg)
                self.ids.pilot_button.disabled = False
                if 'tflite' in self.model_type:
                    self.filters = ['*.tflite']
                elif 'tensorrt' in self.model_type:
                    self.filters = ['*.trt']
                else:
                    self.filters = ['*.h5', '*.savedmodel']

    def on_num(self, e, num):
        """``self.num`` が変更された際に呼び出される。"""
        self.file_path = rc_handler.data.get('pilot_' + self.num, '')
        self.model_type = rc_handler.data.get('model_type_' + self.num, '')


class OverlayImage(FullImage):
    """ユーザーとパイロットの情報を重ねて表示するウィジェット。"""
    pilot = ObjectProperty()
    pilot_record = ObjectProperty()
    throttle_field = StringProperty('user/throttle')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_left = True

    def augment(self, img_arr):
        if pilot_screen().trans_list:
            img_arr = pilot_screen().transformation.run(img_arr)
        if pilot_screen().aug_list:
            img_arr = pilot_screen().augmentation.run(img_arr)
        if pilot_screen().post_trans_list:
            img_arr = pilot_screen().post_transformation.run(img_arr)
        return img_arr

    def get_image(self, record):
        from donkeycar.management.makemovie import MakeMovie
        config = tub_screen().ids.config_manager.config
        orig_img_arr = super().get_image(record)
        aug_img_arr = self.augment(orig_img_arr)
        img_arr = copy(aug_img_arr)
        angle = record.underlying['user/angle']
        throttle = get_norm_value(
            record.underlying[self.throttle_field], config,
            rc_handler.field_properties[self.throttle_field])
        rgb = (0, 255, 0)
        MakeMovie.draw_line_into_image(angle, throttle, False, img_arr, rgb)
        if not self.pilot:
            return img_arr

        output = (0, 0)
        try:
            # すべてのモデルがすべてのインタープリタで動作するわけではない
            output = self.pilot.run(aug_img_arr)
        except Exception as e:
            Logger.error(e)

        rgb = (0, 0, 255)
        MakeMovie.draw_line_into_image(output[0], output[1], True, img_arr, rgb)
        out_record = copy(record)
        out_record.underlying['pilot/angle'] = output[0]
        # スロットル出力のキー名を変更し、正規化を解除
        pilot_throttle_field \
            = rc_handler.data['user_pilot_map'][self.throttle_field]
        out_record.underlying[pilot_throttle_field] \
            = get_norm_value(output[1], tub_screen().ids.config_manager.config,
                             rc_handler.field_properties[self.throttle_field],
                             normalised=False)
        self.pilot_record = out_record
        return img_arr


class TransformationPopup(Popup):
    """画像変換を選択するポップアップウィンドウ。"""
    title = StringProperty()
    transformations = \
        ["TRAPEZE", "CROP", "RGB2BGR", "BGR2RGB", "RGB2HSV", "HSV2RGB",
         "BGR2HSV", "HSV2BGR", "RGB2GRAY", "RBGR2GRAY", "HSV2GRAY", "GRAY2RGB",
         "GRAY2BGR", "CANNY", "BLUR", "RESIZE", "SCALE"]
    transformations_obj = ObjectProperty()
    selected = ListProperty()

    def __init__(self, selected, **kwargs):
        super().__init__(**kwargs)
        for t in self.transformations:
            btn = Button(text=t)
            btn.bind(on_release=self.toggle_transformation)
            self.ids.trafo_list.add_widget(btn)
        self.selected = selected

    def toggle_transformation(self, btn):
        trafo = btn.text
        if trafo in self.selected:
            self.selected.remove(trafo)
        else:
            self.selected.append(trafo)

    def on_selected(self, obj, select):
        self.ids.selected_list.clear_widgets()
        for l in self.selected:
            lab = Label(text=l)
            self.ids.selected_list.add_widget(lab)
        self.transformations_obj.selected = self.selected


class Transformations(Button):
    """画像変換ウィジェットの基底クラス。"""
    title = StringProperty(None)
    pilot_screen = ObjectProperty()
    is_post = False
    selected = ListProperty()

    def open_popup(self):
        popup = TransformationPopup(title=self.title, transformations_obj=self,
                                    selected=self.selected)
        popup.open()

    def on_selected(self, obj, select):
        Logger.info(f"Selected {select}")
        if self.is_post:
            self.pilot_screen.post_trans_list = self.selected
        else:
            self.pilot_screen.trans_list = self.selected


class PilotScreen(Screen):
    """複数のパイロットを比較するための画面。"""
    index = NumericProperty(None, force_dispatch=True)
    current_record = ObjectProperty(None)
    keys_enabled = BooleanProperty(False)
    aug_list = ListProperty(force_dispatch=True)
    augmentation = ObjectProperty()
    trans_list = ListProperty(force_dispatch=True)
    transformation = ObjectProperty()
    post_trans_list = ListProperty(force_dispatch=True)
    post_transformation = ObjectProperty()
    config = ObjectProperty()

    def on_index(self, obj, index):
        """``self.index`` が変化した際にレコードとスライダーを更新する。"""
        if tub_screen().ids.tub_loader.records:
            self.current_record = tub_screen().ids.tub_loader.records[index]
            self.ids.slider.value = index

    def on_current_record(self, obj, record):
        """``self.current_record`` が変化した際に画像と表示を更新する。"""
        if not record:
            return
        i = record.underlying['_index']
        self.ids.pilot_control.record_display = f"Record {i:06}"
        self.ids.img_1.update(record)
        self.ids.img_2.update(record)

    def initialise(self, e):
        self.ids.pilot_loader_1.on_model_type(None, None)
        self.ids.pilot_loader_1.load_action()
        self.ids.pilot_loader_2.on_model_type(None, None)
        self.ids.pilot_loader_2.load_action()
        mapping = copy(rc_handler.data['user_pilot_map'])
        del(mapping['user/angle'])
        self.ids.data_in.ids.data_spinner.values = mapping.keys()
        self.ids.data_in.ids.data_spinner.text = 'user/angle'
        self.ids.data_panel_1.ids.data_spinner.disabled = True
        self.ids.data_panel_2.ids.data_spinner.disabled = True

    def map_pilot_field(self, text):
        """Add/remove 以外の項目をユーザーからパイロットのフィールドへ変換する。"""
        if text == LABEL_SPINNER_TEXT:
            return text
        return rc_handler.data['user_pilot_map'][text]

    def set_brightness(self, val=None):
        if not self.config:
            return
        if self.ids.button_bright.state == 'down':
            self.config.AUG_BRIGHTNESS_RANGE = (val, val)
            if 'BRIGHTNESS' not in self.aug_list:
                self.aug_list.append('BRIGHTNESS')
            else:
                # 設定内容のみ変更した場合 self.on_aug_list() は呼ばれないため、
                # 依存関係を手動で更新する
                self.on_aug_list(None, None)
        elif 'BRIGHTNESS' in self.aug_list:
            self.aug_list.remove('BRIGHTNESS')

    def set_blur(self, val=None):
        if not self.config:
            return
        if self.ids.button_blur.state == 'down':
            self.config.AUG_BLUR_RANGE = (val, val)
            if 'BLUR' not in self.aug_list:
                self.aug_list.append('BLUR')
        elif 'BLUR' in self.aug_list:
            self.aug_list.remove('BLUR')
        # 依存関係を更新
        self.on_aug_list(None, None)

    def on_aug_list(self, obj, aug_list):
        if not self.config:
            return
        self.config.AUGMENTATIONS = self.aug_list
        self.augmentation = ImageAugmentation(
            self.config, 'AUGMENTATIONS', always_apply=True)
        self.on_current_record(None, self.current_record)

    def on_trans_list(self, obj, trans_list):
        if not self.config:
            return
        self.config.TRANSFORMATIONS = self.trans_list
        self.transformation = ImageTransformations(
            self.config, 'TRANSFORMATIONS')
        self.on_current_record(None, self.current_record)

    def on_post_trans_list(self, obj, trans_list):
        if not self.config:
            return
        self.config.POST_TRANSFORMATIONS = self.post_trans_list
        self.post_transformation = ImageTransformations(
            self.config, 'POST_TRANSFORMATIONS')
        self.on_current_record(None, self.current_record)

    def set_mask(self, state):
        if state == 'down':
            self.ids.status.text = '台形マスクを有効化'
            self.trans_list.append('TRAPEZE')
        else:
            self.ids.status.text = '台形マスクを無効化'
            if 'TRAPEZE' in self.trans_list:
                self.trans_list.remove('TRAPEZE')

    def set_crop(self, state):
        if state == 'down':
            self.ids.status.text = 'クロップを有効化'
            self.trans_list.append('CROP')
        else:
            self.ids.status.text = 'クロップを無効化'
            if 'CROP' in self.trans_list:
                self.trans_list.remove('CROP')

    def status(self, msg):
        self.ids.status.text = msg

    def on_keyboard(self, instance, keycode, scancode, key, modifiers):
        if self.keys_enabled:
            self.ids.pilot_control.on_keyboard(key, scancode)


class ScrollableLabel(ScrollView):
    pass


class DataFrameLabel(Label):
    pass


class TransferSelector(BoxLayout, FileChooserBase):
    """転移学習モデルを選択するためのクラス。"""
    filters = ['*.h5']


class TrainScreen(Screen):
    """学習を実行する画面。"""
    config = ObjectProperty(force_dispatch=True, allownone=True)
    database = ObjectProperty()
    pilot_df = ObjectProperty(force_dispatch=True)
    tub_df = ObjectProperty(force_dispatch=True)

    def train_call(self, model_type, *args):
        # パスから車両ディレクトリ名を取り除く
        tub_path = tub_screen().ids.tub_loader.tub.base_path
        transfer = self.ids.transfer_spinner.text
        if transfer != '転移モデルを選択':
            transfer = os.path.join(self.config.MODELS_PATH, transfer + '.h5')
        else:
            transfer = None
        try:
            history = train(self.config, tub_paths=tub_path,
                            model_type=model_type,
                            transfer=transfer,
                            comment=self.ids.comment.text)
            self.ids.status.text = '学習が完了しました'
            self.ids.comment.text = 'コメント'
            self.ids.transfer_spinner.text = '転移モデルを選択'
            self.reload_database()
        except Exception as e:
            Logger.error(e)
            self.ids.status.text = '学習に失敗しました。コンソールを確認してください'
        finally:
            self.ids.train_button.state = 'normal'

    def train(self, model_type):
        self.config.SHOW_PLOT = False
        Thread(target=self.train_call, args=(model_type,)).start()
        self.ids.status.text = '学習を開始しました'

    def set_config_attribute(self, input):
        try:
            val = json.loads(input)
        except ValueError:
            val = input

        att = self.ids.cfg_spinner.text.split(':')[0]
        setattr(self.config, att, val)
        self.ids.cfg_spinner.values = self.value_list()
        self.ids.status.text = f'{att} を {val} (型: {type(val).__name__}) に設定しました'

    def value_list(self):
        if self.config:
            return [f'{k}: {v}' for k, v in self.config.__dict__.items()]
        else:
            return ['select']

    def on_config(self, obj, config):
        if self.config and self.ids:
            self.ids.cfg_spinner.values = self.value_list()
            self.reload_database()

    def reload_database(self):
        if self.config:
            self.database = PilotDatabase(self.config)

    def on_database(self, obj, database):
        group_tubs = self.ids.check.state == 'down'
        pilot_txt, tub_txt, pilot_names = self.database.pretty_print(group_tubs)
        self.ids.scroll_tubs.text = tub_txt
        self.ids.scroll_pilots.text = pilot_txt
        self.ids.transfer_spinner.values \
            = ['転移モデルを選択'] + pilot_names
        self.ids.delete_spinner.values \
            = ['パイロット'] + pilot_names


class CarScreen(Screen):
    """実車と通信するための画面。"""
    config = ObjectProperty(force_dispatch=True, allownone=True)
    files = ListProperty()
    car_dir = StringProperty(rc_handler.data.get('robot_car_dir', '~/mycar'))
    event = ObjectProperty(None, allownone=True)
    connection = ObjectProperty(None, allownone=True)
    pid = NumericProperty(None, allownone=True)
    pilots = ListProperty()
    is_connected = BooleanProperty(False)

    def initialise(self):
        self.event = Clock.schedule_interval(self.connected, 3)

    def list_remote_dir(self, dir):
        if self.is_connected:
            cmd = f'ssh {self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}' + \
                  f' "ls {dir}"'
            listing = os.popen(cmd).read()
            adjusted_listing = listing.split('\n')[1:-1]
            return adjusted_listing
        else:
            return []

    def list_car_dir(self, dir):
        self.car_dir = dir
        self.files = self.list_remote_dir(dir)
        # ディレクトリにファイルがある場合
        if self.files:
            rc_handler.data['robot_car_dir'] = dir

    def update_pilots(self):
        model_dir = os.path.join(self.car_dir, 'models')
        self.pilots = self.list_remote_dir(model_dir)

    def pull(self, tub_dir):
        target = f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}' + \
               f':{os.path.join(self.car_dir, tub_dir)}'
        dest = self.config.DATA_PATH
        if self.ids.create_dir.state == 'normal':
            target += '/'
        cmd = ['rsync', '-rv', '--progress', '--partial', target, dest]
        Logger.info('車両から取得: ' + str(cmd))
        proc = Popen(cmd, shell=False, stdout=PIPE, text=True,
                     encoding='utf-8', universal_newlines=True)
        repeats = 100
        call = partial(self.show_progress, proc, repeats, True)
        event = Clock.schedule_interval(call, 0.0001)

    def send_pilot(self):
        # 末尾に '/' を追加
        src = os.path.join(self.config.MODELS_PATH,'')
        # 同期ボタンが押されていればパスを調整
        buttons = ['h5', 'savedmodel', 'tflite', 'trt']
        select = [btn for btn in buttons if self.ids[f'btn_{btn}'].state
                  == 'down']
        # フィルタを作成: 例として .tflite と .trt のみ同期する場合
        # --include=*.trt/*** --include=*.tflite --exclude=*
        filter = ['--include=database.json']
        for ext in select:
            if ext in ['savedmodel', 'trt']:
                ext += '/***'
            filter.append(f'--include=*.{ext}')
        # 何も選択されていない場合はすべて同期
        if not select:
            filter.append('--include=*')
        else:
            filter.append('--exclude=*')
        dest = f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}:' + \
               f'{os.path.join(self.car_dir, "models")}'
        cmd = ['rsync', '-rv', '--progress', '--partial', *filter, src, dest]
        Logger.info('車両へ送信: ' + ' '.join(cmd))
        proc = Popen(cmd, shell=False, stdout=PIPE,
                     encoding='utf-8', universal_newlines=True)
        repeats = 0
        call = partial(self.show_progress, proc, repeats, False)
        event = Clock.schedule_interval(call, 0.0001)

    def show_progress(self, proc, repeats, is_pull, e):
        # OSX では 'to-check=33/4551)', Linux では 'to-chk=33/4551)' を探す
        # 行末に現れる進捗表示を解析する
        pattern = 'to-(check|chk)=(.*)\)'

        def end():
            # コマンド終了時にスケジュールを停止
            if is_pull:
                button = self.ids.pull_tub
                self.ids.pull_bar.value = 0
                # 以前に削除されたインデックスをマージ（上書きされた可能性あり）
                old_tub = tub_screen().ids.tub_loader.tub
                if old_tub:
                    deleted_indexes = old_tub.manifest.deleted_indexes
                    tub_screen().ids.tub_loader.update_tub()
                    if deleted_indexes:
                        new_tub = tub_screen().ids.tub_loader.tub
                        new_tub.manifest.add_deleted_indexes(deleted_indexes)
            else:
                button = self.ids.send_pilots
                self.ids.push_bar.value = 0
                self.update_pilots()
            button.disabled = False

        if proc.poll() is not None:
            end()
            return False
        # 次の進捗行を取得してバーを更新
        count = 0
        while True:
            stdout_data = proc.stdout.readline()
            if stdout_data:
                res = re.search(pattern, stdout_data)
                if res:
                    if count < repeats:
                        count += 1
                    else:
                        remain, total = tuple(res.group(2).split('/'))
                        bar = 100 * (1. - float(remain) / float(total))
                        if is_pull:
                            self.ids.pull_bar.value = bar
                        else:
                            self.ids.push_bar.value = bar
                        return True
            else:
                # ストリームが終了したら完了
                end()
                return False

    def connected(self, event):
        if not self.config:
            return
        if self.connection is None:
            if not hasattr(self.config, 'PI_USERNAME') or \
                    not hasattr(self.config, 'PI_HOSTNAME'):
                self.ids.connected.text = 'PI_USERNAME と PI_HOSTNAME が必要です'
                return
            # 接続状況を確認するため新たにコマンドを実行
            cmd = ['ssh',
                   '-o ConnectTimeout=3',
                   f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}',
                   'date']
            self.connection = Popen(cmd, shell=False, stdout=PIPE,
                                    stderr=STDOUT, text=True,
                                    encoding='utf-8', universal_newlines=True)
        else:
            # ssh が既に実行中の場合は状態を確認
            return_val = self.connection.poll()
            self.is_connected = False
            if return_val is None:
                # コマンド実行中のため次回に再確認
                status = '接続待機中...'
                self.ids.connected.color = 0.8, 0.8, 0.0, 1
            else:
                # コマンド終了。成功したか確認し接続をリセット
                if return_val == 0:
                    status = '接続済み'
                    self.ids.connected.color = 0, 0.9, 0, 1
                    self.is_connected = True
                else:
                    status = '未接続'
                    self.ids.connected.color = 0.9, 0, 0, 1
                self.connection = None
            self.ids.connected.text = status

    def drive(self):
        model_args = ''
        if self.ids.pilot_spinner.text != 'No pilot':
            model_path = os.path.join(self.car_dir, "models",
                                      self.ids.pilot_spinner.text)
            model_args = f'--type {self.ids.type_spinner.text} ' + \
                         f'--model {model_path}'
        cmd = ['ssh',
               f'{self.config.PI_USERNAME}@{self.config.PI_HOSTNAME}',
               f'source env/bin/activate; cd {self.car_dir}; ./manage.py '
               f'drive {model_args} 2>&1']
        Logger.info(f'車両接続: {cmd}')
        proc = Popen(cmd, shell=False, stdout=PIPE, text=True,
                     encoding='utf-8', universal_newlines=True)
        while True:
            stdout_data = proc.stdout.readline()
            if stdout_data:
                # 'PID: 12345' を検出
                pattern = 'PID: .*'
                res = re.search(pattern, stdout_data)
                if res:
                    try:
                        self.pid = int(res.group(0).split('PID: ')[1])
                        Logger.info(f'車両接続: manage.py drive PID: {self.pid}')
                    except Exception as e:
                        Logger.error(f'車両接続エラー: {e}')
                    return
                Logger.info(f'車両接続: {stdout_data}')
            else:
                return

    def stop(self):
        if self.pid:
            cmd = f'ssh {self.config.PI_USERNAME}@{self.config.PI_HOSTNAME} '\
                  + f'kill {self.pid}'
            out = os.popen(cmd).read()
            Logger.info(f"車両接続: PID {self.pid} を終了 {out}")
            self.pid = None


class StartScreen(Screen):
    """起動画面を表すシンプルなスクリーン。"""
    img_path = os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        '../parts/web_controller/templates/static/donkeycar-logo-sideways.png'))
    pass


class DonkeyApp(App):
    """Kivy ベースの DonkeyCar 管理アプリケーション。"""
    start_screen = None
    tub_screen = None
    train_screen = None
    pilot_screen = None
    car_screen = None
    title = 'Donkey Manager'

    def initialise(self, event):
        self.tub_screen.ids.config_manager.load_action()
        self.pilot_screen.initialise(event)
        self.car_screen.initialise()
        # グラフ生成は他の処理完了後でないと行えないため次のループまで遅延
        Clock.schedule_once(self.tub_screen.ids.tub_loader.update_tub)

    def build(self):
        Window.bind(on_request_close=self.on_request_close)
        self.start_screen = StartScreen(name='donkey')
        self.tub_screen = TubScreen(name='tub')
        self.train_screen = TrainScreen(name='train')
        self.pilot_screen = PilotScreen(name='pilot')
        self.car_screen = CarScreen(name='car')
        Window.bind(on_keyboard=self.tub_screen.on_keyboard)
        Window.bind(on_keyboard=self.pilot_screen.on_keyboard)
        Clock.schedule_once(self.initialise)
        sm = ScreenManager()
        sm.add_widget(self.start_screen)
        sm.add_widget(self.tub_screen)
        sm.add_widget(self.train_screen)
        sm.add_widget(self.pilot_screen)
        sm.add_widget(self.car_screen)
        return sm

    def on_request_close(self, *args):
        tub = self.tub_screen.ids.tub_loader.tub
        if tub:
            tub.close()
        Logger.info("さようなら Donkey")
        return False


def main():
    """アプリケーションを起動するエントリーポイント。"""
    tub_app = DonkeyApp()
    tub_app.run()


if __name__ == '__main__':
    main()
