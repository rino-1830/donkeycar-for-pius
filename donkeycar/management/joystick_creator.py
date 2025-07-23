import sys
import os
import argparse
import json
import time
import math

from donkeycar.utils import *
from donkeycar.parts.controller import JoystickCreatorController

try:
    from prettytable import PrettyTable
except:
    print("必要なモジュールがありません: pip install PrettyTable を実行してください")

class CreateJoystick(object):
    """ジョイスティック設定ウィザードを提供するクラス。"""

    def __init__(self):
        """インスタンスを初期化する。"""
        self.last_button = None
        self.last_axis = None
        self.axis_val = 0
        self.running = False
        self.thread = None
        self.gyro_axis = []
        self.axis_map = []
        self.ignore_axis = False
        self.mapped_controls = []

    def poll(self):
        """ジョイスティック入力を監視するスレッド処理。"""
        while self.running:
            button, button_state, axis, axis_val = self.js.poll()

            if button is not None:
                self.last_button = button
                self.last_axis = None
                self.axis_val = 0.0
            elif axis is not None and not self.ignore_axis:
                if not axis in self.gyro_axis:
                    self.last_axis = axis
                    self.last_button = None
                    self.axis_val = axis_val

    def get_button_press(self, duration=10.0):
        """指定時間内に押されたボタン名を取得する。

        Args:
            duration (float): 待機する秒数。

        Returns:
            str | None: 押されたボタン名。押されなかった場合は ``None``。
        """
        self.last_button = None

        start = time.time()

        while self.last_button is None and time.time() - start < duration:
            time.sleep(0.1)

        return self.last_button

    def get_axis_move(self, duration=2.0):
        """指定時間で軸の動きを計測し、最も動いた軸名を返す。

        Args:
            duration (float): 計測時間の秒数。

        Returns:
            str | None: 動きを検知した軸名。検知できなければ ``None``。
        """
        self.last_axis = None
        axis_samples = {}

        start = time.time()

        while time.time() - start < duration:
            if self.last_axis:
                if self.last_axis in axis_samples:
                    try:
                        axis_samples[self.last_axis] = axis_samples[self.last_axis] + math.fabs(self.axis_val)
                    except:
                        try:
                            axis_samples[self.last_axis] = math.fabs(self.axis_val)
                        except:
                            pass
                else:
                    axis_samples[self.last_axis] = math.fabs(self.axis_val)
            
        most_movement = None
        most_val = 0
        for key, value in axis_samples.items():
            if value > most_val:
                most_movement = key
                most_val = value

        return most_movement

    def clear_scr(self):
        """ターミナル画面をクリアする。"""
        print(chr(27) + "[2J")

    def create_joystick(self, args):
        """対話形式でジョイスティック設定を行い、クラスファイルを生成する。

        Args:
            args (argparse.Namespace): コマンドライン引数。
        """
        
        self.clear_scr()
        print("##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##")
        print("## Joystick Creator ウィザードへようこそ ##")
        print("##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##")
        print("このウィザードは、ドンキー・カーでジョイスティックを使うためのコードを生成します")
        print()
        print("概要:")
        print()
        print("最初に各ボタンを名前付けし、次に各軸を設定します")
        print("次にそれらの名前をアクションへマッピングします")
        print("最後にプロジェクトで利用できる Python ファイルを書き出します")
        print()
        input('続行するには Enter キーを押してください')
        self.clear_scr()
        print("コントローラーを USB か Bluetooth で接続してください。ステータスランプが点灯し、デバイスが認識されていることを確認します")
        input('続行するには Enter キーを押してください ')
        self.clear_scr()
        
        self.init_js_device()
        print()
        
        self.init_polling_js()
        self.clear_scr()

        self.find_gyro()
        self.clear_scr()

        self.explain_config()
        self.clear_scr()

        self.name_buttons()
        self.clear_scr()

        self.name_axes()
        self.clear_scr()

        self.map_steering_throttle()
        self.clear_scr()

        self.map_button_controls()
        self.clear_scr()

        self.revisit_topic()
        self.clear_scr()

        self.write_python_class_file()

        print("新しく生成された Python ファイルを確認し、manage.py にインポートして操作に使用してください")

        self.shutdown()

    def init_js_device(self):
        """ジョイスティックデバイスを初期化する。"""
        from donkeycar.parts.controller import JoystickCreatorController

        js_cr = None

        # デバイスファイルを取得し、ジョイスティック作成ヘルパーを生成
        while js_cr is None:
            print("ジョイスティックのデバイスファイルはどこにありますか?")
            dev_fn = input("既定 /dev/input/js0 を使う場合は Enter、別のパスを入力することもできます: ")
            if len(dev_fn) == 0:
                dev_fn = '/dev/input/js0'

            print()
            print("そのファイルでデバイスを開いてみます...")
            try:
                js_cr = JoystickCreatorController(dev_fn=dev_fn)
                res = js_cr.init_js()
                if res:
                    print("入力デバイスを検出しアクセスできました")
                else:
                    js_cr = None
            except Exception as e:
                print("例外が発生しました:" + str(e))
                js_cr = None

            if js_cr is None:
                ret = input("デバイスを開けませんでした。再試行しますか? [Y/n] : ")
                if ret.upper() == "N":
                    exit(0)

        self.js = js_cr.js
        input("続行するには Enter キーを押してください")

    def init_polling_js(self):
        """入力ポーリング用スレッドを起動する。"""
        self.running = True
        import threading
        self.thread = threading.Thread(target=self.poll)
        self.thread.daemon = True
        self.thread.start()

    def find_gyro(self):
        """ジャイロデータを発生させる軸を検出する。"""
        print("次にジャイロスコープデータを探します")
        input("5 秒間コントローラーを動かし、各軸を回転させてください。Enter を押したら開始します: ")
        start = time.time()
        while time.time() - start < 5.0:
            if self.last_axis is not None and not self.last_axis in self.gyro_axis:
                self.gyro_axis.append(self.last_axis)

        print()
        if len(self.gyro_axis) > 0:
            print("%d 個の軸がジャイロスコープデータを送信しているのを確認しました。これらは名前付けとマッピングの際に無視します" % len(self.gyro_axis))
        else:
            print("イベントは見つかりませんでした。お使いのコントローラーがジャイロデータを出さないだけかもしれません。問題ありません")

        input("続行するには Enter キーを押してください ")

    def get_code_from_button(self, button):
        """ボタン名から数値コードを取得する。

        Args:
            button (str): ボタン名またはコード文字列。

        Returns:
            int | str | None: 解析したコード。失敗した場合は ``None``。
        """
        code = button
        if 'unknown' in button:
            try:
                code_str = button.split('(')[1][:-1]
                code = int(code_str, 16)
            except Exception as e:
                code = None
                print("コードの解析に失敗しました", str(e))
        return code

    def explain_config(self):
        """現在の設定状況を表で表示して説明する。"""
        print("これまでの進捗を次の表で表示します:")
        print()
        self.print_config()
        print("\nボタンに名前を付け、コントロールへマッピングするとこの表が更新されます")
        input("続行するには Enter キーを押してください")


    def name_buttons(self):
        """ボタンに名前を付ける対話式処理。"""
        done = False
        self.ignore_axis = True

        self.print_config()

        print('次に、すべてのボタンに名前を付けます。アナログ入力は後で設定します。')

        while not done:

            print('名前を付けたいボタンを押してください')
            
            self.get_button_press()

            if self.last_button is None:
                print("10 秒間ボタンが押されませんでした。ボタンがすべて軸コマンドを発生させる可能性があります")
                ret = input("ボタンの割り当てを続けますか? [Y, n]")
                if ret == 'n':
                    break
            elif 'unknown' in self.last_button:
                code = self.get_code_from_button(self.last_button)

                if code is not None:
                    if code in self.js.button_names:
                        ret = input("このボタンには既に名前 '%s' が付いています。命名を終了しますか? (y/N) " % self.js.button_names[code])
                        if ret.upper() == "Y":
                            done = True
                            break
                    label = input("このボタンに付ける名前を入力してください:")
                    if len(label) == 0:
                        print("名前が入力されませんでした。スキップします")
                    else:
                        self.clear_scr()
                        self.js.button_names[code] = label
                        self.print_config()
            else:
                print('押されたボタン: ', self.last_button)

            self.clear_scr()
            self.print_config()
            

    def print_config(self):
        """現在のマッピング設定を表形式で表示する。"""
        pt = PrettyTable()
        pt.field_names = ["button code", "button name"]
        for key, value in self.js.button_names.items():
            pt.add_row([str(hex(key)), str(value)])
        print("ボタンマップ:")
        print(pt)

        pt = PrettyTable()
        pt.field_names = ["axis code", "axis name"]
        for key, value in self.js.axis_names.items():
            pt.add_row([str(hex(key)), str(value)])
        print("軸マップ:")
        print(pt)

        pt = PrettyTable()
        pt.field_names = ["control", "action"]
        for button, control in self.mapped_controls:
            pt.add_row([button, control])
        for axis, control in self.axis_map:
            pt.add_row([axis, control])
        print("コントロールマップ:")
        print(pt)

    def name_axes(self):
        """使用する軸に名前を付ける対話式処理。"""
        self.print_config()
        print()
        print('次に使用するすべての軸に名前を付けます')

        done = False
        self.ignore_axis = False

        while not done:
            print('2 秒間コントローラーの軸を動かす準備をしてください')
            ret = input("開始するには Enter、終了する場合は D を押してください ")
            if ret.upper() == 'D':
                break
            
            most_movement = self.get_axis_move()

            if most_movement is None:
                print("動きが検出できませんでした")
                res = input("もう一度試しますか? [Y/n]: ")
                if res == "n":
                    done = True
                    break
                else:
                    continue

            if 'unknown' in most_movement:
                code_str = most_movement.split('(')[1][:-1]
                print('最も動いた軸のコード:', code_str)
                try:
                    code = int(code_str, 16)
                except Exception as e:
                    code = None
                    print("コードの解析に失敗しました", str(e))

                if code is not None:
                    label = input("この軸に付ける名前を入力してください (終了する場合は D): ")
                    if len(label) == 0:
                        print("名前が入力されませんでした。スキップします")
                    elif label.upper() == 'D':
                        done = True
                    else:
                        self.js.axis_names[code] = label
                        self.clear_scr()
                        self.print_config()
            else:
                print('検出した軸: ', self.last_axis)
            print()

    def write_python_class_file(self):
        """設定内容からジョイスティック用の Python クラスファイルを生成する。"""
        pyth_filename = None
        outfile = None
        while pyth_filename is None:
            print("これらの設定を新しい Python ファイルに書き出します")
            pyth_filename = input("ジョイスティックコードを生成するファイル名を入力してください [既定値: my_joystick.py]")
            if len(pyth_filename) == 0:
                pyth_filename = 'my_joystick.py'
            print('使用するファイル名:', pyth_filename)
            print()
            try:
                outfile = open(pyth_filename, "wt")
            except:
                ret = input("ファイルを開けませんでした。別のファイル名を入力しますか? [Y,n]")
                if ret == "n":
                    break
                pyth_filename = None
            print()
            
        if outfile is not None:
            classname = input("生成するジョイスティッククラス名は? [既定値: MyJoystick] ")
            if len(classname) == 0:
                classname = "MyJoystick"
            file_header = \
            '''
from donkeycar.parts.controller import Joystick, JoystickController


class %s(Joystick):
    #/dev/input/js0 で利用できる物理ジョイスティックへのインタフェース
    def __init__(self, *args, **kwargs):
        super(%s, self).__init__(*args, **kwargs)

            \n''' % (classname, classname )

            outfile.write(file_header)

            outfile.write('        self.button_names = {\n')
            for key, value in self.js.button_names.items():
                outfile.write("            %s : '%s',\n" % (str(hex(key)), str(value)))
            outfile.write('        }\n\n\n')
            
            outfile.write('        self.axis_names = {\n')

            for key, value in self.js.axis_names.items():
                outfile.write("            %s : '%s',\n" % (str(hex(key)), str(value)))
            outfile.write('        }\n\n\n')

            js_controller = \
            '''
class %sController(JoystickController):
    #入力をアクションへマッピングするコントローラーオブジェクト
    def __init__(self, *args, **kwargs):
        super(%sController, self).__init__(*args, **kwargs)


    def init_js(self):
        #ジョイスティックの初期化を試みる
        try:
            self.js = %s(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            print(self.dev_fn, "not found.")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        #ボタンから関数呼び出しへのマッピングを初期化
            \n''' % (classname, classname, classname)

            outfile.write(js_controller)

            outfile.write('        self.button_down_trigger_map = {\n')
            for button, control in self.mapped_controls:
                outfile.write("            '%s' : self.%s,\n" % (str(button), str(control)))
            outfile.write('        }\n\n\n')
            
            outfile.write('        self.axis_trigger_map = {\n')
            for axis, control in self.axis_map:
                outfile.write("            '%s' : self.%s,\n" % (str(axis), str(control)))
            outfile.write('        }\n\n\n')

            outfile.close()
            print(pyth_filename, "を書き出しました")

    def map_control_axis(self, control_name, control_fn):
        """指定された操作をコントローラーの軸へマッピングする。"""
        while True:
            axis = self.get_axis_action('%s 用に使用したい軸を動かしてください。2 秒間動かし続けます。' % control_name)
            
            mapped = False

            if axis is None:
                print("%s のマッピングは行われません" % control_name)
            else:
                #print("axis", axis)
                code = self.get_code_from_button(axis)
                for key, value in self.js.axis_names.items():
                    #print('key', key, 'value', value)
                    if key == code or value == code:
                        print('%s を %s に割り当てます\n' % (value, control_name))
                        mapped = value
                        break
            if mapped:
                ret = input('このマッピングでよいですか? (y, N) ')
                if ret.upper() == 'Y':
                    self.axis_map.append((mapped, control_fn))
                    return
            else:
                ret = input('軸が認識できません。再試行しますか? (Y, n) ')
                if ret.upper() == 'N':
                    return


    def map_steering_throttle(self):
        """ステアリングとスロットルの軸を設定する。"""

        self.axis_map = []

        self.print_config()
        print()
        print('これから各コントロールをアクションにマッピングします\n')

        print("まずはステアリングです")
        self.map_control_axis("steering", "set_steering")

        self.clear_scr()
        self.print_config()
        print()
        print("次にスロットルです")
        self.map_control_axis("throttle", "set_throttle")


    def map_button_controls(self):
        """ボタンを各コントロールへ割り当てる処理。"""
        unmapped_controls = [\
            ('toggle_mode', 'ユーザー・ローカル・ローカル角度の各走行モードを切り替え'),
            ('erase_last_N_records', '走行中の最新100件の記録を削除'),
            ('emergency_stop', '車を素早く停止させるため全力で逆転スロットルをかける'),
            ('increase_max_throttle', '最大スロットルを上げる。一定スロットル値にも使用'),
            ('decrease_max_throttle', '最大スロットルを下げる。一定スロットル値にも使用'),
            ('toggle_constant_throttle', '一定スロットル供給モードを切り替える'),
            ('toggle_manual_recording', '記録のオン・オフを切り替える')
        ]
        
        self.mapped_controls = []
        self.print_config()
        print()
        print("次にボタン押下を各コントロールへ割り当てます")
        print()

        while len(unmapped_controls) > 0:

            pt = PrettyTable()
            pt.field_names = ['Num', 'Control', 'Help']
            print("未割り当てのコントロール:")
            for i, td in enumerate(unmapped_controls):
                control, help = td
                pt.add_row([i + 1, control, help])
            print(pt)

            print()
            try:
                ret = " "
                while (not ret.isdigit() and ret.upper() != 'D') or (ret.isdigit() and (int(ret) < 1 or int(ret) > len(unmapped_controls))):
                    ret = input("割り当てるコントロールの番号を押してください (1-%d)。終了する場合は D: " % len(unmapped_controls))

                if ret.upper() == 'D':
                    break

                iControl = int(ret) - 1
            except:
                continue

            
            print('コントロール %s に割り当てるボタンを押してください:' % unmapped_controls[iControl][0])
            self.get_button_press()

            if self.last_button is None:
                print("10 秒間ボタンが押されませんでした")
                ret = input("割り当てを続けますか? [Y, n]")
                if ret == 'n':
                    break
            else:
                code = self.get_code_from_button(self.last_button)
                if code in self.js.button_names: 
                    button_name = self.js.button_names[code]
                else:
                    button_name = self.last_button
                self.mapped_controls.append((button_name, unmapped_controls[iControl][0]))
                unmapped_controls.pop(iControl)
                self.clear_scr()
                self.print_config()
                print()

        print('コントロールの割り当てが完了しました')
        print()

    def revisit_topic(self):
        """設定を再調整するかどうかを尋ねるメニュー。"""
        done = False
        while not done:
            self.clear_scr()
            self.print_config()
            print("設定はほぼ完了しました。この内容でよいですか、それともどこかやり直しますか?")
            print("H) 満足したので Python ファイルを書き出す")
            print("B) ボタン名を付け直す")
            print("A) 軸名を付け直す")
            print("T) スロットルとステアリングの再マッピング")
            print("R) ボタンとコントロールの再マッピング")
            
            ret = input("項目を選択してください ").upper()
            if ret == 'H':
                done = True
            elif ret == 'B':
                self.name_buttons()
            elif ret == 'A':
                self.name_axes()
            elif ret == 'T':
                self.map_steering_throttle()
            elif ret == 'R':
                self.map_button_controls()          


    def get_axis_action(self, prompt):
        """軸操作を検出し、名称を返す補助メソッド。

        Args:
            prompt (str): ユーザーへ表示するメッセージ。

        Returns:
            str | None: 検出した軸名。キャンセルされた場合は ``None``。
        """
        done = False
        while not done:
            print(prompt)
            ret = input("開始するには Enter、終了するには D を押してください ")
            if ret.upper() == 'D':
                return None

            most_movement = self.get_axis_move()

            if most_movement is None:
                print("動きが検出できませんでした")
                res = input("もう一度試しますか? [Y/n]: ")
                if res == "n":
                    return None
                else:
                    continue
            else:
                return most_movement


    def shutdown(self):
        """ポーリングスレッドを停止する。"""
        self.running = False
        if self.thread:
            self.thread = None

    def parse_args(self, args):
        """コマンドライン引数を解析する。"""
        parser = argparse.ArgumentParser(prog='createjs', usage='%(prog)s [options]')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        """ウィザードを実行するエントリーポイント。"""
        args = self.parse_args(args)
        try:
            self.create_joystick(args)
        except KeyboardInterrupt:
            self.shutdown()

