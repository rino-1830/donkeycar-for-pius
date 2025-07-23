import argparse
import os
import shutil
import socket
import stat
import sys
import logging

from progress.bar import IncrementalBar
import donkeycar as dk
from donkeycar.management.joystick_creator import CreateJoystick
from donkeycar.management.tub import TubManager

from donkeycar.utils import normalize_image, load_image, math

PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TEMPLATES_PATH = os.path.join(PACKAGE_PATH, 'templates')
HELP_CONFIG = '使用する設定ファイルの場所。デフォルト: ./config.py'
logger = logging.getLogger(__name__)


def make_dir(path):
    real_path = os.path.expanduser(path)
    print('ディレクトリを作成中 ', real_path)
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    return real_path


def load_config(config_path, myconfig='myconfig.py'):
    """指定されたパスから設定を読み込む。

    Args:
        config_path: 読み込む設定ファイルのパス。
        myconfig: 読み込む追加設定ファイル名。

    Returns:
        読み込んだ設定オブジェクト。失敗した場合は ``None``。
    """
    conf = os.path.expanduser(config_path)
    if not os.path.exists(conf):
        logger.error(
            f"設定ファイルが見つかりません: {conf}. --config を使用して場所を指定する"\
            "か、config.py を含むディレクトリで実行してください。"
        )
        return None

    try:
        cfg = dk.load_config(conf, myconfig)
    except Exception as e:
        logger.error(f"{conf} の読み込み中に例外が発生しました: {e}")
        return None

    return cfg


class BaseCommand(object):
    pass


class CreateCar(BaseCommand):

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='createcar', usage='%(prog)s [options]')
        parser.add_argument('--path', default=None, help='車フォルダーを作成する場所')
        parser.add_argument('--template', default=None, help='使用する車テンプレート名')
        parser.add_argument('--overwrite', action='store_true', help='既存のファイルを置き換えるか')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        self.create_car(path=args.path, template=args.template, overwrite=args.overwrite)

    def create_car(self, path, template='complete', overwrite=False):
        """Donkey が動作するためのフォルダー構成を作成する。

        Donkey がインストールされていない環境でも実行できるようにし、
        Docker 利用者がマウントするフォルダー構成を生成できる。
        """

        # path が None の場合に備えた設定
        path = path or '~/mycar'
        template = template or 'complete'
        print(f"車フォルダーを作成します: {path}")
        path = make_dir(path)

        print("データとモデルのフォルダーを作成します。")
        folders = ['models', 'data', 'logs']
        folder_paths = [os.path.join(path, f) for f in folders]
        for fp in folder_paths:
            make_dir(fp)

        # car アプリケーションと設定ファイルが存在しない場合に追加する
        app_template_path = os.path.join(TEMPLATES_PATH, template+'.py')
        config_template_path = os.path.join(TEMPLATES_PATH, 'cfg_' + template + '.py')
        myconfig_template_path = os.path.join(TEMPLATES_PATH, 'myconfig.py')
        train_template_path = os.path.join(TEMPLATES_PATH, 'train.py')
        calibrate_template_path = os.path.join(TEMPLATES_PATH, 'calibrate.py')
        car_app_path = os.path.join(path, 'manage.py')
        car_config_path = os.path.join(path, 'config.py')
        mycar_config_path = os.path.join(path, 'myconfig.py')
        train_app_path = os.path.join(path, 'train.py')
        calibrate_app_path = os.path.join(path, 'calibrate.py')

        if os.path.exists(car_app_path) and not overwrite:
            print('車アプリケーションは既に存在します。削除してから createcar を再実行してください。')
        else:
            print(f"車アプリケーションのテンプレート {template} をコピーします")
            shutil.copyfile(app_template_path, car_app_path)
            os.chmod(car_app_path, stat.S_IRWXU)

        if os.path.exists(car_config_path) and not overwrite:
            print('車の設定ファイルは既に存在します。削除してから createcar を再実行してください。')
        else:
            print("車の設定のデフォルトをコピーします。車を起動する前に調整してください。")
            shutil.copyfile(config_template_path, car_config_path)

        if os.path.exists(train_app_path) and not overwrite:
            print('train スクリプトは既に存在します。削除してから createcar を再実行してください。')
        else:
            print("train スクリプトをコピーします。車を起動する前に調整してください。")
            shutil.copyfile(train_template_path, train_app_path)
            os.chmod(train_app_path, stat.S_IRWXU)

        if os.path.exists(calibrate_app_path) and not overwrite:
            print('calibrate スクリプトは既に存在します。削除してから createcar を再実行してください。')
        else:
            print("calibrate スクリプトをコピーします。車を起動する前に調整してください。")
            shutil.copyfile(calibrate_template_path, calibrate_app_path)
            os.chmod(calibrate_app_path, stat.S_IRWXU)

        if not os.path.exists(mycar_config_path):
            print("my car 用の設定オーバーライドをコピーします")
            shutil.copyfile(myconfig_template_path, mycar_config_path)
            # config から myconfig へ内容をコピーする。すべての行をコメントアウトする。
            cfg = open(car_config_path, "rt")
            mcfg = open(mycar_config_path, "at")
            copy = False
            for line in cfg:
                if "import os" in line:
                    copy = True
                if copy:
                    mcfg.write("# " + line)
            cfg.close()
            mcfg.close()

        print("Donkey のセットアップが完了しました。")


class UpdateCar(BaseCommand):
    """常に ``~/mycar`` ディレクトリで実行して最新状態へ更新する。"""

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='update', usage='%(prog)s [options]')
        parser.add_argument('--template', default=None, help='使用する車テンプレート名')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        cc = CreateCar()
        cc.create_car(path=".", overwrite=True, template=args.template)


class FindCar(BaseCommand):
    """PC と車の IP アドレスを検索するユーティリティ。"""
    def parse_args(self, args):
        pass

    def run(self, args):
        print('コンピュータのIPアドレスを調べています...')
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        print('あなたのIPアドレス: %s ' % s.getsockname()[0])
        s.close()

        print('車のIPアドレスを検索しています...')
        cmd = "sudo nmap -sP " + ip + "/24 | awk '/^Nmap/{ip=$NF}/B8:27:EB/{print ip}'"
        cmdRPi4 = "sudo nmap -sP " + ip + "/24 | awk '/^Nmap/{ip=$NF}/DC:A6:32/{print ip}'"
        print('車のIPアドレス:')
        os.system(cmd)
        os.system(cmdRPi4)


class CalibrateCar(BaseCommand):
    """PWM 出力を調整するための対話型ユーティリティ。"""

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='calibrate', usage='%(prog)s [options]')
        parser.add_argument(
            '--pwm-pin',
            help="調整したいピンを指定する PwmPin 形式。例: 'RPI_GPIO.BOARD.33' または 'PCA9685.1:40.13'")
        parser.add_argument('--channel', default=None, help="調整したい PCA9685 のチャンネル [0-15]")
        parser.add_argument(
            '--address',
            default='0x40',
            help="調整対象の PCA9685 の I2C アドレス [デフォルト 0x40]")
        parser.add_argument(
            '--bus',
            default=None,
            help="調整対象の PCA9685 の I2C バス [デフォルトは自動検出]")
        parser.add_argument('--pwmFreq', default=60, help="PWM で使用する周波数")
        parser.add_argument(
            '--arduino',
            dest='arduino',
            action='store_true',
            help='Arduino ピンを PWM に使用する (calibrate pin=<channel>)')
        parser.set_defaults(arduino=False)
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)

        if args.arduino:
            from donkeycar.parts.actuator import ArduinoFirmata

            channel = int(args.channel)
            arduino_controller = ArduinoFirmata(servo_pin=channel)
            print('Arduino の PWM をピン %d で初期化します' % (channel))
            input_prompt = "テストする PWM 値を入力してください('q' で終了) (0-180): "

        elif args.pwm_pin is not None:
            from donkeycar.parts.actuator import PulseController
            from donkeycar.parts import pins

            pwm_pin = None
            try:
                pwm_pin = pins.pwm_pin_by_id(args.pwm_pin)
            except ValueError as e:
                print(e)
                print("ピン指定文字列の説明は pins.py を参照してください。")
                exit(-1)
            print(f'ピン {args.pwm_pin} を初期化します')
            freq = int(args.pwmFreq)
            print(f"PWM 周波数: {freq}")
            c = PulseController(pwm_pin)
            input_prompt = "テストする PWM 値を入力してください('q' で終了) (0-1500): "
            print()

        else:
            from donkeycar.parts.actuator import PCA9685
            from donkeycar.parts.sombrero import Sombrero

            Sombrero()  # Sombrero ハット用にピンを設定

            channel = int(args.channel)
            busnum = None
            if args.bus:
                busnum = int(args.bus)
            address = int(args.address, 16)
            print('PCA9685 をチャンネル %d アドレス %s バス %s で初期化します' % (channel, str(hex(address)), str(busnum)))
            freq = int(args.pwmFreq)
            print(f"PWM 周波数: {freq}")
            c = PCA9685(channel, address=address, busnum=busnum, frequency=freq)
            input_prompt = "テストする PWM 値を入力してください('q' で終了) (0-1500): "
            print()

        while True:
            try:
                val = input(input_prompt)
                if val == 'q' or val == 'Q':
                    break
                pmw = int(val)
                if args.arduino == True:
                    arduino_controller.set_pulse(channel, pmw)
                else:
                    c.run(pmw)
            except KeyboardInterrupt:
                print("\nキーボード割り込みを受信しました。終了します。")
                break
            except Exception as ex:
                print(f"エラーが発生しました: {ex}")


class MakeMovieShell(BaseCommand):
    """make movie コマンドを遅延インポートで呼び出すためのラッパー。"""
    def __init__(self):
        self.deg_to_rad = math.pi / 180.0

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='makemovie')
        parser.add_argument('--tub', help='動画を作成する対象の tub データ')
        parser.add_argument(
            '--out',
            default='tub_movie.mp4',
            help='作成する動画ファイル名。デフォルト: tub_movie.mp4')
        parser.add_argument('--config', default='./config.py', help=HELP_CONFIG)
        parser.add_argument('--model', default=None, help='制御出力表示に使用するモデル')
        parser.add_argument('--type', default=None, required=False, help='読み込むモデルタイプ')
        parser.add_argument('--salient', action="store_true", help='活性化を示すサリエントマップを重ねるか')
        parser.add_argument('--start', type=int, default=0, help='処理を開始する最初のフレーム')
        parser.add_argument('--end', type=int, default=-1, help='処理を終了する最後のフレーム')
        parser.add_argument('--scale', type=int, default=2, help='画像フレーム出力を倍にする倍率')
        parser.add_argument(
            '--draw-user-input',
            default=True, action='store_false',
            help='ビデオにユーザー入力を表示する')
        parsed_args = parser.parse_args(args)
        return parsed_args, parser

    def run(self, args):
        """tub 内の画像から動画を生成する。"""
        args, parser = self.parse_args(args)

        from donkeycar.management.makemovie import MakeMovie

        mm = MakeMovie()
        mm.run(args, parser)


class ShowHistogram(BaseCommand):

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='tubhist',
                                         usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='tub のパス')
        parser.add_argument('--record', default=None,
                            help='ヒストグラムを作成するレコード名')
        parser.add_argument('--out', default=None,
                            help='ヒストグラム画像の保存先（.pngで終わるパス）')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def show_histogram(self, tub_paths, record_name, out):
        """指定した tub 内で記録タイプの頻度をヒストグラムとして生成する。"""
        import pandas as pd
        from matplotlib import pyplot as plt
        from donkeycar.parts.tub_v2 import Tub

        output = out or os.path.basename(tub_paths)
        path_list = tub_paths.split(",")
        records = [record for path in path_list for record
                   in Tub(path, read_only=True)]
        df = pd.DataFrame(records)
        df.drop(columns=["_index", "_timestamp_ms"], inplace=True)
        # これは画面に表示するだけ
        if record_name is not None:
            df[record_name].hist(bins=50)
        else:
            df.hist(bins=50)

        try:
            if out is not None:
                filename = output
            else:
                if record_name is not None:
                    filename = f"{output}_hist_{record_name.replace('/', '_')}.png"
                else:
                    filename = f"{output}_hist.png"
            plt.savefig(filename)
            logger.info(f'画像を保存しました: {filename}')
        except Exception as e:
            logger.error(str(e))
        plt.show()

    def run(self, args):
        args = self.parse_args(args)
        if isinstance(args.tub, list):
            args.tub = ','.join(args.tub)
        self.show_histogram(args.tub, args.record, args.out)


class ShowCnnActivations(BaseCommand):

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt

    def get_activations(self, image_path, model_path, cfg):
        """画像から特徴量を抽出し、活性化マップを返す。"""
        from tensorflow.python.keras.models import load_model, Model

        model_path = os.path.expanduser(model_path)
        image_path = os.path.expanduser(image_path)

        model = load_model(model_path, compile=False)
        image = load_image(image_path, cfg)[None, ...]

        conv_layer_names = self.get_conv_layers(model)
        input_layer = model.get_layer(name='img_in').input
        activations = []
        for conv_layer_name in conv_layer_names:
            output_layer = model.get_layer(name=conv_layer_name).output

            layer_model = Model(inputs=[input_layer], outputs=[output_layer])
            activations.append(layer_model.predict(image)[0])
        return activations

    def create_figure(self, activations):
        import math
        cols = 6

        for i, layer in enumerate(activations):
            fig = self.plt.figure()
            fig.suptitle(f'Layer {i+1}')

            print(f'layer {i+1} shape: {layer.shape}')
            feature_maps = layer.shape[2]
            rows = math.ceil(feature_maps / cols)

            for j in range(feature_maps):
                self.plt.subplot(rows, cols, j + 1)

                self.plt.imshow(layer[:, :, j])

        self.plt.show()

    def get_conv_layers(self, model):
        conv_layers = []
        for layer in model.layers:
            if layer.__class__.__name__ == 'Conv2D':
                conv_layers.append(layer.name)
        return conv_layers

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='cnnactivations', usage='%(prog)s [options]')
        parser.add_argument('--image', help='画像へのパス')
        parser.add_argument('--model', default=None, help='モデルへのパス')
        parser.add_argument('--config', default='./config.py', help=HELP_CONFIG)

        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        cfg = load_config(args.config)
        activations = self.get_activations(args.image, args.model, cfg)
        self.create_figure(activations)


class ShowPredictionPlots(BaseCommand):
    """モデルの予測値を可視化するクラス。"""

    def plot_predictions(self, cfg, tub_paths, model_path, limit, model_type,
                         noshow):
        """角度とスロットルの予測結果を tub のデータと比較して描画する。"""
        import matplotlib.pyplot as plt
        import pandas as pd
        from pathlib import Path
        from donkeycar.pipeline.types import TubDataset

        model_path = os.path.expanduser(model_path)
        model = dk.utils.get_model_by_type(model_type, cfg)
        # これはプロットタイトル用の文字列を取得するだけ
        if model_type is None:
            model_type = cfg.DEFAULT_MODEL_TYPE
        model.load(model_path)

        user_angles = []
        user_throttles = []
        pilot_angles = []
        pilot_throttles = []

        base_path = Path(os.path.expanduser(tub_paths)).absolute().as_posix()
        dataset = TubDataset(config=cfg, tub_paths=[base_path],
                             seq_size=model.seq_size())
        records = dataset.get_records()[:limit]
        bar = IncrementalBar('Inferencing', max=len(records))

        output_names = list(model.output_shapes()[1].keys())
        for tub_record in records:
            input_dict = model.x_transform(
                tub_record, lambda x: normalize_image(x))
            pilot_angle, pilot_throttle = \
                model.inference_from_dict(input_dict)
            y_dict = model.y_transform(tub_record)
            user_angle, user_throttle \
                = y_dict[output_names[0]], y_dict[output_names[1]]
            user_angles.append(user_angle)
            user_throttles.append(user_throttle)
            pilot_angles.append(pilot_angle)
            pilot_throttles.append(pilot_throttle)
            bar.next()
        print()  # プログレスバーの後で改行するため

        angles_df = pd.DataFrame({'user_angle': user_angles,
                                  'pilot_angle': pilot_angles})
        throttles_df = pd.DataFrame({'user_throttle': user_throttles,
                                     'pilot_throttle': pilot_throttles})

        fig = plt.figure()
        title = f"Model Predictions\nTubs: {tub_paths}\nModel: {model_path}\n" \
                f"Type: {model_type}"
        fig.suptitle(title)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        angles_df.plot(ax=ax1)
        throttles_df.plot(ax=ax2)
        ax1.legend(loc=4)
        ax2.legend(loc=4)
        plt.savefig(model_path + '_pred.png')
        logger.info(f'{model_path}_pred.png に tubplot を保存しました')
        if not noshow:
            plt.show()

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='tubplot', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='プロットを作成する対象の tub')
        parser.add_argument('--model', default=None, help='予測に使用するモデル')
        parser.add_argument('--limit', type=int, default=1000, help='処理するレコード数')
        parser.add_argument('--type', default=None, help='モデルタイプ')
        parser.add_argument('--noshow', default=False, action="store_true",
                            help='ウィンドウにプロットを表示しない')
        parser.add_argument('--config', default='./config.py', help=HELP_CONFIG)

        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        args.tub = ','.join(args.tub)
        cfg = load_config(args.config)
        self.plot_predictions(cfg, args.tub, args.model, args.limit,
                              args.type, args.noshow)


class Train(BaseCommand):
    """モデル学習を実行するコマンド。"""

    def parse_args(self, args):
        HELP_FRAMEWORK = 'the AI framework to use (tensorflow|pytorch). ' \
                         'Defaults to config.DEFAULT_AI_FRAMEWORK'
        parser = argparse.ArgumentParser(prog='train', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='学習に使用する tub データ')
        parser.add_argument('--model', default=None, help='出力モデル名')
        parser.add_argument('--type', default=None, help='モデルタイプ')
        parser.add_argument('--config', default='./config.py', help=HELP_CONFIG)
        parser.add_argument('--myconfig', default='./myconfig.py',
                            help='myconfig ファイル名。デフォルトは myconfig.py')
        parser.add_argument('--framework',
                            choices=['tensorflow', 'pytorch', None],
                            required=False,
                            help=HELP_FRAMEWORK)
        parser.add_argument('--checkpoint', type=str,
                            help='再開するチェックポイントの場所')
        parser.add_argument('--transfer', type=str, help='転移学習用のモデル')
        parser.add_argument('--comment', type=str,
                            help='モデルデータベースに追加するコメント - 複数単語の場合はダブルクオートを使用')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        args.tub = ','.join(args.tub)
        my_cfg = args.myconfig
        cfg = load_config(args.config, my_cfg)
        framework = args.framework if args.framework \
            else getattr(cfg, 'DEFAULT_AI_FRAMEWORK', 'tensorflow')

        if framework == 'tensorflow':
            from donkeycar.pipeline.training import train
            train(cfg, args.tub, args.model, args.type, args.transfer,
                  args.comment)
        elif framework == 'pytorch':
            from donkeycar.parts.pytorch.torch_train import train
            train(cfg, args.tub, args.model, args.type,
                  checkpoint_path=args.checkpoint)
        else:
            logger.error(
                f"認識できないフレームワークです: {framework}. 'tensorflow' または 'pytorch' を指定してください"
            )


class ModelDatabase(BaseCommand):
    """保存されたモデルの情報を一覧表示する。"""

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='models',
                                         usage='%(prog)s [options]')
        parser.add_argument('--config', default='./config.py', help=HELP_CONFIG)
        parser.add_argument('--group', action="store_true",
                            default=False,
                            help='tub をグループ化して個別にプロットする')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        from donkeycar.pipeline.database import PilotDatabase
        args = self.parse_args(args)
        cfg = load_config(args.config)
        p = PilotDatabase(cfg)
        pilot_txt, tub_txt, _ = p.pretty_print(args.group)
        print(pilot_txt)
        print(tub_txt)


class Gui(BaseCommand):
    """Kivy ベースの GUI を起動するコマンド。"""
    def run(self, args):
        from donkeycar.management.kivy_ui import main
        main()


def execute_from_command_line():
    """"donkey" コマンドから呼び出される関数。"""
    commands = {
        'createcar': CreateCar,
        'findcar': FindCar,
        'calibrate': CalibrateCar,
        'tubclean': TubManager,
        'tubplot': ShowPredictionPlots,
        'tubhist': ShowHistogram,
        'makemovie': MakeMovieShell,
        'createjs': CreateJoystick,
        'cnnactivations': ShowCnnActivations,
        'update': UpdateCar,
        'train': Train,
        'models': ModelDatabase,
        'ui': Gui,
    }

    args = sys.argv[:]

    if len(args) > 1 and args[1] in commands.keys():
        command = commands[args[1]]
        c = command()
        c.run(args[2:])
    else:
        dk.utils.eprint('使用方法: 利用可能なコマンドは次の通りです:')
        dk.utils.eprint(list(commands.keys()))


if __name__ == "__main__":
    execute_from_command_line()
