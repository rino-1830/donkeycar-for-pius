import moviepy.editor as mpy
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
import tensorflow as tf
import cv2
from matplotlib import cm
try:
    from vis.utils import utils
except:
    raise Exception("keras-vis をインストールしてください: pip install git+https://github.com/autorope/keras-vis.git")

import donkeycar as dk
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import *


DEG_TO_RAD = math.pi / 180.0


class MakeMovie(object):

    def run(self, args, parser):
        """tub から画像を読み込み、動画を生成する。

        Args:
            args: コマンドライン引数を格納した ``argparse`` オブジェクト。
            parser: エラー表示に使用する ``argparse`` のパーサー。

        Returns:
            None
        """

        if args.tub is None:
            print("ERR>> --tub 引数がありません")
            parser.print_help()
            return

        conf = os.path.expanduser(args.config)
        if not os.path.exists(conf):
            print("設定ファイルが %s にありません。--config で場所を指定するか、config.py があるディレクトリで実行してください" % conf)
            return

        self.cfg = dk.load_config(conf)

        if args.type is None and args.model is not None:
            args.type = self.cfg.DEFAULT_MODEL_TYPE
            print("モデルタイプが指定されていないため、設定ファイルのデフォルトを使用します")

        if args.salient:
            if args.model is None:
                print("ERR>> salient 可視化にはモデルが必要です。--model 引数で指定してください")
                parser.print_help()

            if args.type not in ['linear', 'categorical']:
                print("モデルタイプ {} はサポートされていません。salient 可視化では linear または categorical のみ対応しています".format(args.type))
                parser.print_help()
                return

        self.model_type = args.type
        self.tub = Tub(args.tub)

        start = args.start
        self.end_index = args.end if args.end != -1 else len(self.tub)
        num_frames = self.end_index - start

        # 適切なオフセット位置まで移動する
        self.current = 0
        self.iterator = self.tub.__iter__()
        while self.current < start:
            self.iterator.next()
            self.current += 1

        self.scale = args.scale
        self.keras_part = None
        self.do_salient = False
        self.user = args.draw_user_input
        if args.model is not None:
            self.keras_part = get_model_by_type(args.type, cfg=self.cfg)
            self.keras_part.load(args.model)
            if args.salient:
                self.do_salient = self.init_salient(self.keras_part.interpreter.model)

        print('動画', args.out, 'を', num_frames, '枚の画像から作成します')
        clip = mpy.VideoClip(self.make_frame, duration=((num_frames - 1) / self.cfg.DRIVE_LOOP_HZ))
        clip.write_videofile(args.out, fps=self.cfg.DRIVE_LOOP_HZ)

    @staticmethod
    def draw_line_into_image(angle, throttle, is_left, img, color):
        """ステアリング値を線として画像に描き込む。"""
        import cv2

        height = img.shape[0]
        width = img.shape[1]
        length = height
        a1 = angle * 45.0
        l1 = throttle * length
        mid = width // 2 + (- 1 if is_left else +1)

        p1 = tuple((mid - 2, height - 1))
        p11 = tuple((int(p1[0] + l1 * math.cos((a1 + 270.0) * DEG_TO_RAD)),
                     int(p1[1] + l1 * math.sin((a1 + 270.0) * DEG_TO_RAD))))

        cv2.line(img, p1, p11, color, 2)

    def draw_user_input(self, record, img, img_drawon):
        """ユーザー操作を画像上に緑色の線として描画する。"""
        user_angle = float(record["user/angle"])
        user_throttle = float(record["user/throttle"])
        green = (0, 255, 0)
        self.draw_line_into_image(user_angle, user_throttle, False, img_drawon, green)

    def draw_model_prediction(self, img, img_drawon):
        """モデルの予測結果を問い合わせ、青い線として画像上に描画する。"""
        if self.keras_part is None:
            return

        expected = tuple(self.keras_part.get_input_shapes()[0][1:])
        actual = img.shape

        # モデルがグレースケールを想定している場合は、RGB から変換する
        if expected[2] == 1 and actual[2] == 3:
            # 変換前に画像を正規化する
            grey_img = rgb2gray(img)
            actual = grey_img.shape
            img = grey_img.reshape(grey_img.shape + (1,))

        if expected != actual:
            print(f"期待する入力次元 {expected} と実際の次元 {actual} が一致しません")
            return

        blue = (0, 0, 255)
        pilot_angle, pilot_throttle = self.keras_part.run(img)
        self.draw_line_into_image(pilot_angle, pilot_throttle, True, img_drawon, blue)

    def draw_steering_distribution(self, img, img_drawon):
        """モデルの予測分布を描画する。KerasCategorical モデル専用。"""
        from donkeycar.parts.keras import KerasCategorical

        if self.keras_part is None or type(self.keras_part) is not KerasCategorical:
            return        
        pred_img = normalize_image(img)
        angle_binned, _ = self.keras_part.interpreter.predict(pred_img, other_arr=None)

        x = 4
        dx = 4
        y = 120 - 4
        iArgMax = np.argmax(angle_binned)
        for i in range(15):
            p1 = (x, y)
            p2 = (x, y - int(angle_binned[i] * 100.0))
            if i == iArgMax:
                cv2.line(img_drawon, p1, p2, (255, 0, 0), 2)
            else:
                cv2.line(img_drawon, p1, p2, (200, 200, 200), 2)
            x += dx

    def init_salient(self, model):
        """salient マスク生成の準備を行う。"""
        # レイヤー名からインデックスを検索するユーティリティ。
        # -1 を指定すれば最後のレイヤーを指すことも可能。
        output_name = []
        layer_idx = []
        for i, layer in enumerate(model.layers):
            if "dropout" not in layer.name.lower() and "out" in layer.name.lower():
                output_name.append(layer.name)
                layer_idx.append(i)

        if output_name is []:
            print("'out' という名前のレイヤーが見つかりません。salient 表示をスキップします")
            return False

        print("####################")
        print("レイヤーの活性化を可視化します:", output_name)
        print("####################")
        
        # 活性化関数を線形に強制する
        for li in layer_idx:
            model.layers[li].activation = activations.linear
        # salient 表示用モデルとオプティマイザを構築
        sal_model = utils.apply_modifications(model)
        self.sal_model = sal_model
        return True

    def compute_visualisation_mask(self, img):
        """画像から可視化用マスクを計算する。"""
        img = img.reshape((1,) + img.shape)
        images = tf.Variable(img, dtype=float)

        if self.model_type == 'linear':
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(images)
                pred_list = self.sal_model(images, training=False)
        elif self.model_type == 'categorical':
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(images)
                pred = self.sal_model(images, training=False)
                pred_list = []
                for p in pred:
                    maxindex = tf.math.argmax(p[0])
                    pred_list.append(p[0][maxindex])
                    
        grads = 0
        for p in pred_list:
            grad = tape.gradient(p, images)
            grads += tf.math.square(grad)
        grads = tf.math.sqrt(grads)

        channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
        grads = np.sum(grads, axis=channel_idx)
        res = utils.normalize(grads)[0]
        return res

    def draw_salient(self, img):
        """salient マスクを重ね合わせた画像を生成する。"""

        alpha = 0.004
        beta = 1.0 - alpha
        expected = self.keras_part.interpreter.model.inputs[0].shape[1:]
        actual = img.shape

        # 入力のチャンネル数を確認し、必要ならグレースケールへ変換する
        if expected[2] == 1 and actual[2] == 3:
            grey_img = rgb2gray(img)
            img = grey_img.reshape(grey_img.shape + (1,))

        norm_img = normalize_image(img)
        salient_mask = self.compute_visualisation_mask(norm_img)
        salient_mask_stacked = cm.inferno(salient_mask)[:,:,0:3]
        salient_mask_stacked = cv2.GaussianBlur(salient_mask_stacked,(3,3),cv2.BORDER_DEFAULT)
        blend = cv2.addWeighted(img.astype('float32'), alpha, salient_mask_stacked.astype('float32'), beta, 0)
        return blend

    def make_frame(self, t):
        """VideoClip からのコールバックでフレーム画像を返す。

        Args:
            t: ``VideoClip`` から渡される時刻。ここでは使用しない。

        Returns:
            numpy.ndarray: 8 ビット RGB 画像。
        """

        if self.current >= self.end_index:
            return None

        rec = self.iterator.next()
        img_path = os.path.join(self.tub.images_base_path, rec['cam/image_array'])
        image_input = img_to_arr(Image.open(img_path))
        image = image_input
        
        if self.do_salient:
            image = self.draw_salient(image_input)
            image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        if self.user: self.draw_user_input(rec, image_input, image)
        if self.keras_part is not None:
            self.draw_model_prediction(image_input, image)
            self.draw_steering_distribution(image_input, image)

        if self.scale != 1:
            h, w, d = image.shape
            dsize = (w * self.scale, h * self.scale)
            image = cv2.resize(image, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            image = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)

        self.current += 1
        # 8 ビット RGB 配列を返す
        return image
