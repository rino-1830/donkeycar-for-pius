"""OpenCV を用いた画像処理パーツ群."""

import time
import cv2
import numpy as np
import logging

from donkeycar.parts.camera import CameraError

logger = logging.getLogger(__name__)


def image_shape(image):
    """画像の形状を返す。

    2 次元配列の場合は高さ、幅、1 を返す。

    Args:
        image (numpy.ndarray): 入力画像。

    Returns:
        tuple: ``(height, width, channels)`` 形式のタプル。
    """

    if image is None:
        return None
    if 2 == len(image.shape):
        height, width = image.shape
        return height, width, 1
    return image.shape


class ImgGreyscale:
    """RGB画像をグレースケールに変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            return img_arr
        except:
            logger.error("RGB画像をグレースケールへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgGRAY2RGB:
    """グレースケール画像を RGB に変換するパーツ."""
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        except:
            logger.error(F"グレースケール画像({img_arr.shape})をRGBへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgGRAY2BGR:
    """グレースケール画像を BGR に変換するパーツ."""
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)
        except:
            logger.error(F"グレースケール画像({img_arr.shape})をRGBへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgRGB2GRAY(ImgGreyscale):
    """RGB画像をグレースケールに変換するエイリアス."""


class ImgBGR2GRAY:
    """BGR画像をグレースケールに変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            return img_arr
        except:
            logger.error("BGR画像をグレースケールへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgHSV2GRAY:
    """HSV画像をグレースケールに変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_HSV2GRAY)
            return img_arr
        except:
            logger.error("HSV画像をグレースケールへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgWriter:
    """画像をファイルへ書き出すパーツ."""

    def __init__(self, filename):
        self.filename = filename

    def run(self, img_arr):
        cv2.imwrite(self.filename, img_arr)

    def shutdown(self):
        pass


class ImgBGR2RGB:
    """BGR画像を RGB に変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            return img_arr
        except:
            logger.error("BGR画像をRGBへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgRGB2BGR:
    """RGB画像を BGR に変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            return img_arr
        except:
            logger.error("RGB画像をBGRへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgHSV2RGB:
    """HSV画像を RGB に変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_HSV2RGB)
            return img_arr
        except:
            logger.error("HSV画像をRGBへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgRGB2HSV:
    """RGB画像を HSV に変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
            return img_arr
        except:
            logger.error("RGB画像をHSVへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgHSV2BGR:
    """HSV画像を BGR に変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_HSV2BGR)
            return img_arr
        except:
            logger.error("HSV画像をBGRへ変換できません")
            return None

    def shutdown(self):
        pass


class ImgBGR2HSV:
    """BGR画像を HSV に変換するパーツ."""

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2HSV)
            return img_arr
        except:
            logger.error("BGR画像をHSVへ変換できません")
            return None

    def shutdown(self):
        pass


class ImageScale:
    """画像を指定倍率で拡大・縮小するパーツ."""

    def __init__(self, scale, scale_height=None):
        if scale is None or scale <= 0:
            raise ValueError("ImageScale: scale must be > 0")
        if scale_height is not None and scale_height <= 0:
            raise ValueError("ImageScale: scale_height must be > 0")
        self.scale = scale
        self.scale_height = scale_height if scale_height is not None else scale

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.resize(img_arr, (0,0), fx=self.scale, fy=self.scale_height)
        except:
            logger.error("画像のスケーリングに失敗しました")
            return None

    def shutdown(self):
        pass


class ImageResize:
    """画像を指定された幅と高さにリサイズするパーツ."""
    def __init__(self, width:int, height:int) -> None:
        if width is None or width <= 0:
            raise ValueError("ImageResize: width must be > 0")
        if height is None or height <= 0:
            raise ValueError("ImageResize: height must be > 0")
        self.width = width
        self.height = height

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.resize(img_arr, (self.width, self.height))
        except:
            logger.error("画像のリサイズに失敗しました")
            return None

    def shutdown(self):
        pass


class ImageRotateBound:
    """画像を回転させるパーツ.

    参考: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """

    def __init__(self, rot_deg):
        self.rot_deg = rot_deg

    def run(self, image):
        if image is None:
            return None

        # 画像のサイズを取得して中心座標を計算する
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # 回転行列を取得し（時計回りに回転させるため角度を負にする）、
        # 行列からサイン・コサイン成分を取り出す
        M = cv2.getRotationMatrix2D((cX, cY), -self.rot_deg, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # 回転後の画像サイズを計算する
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # 移動量を考慮して回転行列を調整する
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # 回転処理を行い画像を返す
        return cv2.warpAffine(image, M, (nW, nH))

    def shutdown(self):
        pass


class ImgCanny:
    """Cannyエッジ検出を行うパーツ."""

    def __init__(self, low_threshold=60, high_threshold=110, aperture_size=3, l2gradient=False):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size   # 3、5、7 のいずれか
        self.l2gradient = l2gradient

    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.Canny(img_arr,
                             self.low_threshold,
                             self.high_threshold,
                             apertureSize=self.aperture_size,
                             L2gradient=self.l2gradient)
        except:
            logger.error("Cannyエッジ検出の適用に失敗しました")
            return None

    def shutdown(self):
        pass


class ImgGaussianBlur:
    """Gaussian ブラーを適用するパーツ."""

    def __init__(self, kernel_size=5, kernel_y=None):
        self.kernel_size = (kernel_size, kernel_y if kernel_y is not None else kernel_size)
        
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.GaussianBlur(img_arr,
                                    self.kernel_size, 
                                    0)
        except:
            logger.error("ガウシアンブラーの適用に失敗しました")
            return None

    def shutdown(self):
        pass


class ImgSimpleBlur:
    """単純な平均化ブラーを適用するパーツ."""

    def __init__(self, kernel_size=5, kernel_y=None):
        self.kernel_size = (kernel_size, kernel_y if kernel_y is not None else kernel_size)
        
    def run(self, img_arr):
        if img_arr is None:
            return None

        try:
            return cv2.blur(img_arr, self.kernel_size)
        except:
            logger.error("単純ブラーの適用に失敗しました")
            return None

    def shutdown(self):
        pass


class ImgTrapezoidalMask:
    """台形マスクを適用し、指定外領域を塗りつぶすパーツ."""

    def __init__(self, left, right, bottom_left, bottom_right, top, bottom, fill=[255, 255, 255]) -> None:
        """初期化処理を行う。

        Args:
            left (int): 上辺左端の X 座標。
            right (int): 上辺右端の X 座標。
            bottom_left (int): 下辺左端の X 座標。
            bottom_right (int): 下辺右端の X 座標。
            top (int): 上辺の Y 座標。
            bottom (int): 下辺の Y 座標。
            fill (list[int]): 塗りつぶし色。
        """
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_left = left
        self.top_right = right
        self.top = top
        self.bottom = bottom
        self.fill = fill
        self.masks = {}

    def run(self, image):
        """台形マスクを適用する."""
        transformed = None
        if image is not None:
            mask = None
            key = str(image.shape)
            if self.masks.get(key) is None:
                mask = np.zeros(image.shape, dtype=np.int32)
                points = [
                    [self.top_left, self.top],
                    [self.top_right, self.top],
                    [self.bottom_right, self.bottom],
                    [self.bottom_left, self.bottom]
                ]
                cv2.fillConvexPoly(mask,
                                    np.array(points, dtype=np.int32),
                                    self.fill)
                mask = np.asarray(mask, dtype='bool')
                self.masks[key] = mask

            mask = self.masks[key]
            transformed = np.multiply(image, mask)

        return transformed

    def shutdown(self):
        self.masks = {}  # キャッシュしたマスクを解放


class ImgCropMask:
    """画像の一部領域のみを残すマスクを適用するパーツ."""

    def __init__(self, left=0, top=0, right=0, bottom=0, fill=[255, 255, 255]) -> None:
        """初期化処理を行う。

        Args:
            left (int): 左端の X 座標。
            top (int): 上端の Y 座標。
            right (int): 右端からの距離。
            bottom (int): 下端からの距離。
            fill (list[int]): 塗りつぶし色。
        """
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.fill = fill
        self.masks = {}

    def run(self, image):
        """指定した領域のみを残すマスクを適用する."""
        transformed = None
        if image is not None:
            mask = None
            key = str(image.shape)
            if self.masks.get(key) is None:
                height, width, depth = image_shape(image)
                top = self.top if self.top is not None else 0
                bottom = (height - self.bottom) if self.bottom is not None else height
                left = self.left if self.left is not None else 0
                right = (width - self.right) if self.right is not None else width
                mask = np.zeros(image.shape, dtype=np.int32)
                points = [
                    [left, top],
                    [right, top],
                    [right, bottom],
                    [left, bottom]
                ]
                cv2.fillConvexPoly(mask,
                                    np.array(points, dtype=np.int32),
                                    self.fill)
                mask = np.asarray(mask, dtype='bool')
                self.masks[key] = mask

            mask = self.masks[key]
            transformed = np.multiply(image, mask)

        return transformed

    def shutdown(self):
        self.masks = {}  # キャッシュしたマスクを解放


class ArrowKeyboardControls:
    """矢印キーのみで操作する簡易コントローラ."""
    def __init__(self):
        self.left = 2424832
        self.right = 2555904
        self.up = 2490368
        self.down = 2621440
        self.codes = [self.left, self.right, self.down, self.up]
        self.vec = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def run(self):
        code = cv2.waitKeyEx(delay=100)
        for iCode, keyCode in enumerate(self.codes):
            if keyCode == code:
                return self.vec[iCode]
        return 0., 0.


class Pipeline:
    """一連の処理を順に適用するパイプライン."""
    def __init__(self, steps):
        self.steps = steps
    
    def run(self, val):
        for step in self.steps:
            f = step['f']
            args = step['args']
            kwargs = step['kwargs']
            
            val = f(val, *args, **kwargs)
        return val


class CvImgFromFile(object):
    """画像ファイルを読み込みRGB形式で提供するパーツ."""
    def __init__(self, file_path, image_w=None, image_h=None, image_d=None, copy=False):
        """ファイルから画像を読み込みRGB画像として保持する."""
        if file_path is None:
            raise ValueError("CvImage passed empty file_path")

        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"CvImage file_path did not resolve to a readable image file: {file_path}")
        
        #
        # 指定があればリサイズする
        #
        height, width, depth = image_shape(image)
        if (image_h is not None and image_h != height) or (image_w is not None and image_w != width):
            if image_h is not None:
                height = image_h
            if image_w is not None:
                width = image_w
            image = cv2.resize(image, (width, height))

        #
        # 指定があれば色空間を変換する。
        # 既定ではカラー画像は BGR で読み込まれるため、RGB に変換する。
        #
        if image_d is not None and image_d != depth:
            if 1 == image_d:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif 3 == image_d:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif 3 == depth:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image = image
        self.copy = copy

    def run(self):
        if self.copy:
            return self.image.copy()
        return self.image


class CvCam(object):
    """OpenCV を利用したカメラ入力パーツ."""
    def __init__(self, image_w=160, image_h=120, image_d=3, iCam=0, warming_secs=5):
        self.width = image_w
        self.height = image_h
        self.depth = image_d

        self.frame = None
        self.cap = cv2.VideoCapture(iCam)

        # フレームを取得できるまでウォームアップする
        if self.cap is not None:
            # self.cap.set(3, image_w)
            # self.cap.set(4, image_h)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_h)
            logger.info('CvCam をオープンしました...')
            warming_time = time.time() + warming_secs  # おおよそ5秒で完了
            while self.frame is None and time.time() < warming_time:
                logger.info("...カメラをウォームアップ中")
                self.run()
                time.sleep(0.2)

            if self.frame is None:
                raise CameraError("CvCam を開始できません")
        else:
            raise CameraError("CvCam を開けません")

        self.running = True
        logger.info("CvCam の準備ができました")

    def poll(self):
        if self.cap.isOpened():
            _, self.frame = self.cap.read()
            if self.frame is not None:
                width, height = self.frame.shape[:2]
                if width != self.width or height != self.height:
                    self.frame = cv2.resize(self.frame, (self.width, self.height))

    def update(self):
        """フレーム取得を継続するスレッドループ."""
        while self.running:
            self.poll()

    def run_threaded(self):
        return self.frame

    def run(self):
        self.poll()
        return self.frame

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()


class CvImageView(object):
    """OpenCV で画像を表示する簡易ビューア."""

    def run(self, image):
        if image is None:
            return

        try:
            cv2.imshow('frame', image)
            cv2.waitKey(1)
        except:
            logger.error("画像ウィンドウを開けません")

    def shutdown(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import sys
    
    # 引数を解析
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=0,
                        help="複数カメラを使用する場合のカメラインデックス")
    parser.add_argument("-wd", "--width", type=int, default=160,
                        help="取得する画像の幅")
    parser.add_argument("-ht", "--height", type=int, default=120,
                        help="取得する画像の高さ")
    parser.add_argument("-f", "--file", type=str,
                        help="カメラの代わりに使用する画像ファイルのパス")
    parser.add_argument("-a", "--aug", required=True, type=str.upper,
                        choices=['CROP', 'TRAPEZE',
                                 "RGB2HSV", "HSV2RGB", "RGB2BGR", "BGR2RGB", "BGR2HSV", "HSV2BRG",
                                 "RGB2GREY", "BGR2GREY", "HSV2GREY",
                                 "CANNY",
                                 "BLUR", "GBLUR",
                                 "RESIZE", "SCALE"],
                        help="適用するオーグメンテーション")
    parser.add_argument("-l", "--left", type=int, default=0,
                        help="左端のピクセル位置(デフォルト0)")
    parser.add_argument("-lb", "--left-bottom", type=int, default=None,
                        help="左下のピクセル位置(デフォルト0)")
    parser.add_argument("-r", "--right", type=int, default=None,
                        help="右端のピクセル位置(デフォルトは画像幅)")
    parser.add_argument("-rb", "--right-bottom", type=int, default=None,
                        help="右下のピクセル位置(デフォルトは画像幅)")
    parser.add_argument("-t", "--top", type=int, default=0,
                        help="上端のピクセル位置(デフォルト0)")
    parser.add_argument("-b", "--bottom", type=int, default=None,
                        help="下端のピクセル位置(デフォルトは画像高さ)")
    parser.add_argument("-cl", "--canny-low", type=int, default=60,
                        help="Canny エッジ検出の下限閾値")
    parser.add_argument("-ch", "--canny-high", type=int, default=110,
                        help="Canny エッジ検出の上限閾値")
    parser.add_argument("-ca", "--canny-aperture", type=int, choices=[3, 5, 7], default=3,
                        help="Canny エッジ検出のアパーチャサイズ")
    parser.add_argument("-gk", "--guassian-kernel", type=int, choices=[3, 5, 7, 9], default=3,
                        help="ガウシアンブラーのカーネルサイズ")
    parser.add_argument("-gky", "--guassian-kernel-y", type=int, choices=[3, 5, 7, 9],
                        help="ガウシアンブラーの縦カーネルサイズ。省略時は正方形")
    parser.add_argument("-bk", "--blur-kernel", type=int, choices=[3, 5, 7, 9], default=3,
                        help="単純ブラーのカーネルサイズ")
    parser.add_argument("-bky", "--blur-kernel-y", type=int, choices=[3, 5, 7, 9],
                        help="単純ブラーの縦カーネルサイズ。省略時は正方形")
    parser.add_argument("-sw", "--scale", type=float,
                        help="画像幅のスケール係数")
    parser.add_argument("-sh", "--scale-height", type=float,
                        help="画像高さのスケール係数。省略時は幅と同じ")

    #
    # オーグメンテーションの設定
    #
    transformations = {}

    # コマンドライン引数を読み込む
    args = parser.parse_args()
    
    image_source = None
    help = []
    if args.file is None:
        if args.camera < 0:
            help.append("-c/--camera は 0 以上で指定してください")
        if args.width is None or args.width < 160:
            help.append("-wd/--width は 160 以上で指定してください")
        if args.height is None or args.height < 120:
            help.append("-ht/--height は 120 以上で指定してください")

    if "SCALE" == args.aug:
        if args.scale is None or args.scale <= 0:
            help.append("-sw/--scale は 0 より大きい値を指定してください")
        elif args.scale_height is not None and args.scale_height <= 0:
            help.append("-sh/--scale-height は 0 より大きい値を指定してください")

    if "RESIZE" == args.aug:
        if args.width is None or args.width < 160:
            help.append("-wd/--width は 160 以上で指定してください")
        if args.height is None or args.height < 120:
            help.append("-ht/--height は 120 以上で指定してください")


    if len(help) > 0:
        parser.print_help()
        for h in help:
            print("  " + h)
        sys.exit(1)

    #
    # ファイルを読み込むかカメラを初期化する
    #
    cap = None
    width = None
    height = None
    depth = 3
    if args.file is not None:
        image_source = CvImgFromFile(args.file, image_w=args.width, image_h=args.height, copy=True)
        height, width, depth = image_shape(image_source.run())
    else:
        width = args.width
        height = args.height
        image_source = CvCam(image_w=width, image_h=height, iCam=args.camera)

    transformer = None
    transformation = args.aug

    #
    # マスク処理
    if "TRAPEZE" == transformation or "CROP" == transformation:
        # マスク変換
        if "TRAPEZE" == transformation:
            transformer = ImgTrapezoidalMask(
                args.left if args.left is not None else 0,
                args.right if args.right is not None else width,
                args.left_bottom if args.left_bottom is not None else 0,
                args.right_bottom if args.right_bottom is not None else width,
                args.top if args.top is not None else 0,
                args.bottom if args.bottom is not None else height
            )
        else:
            transformer = ImgCropMask(
                args.left if args.left is not None else 0, 
                args.top if args.top is not None else 0, 
                args.right if args.right is not None else 0, 
                args.bottom if args.bottom is not None else 0)
    #
    # 色空間の変換
    #
    elif "RGB2BGR" == transformation:
        transformer = ImgRGB2BGR()
    elif "BGR2RGB" == transformation:
        transformer = ImgBGR2RGB()
    elif "RGB2HSV" == transformation:
        transformer = ImgRGB2HSV()
    elif "HSV2RGB" == transformation:
        transformer = ImgHSV2RGB()
    elif "BGR2HSV" == transformation:
        transformer = ImgBGR2HSV()
    elif "HSV2BGR" == transformation:
        transformer = ImgHSV2BGR()
    elif "RGB2GREY" == transformation:
        transformer = ImgRGB2GRAY()
    elif "RBGR2GREY" == transformation:
        transformer = ImgBGR2GRAY()
    elif "HSV2GREY" == transformation:
        transformer = ImgHSV2GRAY()
    elif "CANNY" == transformation:
        # Canny エッジ検出
        transformer = ImgCanny(args.canny_low, args.canny_high, args.canny_aperture)
    # 
    # ぼかし処理
    #
    elif "GBLUR" == transformation:
        transformer = ImgGaussianBlur(args.guassian_kernel, args.guassian_kernel_y)
    elif "BLUR" == transformation:
        transformer = ImgSimpleBlur(args.blur_kernel, args.blur_kernel_y)
    # 
    # リサイズ処理
    #
    elif "RESIZE" == transformation:
        transformer = ImageResize(args.width, args.height)
    elif "SCALE" == transformation:
        transformer = ImageScale(args.scale, args.scale_height)
    else:
        print("-a/--aug は有効なオーグメンテーションではありません")
        exit()

    # ウィンドウを作成して表示
    window_name = 'hsv_range_picker'
    cv2.namedWindow(window_name)

    while(1):

        frame = image_source.run()

        #
        # オーグメンテーションを適用
        #
        transformed_image = transformer.run(frame)

        #
        # 変換後の画像を表示
        #
        cv2.imshow(window_name, transformed_image)

        k = cv2.waitKey(5) & 0xFF
        if k == ord('q') or k == ord('Q'):  # 'Q' または 'q'
            break
    
    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()
