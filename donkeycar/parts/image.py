
from PIL import Image
import numpy as np
from donkeycar.utils import img_to_binary, binary_to_img, arr_to_img, \
    img_to_arr, normalize_image


class ImgArrToJpg():
    """Numpy 配列の画像を JPEG バイト列へ変換するクラス。"""

    def run(self, img_arr):
        """画像配列を JPEG のバイト列に変換する。

        Args:
            img_arr: 画像の ``numpy.ndarray``。

        Returns:
            ``bytes`` もしくは ``None``。
        """
        if img_arr is None:
            return None
        try:
            image = arr_to_img(img_arr)
            jpg = img_to_binary(image)
            return jpg
        except:
            return None


class JpgToImgArr():
    """JPEG バイト列を ``numpy.ndarray`` に変換するクラス。"""

    def run(self, jpg):
        """JPEG データを画像配列に変換する。

        Args:
            jpg: JPEG 形式のバイト列。

        Returns:
            ``numpy.ndarray`` もしくは ``None``。
        """
        if jpg is None:
            return None
        image = binary_to_img(jpg)
        img_arr = img_to_arr(image)
        return img_arr


class StereoPair:
    """2 枚の画像を 1 枚にまとめるクラス。"""
    def run(self, image_a, image_b):
        """2 枚の画像を合成し、赤と緑に配置し差分を青チャンネルへ入れる。

        Args:
            image_a: 1 枚目の ``numpy.ndarray``。
            image_b: 2 枚目の ``numpy.ndarray``。

        Returns:
            3 チャンネルの ``numpy.ndarray``。
        """
        if image_a is not None and image_b is not None:
            width, height, _ = image_a.shape
            grey_a = dk.utils.rgb2gray(image_a)
            grey_b = dk.utils.rgb2gray(image_b)
            grey_c = grey_a - grey_b
            
            stereo_image = np.zeros([width, height, 3], dtype=np.dtype('B'))
            stereo_image[...,0] = np.reshape(grey_a, (width, height))
            stereo_image[...,1] = np.reshape(grey_b, (width, height))
            stereo_image[...,2] = np.reshape(grey_c, (width, height))
        else:
            stereo_image = []

        return np.array(stereo_image)


class ImgCrop:
    """興味領域を指定して画像を切り抜くクラス。"""
    def __init__(self, top=0, bottom=0, left=0, right=0):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        
    def run(self, img_arr):
        """画像を指定範囲で切り抜く。

        Args:
            img_arr: 切り抜き対象の ``numpy.ndarray``。

        Returns:
            切り抜かれた ``numpy.ndarray`` もしくは ``None``。
        """
        if img_arr is None:
            return None
        width, height, _ = img_arr.shape
        img_arr = img_arr[self.top:height-self.bottom,
                          self.left: width-self.right]
        return img_arr

    def shutdown(self):
        pass


class ImgStack:
    """過去 ``N`` 枚のグレースケール画像をチャンネル方向に積み重ねるクラス。"""
    def __init__(self, num_channels=3):
        self.img_arr = None
        self.num_channels = num_channels

    def rgb2gray(self, rgb):
        """RGB 画像をグレースケールに変換する。"""
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
    def run(self, img_arr):
        """画像をスタックし ``num_channels`` チャンネルの配列を返す。

        Args:
            img_arr: 最新の ``numpy.ndarray`` 画像。

        Returns:
            スタックされた ``numpy.ndarray``。
        """
        width, height, _ = img_arr.shape
        gray = self.rgb2gray(img_arr)
        
        if self.img_arr is None:
            self.img_arr = np.zeros([width, height, self.num_channels], dtype=np.dtype('B'))

        for ch in range(self.num_channels - 1):
            self.img_arr[...,ch] = self.img_arr[...,ch+1]

        self.img_arr[...,self.num_channels - 1:] = np.reshape(gray, (width, height, 1))

        return self.img_arr

    def shutdown(self):
        pass
