"""
utils.py

Functions that don't fit anywhere else.

"""

import glob
import itertools
import logging
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
import zipfile
from io import BytesIO
from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


ONE_BYTE_SCALE = 1.0 / 255.0


class EqMemorizedString:
    """比較された値を記憶する文字列"""

    def __init__(self, string):
        self.string = string
        self.mem = set()

    def __eq__(self, other):
        self.mem.add(other)
        return self.string == other

    def mem_as_str(self):
        return ", ".join(self.mem)


"""
IMAGES
"""


def scale(im, size=128):
    """
    PIL 画像を受け取り、辺の長さが size になるように縮小した画像を返す
    """
    size = (size, size)
    im.thumbnail(size, Image.ANTIALIAS)
    return im


def img_to_binary(img, format="jpeg"):
    """
    PIL 画像を受け取り、バイナリストリームを返す（データベース保存用）
    """
    f = BytesIO()
    try:
        img.save(f, format=format)
    except Exception as e:
        raise e
    return f.getvalue()


def arr_to_binary(arr):
    """
    (高さ、幅、チャンネル) の形状をもつ NumPy 配列を受け取り、
    バイナリストリームを返す（データベース保存用）
    """
    img = arr_to_img(arr)
    return img_to_binary(img)


def arr_to_img(arr):
    """
    (高さ、幅、チャンネル) の形状をもつ NumPy 配列を受け取り、
    PIL 画像を返す
    """
    arr = np.uint8(arr)
    img = Image.fromarray(arr)
    return img


def img_to_arr(img):
    """
    PIL 画像を受け取り、NumPy uint8 配列を返す
    """
    return np.array(img)


def binary_to_img(binary):
    """
    BytesIO からのバイナリオブジェクトを受け取り、PIL 画像を返す
    """
    if binary is None or len(binary) == 0:
        return None

    img = BytesIO(binary)
    try:
        img = Image.open(img)
        return img
    except:
        return None


def norm_img(img):
    return (img - img.mean() / np.std(img)) * ONE_BYTE_SCALE


def rgb2gray(rgb):
    """
    正規化された形状 (w, h, 3) の NumPy 画像配列を受け取り、
    グレースケール画像 (w, h) を返す
    :param rgb:     正規化済み [0,1] float32 または [0,255] uint8 の
                    形状(w,h,3) 画像配列
    :return:        正規化済み [0,1] float32 配列 (w,h) または
                    [0,255] uint8 のグレースケール配列
    """
    # ここでは uint8 配列を float64 に変換する
    grey = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    # 入力が uint8 配列なら元の型に戻す
    if rgb.dtype.type is np.uint8:
        grey = round(grey).astype(np.uint8)
    return grey


def img_crop(img_arr, top, bottom):
    if bottom == 0:
        end = img_arr.shape[0]
    else:
        end = -bottom
    return img_arr[top:end, ...]


def normalize_image(img_arr_uint):
    """
    uint8 の NumPy 画像配列を [0,1] の浮動小数点画像配列に変換する
    :param img_arr_uint:    [0,255]uint8 の NumPy 画像配列
    :return:                [0,1] float32 の NumPy 画像配列
    """
    return img_arr_uint.astype(np.float64) * ONE_BYTE_SCALE


def denormalize_image(img_arr_float):
    """
    :param img_arr_float:   [0,1] 浮動小数点 NumPy 画像配列
    :return:                [0,255]uint8 の NumPy 画像配列
    """
    return (img_arr_float * 255.0).astype(np.uint8)


def load_pil_image(filename, cfg):
    """ファイルパスから画像を読み込み、必要ならリサイズした PIL 画像を返す。

    Args:
        filename (string): 画像ファイルへのパス
        cfg (object): Donkey の設定

    Returns: PIL 画像
    """
    try:
        img = Image.open(filename)
        if img.height != cfg.IMAGE_H or img.width != cfg.IMAGE_W:
            img = img.resize((cfg.IMAGE_W, cfg.IMAGE_H))

        if cfg.IMAGE_DEPTH == 1:
            img = img.convert("L")

        return img

    except Exception as e:
        logger.error(f"{filename} から画像を読み込めませんでした: {e.message}")
        return None


def load_image(filename, cfg):
    """
    :param string filename:     画像ファイルへのパス
    :param cfg:                 Donkey の設定
    :return np.ndarray:         NumPy uint8 画像配列
    """
    img_arr = load_image_sized(filename, cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)

    return img_arr


def load_image_sized(filename, image_width, image_height, image_depth):
    """ファイルパスから画像を読み込みリサイズした PIL 画像を NumPy 配列で返す。

    Args:
        filename (string): 画像ファイルへのパス
        image_width: 出力画像の幅（ピクセル）
        image_height: 出力画像の高さ（ピクセル）
        image_depth: 出力画像の深度（1 はグレースケール）

    Returns:
        (np.ndarray):         NumPy uint8 画像配列
    """
    try:
        img = Image.open(filename)
        if img.height != image_height or img.width != image_width:
            img = img.resize((image_width, image_height))

        if image_depth == 1:
            img = img.convert("L")

        img_arr = np.asarray(img)

        # PIL 画像がグレースケールの場合、配列の形は (H, W) になる
        # そのためチャンネルを追加して (H, W, 1) に拡張する
        if img.mode == "L":
            h, w = img_arr.shape[:2]
            img_arr = img_arr.reshape(h, w, 1)

        return img_arr

    except Exception as e:
        logger.error(f"{filename} から画像を読み込めませんでした: {e.message}")
        return None


"""
FILES
"""


def most_recent_file(dir_path, ext=""):
    """
    ディレクトリと拡張子を指定して最新のファイルを返す
    """
    query = dir_path + "/*" + ext
    newest = min(glob.iglob(query), key=os.path.getctime)
    return newest


def make_dir(path):
    real_path = os.path.expanduser(path)
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    return real_path


def zip_dir(dir_path, zip_path):
    """
    単一階層のディレクトリを zip 化して保存する
    """
    file_paths = glob.glob(dir_path + "/*")  # ファイル探索用のパスを作成

    zf = zipfile.ZipFile(zip_path, "w")
    dir_name = os.path.basename(dir_path)
    for p in file_paths:
        file_name = os.path.basename(p)
        zf.write(p, arcname=os.path.join(dir_name, file_name))
    zf.close()
    return zip_path


"""
BINNING
functions to help converte between floating point numbers and categories.
"""


def clamp(n, min, max):
    if min > max:
        return clamp(n, max, min)

    if n < min:
        return min
    if n > max:
        return max
    return n


def linear_bin(a, N=15, offset=1, R=2.0):
    """
    長さ N のビンを作成し、値 a を範囲 R にマッピングする。
    オフセット分だけワンホットビンをずらす（通常は R/2）
    """
    a = a + offset
    b = round(a / (R / (N - offset)))
    arr = np.zeros(N)
    b = clamp(b, 0, N - 1)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr, N=15, offset=-1, R=2.0):
    """
    linear_bin の逆変換を行い、ワンホット配列から最大値を取得して
    指定の範囲 R とオフセットへ戻す
    """
    b = np.argmax(arr)
    a = b * (R / (N + offset)) + offset
    return a


def map_range(x, X_min, X_max, Y_min, Y_max):
    """
    2 つの値の範囲を線形変換する
    """
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range / Y_range

    y = ((x - X_min) / XY_ratio + Y_min) // 1

    return int(y)


def map_range_float(x, X_min, X_max, Y_min, Y_max):
    """
    map_range と同じだが、戻り値を少数で扱い小数点以下2桁に丸める
    """
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range / Y_range

    y = (x - X_min) / XY_ratio + Y_min

    # print("y= {}".format(y))

    return round(y, 2)


"""
ANGLES
"""


def norm_deg(theta):
    while theta > 360:
        theta -= 360
    while theta < 0:
        theta += 360
    return theta


DEG_TO_RAD = math.pi / 180.0


def deg2rad(theta):
    return theta * DEG_TO_RAD


"""
VECTORS
"""


def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))


"""
NETWORKING
"""


def my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("192.0.0.8", 1027))
    return s.getsockname()[0]


"""
THROTTLE
"""

STEERING_MIN = -1.0
STEERING_MAX = 1.0
# ステアリング角に応じてスロットルを約0.5〜1.0に調整する
EXP_SCALING_FACTOR = 0.5
DAMPENING = 0.05


def _steering(input_value):
    input_value = clamp(input_value, STEERING_MIN, STEERING_MAX)
    return (input_value - STEERING_MIN) / (STEERING_MAX - STEERING_MIN)


def throttle(input_value):
    magnitude = _steering(input_value)
    decay = math.exp(magnitude * EXP_SCALING_FACTOR)
    dampening = DAMPENING * magnitude
    return (1 / decay) - dampening


"""
OTHER
"""


def is_number_type(i):
    return type(i) == int or type(i) == float


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def compare_to(
    value: float,  # 入力値
    toValue: float,  # 許容誤差付きで比較する値
    tolerance: float,
):  # 非負の許容誤差
    # 戻り値: value > toValue + tolerance なら 1
    #         value < toValue - tolerance なら -1
    #         それ以外は 0
    if (toValue - value) > tolerance:
        return -1
    if (value - toValue) > tolerance:
        return 1
    return 0


def map_frange(x, X_min, X_max, Y_min, Y_max):
    """
    Linear mapping between two ranges of values
    map from x range to y range
    """
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range / Y_range

    y = (x - X_min) / XY_ratio + Y_min

    return y


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def param_gen(params):
    """
    パラメータ候補を辞書で受け取り、全ての組み合わせを辞書として返す
    """
    for p in itertools.product(*params.values()):
        yield dict(zip(params.keys(), p))


def run_shell_command(cmd, cwd=None, timeout=15):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    out = []
    err = []

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill(proc.pid)

    for line in proc.stdout.readlines():
        out.append(line.decode())

    for line in proc.stderr.readlines():
        err.append(line)
    return out, err, proc.pid


def kill(proc_id):
    os.kill(proc_id, signal.SIGINT)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_model_by_type(
    model_type: str, cfg: "Config"
) -> Union["KerasPilot", "FastAiPilot"]:
    """
    文字列 model_type と設定 cfg に基づき、対応する Keras モデルを生成して返す。
    """
    from donkeycar.parts.interpreter import (
        FastAIInterpreter,
        KerasInterpreter,
        TensorRT,
        TfLite,
    )
    from donkeycar.parts.keras import (
        Keras3D_CNN,
        KerasBehavioral,
        KerasCategorical,
        KerasIMU,
        KerasInferred,
        KerasLinear,
        KerasLocalizer,
        KerasLSTM,
        KerasMemory,
    )

    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
    logger.info(f"モデルタイプ {model_type} を取得します")
    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    if "tflite_" in model_type:
        interpreter = TfLite()
        used_model_type = model_type.replace("tflite_", "")
    elif "tensorrt_" in model_type:
        interpreter = TensorRT()
        used_model_type = model_type.replace("tensorrt_", "")
    elif "fastai_" in model_type:
        interpreter = FastAIInterpreter()
        used_model_type = model_type.replace("fastai_", "")
        if used_model_type == "linear":
            from donkeycar.parts.fastai import FastAILinear

            return FastAILinear(interpreter=interpreter, input_shape=input_shape)
    else:
        interpreter = KerasInterpreter()
        used_model_type = model_type

    used_model_type = EqMemorizedString(used_model_type)
    if used_model_type == "linear":
        kl = KerasLinear(interpreter=interpreter, input_shape=input_shape)
    elif used_model_type == "categorical":
        kl = KerasCategorical(
            interpreter=interpreter,
            input_shape=input_shape,
            throttle_range=cfg.MODEL_CATEGORICAL_MAX_THROTTLE_RANGE,
        )
    elif used_model_type == "inferred":
        kl = KerasInferred(interpreter=interpreter, input_shape=input_shape)
    elif used_model_type == "imu":
        kl = KerasIMU(interpreter=interpreter, input_shape=input_shape)
    elif used_model_type == "memory":
        mem_length = getattr(cfg, "SEQUENCE_LENGTH", 3)
        mem_depth = getattr(cfg, "MEM_DEPTH", 0)
        kl = KerasMemory(
            interpreter=interpreter,
            input_shape=input_shape,
            mem_length=mem_length,
            mem_depth=mem_depth,
        )
    elif used_model_type == "behavior":
        kl = KerasBehavioral(
            interpreter=interpreter,
            input_shape=input_shape,
            throttle_range=cfg.MODEL_CATEGORICAL_MAX_THROTTLE_RANGE,
            num_behavior_inputs=len(cfg.BEHAVIOR_LIST),
        )
    elif used_model_type == "localizer":
        kl = KerasLocalizer(
            interpreter=interpreter,
            input_shape=input_shape,
            num_locations=cfg.NUM_LOCATIONS,
        )
    elif used_model_type == "rnn":
        kl = KerasLSTM(
            interpreter=interpreter,
            input_shape=input_shape,
            seq_length=cfg.SEQUENCE_LENGTH,
        )
    elif used_model_type == "3d":
        kl = Keras3D_CNN(
            interpreter=interpreter,
            input_shape=input_shape,
            seq_length=cfg.SEQUENCE_LENGTH,
        )
    else:
        known = [
            k + u for k in ("", "tflite_", "tensorrt_") for u in used_model_type.mem
        ]
        raise ValueError(
            f"Unknown model type {model_type}, supported types are {', '.join(known)}"
        )
    return kl


def get_test_img(keras_pilot):
    """
    入力形式を調べ、テストモデルで使用可能な画像を生成する
    :param keras_pilot:             入力となる Keras パイロット
    :return np.ndarry(np.uint8):    ランダム生成した NumPy 画像配列
    """
    try:
        count, h, w, ch = keras_pilot.get_input_shapes()[0]
        seq_len = 0
    except Exception:
        count, seq_len, h, w, ch = keras_pilot.get_input_shapes()[0]

    # generate random array in the right shape
    img = np.random.randint(0, 255, size=(h, w, ch))
    return img.astype(np.uint8)


def train_test_split(
    data_list: List[Any], shuffle: bool = True, test_size: float = 0.2
) -> Tuple[List[Any], List[Any]]:
    """
    リストを受け取り、テストサイズの割合でランダムに分割する。
    シャッフルは常に有効で、後方互換性のために残してある。
    """
    target_train_size = int(len(data_list) * (1.0 - test_size))

    if shuffle:
        train_data = []
        i_sample = 0
        while i_sample < target_train_size and len(data_list) > 1:
            i_choice = random.randint(0, len(data_list) - 1)
            train_data.append(data_list.pop(i_choice))
            i_sample += 1

        # remainder of the original list is the validation set
        val_data = data_list

    else:
        train_data = data_list[:target_train_size]
        val_data = data_list[target_train_size:]

    return train_data, val_data


"""
タイマー
"""


class FPSTimer(object):
    def __init__(self):
        self.t = time.time()
        self.iter = 0

    def reset(self):
        self.t = time.time()
        self.iter = 0

    def on_frame(self):
        self.iter += 1
        if self.iter == 100:
            e = time.time()
            print("fps", 100.0 / (e - self.t))
            self.t = time.time()
            self.iter = 0
