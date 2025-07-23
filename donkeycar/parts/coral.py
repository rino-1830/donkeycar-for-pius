"""推論タスクで使用する `InferenceEngine` を提供するモジュール。"""

from edgetpu.basic.basic_engine import BasicEngine
import numpy
from PIL import Image


class InferenceEngine(BasicEngine):
    """推論タスクに使用するエンジン。"""

    def __init__(self, model_path, device_path=None):
        """指定されたモデルで :class:`BasicEngine` を生成する。

        Args:
            model_path (str): TF-Lite Flatbuffer ファイルのパス。
            device_path (str, optional): 指定した場合、Edge TPU を ``device_path`` でバインドする。

        Raises:
            ValueError: モデルの出力形式が不正な場合に送出される。
        """
        if device_path:
            super().__init__(model_path, device_path)
        else:
            super().__init__(model_path)
        output_tensors_sizes = self.get_all_output_tensors_sizes()
        if output_tensors_sizes.size > 2:
            raise ValueError(
                ('推論モデルの出力テンソルは 2 つ以下である必要があります。'
                 'このモデルの出力数は {} です。'.format(output_tensors_sizes.size)))

    def Inference(self, img):
        """numpy 配列の画像を用いて推論を実行する。

        このインターフェースは読み込んだモデルが画像分類用に学習済みであることを想定する。

        Args:
            img (numpy.ndarray): 入力画像。

        Returns:
            List[float]: 推論結果を表す数値のリスト。

        Raises:
            RuntimeError: 入力テンソルが単一の 3 チャンネル画像でない場合。
            AssertionError: 画像サイズが正しくない場合。
        """
        input_tensor_shape = self.get_input_tensor_shape()
        if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 3 or
                input_tensor_shape[0] != 1):
            raise RuntimeError(
                '入力テンソルの形状が不正です。期待値: [1, height, width, 3]')
        _, height, width, _ = input_tensor_shape
        assert height == img.shape[0]
        assert width == img.shape[1]
        input_tensor = img.flatten()
        return self.RunInferenceWithInputTensor(input_tensor)

    def RunInferenceWithInputTensor(self, input_tensor):
        """生の入力テンソルで推論を実行する。

        このインターフェースでは、ユーザーが入力データを処理してフォーマット済みのテンソルに変換しておく必要がある。

        Args:
            input_tensor (numpy.ndarray): 入力テンソル。

        Returns:
            List[float]: 推論結果を表す数値のリスト。

        Raises:
            ValueError: 入力パラメータが不正な場合。
        """
        _, self._raw_result = self.RunInference(
            input_tensor)
        return [self._raw_result]


class CoralLinearPilot(object):
    """車を操作するためのステアリングとスロットルを出力する TFLite モデルの基底クラス。"""

    def __init__(self):
        """モデルと推論エンジンのプレースホルダーを初期化する。"""
        self.model = None
        self.engine = None

    def load(self, model_path):
        """Coral EdgeTPU 用の TFLite モデルを読み込み、テンソルを割り当てる。

        Args:
            model_path (str): モデルファイルのパス。
        """
        # Coral EdgeTPU 用の TFLite モデルを読み込み、テンソルを割り当てる。
        self.engine = InferenceEngine(model_path)

    def run(self, image):
        """画像を入力し、ステアリングとスロットルを推論する。

        Args:
            image (numpy.ndarray): 入力画像。

        Returns:
            Tuple[float, float]: ステアリング角度とスロットル値。
        """
        steering, throttle = self.engine.Inference(image)[0]
        return steering, throttle
