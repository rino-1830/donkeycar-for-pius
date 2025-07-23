"""モデル推論に用いるインタープリタを提供するモジュール."""

import os
from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Union, Sequence, List

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2 as convert_var_to_const
from tensorflow.python.saved_model import tag_constants, signature_constants

logger = logging.getLogger(__name__)


def keras_model_to_tflite(in_filename, out_filename, data_gen=None):
    """Keras モデルを読み込み TFLite 形式に変換する.

    Args:
        in_filename: 変換元 Keras モデルのファイル名。
        out_filename: 生成される TFLite ファイル名。
        data_gen: 量子化用のデータジェネレータ。省略可能。
    """
    logger.info(f'{in_filename} を TFLite {out_filename} に変換します')
    model = tf.keras.models.load_model(in_filename)
    keras_to_tflite(model, out_filename, data_gen)
    logger.info('TFLite への変換が完了しました。')


def keras_to_tflite(model, out_filename, data_gen=None):
    """Keras モデルを TFLite ファイルへ変換する.

    Args:
        model: 変換対象の Keras モデル。
        out_filename: 出力する TFLite ファイル名。
        data_gen: 量子化に用いるデータジェネレータ。省略可能。
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    if data_gen is not None:
        # data_gen が指定されていればそれを使用して整数ウェイトを生成し
        # キャリブレーションを行う。注意: このモデルは標準の tflite
        # エンジン（浮動小数点のみ）では動作しなくなる。
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = data_gen
        try:
            converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        except:
            pass
        try:
            converter.target_spec.supported_ops \
                = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        except:
            pass
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        logger.info(
            "データジェネレータを使用して Coral TPU 用の整数最適化ウェイトを作成します"
        )
    tflite_model = converter.convert()
    open(out_filename, "wb").write(tflite_model)


def saved_model_to_tensor_rt(saved_path: str, tensor_rt_path: str):
    """TF SavedModel 形式を TensorRT 用に変換する.

    CUDA 環境がなくても動作し、GPU 固有の処理は TensorFlow 内で実行される。

    Args:
        saved_path: 変換元となる SavedModel のパス。
        tensor_rt_path: 変換後の TensorRT モデルを保存するパス。
    """
    logger.info(
        f'SavedModel {saved_path} を TensorRT {tensor_rt_path} に変換します'
    )
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    params = params._replace(max_workspace_size_bytes=(1 << 32))
    params = params._replace(precision_mode="FP16")
    params = params._replace(maximum_cached_engines=100)
    try:
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=saved_path,
            conversion_params=params)
        converter.convert()
        converter.save(tensor_rt_path)
        logger.info('TensorRT への変換が完了しました。')
    except Exception as e:
        logger.error(f'TensorRT への変換に失敗しました: {e}')


class Interpreter(ABC):
    """Keras、TFLite、TensorRT 間で処理を振り分ける基底クラス."""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """モデルを読み込む."""
        pass

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        """重みを読み込む."""
        raise NotImplementedError('実装が必要です')

    def set_model(self, pilot: 'KerasPilot') -> None:
        """一部のインタープリターではモデルが必要となる."""
        pass

    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """オプティマイザを設定する."""
        pass

    def compile(self, **kwargs):
        """モデルをコンパイルする."""
        raise NotImplementedError('実装が必要です')

    @abstractmethod
    def get_input_shapes(self) -> List[tf.TensorShape]:
        """入力テンソルの形状を取得する."""
        pass

    @abstractmethod
    def predict(self, img_arr: np.ndarray, other_arr: np.ndarray) \
            -> Sequence[Union[float, np.ndarray]]:
        """推論を実行する."""
        pass

    def predict_from_dict(self, input_dict) -> Sequence[Union[float, np.ndarray]]:
        """辞書形式の入力で推論を実行する."""
        pass

    def summary(self) -> str:
        """モデルの概要を返す."""
        pass

    def __str__(self) -> str:
        """インタープリター名を文字列として返す."""
        return type(self).__name__


class KerasInterpreter(Interpreter):
    """Keras モデルによる推論を行うインタープリター."""

    def __init__(self):
        """インスタンスを初期化する."""
        super().__init__()
        self.model: tf.keras.Model = None

    def set_model(self, pilot: 'KerasPilot') -> None:
        """パイロットからモデルを生成して保持する."""
        self.model = pilot.create_model()

    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """オプティマイザを設定する."""
        self.model.optimizer = optimizer

    def get_input_shapes(self) -> List[tf.TensorShape]:
        """入力テンソルの形状を取得する."""
        assert self.model, 'モデルが設定されていません'
        return [inp.shape for inp in self.model.inputs]

    def compile(self, **kwargs):
        """モデルをコンパイルする."""
        assert self.model, 'モデルが設定されていません'
        self.model.compile(**kwargs)

    def invoke(self, inputs):
        """内部モデルを呼び出し推論結果を取得する."""
        outputs = self.model(inputs, training=False)
        # 関数型モデルの場合、出力はリストとなる
        if type(outputs) is list:
            # バッチサイズ 1 で呼び出すため、余分な次元を取り除く
            output = [output.numpy().squeeze(axis=0) for output in outputs]
            return output
        # 順次モデルの場合、出力形状は (1, n) となる
        else:
            return outputs.numpy().squeeze(axis=0)

    def predict(self, img_arr: np.ndarray, other_arr: np.ndarray) \
            -> Sequence[Union[float, np.ndarray]]:
        """画像と補助入力から推論を実行する."""
        img_arr = np.expand_dims(img_arr, axis=0)
        inputs = img_arr
        if other_arr is not None:
            other_arr = np.expand_dims(other_arr, axis=0)
            inputs = [img_arr, other_arr]
        return self.invoke(inputs)

    def predict_from_dict(self, input_dict):
        """辞書形式の入力データで推論を実行する."""
        for k, v in input_dict.items():
            input_dict[k] = np.expand_dims(v, axis=0)
        return self.invoke(input_dict)

    def load(self, model_path: str) -> None:
        """Keras モデルを読み込む."""
        logger.info(f'モデル {model_path} を読み込みます')
        self.model = keras.models.load_model(model_path, compile=False)

    def load_weights(self, model_path: str, by_name: bool = True) -> \
            None:
        """モデルに重みを読み込む."""
        assert self.model, 'モデルが設定されていません'
        self.model.load_weights(model_path, by_name=by_name)

    def summary(self) -> str:
        """モデルのサマリーを返す."""
        return self.model.summary()


class FastAIInterpreter(Interpreter):
    """FastAI ライブラリを利用したモデルの推論を行うインタープリター."""

    def __init__(self):
        """インスタンスを初期化する."""
        super().__init__()
        self.model: None
        from fastai import learner as fastai_learner
        from fastai import optimizer as fastai_optimizer

    def set_model(self, pilot: 'FastAiPilot') -> None:
        """パイロットからモデルを生成して保持する."""
        self.model = pilot.create_model()

    def set_optimizer(self, optimizer: 'fastai_optimizer') -> None:
        """オプティマイザを設定する."""
        self.model.optimizer = optimizer

    def get_input_shapes(self):
        """入力テンソルの形状を取得する."""
        assert self.model, 'モデルが設定されていません'
        return [inp.shape for inp in self.model.inputs]

    def compile(self, **kwargs):
        """モデルのコンパイルは行わない."""
        pass

    def invoke(self, inputs):
        """内部モデルを呼び出し推論結果を取得する."""
        outputs = self.model(inputs)
        # 関数型モデルの場合、出力はリストとなる
        if type(outputs) is list:
            # バッチサイズ 1 で呼び出すため、余分な次元を取り除く
            output = [output.numpy().squeeze(axis=0) for output in outputs]
            return output
        # 順次モデルの場合、出力形状は (1, n) となる
        else:
            return outputs.detach().numpy().squeeze(axis=0)

    def predict(self, img_arr: np.ndarray, other_arr: np.ndarray) \
            -> Sequence[Union[float, np.ndarray]]:
        """画像と補助入力から推論を実行する."""
        import torch
        inputs = torch.unsqueeze(img_arr, 0)
        if other_arr is not None:
            # other_arr = np.expand_dims(other_arr, axis=0)  # 使用例
            inputs = [img_arr, other_arr]
        return self.invoke(inputs)

    def load(self, model_path: str) -> None:
        """学習済みモデルを読み込む."""
        import torch
        logger.info(f'モデル {model_path} を読み込みます')
        if torch.cuda.is_available():
            logger.info("CUDA を使用して PyTorch 推論を行います")
            self.model = torch.load(model_path)
        else:
            logger.info("PyTorch 推論に CUDA は利用できません")
            self.model = torch.load(model_path, map_location=torch.device('cpu'))

        logger.info(self.model)
        self.model.eval()

    def summary(self) -> str:
        """モデルのサマリーを返す."""
        return self.model


class TfLite(Interpreter):
    """TensorFlow Lite インタープリタをラップするクラス."""

    def __init__(self):
        """インスタンスを初期化する."""
        super().__init__()
        self.interpreter = None
        self.input_shapes = None
        self.input_details = None
        self.output_details = None
    
    def load(self, model_path):
        """Tflite モデルを読み込む."""
        assert os.path.splitext(model_path)[1] == '.tflite', \
            'TFlitePilot は .tflite ファイルのみを読み込みます'
        logger.info(f'モデル {model_path} を読み込みます')
        # TFLite モデルを読み込みテンソルを割り当てる
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # 入力および出力テンソルを取得する
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # 入力の形状を取得する
        self.input_shapes = []
        logger.info('tflite 入力テンソルの詳細を読み込みます')
        for detail in self.input_details:
            logger.debug(detail)
            self.input_shapes.append(detail['shape'])

    def compile(self, **kwargs):
        """Tflite ではコンパイル処理を行わない."""
        pass

    def invoke(self) -> Sequence[Union[float, np.ndarray]]:
        """インタープリタを実行し出力を取得する."""
        self.interpreter.invoke()
        outputs = []
        for tensor in self.output_details:
            output_data = self.interpreter.get_tensor(tensor['index'])
            # バッチサイズ 1 で呼び出すため、余分な次元を取り除く
            outputs.append(output_data[0])
        # 出力が 1 次元の場合はリストにしない
        return outputs if len(outputs) > 1 else outputs[0]

    def predict(self, img_arr, other_arr) \
            -> Sequence[Union[float, np.ndarray]]:
        """画像と補助入力から推論を実行する."""
        assert self.input_shapes and self.input_details, \
            "Tflite モデルが読み込まれていません"
        input_arrays = (img_arr, other_arr)
        for arr, shape, detail \
                in zip(input_arrays, self.input_shapes, self.input_details):
            in_data = arr.reshape(shape).astype(np.float32)
            self.interpreter.set_tensor(detail['index'], in_data)
        return self.invoke()

    def predict_from_dict(self, input_dict):
        """辞書形式の入力データで推論を実行する."""
        for detail in self.input_details:
            k = detail['name']
            inp_k = input_dict[k]
            inp_k_res = inp_k.reshape(detail['shape']).astype(np.float32)
            self.interpreter.set_tensor(detail['index'], inp_k_res)
        return self.invoke()

    def get_input_shapes(self):
        """入力テンソルの形状を取得する."""
        assert self.input_shapes is not None, "モデルを先に読み込む必要があります"
        return self.input_shapes


class TensorRT(Interpreter):
    """TensorRT を用いて推論を実行するクラス."""

    def __init__(self):
        """インスタンスを初期化する."""
        self.frozen_func = None
        self.input_shapes = None

    def get_input_shapes(self) -> List[tf.TensorShape]:
        """入力テンソルの形状を取得する."""
        return self.input_shapes

    def compile(self, **kwargs):
        """このクラスではコンパイル処理を行わない."""
        pass

    def load(self, model_path: str) -> None:
        """モデルを読み込む."""
        saved_model_loaded = tf.saved_model.load(model_path,
                                                 tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.frozen_func = convert_var_to_const(graph_func)
        self.input_shapes = [inp.shape for inp in graph_func.inputs]

    def predict(self, img_arr: np.ndarray, other_arr: np.ndarray) \
            -> Sequence[Union[float, np.ndarray]]:
        """画像と補助入力から推論を実行する."""
        # まず通常通り reshape する
        img_arr = np.expand_dims(img_arr, axis=0).astype(np.float32)
        img_tensor = self.convert(img_arr)
        if other_arr is not None:
            other_arr = np.expand_dims(other_arr, axis=0).astype(np.float32)
            other_tensor = self.convert(other_arr)
            output_tensors = self.frozen_func(img_tensor, other_tensor)
        else:
            output_tensors = self.frozen_func(img_tensor)

        # バッチサイズ 1 のため先頭要素を取得
        outputs = [out.numpy().squeeze(axis=0) for out in output_tensors]
        # 出力が 1 次元の場合はリストにしない
        return outputs if len(outputs) > 1 else outputs[0]

    def predict_from_dict(self, input_dict):
        """辞書形式の入力データで推論を実行する."""
        args = []
        for inp in self.frozen_func.inputs:
            name = inp.name.split(':')[0]
            val = input_dict[name]
            val_res = np.expand_dims(val, axis=0).astype(np.float32)
            val_conv = self.convert(val_res)
            args.append(val_conv)
        output_tensors = self.frozen_func(*args)
        # バッチサイズ 1 のため先頭要素を取得
        outputs = [out.numpy().squeeze(axis=0) for out in output_tensors]
        # 出力が 1 次元の場合はリストにしない
        return outputs if len(outputs) > 1 else outputs[0]

    @staticmethod
    def convert(arr):
        """補助用の変換関数."""
        value = tf.compat.v1.get_variable("features", dtype=tf.float32,
                                          initializer=tf.constant(arr))
        return tf.convert_to_tensor(value=value)
