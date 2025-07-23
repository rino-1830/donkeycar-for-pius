"""
keras.py

パイロットを作成、利用、保存、読み込むためのメソッド群。
パイロットは車両の角度とスロットルを決定する高レベルなロジックを保持する。
複数のモデルを組み合わせることで車両の動きを制御できる。
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable
from logging import getLogger

from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2

import donkeycar as dk
from donkeycar.utils import normalize_image, linear_bin
from donkeycar.pipeline.types import TubRecord
from donkeycar.parts.interpreter import Interpreter, KerasInterpreter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2DTranspose
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

ONE_BYTE_SCALE = 1.0 / 255.0

# x の型
XY = Union[float, np.ndarray, Tuple[Union[float, np.ndarray], ...]]


logger = getLogger(__name__)


class KerasPilot(ABC):
    """Keras モデルを用いてステアリングとスロットルを決定する基底クラス。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        # self.model: Optional[Model] = None  # 未使用
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        logger.info(f'{self} を作成しました。インタープリタ: {interpreter}')

    def load(self, model_path: str) -> None:
        logger.info(f'モデル {model_path} を読み込みます')
        self.interpreter.load(model_path)

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self) -> None:
        pass

    def compile(self) -> None:
        pass

    @abstractmethod
    def create_model(self):
        pass

    def set_optimizer(self, optimizer_type: str,
                      rate: float, decay: float) -> None:
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception(f"未知のオプティマイザタイプです: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    def get_input_shapes(self) -> List[tf.TensorShape]:
        return self.interpreter.get_input_shapes()

    def seq_size(self) -> int:
        return 0

    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ループ内でパーツを実行するインターフェース。

        Args:
            img_arr: 画像データの ``uint8`` 配列。
            other_arr: IMU モデル用の IMU 配列や行動モデルの状態ベクトルなど、
                追加データの配列。

        Returns:
            角度とスロットルのタプル。
        """
        norm_arr = normalize_image(img_arr)
        np_other_array = np.array(other_arr) if other_arr else None
        return self.inference(norm_arr, np_other_array)

    def inference(self, img_arr: np.ndarray, other_arr: Optional[np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """インタープリタを用いた推論を行う。

        Args:
            img_arr: 正規化済み画像 ``float32`` 配列。
            other_arr: IMU モデル用の IMU 配列や行動モデルの状態ベクトルなど、
                追加データの配列。

        Returns:
            角度とスロットルのタプル。
        """
        out = self.interpreter.predict(img_arr, other_arr)
        return self.interpreter_to_output(out)

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """入力辞書を用いて推論する。

        Args:
            input_dict: 文字列と ``np.ndarray`` の辞書。

        Returns:
            通常は ``(angle, throttle)`` のタプル。
        """
        output = self.interpreter.predict_from_dict(input_dict)
        return self.interpreter_to_output(output)

    @abstractmethod
    def interpreter_to_output(
            self,
            interpreter_out: Sequence[Union[float, np.ndarray]]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """子クラスで実装されるべき出力変換処理。"""
        pass

    def train(self,
              model_path: str,
              train_data: Union[DatasetV1, DatasetV2],
              train_steps: int,
              batch_size: int,
              validation_data: Union[DatasetV1, DatasetV2],
              validation_steps: int,
              epochs: int,
              verbose: int = 1,
              min_delta: float = .0005,
              patience: int = 5,
              show_plot: bool = False) -> tf.keras.callbacks.History:
        """モデルを学習する。

        Args:
            model_path: モデル保存先パス。
            train_data: 学習データセット。
            train_steps: 学習ステップ数。
            batch_size: バッチサイズ。
            validation_data: 検証データセット。
            validation_steps: 検証ステップ数。
            epochs: エポック数。
            verbose: 表示レベル。
            min_delta: 早期終了の最小改善量。
            patience: 早期終了までの待機エポック数。
            show_plot: 損失グラフを表示するかどうか。

        Returns:
            ``History`` オブジェクト。
        """
        assert isinstance(self.interpreter, KerasInterpreter)
        model = self.interpreter.model
        self.compile()

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          min_delta=min_delta),
            ModelCheckpoint(monitor='val_loss',
                            filepath=model_path,
                            save_best_only=True,
                            verbose=verbose)]

        history: tf.keras.callbacks.History = model.fit(
            x=train_data,
            steps_per_epoch=train_steps,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False)
            
        if show_plot:
            try:
                import matplotlib.pyplot as plt
                from pathlib import Path

                plt.figure(1)
                # 精度データがある場合のみ実行する
                # （例: カテゴリ出力）
                if 'angle_out_acc' in history.history:
                    plt.subplot(121)

                # 損失の履歴をまとめる
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('モデル損失')
                plt.ylabel('損失')
                plt.xlabel('エポック')
                plt.legend(['学習', '検証'], loc='upper right')

                # 精度の履歴をまとめる
                if 'angle_out_acc' in history.history:
                    plt.subplot(122)
                    plt.plot(history.history['angle_out_acc'])
                    plt.plot(history.history['val_angle_out_acc'])
                    plt.title('ステアリング精度')
                    plt.ylabel('精度')
                    plt.xlabel('エポック')

                plt.savefig(Path(model_path).with_suffix('.png'))
                # plt.show()

            except Exception as ex:
                print(f"損失グラフの生成に失敗しました: {ex}")
            
        return history.history

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        """学習用の ``x`` を生成し画像の拡張を行う。

        ここではモデルが画像のみを入力と仮定する。モデルの入力レイヤー名と
        辞書のキーは一致していなければならない。
        """
        assert isinstance(record, TubRecord), "TubRecord が必要です"
        img_arr = record.image(processor=img_processor)
        return {'img_in': img_arr}

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """学習用の ``y`` を生成する。

        モデルの出力レイヤー名と辞書のキーは一致していなければならない。
        """
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self) -> Tuple[Dict[str, np.typename], ...]:
        """``tf.data`` で使用する型情報を返す。全て ``float64`` を想定する。"""
        shapes = self.output_shapes()
        types = tuple({k: tf.float64 for k in d} for d in shapes)
        return types

    def output_shapes(self) -> Dict[str, tf.TensorShape]:
        return {}

    def __str__(self) -> str:
        """モデル初期化時の表示用文字列を返す。"""
        return type(self).__name__


class KerasCategorical(KerasPilot):
    """角度とスロットルを離散化して学習するパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 throttle_range: float = 0.5):
        super().__init__(interpreter, input_shape)
        self.throttle_range = throttle_range

    def create_model(self):
        return default_categorical(self.input_shape)

    def compile(self):
        self.interpreter.compile(
            optimizer=self.optimizer,
            metrics=['accuracy'],
            loss={'angle_out': 'categorical_crossentropy',
                  'throttle_out': 'categorical_crossentropy'},
            loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})

    def interpreter_to_output(self, interpreter_out):
        angle_binned, throttle_binned = interpreter_out
        N = len(throttle_binned)
        throttle = dk.utils.linear_unbin(throttle_binned, N=N,
                                         offset=0.0, R=self.throttle_range)
        angle = dk.utils.linear_unbin(angle_binned)
        return angle, throttle

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord を期待します"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        angle = linear_bin(angle, N=15, offset=1, R=2.0)
        throttle = linear_bin(throttle, N=20, offset=0.0, R=self.throttle_range)
        return {'angle_out': angle, 'throttle_out': throttle}

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle_out': tf.TensorShape([15]),
                   'throttle_out': tf.TensorShape([20])})
        return shapes


class KerasLinear(KerasPilot):
    """線形活性化で連続値を出力するパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(self.num_outputs, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord を期待します'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes


class KerasMemory(KerasLinear):
    """過去の操作履歴を入力に加えることで操舵を滑らかにするパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 mem_length: int = 3,
                 mem_depth: int = 0,
                 mem_start_speed: float = 0.0):
        self.mem_length = mem_length
        self.mem_start_speed = mem_start_speed
        self.mem_seq = deque([[0, mem_start_speed]] * mem_length)
        self.mem_depth = mem_depth
        super().__init__(interpreter, input_shape)

    def seq_size(self) -> int:
        return self.mem_length + 1

    def create_model(self):
        return default_memory(self.input_shape,
                              self.mem_length, self.mem_depth, )

    def load(self, model_path: str) -> None:
        super().load(model_path)
        self.mem_length = self.interpreter.get_input_shapes()[1][1] // 2
        self.mem_seq = deque([[0, self.mem_start_speed]] * self.mem_length)
        logger.info(f'メモリモデルを読み込みました。履歴長さ {self.mem_length}')

    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) -> \
            Tuple[Union[float, np.ndarray], ...]:
        # 起動時に以前の値を埋めるために一度だけ呼ばれる

        np_mem_arr = np.array(self.mem_seq).reshape((2 * self.mem_length,))
        img_arr_norm = normalize_image(img_arr)
        angle, throttle = super().inference(img_arr_norm, np_mem_arr)
        # 新しい値を履歴の末尾に追加する
        self.mem_seq.popleft()
        self.mem_seq.append([angle, throttle])
        return angle, throttle

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(record, list), 'List[TubRecord] を期待します'
        assert len(record) == self.mem_length + 1, \
            f"{self.mem_length} 個のレコードが必要ですが {len(record)} 個渡されました"
        img_arr = record[-1].image(processor=img_processor)
        mem = [[r.underlying['user/angle'], r.underlying['user/throttle']]
               for r in record[:-1]]
        np_mem = np.array(mem).reshape((2 * self.mem_length,))
        return {'img_in': img_arr, 'mem_in': np_mem}

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(records, list), 'List[TubRecord] を期待します'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'mem_in': tf.TensorShape(2 * self.mem_length)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes

    def __str__(self) -> str:
        """ For printing model initialisation """
        return super().__str__() \
            + f'-L:{self.mem_length}-D:{self.mem_depth}'


class KerasInferred(KerasPilot):
    """ステアリングからスロットルを推定するパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)):
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(1, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        return steering, dk.utils.throttle(steering)

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord を期待します"
        angle: float = record.underlying['user/angle']
        return {'n_outputs0': angle}

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([])})
        return shapes


class KerasIMU(KerasPilot):
    """画像と IMU ベクトルを入力として扱うパイロット。"""
    # TubRecord から取得する IMU データのキー
    imu_vec = [f'imu/{f}_{x}' for f in ('acl', 'gyr') for x in 'xyz']

    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2, num_imu_inputs: int = 6):
        self.num_outputs = num_outputs
        self.num_imu_inputs = num_imu_inputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_imu(num_outputs=self.num_outputs,
                           num_imu_inputs=self.num_imu_inputs,
                           input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        # 学習用の x へレコードを変換する
        assert isinstance(record, TubRecord), 'TubRecord を期待します'
        img_arr = record.image(processor=img_processor)
        imu_arr = np.array([record.underlying[k] for k in self.imu_vec])
        return {'img_in': img_arr, 'imu_in': imu_arr}

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord を期待します"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'out_0': angle, 'out_1': throttle}

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        # モデルの入出力レイヤー名にキーを合わせる必要がある
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'imu_in': tf.TensorShape([self.num_imu_inputs])},
                  {'out_0': tf.TensorShape([]),
                   'out_1': tf.TensorShape([])})
        return shapes


class KerasBehavioral(KerasCategorical):
    """画像と行動ベクトルを入力として扱うパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 throttle_range: float = 0.5,
                 num_behavior_inputs: int = 2):
        self.num_behavior_inputs = num_behavior_inputs
        super().__init__(interpreter, input_shape, throttle_range)

    def create_model(self):
        return default_bhv(num_bvh_inputs=self.num_behavior_inputs,
                           input_shape=self.input_shape)

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(record, TubRecord), 'TubRecord を期待します'
        # 学習用の x へレコードを変換する
        img_arr = record.image(processor=img_processor)
        bhv_arr = np.array(record.underlying['behavior/one_hot_state_array'])
        return {'img_in': img_arr, 'xbehavior_in': bhv_arr}

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        # モデルの入出力レイヤー名にキーを合わせる必要がある
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'xbehavior_in': tf.TensorShape([self.num_behavior_inputs])},
                  {'angle_out': tf.TensorShape([15]),
                   'throttle_out': tf.TensorShape([20])})
        return shapes


class KerasLocalizer(KerasPilot):
    """画像から位置を推定するパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_locations: int = 8):
        self.num_locations = num_locations
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_loc(num_locations=self.num_locations,
                           input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, metrics=['acc'],
                                 loss='mse')
        
    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        angle, throttle, track_loc = interpreter_out
        loc = np.argmax(track_loc)
        return angle[0], throttle[0], loc

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), "TubRecord を期待します"
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        loc = record.underlying['localizer/location']
        loc_one_hot = np.zeros(self.num_locations)
        loc_one_hot[loc] = 1
        return {'angle': angle, 'throttle': throttle, 'zloc': loc_one_hot}

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        # モデルの入出力レイヤー名にキーを合わせる必要がある
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle': tf.TensorShape([]),
                   'throttle': tf.TensorShape([]),
                   'zloc': tf.TensorShape([self.num_locations])})
        return shapes


class KerasLSTM(KerasPilot):
    """LSTM を用いて時系列画像を処理するパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 seq_length=3,
                 num_outputs=2):
        self.num_outputs = num_outputs
        self.seq_length = seq_length
        super().__init__(interpreter, input_shape)
        self.img_seq = deque()
        self.optimizer = "rmsprop"

    def seq_size(self) -> int:
        return self.seq_length

    def create_model(self):
        return rnn_lstm(seq_length=self.seq_length,
                        num_outputs=self.num_outputs,
                        input_shape=self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def x_transform(
            self,
            records: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
        -> Dict[str, Union[float, np.ndarray]]:
        """レコードのシーケンスを ``x`` に変換して学習に用いる。"""
        assert isinstance(records, list), 'List[TubRecord] を期待します'
        assert len(records) == self.seq_length, \
            f"{self.seq_length} 個のレコードが必要ですが {len(records)} 個渡されました"
        img_arrays = [rec.image(processor=img_processor) for rec in records]
        return {'img_in': np.array(img_arrays)}

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """角度とスロットルの最後の値のみを返す。"""
        assert isinstance(records, list), 'List[TubRecord] を期待します'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return {'model_outputs': [angle, throttle]}

    def run(self, img_arr, other_arr=None):
        if img_arr.shape[2] == 3 and self.input_shape[2] == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq.popleft()
        self.img_seq.append(img_arr)
        new_shape = (self.seq_length, *self.input_shape)
        img_arr = np.array(self.img_seq).reshape(new_shape)
        img_arr_norm = normalize_image(img_arr)
        return self.inference(img_arr_norm, other_arr)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        # モデルの入出力レイヤー名にキーを合わせる必要がある
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'model_outputs': tf.TensorShape([self.num_outputs])})
        return shapes

    def __str__(self) -> str:
        """ For printing model initialisation """
        return f'{super().__str__()}-L:{self.seq_length}'


class Keras3D_CNN(KerasPilot):
    """3D CNN を利用した時系列画像パイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 seq_length=20,
                 num_outputs=2):
        self.num_outputs = num_outputs
        self.seq_length = seq_length
        super().__init__(interpreter, input_shape)
        self.img_seq = deque()

    def seq_size(self) -> int:
        return self.seq_length

    def create_model(self):
        return build_3d_cnn(self.input_shape, s=self.seq_length,
                            num_outputs=self.num_outputs)

    def compile(self):
        self.interpreter.compile(loss='mse', optimizer=self.optimizer)

    def x_transform(
            self,
            records: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        """レコードのシーケンスを ``x`` に変換して学習に用いる。"""
        assert isinstance(records, list), 'List[TubRecord] を期待します'
        assert len(records) == self.seq_length, \
            f"{self.seq_length} 個のレコードが必要ですが {len(records)} 個渡されました"
        img_seq = [rec.image(processor=img_processor) for rec in records]
        return {'img_in': np.array(img_seq)}

    def y_transform(self, records: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """角度とスロットルの最後の値のみを返す。"""
        assert isinstance(records, list), 'List[TubRecord] を期待します'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return {'outputs': [angle, throttle]}

    def run(self, img_arr, other_arr=None):
        if img_arr.shape[2] == 3 and self.input_shape[2] == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq.popleft()
        self.img_seq.append(img_arr)
        new_shape = (self.seq_length, *self.input_shape)
        img_arr = np.array(self.img_seq).reshape(new_shape)
        img_arr_norm = normalize_image(img_arr)
        return self.inference(img_arr_norm, other_arr)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def output_shapes(self):
        # [None, 120, 160, 3] から None を除く必要がある
        img_shape = self.get_input_shapes()[0][1:]
        # モデルの入出力レイヤー名にキーを合わせる必要がある
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'outputs': tf.TensorShape([self.num_outputs])})
        return shapes


class KerasLatent(KerasPilot):
    """画像の潜在表現を学習するオートエンコーダパイロット。"""
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_latent(self.num_outputs, self.input_shape)

    def compile(self):
        loss = {"img_out": "mse", "n_outputs0": "mse", "n_outputs1": "mse"}
        weights = {"img_out": 100.0, "n_outputs0": 2.0, "n_outputs1": 1.0}
        self.interpreter.compile(optimizer=self.optimizer,
                                 loss=loss, loss_weights=weights)

    def interpreter_to_output(self, interpreter_out) \
            -> Tuple[Union[float, np.ndarray], ...]:
        steering = interpreter_out[1]
        throttle = interpreter_out[2]
        return steering[0][0], throttle[0][0]


def conv2d(filters, kernel, strides, layer_num, activation='relu'):
    """標準的な畳み込み層を作成するヘルパー関数。

    Args:
        filters: チャネル数。
        kernel: カーネルサイズ。
        strides: ストライド。
        layer_num: レイヤー番号。
        activation: 活性化関数。デフォルトは ``relu``。

    Returns:
        ``Convolution2D`` レイヤー。
    """
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))


def core_cnn_layers(img_in, drop, l4_stride=1):
    """複数モデルで共有される CNN の中核部分を返す。

    Args:
        img_in: ネットワークの入力レイヤー。
        drop: ドロップアウト率。
        l4_stride: 第4層のストライド。デフォルトは ``1``。

    Returns:
        CNN レイヤーのスタック。
    """
    x = img_in
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, l4_stride, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x


def default_n_linear(num_outputs, input_shape=(120, 160, 3)):
    """線形出力を複数持つ基本モデルを生成する。"""
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear')
    return model


def default_memory(input_shape=(120, 160, 3), mem_length=3, mem_depth=0):
    """メモリ入力を持つモデルを生成する。"""
    drop = 0.2
    drop2 = 0.1
    logger.info(f'メモリモデルを作成します。長さ {mem_length}, 深さ {mem_depth}')
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    mem_in = Input(shape=(2 * mem_length,), name='mem_in')
    y = mem_in
    for i in range(mem_depth):
        y = Dense(4 * mem_length, activation='relu', name=f'mem_{i}')(y)
        y = Dropout(drop2)(y)
    for i in range(1, mem_length):
        y = Dense(2 * (mem_length - i), activation='relu', name=f'mem_c_{i}')(y)
        y = Dropout(drop2)(y)
    x = concatenate([x, y])
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)
    activation = ['tanh', 'sigmoid']
    outputs = [Dense(1, activation=activation[i], name='n_outputs' + str(i))(x)
               for i in range(2)]
    model = Model(inputs=[img_in, mem_in], outputs=outputs, name='memory')
    return model


def default_categorical(input_shape=(120, 160, 3)):
    """角度とスロットルを離散化するモデルを生成する。"""
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop, l4_stride=2)
    x = Dense(100, activation='relu', name="dense_1")(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name="dense_2")(x)
    x = Dropout(drop)(x)
    # 角度を 15 ビンに分類して出力
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    # スロットルを 20 ビンに分類して出力
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out],
                  name='categorical')
    return model


def default_imu(num_outputs, num_imu_inputs, input_shape):
    """IMU 入力を併用するモデルを生成する。"""
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    imu_in = Input(shape=(num_imu_inputs,), name="imu_in")

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    
    y = imu_in
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    y = Dense(14, activation='relu')(y)
    
    z = concatenate([x, y])
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)

    outputs = []
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='out_' + str(i))(z))
        
    model = Model(inputs=[img_in, imu_in], outputs=outputs, name='imu')
    return model


def default_bhv(num_bvh_inputs, input_shape):
    """行動入力を組み合わせるモデルを生成する。"""
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    # TensorRT では入力レイヤーがアルファベット順に並ぶため、
    # behavior は image の後になるよう先頭に x を付ける
    bvh_in = Input(shape=(num_bvh_inputs,), name="xbehavior_in")

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    
    y = bvh_in
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    y = Dense(num_bvh_inputs * 2, activation='relu')(y)
    
    z = concatenate([x, y])
    z = Dense(100, activation='relu')(z)
    z = Dropout(.1)(z)
    z = Dense(50, activation='relu')(z)
    z = Dropout(.1)(z)
    
    # 角度を 15 ビンに分類して出力
    angle_out = Dense(15, activation='softmax', name='angle_out')(z)
    # スロットルを 20 ビンに分類して出力
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(z)
        
    model = Model(inputs=[img_in, bvh_in], outputs=[angle_out, throttle_out],
                  name='behavioral')
    return model


def default_loc(num_locations, input_shape):
    """位置推定を含むモデルを生成する。"""
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')

    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    
    z = Dense(50, activation='relu')(x)
    z = Dropout(drop)(z)

    # 角度の線形出力
    angle_out = Dense(1, activation='linear', name='angle')(z)
    # スロットルの線形出力
    throttle_out = Dense(1, activation='linear', name='throttle')(z)
    # 位置を表すカテゴリカル出力
    # TF Lite のバグで出力がレイヤー名のアルファベット順に並ぶため
    # この出力が最後になるようにしておく
    loc_out = Dense(num_locations, activation='softmax', name='zloc')(z)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out, loc_out],
                  name='localizer')
    return model


def rnn_lstm(seq_length=3, num_outputs=2, input_shape=(120, 160, 3)):
    """LSTM を用いた時系列モデルを生成する。"""
    # keras の TimeDistributed が要求する形状 (num_samples, seq_length, input_shape)
    # を作るためシーケンス次元を追加する
    img_seq_shape = (seq_length,) + input_shape
    img_in = Input(shape=img_seq_shape, name='img_in')
    drop_out = 0.3

    x = img_in
    x = TD(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(Convolution2D(32, (3, 3), strides=(1, 1), activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)
    x = TD(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TD(Flatten(name='flattened'))(x)
    x = TD(Dense(100, activation='relu'))(x)
    x = TD(Dropout(drop_out))(x)

    x = LSTM(128, return_sequences=True, name="LSTM_seq")(x)
    x = Dropout(.1)(x)
    x = LSTM(128, return_sequences=False, name="LSTM_fin")(x)
    x = Dropout(.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    out = Dense(num_outputs, activation='linear', name='model_outputs')(x)
    model = Model(inputs=[img_in], outputs=[out], name='lstm')
    return model


def build_3d_cnn(input_shape, s, num_outputs):
    """3D CNN モデルを生成する。

    Credit: https://github.com/jessecha/DNRacing/blob/master/3D_CNN_Model/model.py

    Args:
        input_shape: 画像入力の形状。
        s: シーケンス長。
        num_outputs: 出力次元。

    Returns:
        Keras モデル。
    """
    drop = 0.5
    input_shape = (s, ) + input_shape
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    # 第2層
    x = Conv3D(
            filters=16, kernel_size=(3, 3, 3), strides=(1, 3, 3),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # 第3層
    x = Conv3D(
            filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
        data_format=None)(x)
    # 第4層
    x = Conv3D(
            filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # 第5層
    x = Conv3D(
            filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1),
            data_format='channels_last', padding='same', activation='relu')(x)
    x = MaxPooling3D(
            pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid',
            data_format=None)(x)
    # 全結合層
    x = Flatten()(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop)(x)

    out = Dense(num_outputs, name='outputs')(x)
    model = Model(inputs=[img_in], outputs=out, name='3dcnn')
    return model


def default_latent(num_outputs, input_shape):
    """潜在表現と画像を同時に出力するモデルを生成する。"""
    # TODO: このオートエンコーダは標準の CNN を利用してエンコードし、
    #  対応するデコーダを持つべき。さらに出力の順序を画像が最後になるよう
    #  逆にする必要がある。
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, 5, strides=2, activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 5, strides=2, activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, 3, strides=1, activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_6")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 3, strides=2, activation='relu', name="conv2d_7")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, 1, strides=2, activation='relu', name="latent")(x)
    
    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_1")(x)
    y = Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                        name="deconv2d_2")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_3")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_4")(y)
    y = Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                        name="deconv2d_5")(y)
    y = Conv2DTranspose(filters=1, kernel_size=3, strides=2, name="img_out")(y)
    
    x = Flatten(name='flattened')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = [y]
    for i in range(num_outputs):
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=outputs, name='latent')
    return model
