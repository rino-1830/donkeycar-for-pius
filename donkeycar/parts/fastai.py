"""fastai.py モジュール

パイロットの作成、利用、保存、読み込みを行うためのメソッドを提供する。
パイロットは車両の角度とスロットルを決定する高レベルのロジックを持ち、
車両の動きを導くために一つ以上のモデルを含むことがある。
"""
from abc import ABC, abstractmethod

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List, Sequence, Callable
from logging import getLogger

import donkeycar as dk
import torch
from donkeycar.utils import normalize_image, linear_bin
from donkeycar.pipeline.types import TubRecord, TubDataset
from donkeycar.pipeline.sequence import TubSequence
from donkeycar.parts.interpreter import FastAIInterpreter, Interpreter, KerasInterpreter
from donkeycar.parts.pytorch.torch_data import TorchTubDataset, get_default_transform

from fastai.vision.all import *
from fastai.data.transforms import *
from fastai import optimizer as fastai_optimizer
from torch.utils.data import IterableDataset, DataLoader

from torchvision import transforms

ONE_BYTE_SCALE = 1.0 / 255.0

# x の型
XY = Union[float, np.ndarray, Tuple[Union[float, np.ndarray], ...]]

logger = getLogger(__name__)


class FastAiPilot(ABC):
    """FastAIモデル用パイロットの基底クラス。

    ステアリング角とスロットルを計算して車両を誘導する。
    """

    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        """クラスを初期化する。

        Args:
            interpreter: 使用するインタープリタ。
            input_shape: 入力画像の形状。
        """
        self.model: Optional[Model] = None
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        self.learner = None
        logger.info(f'{self} をインタープリタ {interpreter} で作成しました')

    def load(self, model_path):
        """モデルを読み込む。"""
        logger.info(f'モデル {model_path} を読み込みます')
        self.interpreter.load(model_path)

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        """重みのみを読み込む。"""
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self) -> None:
        """シャットダウン処理。"""
        pass

    def compile(self) -> None:
        """モデルのコンパイルを行う。"""
        pass

    @abstractmethod
    def create_model(self):
        """モデルを生成して返す。"""
        pass

    def set_optimizer(self, optimizer_type: str,
                      rate: float, decay: float) -> None:
        """オプティマイザを設定する。"""
        if optimizer_type == "adam":
            optimizer = fastai_optimizer.Adam(lr=rate, wd=decay)
        elif optimizer_type == "sgd":
            optimizer = fastai_optimizer.SGD(lr=rate, wd=decay)
        elif optimizer_type == "rmsprop":
            optimizer = fastai_optimizer.RMSprop(lr=rate, wd=decay)
        else:
            raise Exception(f"未知のオプティマイザタイプ: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    # 形状
    def get_input_shapes(self):
        """入力形状を取得する。"""
        return self.interpreter.get_input_shapes()

    def seq_size(self) -> int:
        """シーケンス長を返す。"""
        return 0

    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """Donkeycar パーツループでこのパートを実行する。

        Args:
            img_arr: uint8 [0,255] の画像データ。
            other_arr: IMU モデル用の IMU 配列や状態ベクトルなどの追加データ。

        Returns:
            角度とスロットルのタプル。
        """
        transform = get_default_transform(resize=False)
        norm_arr = transform(img_arr)
        tensor_other_array = torch.FloatTensor(other_arr) if other_arr else None
        return self.inference(norm_arr, tensor_other_array)

    def inference(self, img_arr: torch.tensor, other_arr: Optional[torch.tensor]) \
            -> Tuple[Union[float, torch.tensor], ...]:
        """インタープリタを用いて推論を実行する。

        Args:
            img_arr: float32 [0,1] の正規化済み画像データ。
            other_arr: IMU 配列や状態ベクトルなどの追加データを含むテンソル。

        Returns:
            角度とスロットルのタプル。
        """
        out = self.interpreter.predict(img_arr, other_arr)
        return self.interpreter_to_output(out)

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """辞書から入力を受け取りインタープリタで推論する。

        Args:
            input_dict: 文字列キーと ``np.ndarray`` の入力辞書。

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
        """子クラスで実装されるべき出力変換処理。

        Args:
            interpreter_out: 変換前のデータ。

        Returns:
            出力値。 ``np.ndarray`` のタプルなどを返すことがある。
        """
        pass

    def train(self,
              model_path: str,
              train_data: TorchTubDataset,
              train_steps: int,
              batch_size: int,
              validation_data: TorchTubDataset,
              validation_steps: int,
              epochs: int,
              verbose: int = 1,
              min_delta: float = .0005,
              patience: int = 5,
              show_plot: bool = False):
        """モデルの学習を実行する。

        Args:
            model_path: 学習済みモデルを保存するパス。
            train_data: 学習用データセット。
            train_steps: 学習ステップ数。
            batch_size: バッチサイズ。
            validation_data: 検証用データセット。
            validation_steps: 検証ステップ数。
            epochs: エポック数。
            verbose: ログ出力の詳細度。
            min_delta: 早期終了判定に使用する最小変化量。
            patience: 早期終了までの待機エポック数。
            show_plot: 損失グラフを保存するかどうか。

        Returns:
            損失履歴を含む辞書。
        """
        assert isinstance(self.interpreter, FastAIInterpreter)
        model = self.interpreter.model

        dataLoader = DataLoaders.from_dsets(train_data, validation_data, bs=batch_size, shuffle=False)
        if torch.cuda.is_available():
            dataLoader.cuda()

        #dataLoaderTest = self.dataBlock.dataloaders.test_dl(validation_data, with_labels=True)
        #print(dataLoader.train[0])

        callbacks = [
            EarlyStoppingCallback(monitor='valid_loss',
                                  patience=patience,
                                  min_delta=min_delta),
            SaveModelCallback(monitor='valid_loss',
                              every_epoch=False
                              )
        ]

        self.learner = Learner(dataLoader, model, loss_func=self.loss, path=Path(model_path).parent)

        logger.info(self.learner.summary())
        logger.info(self.learner.loss_func)

        lr_result = self.learner.lr_find()
        suggestedLr = float(lr_result[0])

        logger.info(f"推奨学習率 {suggestedLr}")

        self.learner.fit_one_cycle(epochs, suggestedLr, cbs=callbacks)

        torch.save(self.learner.model, model_path)

        if show_plot:
            self.learner.recorder.plot_loss()
            plt.savefig(Path(model_path).with_suffix('.png'))

        history = { "loss" : list(map((lambda x: x.item()), self.learner.recorder.losses)) }
        return history

    def __str__(self) -> str:
        """モデル初期化時の表示用文字列を返す。"""
        return type(self).__name__


class FastAILinear(FastAiPilot):
    """線形出力を行うシンプルなパイロット。

    Keras の ``Dense`` レイヤを線形活性化で用い、ステアリングとスロットル
    の 2 つの値を出力する。出力には範囲制限がない。
    """

    def __init__(self,
                 interpreter: Interpreter = FastAIInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        """インスタンスを生成する。

        Args:
            interpreter: 使用するインタープリタ。
            input_shape: 入力画像の形状。
            num_outputs: 出力数。
        """
        self.num_outputs = num_outputs
        self.loss = MSELossFlat()

        super().__init__(interpreter, input_shape)

    def create_model(self):
        """モデルを生成して返す。"""
        return Linear()

    def compile(self):
        """モデルのコンパイル設定を行う。"""
        self.optimizer = self.optimizer
        self.loss = 'mse'

    def interpreter_to_output(self, interpreter_out):
        """インタープリタの出力を角度とスロットルに変換する。"""
        interpreter_out = (interpreter_out * 2) - 1
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering, throttle

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """学習用ターゲットデータを生成する。"""
        assert isinstance(record, TubRecord), 'TubRecord が必要です'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        """出力テンソルの形状を返す。"""
        # [None, 120, 160, 3] のテンソル形状から None を取り除く必要がある
        img_shape = self.get_input_shapes()[0][1:]


class Linear(nn.Module):
    """FastAI 用のシンプルな CNN モデル。"""

    def __init__(self):
        """各レイヤーを初期化する。"""
        super(Linear, self).__init__()
        self.dropout = 0.1
        # 層を初期化する
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(6656, 100)
        self.fc2 = nn.Linear(100, 50)
        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.output1 = nn.Linear(50, 1)
        self.output2 = nn.Linear(50, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """順伝播を実行する。"""
        x = self.relu(self.conv24(x))
        x = self.drop(x)
        x = self.relu(self.conv32(x))
        x = self.drop(x)
        x = self.relu(self.conv64_5(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.relu(self.conv64_3(x))
        x = self.drop(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x1 = self.drop(x)
        angle = self.output1(x1)
        throttle = self.output2(x1)
        return torch.cat((angle, throttle), 1)
