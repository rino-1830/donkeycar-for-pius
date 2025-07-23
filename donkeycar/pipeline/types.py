import copy
import os
from typing import Any, List, Optional, TypeVar, Iterator, Iterable
import logging
import numpy as np
from donkeycar.config import Config
from donkeycar.parts.tub_v2 import Tub
from donkeycar.utils import load_image, load_pil_image
from typing_extensions import TypedDict


logger = logging.getLogger(__name__)

X = TypeVar("X", covariant=True)

TubRecordDict = TypedDict(
    "TubRecordDict",
    {
        "_index": int,
        "cam/image_array": str,
        "user/angle": float,
        "user/throttle": float,
        "user/mode": str,
        "imu/acl_x": Optional[float],
        "imu/acl_y": Optional[float],
        "imu/acl_z": Optional[float],
        "imu/gyr_x": Optional[float],
        "imu/gyr_y": Optional[float],
        "imu/gyr_z": Optional[float],
        "behavior/one_hot_state_array": Optional[List[float]],
        "localizer/location": Optional[int],
    },
)


class TubRecord(object):
    """Tub 内の単一レコードを表す。"""

    def __init__(
        self, config: Config, base_path: str, underlying: TubRecordDict
    ) -> None:
        """インスタンスを初期化する。"""
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._cache_images = getattr(self.config, "CACHE_IMAGES", True)
        self._image: Optional[Any] = None

    def image(self, processor=None, as_nparray=True) -> np.ndarray:
        """画像を読み込む。"""

        # ``processor`` にオーグメンテーションやクロップ処理を渡すことができる
        # ``as_nparray`` が ``True`` の場合は ``uint8`` の ``ndarray`` に変換し、
        # ``False`` の場合は ``PIL.Image`` をそのまま返す
        if self._image is None:
            image_path = self.underlying["cam/image_array"]
            full_path = os.path.join(self.base_path, "images", image_path)

            if as_nparray:
                _image = load_image(full_path, cfg=self.config)
            else:
                # 生の ``PIL.Image`` が欲しい場合
                _image = load_pil_image(full_path, cfg=self.config)
            if processor:
                _image = processor(_image)
            # 設定で禁止されていなければ画像をキャッシュする
            if self._cache_images:
                self._image = _image
        else:
            _image = self._image
        return _image

    def __repr__(self) -> str:
        return repr(self.underlying)


class TubDataset(object):
    """Tub を読み込み ``TubRecord`` のリストを生成する。"""

    def __init__(self, config: Config, tub_paths: List[str], seq_size: int = 0) -> None:
        """データセットを初期化する。"""
        self.config = config
        self.tub_paths = tub_paths
        self.tubs: List[Tub] = [
            Tub(tub_path, read_only=True) for tub_path in self.tub_paths
        ]
        self.records: List[TubRecord] = list()
        self.train_filter = getattr(config, "TRAIN_FILTER", None)
        self.seq_size = seq_size

    def get_records(self):
        """``TubRecord`` のリストを取得する。"""
        if not self.records:
            logger.info(f"パス {self.tub_paths} から tub を読み込みます")
            for tub in self.tubs:
                for underlying in tub:
                    record = TubRecord(self.config, tub.base_path, underlying)
                    if not self.train_filter or self.train_filter(record):
                        self.records.append(record)
            if self.seq_size > 0:
                seq = Collator(self.seq_size, self.records)
                self.records = list(seq)
        return self.records


class Collator(Iterable[List[TubRecord]]):
    """連続した TubRecord を束ねてシーケンス化する。"""

    def __init__(self, seq_length: int, records: List[TubRecord]):
        """シーケンス長とレコードを指定して初期化する。"""
        self.records = records
        self.seq_length = seq_length

    @staticmethod
    def is_continuous(rec_1: TubRecord, rec_2: TubRecord) -> bool:
        """2 つのレコードが連続しているか判定する。"""
        it_is = (
            rec_1.underlying["_index"] == rec_2.underlying["_index"] - 1
            and "__empty__" not in rec_1.underlying
            and "__empty__" not in rec_2.underlying
        )
        return it_is

    def __iter__(self) -> Iterator[List[TubRecord]]:
        """イテレータとして連続レコードのリストを返す。"""
        it = iter(self.records)
        for this_record in it:
            seq = [this_record]
            seq_it = copy.copy(it)
            for next_record in seq_it:
                if (
                    self.is_continuous(this_record, next_record)
                    and len(seq) < self.seq_length
                ):
                    seq.append(next_record)
                    this_record = next_record
                else:
                    break
            if len(seq) == self.seq_length:
                yield seq
