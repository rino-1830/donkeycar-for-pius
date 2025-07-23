from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Sized,
    Tuple,
    TypeVar,
)

from donkeycar.pipeline.types import TubRecord

# `TypeVar` を再利用する際は注意すること。
# 一貫しないと mypy が混乱しやすい。

R = TypeVar("R", covariant=True)
X = TypeVar("X", covariant=True)
Y = TypeVar("Y", covariant=True)
XOut = TypeVar("XOut", covariant=True)
YOut = TypeVar("YOut", covariant=True)


class SizedIterator(Generic[X], Iterator[X], Sized):
    """サイズ情報を持つイテレータの疑似プロトコル。"""

    def __init__(self) -> None:
        """Python 3.7 以前でも ``Protocol`` なしに表現するための空実装。"""
        pass


class TubSeqIterator(SizedIterator[TubRecord]):
    """``TubRecord`` のリストを順に返すイテレータ。"""

    def __init__(self, records: List[TubRecord]) -> None:
        """イテレータを初期化する。"""
        self.records = records or list()
        self.current_index = 0

    def __len__(self):
        """要素数を返す。"""
        return len(self.records)

    def __iter__(self) -> SizedIterator[TubRecord]:
        """新しい ``TubSeqIterator`` を返す。"""
        return TubSeqIterator(self.records)

    def __next__(self):
        """次の ``TubRecord`` を取得する。"""
        if self.current_index >= len(self.records):
            raise StopIteration("記録はこれ以上ありません")

        record = self.records[self.current_index]
        self.current_index += 1
        return record

    next = __next__


class TfmIterator(Generic[R, XOut, YOut], SizedIterator[Tuple[XOut, YOut]]):
    """任意のレコードから ``x`` と ``y`` を生成するイテレータ。"""

    def __init__(
        self,
        iterable: Iterable[R],
        x_transform: Callable[[R], XOut],
        y_transform: Callable[[R], YOut],
    ) -> None:

        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iterator = BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform,
        )

    def __len__(self):
        """内部イテレータの長さを返す。"""
        return len(self.iterator)

    def __iter__(self) -> SizedIterator[Tuple[XOut, YOut]]:
        """新しい ``TfmIterator`` を返す。"""
        return BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform,
        )

    def __next__(self):
        """次の ``(x, y)`` タプルを返す。"""
        return next(self.iterator)


class TfmTupleIterator(Generic[X, Y, XOut, YOut], SizedIterator[Tuple[XOut, YOut]]):
    """``(x, y)`` タプルを受け取り別の ``(x, y)`` に変換するイテレータ。"""

    def __init__(
        self,
        iterable: Iterable[Tuple[X, Y]],
        x_transform: Callable[[X], XOut],
        y_transform: Callable[[Y], YOut],
    ) -> None:

        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iterator = BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform,
        )

    def __len__(self):
        """内部イテレータの長さを返す。"""
        return len(self.iterator)

    def __iter__(self) -> SizedIterator[Tuple[XOut, YOut]]:
        """新しい ``TfmTupleIterator`` を返す。"""
        return BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform,
        )

    def __next__(self):
        """次の変換済み ``(x, y)`` を返す。"""
        return next(self.iterator)


class BaseTfmIterator_(Generic[XOut, YOut], SizedIterator[Tuple[XOut, YOut]]):
    """基本的な変換イテレータ。直接利用しないこと。"""

    def __init__(
        self,
        # 共通実装のために若干型安全性を失う
        iterable: Iterable[Any],
        x_transform: Callable[[R], XOut],
        y_transform: Callable[[R], YOut],
    ) -> None:

        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iterator = iter(self.iterable)

    def __len__(self):
        """長さを返す。"""
        return len(self.iterator)

    def __iter__(self) -> SizedIterator[Tuple[XOut, YOut]]:
        """新しい ``BaseTfmIterator_`` を返す。"""
        return BaseTfmIterator_(self.iterable, self.x_transform, self.y_transform)

    def __next__(self):
        """次の要素を ``(x, y)`` に変換して返す。"""
        record = next(self.iterator)
        if isinstance(record, tuple) and len(record) == 2:
            x, y = record
            return self.x_transform(x), self.y_transform(y)
        else:
            return self.x_transform(record), self.y_transform(record)


class TubSequence(Iterable[TubRecord]):
    """``TubRecord`` のコンテナ。``Iterable`` として振る舞う。"""

    def __init__(self, records: List[TubRecord]) -> None:
        """レコードを保持する。"""
        self.records = records

    def __iter__(self) -> SizedIterator[TubRecord]:
        """イテレータを返す。"""
        return TubSeqIterator(self.records)

    def __len__(self):
        """保持しているレコード数を返す。"""
        return len(self.records)

    def build_pipeline(
        self,
        x_transform: Callable[[TubRecord], X],
        y_transform: Callable[[TubRecord], Y],
    ) -> TfmIterator:
        """``TubRecord`` から ``(x, y)`` への変換パイプラインを構築する。"""
        return TfmIterator(self, x_transform=x_transform, y_transform=y_transform)

    @classmethod
    def map_pipeline(
        cls,
        x_transform: Callable[[X], XOut],
        y_transform: Callable[[Y], YOut],
        pipeline: SizedIterator[Tuple[X, Y]],
    ) -> SizedIterator[Tuple[XOut, YOut]]:
        """既存のパイプラインに変換を適用する。"""
        return TfmTupleIterator(
            pipeline, x_transform=x_transform, y_transform=y_transform
        )

    @classmethod
    def map_pipeline_factory(
        cls,
        x_transform: Callable[[X], XOut],
        y_transform: Callable[[Y], YOut],
        factory: Callable[[], SizedIterator[Tuple[X, Y]]],
    ) -> SizedIterator[Tuple[XOut, YOut]]:
        """ファクトリーから生成したパイプラインに変換を適用する。"""

        pipeline = factory()
        return cls.map_pipeline(
            pipeline=pipeline, x_transform=x_transform, y_transform=y_transform
        )
