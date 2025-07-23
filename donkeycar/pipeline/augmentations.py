import albumentations.core.transforms_interface
import logging
import albumentations as A
from albumentations import GaussianBlur
from albumentations.augmentations.transforms import RandomBrightnessContrast

from donkeycar.config import Config


logger = logging.getLogger(__name__)


class ImageAugmentation:
    """画像の前処理用オーグメンテーションを生成するクラス。"""

    def __init__(self, cfg, key, prob=0.5, always_apply=False):
        """設定からオーグメンテーションを組み立てる。"""
        aug_list = getattr(cfg, key, [])
        augmentations = [
            ImageAugmentation.create(a, cfg, prob, always_apply) for a in aug_list
        ]
        self.augmentations = A.Compose(augmentations)

    @classmethod
    def create(
        cls, aug_type: str, config: Config, prob, always
    ) -> albumentations.core.transforms_interface.BasicTransform:
        """オーグメンテーション生成ファクトリー。

        Cropping や台形マスクは学習・検証・推論のすべてで適用すべき変換であり、
        Multiply や Blur などは学習時のみ使用する。

        Args:
            aug_type: オーグメンテーション種別。
            config: 設定オブジェクト。
            prob: 適用確率。
            always: 必ず適用するかどうか。

        Returns:
            Albumentations の変換インスタンス。
        """

        if aug_type == "BRIGHTNESS":
            b_limit = getattr(config, "AUG_BRIGHTNESS_RANGE", 0.2)
            logger.info(f"オーグメンテーション {aug_type} {b_limit} を作成")
            return RandomBrightnessContrast(
                brightness_limit=b_limit,
                contrast_limit=b_limit,
                p=prob,
                always_apply=always,
            )

        elif aug_type == "BLUR":
            b_range = getattr(config, "AUG_BLUR_RANGE", 3)
            logger.info(f"オーグメンテーション {aug_type} {b_range} を作成")
            return GaussianBlur(
                sigma_limit=b_range, blur_limit=(13, 13), p=prob, always_apply=always
            )

    # パーツインターフェース
    def run(self, img_arr):
        """画像配列にオーグメンテーションを適用する。"""
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr
