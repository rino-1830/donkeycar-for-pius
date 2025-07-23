import logging
from typing import List
from donkeycar.config import Config
from donkeycar.parts import cv as cv_parts


logger = logging.getLogger(__name__)


class ImageTransformations:
    def __init__(self, config: Config, transformation: str,
                 post_transformation: str = None) -> object:
        """画像変換パーツのリストを作成し、順番に実行して変換後の画像を生成する。

        Args:
            config: ``Config`` オブジェクト。
            transformation: 使用する変換名。
            post_transformation: 追加で適用する変換名。

        Returns:
            object: この ``ImageTransformations`` インスタンス。
        """
        transformations = getattr(config, transformation, [])
        if post_transformation:
            transformations += getattr(config, post_transformation, [])
        self.transformations = [image_transformer(name, config) for name in
                                transformations]
        logger.info(f'ImageTransformations {transformations} を生成')
    
    def run(self, image):
        """変換パーツを順番に実行して変換後の画像を返す。

        Args:
            image: 入力画像。

        Returns:
            変換後の画像。
        """
        for transformer in self.transformations:
            image = transformer.run(image)
        return image


def image_transformer(name: str, config):
    """cv 画像変換パーツのファクトリ。

    Args:
        name: 変換の名称。
        config: パーツ生成時に使用する設定オブジェクト。

    Returns:
        object: cv 画像変換パーツ。
    """
    #
    # マスキング処理
    #
    if "TRAPEZE" == name:
        return cv_parts.ImgTrapezoidalMask(
            config.ROI_TRAPEZE_UL,
            config.ROI_TRAPEZE_UR,
            config.ROI_TRAPEZE_LL,
            config.ROI_TRAPEZE_LR,
            config.ROI_TRAPEZE_MIN_Y,
            config.ROI_TRAPEZE_MAX_Y
        )

    elif "CROP" == name:
        return cv_parts.ImgCropMask(
            config.ROI_CROP_LEFT,
            config.ROI_CROP_TOP,
            config.ROI_CROP_RIGHT,
            config.ROI_CROP_BOTTOM
        )
    #
    # カラースペース変換
    #
    elif "RGB2BGR" == name:
        return cv_parts.ImgRGB2BGR()
    elif "BGR2RGB" == name:
        return cv_parts.ImgBGR2RGB()
    elif "RGB2HSV" == name:
        return cv_parts.ImgRGB2HSV()
    elif "HSV2RGB" == name:
        return cv_parts.ImgHSV2RGB()
    elif "BGR2HSV" == name:
        return cv_parts.ImgBGR2HSV()
    elif "HSV2BGR" == name:
        return cv_parts.ImgHSV2BGR()
    elif "RGB2GRAY" == name:
        return cv_parts.ImgRGB2GRAY()
    elif "RBGR2GRAY" == name:
        return cv_parts.ImgBGR2GRAY()
    elif "HSV2GRAY" == name:
        return cv_parts.ImgHSV2GRAY()
    elif "GRAY2RGB" == name:
        return cv_parts.ImgGRAY2RGB()
    elif "GRAY2BGR" == name:
        return cv_parts.ImgGRAY2BGR()
    elif "CANNY" == name:
        # Canny エッジ検出
        return cv_parts.ImgCanny(config.CANNY_LOW_THRESHOLD,
                                 config.CANNY_HIGH_THRESHOLD,
                                 config.CANNY_APERTURE)
    #
    # ぼかし処理
    #
    elif "BLUR" == name:
        if config.BLUR_GAUSSIAN:
            return cv_parts.ImgGaussianBlur(config.BLUR_KERNEL,
                                            config.BLUR_KERNEL_Y)
        else:
            return cv_parts.ImgSimpleBlur(config.BLUR_KERNEL,
                                          config.BLUR_KERNEL_Y)
    #
    # リサイズ処理
    #
    elif "RESIZE" == name:
        return cv_parts.ImageResize(config.RESIZE_WIDTH, config.RESIZE_HEIGHT)
    elif "SCALE" == name:
        return cv_parts.ImageScale(config.SCALE_WIDTH, config.SCALE_HEIGHT)
    elif name.startswith("CUSTOM"):
        return custom_transformer(name, config)
    else:
        msg = f"{name} は有効な拡張ではありません"
        logger.error(msg)
        raise ValueError(msg)


def custom_transformer(name: str,
                       config: Config,
                       file_path: str = None,
                       class_name: str = None) -> object:
    """カスタム画像変換パーツを生成する。

    カスタム変換パーツは ``Config`` オブジェクトをコンストラクタで受け取り、
    ``run()`` メソッドで画像を受け取って変換後の画像を返すクラスである。
    例::

        class CustomImageTransformer:
            def __init__(self, config: Config):
                # 設定をインスタンス変数に保存する
                self.foo = config.foo

            def run(self, image):
                # 画像を処理して変換結果を返す
                return image.copy()

    変換名は ``CUSTOM`` で始まり、たとえば ``CUSTOM_BLUR`` のように指定する。
    モジュール名とクラス名は設定に記載し、モジュール名は ``_MODULE``、
    クラス名は ``_CLASS`` というサフィックスを付けたキーで参照する。

    Args:
        name: 変換名。
        config: ``Config`` オブジェクト。
        file_path: モジュールのファイルパス。指定しない場合は設定から取得する。
        class_name: クラス名。指定しない場合は設定から取得する。

    Returns:
        object: カスタム変換パーツのインスタンス。``run()`` は画像を受け取り
        変換後の画像を返す。

    Raises:
        ValueError: モジュールまたはクラスが見つからない場合。
    """
    if file_path is None:
        file_path = getattr(config, name + "_MODULE", None)
    if file_path is None:
        raise ValueError(f"カスタム画像変換器 {name} 用のモジュールファイルパスが宣言されていません")
    if class_name is None:
        class_name = getattr(config, name + "_CLASS", None)
    if class_name is None:
        raise ValueError(f"カスタム画像変換器 {name} のクラスが宣言されていません")
    
    import os
    import sys
    import importlib.util
    # 読み込むモジュールを指定する
    # パスを基準に相対的にインポートする
    # モジュール

    #
    # ベースファイル名を取り出してカスタム名前空間に
    # 付加することでモジュール名を作成する
    #
    namespace = "custom.transformation." + os.path.split(file_path)[1].split('.')[0]
    module = sys.modules.get(namespace)
    if module:
        # 既にロード済み
        logger.info(f"既存のカスタム変換モジュールを使用: {namespace}")
    else:
        logger.info(f"カスタム変換モジュール {namespace} を {file_path} から読み込み")

        # Python ファイルから動的にロード
        spec=importlib.util.spec_from_file_location(namespace, file_path)
        if spec:
            # spec から新しいモジュールを作成
            module = importlib.util.module_from_spec(spec)
            if module:
                logger.info(f"カスタム変換モジュール {namespace} をキャッシュ")
                sys.modules[namespace] = module

                # モジュールを独自の名前空間で実行する
                # モジュールがインポートまたは再読み込みされたとき
                spec.loader.exec_module(module)
            else:
                logger.error(f"spec からカスタム変換モジュール {namespace} を {file_path} から読み込めませんでした")
        else:
            logger.error(f"{file_path} からカスタム変換モジュール {namespace} の spec を読み込めませんでした")

    if module:
        # モジュールからクラスを取得
        my_class = getattr(module, class_name, None)
        if my_class is None:
            raise ValueError(f"モジュール {namespace} ({file_path}) にクラス {class_name} が見つかりません")

        #
        # クラスのインスタンスを生成する
        # ``__init__()`` は ``Config`` オブジェクトを受け取る必要がある
        #
        return my_class(config)
    else:
        raise ValueError(f"カスタム変換モジュールを {file_path} から読み込めません")

