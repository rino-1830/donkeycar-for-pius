import glob
import logging
import os
import time

import numpy as np
from PIL import Image

from donkeycar.utils import rgb2gray

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CameraError(Exception):
    pass


class BaseCamera:
    """カメラ部品の基本クラス。"""

    def run_threaded(self):
        """スレッド用に最新のフレームを返す。"""
        return self.frame


class PiCamera(BaseCamera):
    """Raspberry Pi カメラモジュールを制御するクラス。"""

    def __init__(
        self,
        image_w=160,
        image_h=120,
        image_d=3,
        framerate=20,
        vflip=False,
        hflip=False,
    ):
        """PiCamera の初期化を行う。

        Args:
            image_w (int): 画像の幅。
            image_h (int): 画像の高さ。
            image_d (int): 画像のチャンネル数。
            framerate (int): フレームレート。
            vflip (bool): 垂直反転の有無。
            hflip (bool): 水平反転の有無。
        """
        from picamera import PiCamera
        from picamera.array import PiRGBArray

        resolution = (image_w, image_h)
        # カメラとストリームを初期化
        self.camera = PiCamera()  # PiCamera は (高さ, 幅) 順で解像度を受け取る
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.vflip = vflip
        self.camera.hflip = hflip
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(
            self.rawCapture, format="rgb", use_video_port=True
        )

        # フレームとスレッド停止フラグを初期化
        self.frame = None
        self.on = True
        self.image_d = image_d

        # 最初のフレームを取得するまで待機
        logger.info("PiCamera をロードしました...")
        if self.stream is not None:
            logger.info("PiCamera をオープンしました...")
            warming_time = time.time() + 5  # 5 秒後にタイムアウト
            while self.frame is None and time.time() < warming_time:
                logger.info("...カメラをウォームアップ中")
                self.run()
                time.sleep(0.2)

            if self.frame is None:
                raise CameraError("PiCamera を起動できません。")
        else:
            raise CameraError("PiCamera を開けません。")
        logger.info("PiCamera の準備ができました。")

    def run(self):
        """ストリームからフレームを取得し次のフレーム準備を行う。"""
        # ストリームからフレームを取得し、次のフレームのためにバッファをクリア
        if self.stream is not None:
            f = next(self.stream)
            if f is not None:
                self.frame = f.array
                self.rawCapture.truncate(0)
                if self.image_d == 1:
                    self.frame = rgb2gray(self.frame)

        return self.frame

    def update(self):
        """スレッドが停止されるまでループし続ける。"""
        # スレッドが停止されるまで無限ループ
        while self.on:
            self.run()

    def shutdown(self):
        """カメラを停止しリソースを解放する。"""
        # スレッド停止を通知
        self.on = False
        logger.info("PiCamera を停止します")
        time.sleep(0.5)
        self.stream.close()
        self.rawCapture.close()
        self.camera.close()
        self.stream = None
        self.rawCapture = None
        self.camera = None


class Webcam(BaseCamera):
    """USB など汎用 Web カメラを利用するクラス。"""

    def __init__(
        self, image_w=160, image_h=120, image_d=3, framerate=20, camera_index=0
    ):
        """Web カメラの初期化を行う。

        Note:
            `pygame` はデフォルトではインストールされていません。RaspberryPi で使用する場合は次のコマンドを実行してください::

                sudo apt-get install libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0
                pip install pygame

        Args:
            image_w (int): 画像の幅。
            image_h (int): 画像の高さ。
            image_d (int): 画像のチャンネル数。
            framerate (int): フレームレート。
            camera_index (int): 使用するカメラのインデックス。
        """
        super().__init__()
        self.cam = None
        self.framerate = framerate

        # スレッド停止フラグとして使う変数を初期化
        self.frame = None
        self.image_d = image_d
        self.image_w = image_w
        self.image_h = image_h

        self.init_camera(image_w, image_h, image_d, camera_index)
        self.on = True

    def init_camera(self, image_w, image_h, image_d, camera_index=0):
        """Web カメラを開いて設定する。"""
        try:
            import pygame
            import pygame.camera
        except ModuleNotFoundError as e:
            logger.error(
                "pygame をインポートできません。次を実行してインストールしてください:\n"
                "    sudo apt-get install libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0\n"
                "    pip install pygame"
            )
            raise e

        logger.info("Web カメラを開いています...")

        self.resolution = (image_w, image_h)

        try:
            pygame.init()
            pygame.camera.init()
            l = pygame.camera.list_cameras()

            if len(l) == 0:
                raise CameraError("利用可能なカメラがありません")

            logger.info(f"利用可能なカメラ {l}")
            if camera_index < 0 or camera_index >= len(l):
                raise CameraError(
                    f"myconfig.py の 'CAMERA_INDEX={camera_index}' の値が範囲外です"
                )

            self.cam = pygame.camera.Camera(l[camera_index], self.resolution, "RGB")
            self.cam.start()

            logger.info(f"Web カメラ {l[camera_index]} をオープンしました...")
            warming_time = time.time() + 5  # 5 秒後にタイムアウト
            while self.frame is None and time.time() < warming_time:
                logger.info("...カメラをウォームアップ中")
                self.run()
                time.sleep(0.2)

            if self.frame is None:
                raise CameraError(
                    'Web カメラを起動できません。\n複数のカメラがある場合は myconfig.py の "CAMERA_INDEX" が正しいか確認してください'
                )

        except CameraError:
            raise
        except Exception as e:
            raise CameraError(
                'Web カメラを開けません。\n複数のカメラがある場合は myconfig.py の "CAMERA_INDEX" が正しいか確認してください'
            ) from e
        logger.info("Web カメラの準備ができました。")

    def run(self):
        """カメラから画像を取得して返す。"""
        import pygame.image

        if self.cam.query_image():
            snapshot = self.cam.get_image()
            if snapshot is not None:
                snapshot1 = pygame.transform.scale(snapshot, self.resolution)
                self.frame = pygame.surfarray.pixels3d(
                    pygame.transform.rotate(
                        pygame.transform.flip(snapshot1, True, False), 90
                    )
                )
                if self.image_d == 1:
                    self.frame = rgb2gray(frame)

        return self.frame

    def update(self):
        """設定されたフレームレートになるよう待機しつつフレームを更新する。"""
        from datetime import datetime

        while self.on:
            start = datetime.now()
            self.run()
            stop = datetime.now()
            s = 1 / self.framerate - (stop - start).total_seconds()
            if s > 0:
                time.sleep(s)

    def run_threaded(self):
        """最新のフレームを返す。"""
        return self.frame

    def shutdown(self):
        """カメラを停止しリソースを解放する。"""
        # スレッド停止を通知
        self.on = False
        if self.cam:
            logger.info("Web カメラを停止します")
            self.cam.stop()
            self.cam = None
        time.sleep(0.5)


class CSICamera(BaseCamera):
    """Jetson Nano 用 IMX219 カメラを扱うクラス。"""

    def gstreamer_pipeline(
        self,
        capture_width=3280,
        capture_height=2464,
        output_width=224,
        output_height=224,
        framerate=21,
        flip_method=0,
    ):
        """GStreamer パイプライン文字列を生成する。"""
        return (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d "
            "! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx "
            "! videoconvert ! appsink"
        ) % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            output_width,
            output_height,
        )

    def __init__(
        self,
        image_w=160,
        image_h=120,
        image_d=3,
        capture_width=3280,
        capture_height=2464,
        framerate=60,
        gstreamer_flip=0,
    ):
        """CSI カメラを初期化する。

        Args:
            image_w (int): 画像の幅。
            image_h (int): 画像の高さ。
            image_d (int): 画像のチャンネル数。
            capture_width (int): キャプチャ解像度の幅。
            capture_height (int): キャプチャ解像度の高さ。
            framerate (int): フレームレート。
            gstreamer_flip (int): 回転・反転設定。

        Note:
            ``gstreamer_flip`` の値は次の通り::

                0 - 反転なし
                1 - 反時計回りに 90 度回転
                2 - 垂直反転
                3 - 時計回りに 90 度回転
        """
        self.w = image_w
        self.h = image_h
        self.flip_method = gstreamer_flip
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.framerate = framerate
        self.frame = None
        self.init_camera()
        self.running = True

    def init_camera(self):
        """カメラとストリームを初期化する。"""
        import cv2

        # カメラとストリームを初期化
        self.camera = cv2.VideoCapture(
            self.gstreamer_pipeline(
                capture_width=self.capture_width,
                capture_height=self.capture_height,
                output_width=self.w,
                output_height=self.h,
                framerate=self.framerate,
                flip_method=self.flip_method,
            ),
            cv2.CAP_GSTREAMER,
        )

        if self.camera and self.camera.isOpened():
            logger.info("CSI カメラをオープンしました...")
            warming_time = time.time() + 5  # 5 秒後にタイムアウト
            while self.frame is None and time.time() < warming_time:
                logger.info("...カメラをウォームアップ中")
                self.poll_camera()
                time.sleep(0.2)

            if self.frame is None:
                raise RuntimeError("CSI カメラを起動できません。")
        else:
            raise RuntimeError("CSI カメラを開けません。")
        logger.info("CSI カメラの準備ができました。")

    def update(self):
        """フレームを継続的に取得する。"""
        while self.running:
            self.poll_camera()

    def poll_camera(self):
        """カメラからデータを取得して RGB 形式に変換する。"""
        import cv2

        self.ret, frame = self.camera.read()
        if frame is not None:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def run(self):
        """フレームを取得して返す。"""
        self.poll_camera()
        return self.frame

    def run_threaded(self):
        """最新のフレームを返す。"""
        return self.frame

    def shutdown(self):
        """カメラを停止しリソースを解放する。"""
        self.running = False
        logger.info("CSI カメラを停止します")
        time.sleep(0.5)
        del self.camera


class V4LCamera(BaseCamera):
    """v4l2capture ライブラリを利用したカメラクラス。"""

    def __init__(
        self,
        image_w=160,
        image_h=120,
        image_d=3,
        framerate=20,
        dev_fn="/dev/video0",
        fourcc="MJPG",
    ):
        """V4L2 カメラを初期化する。

        Args:
            image_w (int): 画像の幅。
            image_h (int): 画像の高さ。
            image_d (int): 画像のチャンネル数。
            framerate (int): フレームレート。
            dev_fn (str): デバイスファイル名。
            fourcc (str): フォーマット指定子。
        """

        self.running = True
        self.frame = None
        self.image_w = image_w
        self.image_h = image_h
        self.dev_fn = dev_fn
        self.fourcc = fourcc

    def init_video(self):
        """v4l2 デバイスを初期化する。"""
        import v4l2capture

        self.video = v4l2capture.Video_device(self.dev_fn)

        # デバイスへ画像サイズを提案。サポートしていない場合は別のサイズが返される。
        self.size_x, self.size_y = self.video.set_format(
            self.image_w, self.image_h, fourcc=self.fourcc
        )

        logger.info(
            "V4L カメラが %d x %d の解像度を受け入れました" % (self.size_x, self.size_y)
        )

        # バッファを作成する。libv4l2 でコンパイルされた場合 'start' 前に必要。
        self.video.create_buffers(30)

        # 一部のデバイスでは 'start' 前にバッファをキューする必要がある。
        self.video.queue_all_buffers()

        # デバイスを開始。LED 付きカメラなら点灯する。
        self.video.start()

    def update(self):
        """連続的にフレームを読み込む。"""
        import select

        from donkeycar.parts.image import JpgToImgArr

        self.init_video()
        jpg_conv = JpgToImgArr()

        while self.running:
            # デバイスがバッファを満たすまで待機
            select.select((self.video,), (), ())
            image_data = self.video.read_and_queue()
            self.frame = jpg_conv.run(image_data)

    def shutdown(self):
        """カメラを停止する。"""
        self.running = False
        time.sleep(0.5)


class MockCamera(BaseCamera):
    """単一の静止画像のみを返すモックカメラ。"""

    def __init__(self, image_w=160, image_h=120, image_d=3, image=None):
        """モックカメラを初期化する。"""
        if image is not None:
            self.frame = image
        else:
            self.frame = np.array(Image.new("RGB", (image_w, image_h)))

    def update(self):
        """何もしない。"""
        pass

    def shutdown(self):
        """何もしない。"""
        pass


class ImageListCamera(BaseCamera):
    """tub 内の画像をカメラ出力として利用するクラス。"""

    def __init__(self, path_mask="~/mycar/data/**/images/*.jpg"):
        """画像ファイル群を読み込み順に返す。

        Args:
            path_mask (str): 画像ファイルを検索するパスのマスク。
        """
        self.image_filenames = glob.glob(os.path.expanduser(path_mask), recursive=True)

        def get_image_index(fnm):
            sl = os.path.basename(fnm).split("_")
            return int(sl[0])

        """更新日時ではなく画像番号でソートした方が適切な場合が多い。"""
        self.image_filenames.sort(key=get_image_index)
        # self.image_filenames.sort(key=os.path.getmtime)
        self.num_images = len(self.image_filenames)
        logger.info("%d 枚の画像を読み込みました" % self.num_images)
        logger.info(self.image_filenames[:10])
        self.i_frame = 0
        self.frame = None
        self.update()

    def update(self):
        """何もしない。"""
        pass

    def run_threaded(self):
        """画像を順番に読み込み NumPy 配列で返す。"""
        if self.num_images > 0:
            self.i_frame = (self.i_frame + 1) % self.num_images
            self.frame = Image.open(self.image_filenames[self.i_frame])

        return np.asarray(self.frame)

    def shutdown(self):
        """何もしない。"""
        pass
