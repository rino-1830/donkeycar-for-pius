import cv2
from donkeycar.parts.camera import BaseCamera
from donkeycar.parts.fast_stretch import fast_stretch
import time

"""Leopard Imaging カメラ部品."""


class LICamera(BaseCamera):
    """Fast-Stretch を組み込んだ Leopard Imaging カメラ."""
    def __init__(self, width=224, height=224, capture_width=1280, capture_height=720, fps=60):
        """カメラを初期化する。

        Args:
            width: 取得後の画像幅。
            height: 取得後の画像高さ。
            capture_width: センサーから取得する画像の幅。
            capture_height: センサーから取得する画像の高さ。
            fps: フレームレート(Frames Per Second)。
        """
        super(LICamera, self).__init__()
        self.width = width
        self.height = height
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.fps = fps
        self.camera_id = LICamera.camera_id(self.capture_width, self.capture_height, self.width, self.height, self.fps)
        self.frame = None
        print('Leopard Imaging カメラへ接続します')
        self.capture = cv2.VideoCapture(self.camera_id)
        time.sleep(2)
        if self.capture.isOpened():
            print('Leopard Imaging カメラに接続しました。')
            self.on = True
        else:
            self.on = False
            print('接続できません。カメラパラメータは正しいですか?')

    def read_frame(self):
        """カメラからフレームを取得する。"""
        success, frame = self.capture.read()
        if success:
            # RGB フレームを返す
            frame = fast_stretch(frame)
            self.frame = frame

    def run(self):
        """最新フレームを取得して返す。"""
        self.read_frame()
        return self.frame

    def update(self):
        """スレッドが停止されるまでフレームを読み続ける。"""
        # スレッド終了フラグが立つまで無限ループ
        while self.on:
            self.read_frame()

    def shutdown(self):
        """カメラを解放して停止する。"""
        # スレッドを停止させる指示
        self.on = False
        print('Leopard Imaging カメラを停止します')
        self.capture.release()
        time.sleep(.5)

    @classmethod
    def camera_id(cls, capture_width, capture_height, width, height, fps):
        """GStreamer パイプラインの文字列を生成する。

        Args:
            capture_width: センサーから取得する画像の幅。
            capture_height: センサーから取得する画像の高さ。
            width: 出力画像の幅。
            height: 出力画像の高さ。
            fps: フレームレート。

        Returns:
            Leopard Imaging カメラ用の GStreamer パイプライン文字列。
        """
        return (
            'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx '
            '! videoconvert ! appsink'
        ) % (capture_width, capture_height, fps, width, height)