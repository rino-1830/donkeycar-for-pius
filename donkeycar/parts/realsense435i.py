"""
著者: Ed Murphy
ファイル: realsense435i.py
日付: 2019年4月14日
メモ: Intel RealSense 深度カメラ D435/D435i 用の Donkeycar パーツ。
"""
import argparse
import time
import logging
import sys

import numpy as np
import pyrealsense2 as rs

#
# NOTE: Jetson Nano ユーザーは Jetson Hacks プロジェクト
#       https://github.com/JetsonHacksNano/installLibrealsense
#       をクローンし、Python バインディングを得るために
#       librealsense をソースからビルドしてください。
#       ./buildLibrealsense.sh
#

#
# Realsense D435 は任意のサイズの画像を出力できません。
# 使用できるサイズは決められたリストから選ぶ必要があります。
# 本コードではカメラでサポートされている無難な解像度を選択しています。
# もし別のサイズで初期化された場合は、opencv を使って返却前にリサイズします。
#
WIDTH = 424
HEIGHT = 240
CHANNELS = 3

class RealSense435i(object):
    """RealSense D435/D435i 用カメラパーツ。

    IMU と二つの魚眼カメラ、Movidius チップを備えた D435i では、RGB 画像と深度
    マップに加えてオプションで加速度とジャイロデータを取得できます（"i" が付く
    モデルのみ IMU を搭載）。画像は常に 424x240 ピクセル RGB 60fps で取得され、
    異なるサイズが指定された場合は出力時にリサイズされます。

    Args:
        width (int): 画像の幅。
        height (int): 画像の高さ。
        channels (int): チャンネル数。
        enable_rgb (bool): RGB 画像を取得するかどうか。
        enable_depth (bool): 深度画像を取得するかどうか。
        enable_imu (bool): IMU を有効にするかどうか。
        device_id (str | None): 使用するカメラのシリアル番号。
    """

    def __init__(self, width=WIDTH, height=HEIGHT, channels=CHANNELS,
                 enable_rgb=True, enable_depth=True, enable_imu=False,
                 device_id=None):
        """カメラを初期化する。"""

        self.device_id = device_id  # "923322071108" デフォルトを使う場合は None
        self.enable_imu = enable_imu
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth

        self.width = width
        self.height = height
        self.channels = channels
        self.resize = (width != WIDTH) or (height != height) or (channels != CHANNELS)
        if self.resize:
            print(
                "出力画像は {} から {} にリサイズされます。opencv が必要です".format(
                    (WIDTH, HEIGHT, CHANNELS),
                    (self.width, self.height, self.channels),
                )
            )

        # ストリームの設定
        self.imu_pipeline = None
        if self.enable_imu:
            self.imu_pipeline = rs.pipeline()
            imu_config = rs.config()
            if None != self.device_id:
                imu_config.enable_device(self.device_id)
            imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)  # acceleration
            imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope
            imu_profile = self.imu_pipeline.start(imu_config)
            # IMU が安定するまで数フレーム捨てる
            for i in range(0, 5):
                self.imu_pipeline.wait_for_frames()

        self.pipeline = None
        if self.enable_depth or self.enable_rgb:
            self.pipeline = rs.pipeline()
            config = rs.config()

            # 特定のデバイスが指定されていればそれを有効にする
            if None != self.device_id:
                config.enable_device(self.device_id)

            if self.enable_depth:
                config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)  # depth

            if self.enable_rgb:
                config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 60)  # rgb

            # ストリーミング開始
            profile = self.pipeline.start(config)

            # 深度センサーのスケールを取得（rs-align の例を参照）
            if self.enable_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                print("Depth Scale は", depth_scale)
                if self.enable_rgb:
                    # align オブジェクトを作成
                    # rs.align を使用すると、深度フレームを他のフレームに合わせられる
                    # "align_to" は深度フレームを合わせる対象ストリームタイプ
                    align_to = rs.stream.color
                    self.align = rs.align(align_to)

            # 露出制御が落ち着くまで数フレーム捨てる
            for i in range(0, 5):
                self.pipeline.wait_for_frames()

        time.sleep(2)   # カメラが温まるまで待つ

        # フレームの初期状態を設定
        self.color_image = None
        self.depth_image = None
        self.acceleration_x = None
        self.acceleration_y = None
        self.acceleration_z = None
        self.gyroscope_x = None
        self.gyroscope_y = None
        self.gyroscope_z = None
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_time = self.start_time

        self.running = True

    def _stop_pipeline(self):
        """内部パイプラインを停止する。"""
        if self.imu_pipeline is not None:
            self.imu_pipeline.stop()
            self.imu_pipeline = None
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def _poll(self):
        """カメラと IMU から最新のデータを取得する。"""
        last_time = self.frame_time
        self.frame_time = time.time() - self.start_time
        self.frame_count += 1

        #
        # フレームの取得
        #
        try:
            if self.enable_imu:
                imu_frames = self.imu_pipeline.wait_for_frames()

            if self.enable_rgb or self.enable_depth:
                frames = self.pipeline.wait_for_frames()
        except Exception as e:
            logging.error(e)
            return

        #
        # カメラフレームを画像に変換
        #
        if self.enable_rgb or self.enable_depth:
            # 深度フレームをカラーに整列
            aligned_frames = self.align.process(frames) if self.enable_depth and self.enable_rgb else None
            depth_frame = aligned_frames.get_depth_frame() if aligned_frames is not None else frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame() if aligned_frames is not None else frames.get_color_frame()

            # 深度は16bit配列に、RGBは8bit平面配列に変換
            self.depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16) if self.enable_depth else None
            self.color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8) if self.enable_rgb else None

            if self.resize:
                import cv2
                if self.width != WIDTH or self.height != HEIGHT:
                    self.color_image = cv2.resize(self.color_image, (self.width, self.height), cv2.INTER_NEAREST) if self.enable_rgb else None
                    self.depth_image = cv2.resize(self.depth_image, (self.width, self.height), cv2.INTER_NEAREST) if self.enable_depth else None
                if self.channels != CHANNELS:
                    self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2GRAY) if self.enable_rgb else None

        #
        # IMU データを Mpu6050 コードと同じ順序の個別値として出力する。
        # これにより他の IMU 消費パーツとの互換性を保つ。
        #
        if self.enable_imu:
            acceleration = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f).as_motion_frame().get_motion_data()
            self.acceleration_x = acceleration.x
            self.acceleration_y = acceleration.y
            self.acceleration_z = acceleration.z
            gyroscope = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f).as_motion_frame().get_motion_data()
            self.gyroscope_x = gyroscope.x
            self.gyroscope_y = gyroscope.y
            self.gyroscope_z = gyroscope.z
            logging.debug(
                "IMU フレーム {} を取得 ({:.2f} 秒):\n\taccel = {}\n\tgyro = {}".format(
                    self.frame_count,
                    self.frame_time - last_time,
                    (self.acceleration_x, self.acceleration_y, self.acceleration_z),
                    (self.gyroscope_x, self.gyroscope_y, self.gyroscope_z),
                )
            )

    def update(self):
        """スレッド実行時にバックグラウンドで状態を更新する。"""
        while self.running:
            self._poll()

    def run_threaded(self):
        """最新の状態を返す。ブロックしない。

        Returns:
            Tuple[numpy.ndarray | None, numpy.ndarray | None, float | None,
            float | None, float | None, float | None, float | None, float | None]:
            RGB 画像、深度画像、加速度 (x, y, z)、ジャイロ (x, y, z)。
            無効な機能は ``None`` が返る。ジャイロの x はピッチ、y はヨー、z はロール。
        """
        return self.color_image, self.depth_image, self.acceleration_x, self.acceleration_y, self.acceleration_z, self.gyroscope_x, self.gyroscope_y, self.gyroscope_z

    def run(self):
        """カメラからフレームを取得して返す。ブロック処理。

        Returns:
            Tuple[numpy.ndarray | None, numpy.ndarray | None, float | None,
            float | None, float | None, float | None, float | None, float | None]:
            ``run_threaded`` と同じ内容。
        """
        self._poll()
        return self.run_threaded()

    def shutdown(self):
        """カメラを停止して後処理を行う。"""
        self.running = False
        time.sleep(2) # スレッド終了の猶予

        # 処理終了
        self._stop_pipeline()


#
# 自己テスト
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb", default=False, action='store_true', help="RGB カメラをストリーム")
    parser.add_argument("--depth", default=False, action='store_true', help="深度カメラをストリーム")
    parser.add_argument("--imu", default=False, action='store_true', help="IMU をコンソール出力")
    parser.add_argument("--device_id", help="複数台接続時のカメラ ID")
    args = parser.parse_args()

    if not (args.rgb or args.depth or args.imu):
        print("--rgb、--depth、--imu のいずれかを指定してください")
        parser.print_help()
        sys.exit(0)


    show_opencv_window = args.rgb or args.depth  # 画像をウィンドウ表示する場合は True。デフォルト環境では非対応。
    if show_opencv_window:
        import cv2

    enable_rgb = args.rgb
    enable_depth = args.depth
    enable_imu = args.imu
    device_id = args.device_id

    width = 212
    height = 120
    channels = 3

    profile_frames = 0  # 非ゼロにすると指定フレーム数で最大フレームレートを計測

    try:
        #
        # D435i では enable_imu を True にできるが、D435 では False 推奨
        #
        camera = RealSense435i(
            width=width, height=height, channels=channels,
            enable_rgb=enable_rgb, enable_depth=enable_depth, enable_imu=enable_imu, device_id=device_id)

        frame_count = 0
        start_time = time.time()
        frame_time = start_time
        while True:
            #
            # カメラからデータを読み込む
            #
            color_image, depth_image, acceleration_x, acceleration_y, acceleration_z, gyroscope_x, gyroscope_y, gyroscope_z = camera.run()

            # フレーム時間を計測
            frame_count += 1
            last_time = frame_time
            frame_time = time.time()

            if enable_imu and not profile_frames:
                print(
                    "IMU フレーム {} ({:.2f} 秒)\n\taccel = {}\n\tgyro = {}".format(
                        frame_count,
                        frame_time - last_time,
                        (acceleration_x, acceleration_y, acceleration_z),
                        (gyroscope_x, gyroscope_y, gyroscope_z),
                    )
                )

            # 画像を表示
            if show_opencv_window and not profile_frames:
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                if enable_rgb or enable_depth:
                    # ウィンドウ表示のため色と深度のチャンネル数をそろえる
                    if 3 == channels:
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) if enable_depth else None
                    else:
                        depth_colormap = cv2.cvtColor(cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET), cv2.COLOR_RGB2GRAY) if enable_depth else None


                    # 2 つの画像を横に並べる
                    images = None
                    if enable_rgb:
                        images = np.hstack((color_image, depth_colormap)) if enable_depth else color_image
                    elif enable_depth:
                        images = depth_colormap

                    if images is not None:
                        cv2.imshow('RealSense', images)

                # ESC もしくは 'q' でウィンドウを閉じる
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
            if profile_frames > 0:
                if frame_count == profile_frames:
                    print(
                        "{} フレームを {:.2f} 秒で取得、{} fps".format(
                            frame_count,
                            frame_time - start_time,
                            frame_count / (frame_time - start_time),
                        )
                    )
                    break
            else:
                time.sleep(0.05)
    finally:
        camera.shutdown()
