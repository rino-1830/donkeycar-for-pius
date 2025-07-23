"""Intel Realsense 2 カメラからデータを取り込むためのパーツ群。

Author:
    Tawn Kramer

File:
    realsense2.py

Date:
    April 14 2019
"""
import time
import logging

import numpy as np
import pyrealsense2 as rs


class RS_T265(object):
    """Intel Realsense T265 カメラを利用して姿勢推定を行うパーツ。

    Attributes:
        image_output (bool): 魚眼カメラ画像を返すかどうか。
        enc_vel_ms (float): エンコーダから取得する速度[m/s]。
        wheel_odometer: Realsense のホイールオドメトリセンサー。
    """


    def __init__(self, image_output=False, calib_filename=None):
        """T265 カメラを初期化する。

        Args:
            image_output (bool): 魚眼カメラの画像も取得するかどうか。
            calib_filename (str | None): ホイールオドメトリのキャリブレーションファイル。
        """
        # image_output を有効にすると 2 つの魚眼カメラストリームを取得するが、戻り値は 1 つのみ。
        # USB2 では負荷が高い場合があり、USB3 接続が推奨されている。
        self.image_output = image_output
        
        # エンコーダを使用する場合、ここに直近の速度を保持する。
        self.enc_vel_ms = 0.0
        self.wheel_odometer = None

        # 実デバイスとセンサーをカプセル化した RealSense パイプラインを生成
        print("T265 を起動中")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        profile = cfg.resolve(self.pipe)
        dev = profile.get_device()
        tm2 = dev.as_tm2()
        

        if self.image_output:
            # 現在は両方のストリームを有効にする必要がある
            cfg.enable_stream(rs.stream.fisheye, 1)  # 左カメラ
            cfg.enable_stream(rs.stream.fisheye, 2)  # 右カメラ

        if calib_filename is not None:
            pose_sensor = tm2.first_pose_sensor()
            self.wheel_odometer = pose_sensor.as_wheel_odometer()

            # キャリブレーションファイルを uint8 のリストへ変換
            f = open(calib_filename)
            chars = []
            for line in f:
                for c in line:
                    chars.append(ord(c))  # 文字を uint8 に変換

            # ホイールオドメトリをロードして設定
            print("ホイール設定を読み込み中", calib_filename)
            self.wheel_odometer.load_wheel_odometery_config(chars)


        # 設定した内容でストリーミングを開始
        self.pipe.start(cfg)
        self.running = True
        print("T265 はトラッキングデータを出力する前に数秒のウォームアップが必要です")
        
        zero_vec = (0.0, 0.0, 0.0)
        self.pos = zero_vec
        self.vel = zero_vec
        self.acc = zero_vec
        self.img = None

    def poll(self):
        """Realsense から最新データを取得する。"""

        if self.wheel_odometer:
            wo_sensor_id = 0  # キャリブレーションファイルと同じ順序で 0 から始まる
            frame_num = 0  # 使用しない
            v = rs.vector()
            v.x = -1.0 * self.enc_vel_ms  # m/s
            #v.z = -1.0 * self.enc_vel_ms  # m/s
            self.wheel_odometer.send_wheel_odometry(wo_sensor_id, frame_num, v)

        try:
            frames = self.pipe.wait_for_frames()
        except Exception as e:
            logging.error(e)
            return

        if self.image_output:
            # ひとまず片側の画像のみ取得する
            # 左側魚眼カメラのフレーム
            left = frames.get_fisheye_frame(1)
            self.img = np.asanyarray(left.get_data())


        # 姿勢フレームを取得
        pose = frames.get_pose_frame()

        if pose:
            data = pose.get_pose_data()
            self.pos = data.translation
            self.vel = data.velocity
            self.acc = data.acceleration
            logging.debug('realsense 座標(%f, %f, %f)' % (self.pos.x, self.pos.y, self.pos.z))


    def update(self):
        """ループして常に最新データを取得する。"""
        while self.running:
            self.poll()

    def run_threaded(self, enc_vel_ms):
        """スレッドモードでの呼び出し時に値を返す。"""
        self.enc_vel_ms = enc_vel_ms
        return self.pos, self.vel, self.acc, self.img

    def run(self, enc_vel_ms):
        """単発実行モードでデータを取得する。"""
        self.enc_vel_ms = enc_vel_ms
        self.poll()
        return self.run_threaded()

    def shutdown(self):
        """センサーを停止する。"""
        self.running = False
        time.sleep(0.1)
        self.pipe.stop()



if __name__ == "__main__":
    c = RS_T265()
    while True:
        pos, vel, acc = c.run()
        print(pos)
        time.sleep(0.1)
    c.shutdown()
