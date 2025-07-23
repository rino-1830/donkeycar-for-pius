"""DonkeyGym を用いた学習環境の部品群."""

import os
import time
import gym
import gym_donkeycar


def is_exe(fpath):
    """実行可能ファイルかどうかを判定する.

    Args:
        fpath (str): ファイルパス

    Returns:
        bool: 実行可能なら ``True`` を返す
    """

    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


class DonkeyGymEnv(object):
    """DonkeyGym 環境とのインターフェース."""

    def __init__(
        self,
        sim_path,
        host="127.0.0.1",
        port=9091,
        headless=0,
        env_name="donkey-generated-track-v0",
        sync="asynchronous",
        conf={},
        record_location=False,
        record_gyroaccel=False,
        record_velocity=False,
        record_lidar=False,
        delay=0,
    ):
        """DonkeyGym 環境を初期化する.

        Args:
            sim_path (str): シミュレータの実行ファイルパスまたは ``"remote"``
            host (str, optional): 接続先ホスト. Defaults to ``"127.0.0.1"``.
            port (int, optional): 接続ポート. Defaults to ``9091``.
            headless (int, optional): ヘッドレスモードの有無. Defaults to ``0``.
            env_name (str, optional): ジム環境名. Defaults to ``"donkey-generated-track-v0"``.
            sync (str, optional): 同期設定. Defaults to ``"asynchronous"``.
            conf (dict, optional): その他設定.
            record_location (bool, optional): 位置情報を記録するかどうか.
            record_gyroaccel (bool, optional): ジャイロ・加速度データを記録するかどうか.
            record_velocity (bool, optional): 速度情報を記録するかどうか.
            record_lidar (bool, optional): LIDAR 情報を記録するかどうか.
            delay (int, optional): 遅延時間(ミリ秒).

        Raises:
            Exception: シミュレータのパスが存在しない場合
            Exception: シミュレータのパスが実行可能でない場合
        """

        if sim_path != "remote":
            if not os.path.exists(sim_path):
                raise Exception("シミュレータのパスが存在しません")

            if not is_exe(sim_path):
                raise Exception("指定されたパスは実行可能ファイルではありません")

        conf["exe_path"] = sim_path
        conf["host"] = host
        conf["port"] = port
        conf["guid"] = 0
        conf["frame_skip"] = 1
        self.env = gym.make(env_name, conf=conf)
        self.frame = self.env.reset()
        self.action = [0.0, 0.0, 0.0]
        self.running = True
        self.info = {'pos': (0., 0., 0.),
                     'speed': 0,
                     'cte': 0,
                     'gyro': (0., 0., 0.),
                     'accel': (0., 0., 0.),
                     'vel': (0., 0., 0.),
                     'lidar': []}
        self.delay = float(delay) / 1000
        self.record_location = record_location
        self.record_gyroaccel = record_gyroaccel
        self.record_velocity = record_velocity
        self.record_lidar = record_lidar

        self.buffer = []

    def delay_buffer(self, frame, info):
        """遅延バッファを操作してフレームを取り出す.

        Args:
            frame (np.ndarray): 画像フレーム
            info (dict): 環境から得られる各種情報
        """

        now = time.time()
        buffer_tuple = (now, frame, info)
        self.buffer.append(buffer_tuple)

        # バッファ内を走査する
        num_to_remove = 0
        for buf in self.buffer:
            if now - buf[0] >= self.delay:
                num_to_remove += 1
                self.frame = buf[1]
            else:
                break

        # 古いエントリーを削除する
        del self.buffer[:num_to_remove]

    def update(self):
        """環境を更新し続けるループ.

        このメソッドは ``run_threaded`` で設定されたアクションを用いて
        Gym 環境からフレームと情報を取得し続ける。
        """

        while self.running:
            if self.delay > 0.0:
                current_frame, _, _, current_info = self.env.step(self.action)
                self.delay_buffer(current_frame, current_info)
            else:
                self.frame, _, _, self.info = self.env.step(self.action)

    def run_threaded(self, steering, throttle, brake=None):
        """コントローラからの入力を処理し、結果を返す.

        Args:
            steering (float): ステアリング角
            throttle (float): スロットル値
            brake (float, optional): ブレーキ値

        Returns:
            Any: 設定に応じた画像と各種情報
        """

        if steering is None or throttle is None:
            steering = 0.0
            throttle = 0.0
        if brake is None:
            brake = 0.0

        self.action = [steering, throttle, brake]

        # 必要に応じてシムの位置情報を出力
        outputs = [self.frame]
        if self.record_location:
            outputs += self.info['pos'][0], self.info['pos'][1], self.info['pos'][2], self.info['speed'], self.info['cte']
        if self.record_gyroaccel:
            outputs += self.info['gyro'][0], self.info['gyro'][1], self.info['gyro'][2], self.info['accel'][0], self.info['accel'][1], self.info['accel'][2]
        if self.record_velocity:
            outputs += self.info['vel'][0], self.info['vel'][1], self.info['vel'][2]
        if self.record_lidar:
            outputs += [self.info['lidar']]
        if len(outputs) == 1:
            return self.frame
        else:
            return outputs

    def shutdown(self):
        """環境を終了する処理を行う."""

        self.running = False
        time.sleep(0.2)
        self.env.close()
