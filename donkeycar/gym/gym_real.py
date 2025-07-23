"""gym_real モジュール.

作者: Tawn Kramer
日付: 2019-01-24
説明: ジムインターフェース経由で実機 Donkey ロボットを制御する
"""
import os
import time

import gym
import numpy as np
from gym import error, spaces, utils

from .remote_controller import DonkeyRemoteContoller


class DonkeyRealEnv(gym.Env):
    """実機 Donkey 用の OpenAI Gym 環境。"""

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION_NAMES = ["steer", "throttle"]
    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0
    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 5.0
    VAL_PER_PIXEL = 255

    def __init__(self, time_step=0.05, frame_skip=2):
        """コンストラクタ."""

        print("DonkeyGym 環境を開始します")
        
        try:
            donkey_name = str(os.environ['DONKEY_NAME'])
        except:
            donkey_name = 'my_robot1234'
            print("DONKEY_NAME 環境変数が見つかりません。既定値を使用します:", donkey_name)

        try:
            mqtt_broker = str(os.environ['DONKEY_MQTT_BROKER'])
        except:
            mqtt_broker = "iot.eclipse.org"
            print("DONKEY_MQTT_BROKER 環境変数が見つかりません。既定値を使用します:", mqtt_broker)
            
        # コントローラーを起動する
        self.controller = DonkeyRemoteContoller(donkey_name=donkey_name, mqtt_broker=mqtt_broker)
        
        # ステアリングとスロットル
        self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
            high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]), dtype=np.float32 )

        # カメラセンサーのデータ
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.controller.get_sensor_size(), dtype=np.uint8)

        # フレームスキップ
        self.frame_skip = frame_skip

        # 接続完了まで待機する
        self.controller.wait_until_connected()
        

    def close(self):
        """コントローラーを終了する。"""

        self.controller.quit()

    def step(self, action):
        """一ステップ分のアクションを実行する。"""
        for i in range(self.frame_skip):
            self.controller.take_action(action)
            time.sleep(0.05)
            observation = self.controller.observe()
            reward, done, info = 0.1, False, None
        return observation, reward, done, info

    def reset(self):
        """環境をリセットする。"""
        observation = self.controller.observe()
        reward, done, info = 0.1, False, None
        return observation

    def render(self, mode="human", close=False):
        """現在の観測画像を取得する。"""
        if close:
            self.controller.quit()

        return self.controller.observe()

    def is_game_over(self):
        """常に ``False`` を返す。"""

        return False
