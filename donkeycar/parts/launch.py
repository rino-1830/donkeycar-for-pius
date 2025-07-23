"""AIで自動運転を開始するときに強いスロットルを一時的に与えるパーツ."""

import time

class AiLaunch():
    """初回起動時に大きなスロットルを適用するパーツ。

    レースで素早くスタートするためのもので、十分な速度に達するとすぐにAIが制御を引き継ぎます。
    """

    def __init__(self, launch_duration=1.0, launch_throttle=1.0, keep_enabled=False):
        """オブジェクトを初期化する。

        Args:
            launch_duration: スロットルを維持する秒数。
            launch_throttle: 適用するスロットル値。
            keep_enabled: モード切替時に自動で起動を有効にするかどうか。
        """
        self.active = False
        self.enabled = False
        self.timer_start = None
        self.timer_duration = launch_duration
        self.launch_throttle = launch_throttle
        self.prev_mode = None
        self.trigger_on_switch = keep_enabled
        
    def enable_ai_launch(self):
        """AIローンチを有効にする。"""
        self.enabled = True
        print('AiLauncherが有効になりました。')

    def run(self, mode, ai_throttle):
        """スロットルを更新する。

        Args:
            mode: 現在のドライブモード。
            ai_throttle: AIが算出したスロットル値。

        Returns:
            float: 実際に使用するスロットル値。
        """

        new_throttle = ai_throttle

        if mode != self.prev_mode:
            self.prev_mode = mode
            if mode == "local" and self.trigger_on_switch:
                self.enabled = True

        if mode == "local" and self.enabled:
            if not self.active:
                self.active = True
                self.timer_start = time.time()
            else:
                duration = time.time() - self.timer_start
                if duration > self.timer_duration:
                    self.active = False
                    self.enabled = False
        else:
            self.active = False

        if self.active:
            print('AiLauncherがアクティブです!!!')
            new_throttle = self.launch_throttle

        return new_throttle

