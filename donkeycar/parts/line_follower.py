import cv2
import numpy as np
from simple_pid import PID
import logging

logger = logging.getLogger(__name__)


class LineFollower:
    """OpenCV を用いたラインフォロワーコントローラ。

    画像の指定された Y 座標で水平に切り出し、HSV 変換後に黄色を検出する。ヒストグラム
    から黄色が最も多い画素を求め、その画素位置を PID コントローラで維持する。
    """
    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.scan_y = cfg.SCAN_Y   # 画像の上端から水平走査を開始するピクセル数
        self.scan_height = cfg.SCAN_HEIGHT  # 水平走査で取得する画像の高さ
        self.color_thr_low = np.asarray(cfg.COLOR_THRESHOLD_LOW)  # HSV での黄色下限
        self.color_thr_hi = np.asarray(cfg.COLOR_THRESHOLD_HIGH)  # HSV での黄色上限
        self.target_pixel = cfg.TARGET_PIXEL  # 理想となる位置を表すインデックス
        self.target_threshold = cfg.TARGET_THRESHOLD # target_pixel から外れたと判断する最小距離
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD  # target_pixel スライスに必要な黄色ピクセルの割合
        self.steering = 0.0 # -1 から 1 の範囲の操舵値
        self.throttle = cfg.THROTTLE_INITIAL # -1 から 1 の範囲のスロットル値
        self.delta_th = cfg.THROTTLE_STEP  # スロットルを変化させる量
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        self.pid_st = pid


    def get_i_color(self, cam_img):
        """指定した水平ラインでの色インデックスを取得する。

        Args:
            cam_img: RGB の NumPy 配列。

        Returns:
            Tuple[int, float, np.ndarray]:
            最大色のインデックス、そのインデックスの総色量、検出したマスク。
        """
        # 画像から水平スライスを取得
        iSlice = self.scan_y
        scan_line = cam_img[iSlice : iSlice + self.scan_height, :, :]

        # HSV 色空間へ変換
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)

        # 対象色のマスクを作成
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        # 黄色が最も多いインデックスを取得
        hist = np.sum(mask, axis=0)
        max_yellow = np.argmax(hist)

        return max_yellow, hist[max_yellow], mask


    def run(self, cam_img):
        """CV コントローラのメインループ。

        Args:
            cam_img: RGB の NumPy 配列。

        Returns:
            Tuple[float, float, np.ndarray]:
            ステアリング値、スロットル値、出力画像。 ``overlay_image`` が ``True``
            の場合は処理内容を示すオーバーレイを付加し、 ``False`` なら画像をそのまま返す。
        """
        if cam_img is None:
            return 0, 0, False, None

        max_yellow, confidence, mask = self.get_i_color(cam_img)
        conf_thresh = 0.001

        if self.target_pixel is None:
            # 最初の get_i_color 実行時に黄色ラインとの位置関係を決定する
            # 任意に target_pixel を初期化しておくこともできる
            self.target_pixel = max_yellow
            logger.info(f"ライン位置を自動設定しました: {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            # 操舵 PID コントローラの目標値を設定
            self.pid_st.setpoint = self.target_pixel

        if confidence >= self.confidence_threshold:
            # 現在の黄色ライン位置で制御を行う
            # 理想値へ近づくよう新しい操舵値を得る
            self.steering = self.pid_st(max_yellow)

            # 理想から離れるほど減速し、近づけば加速する
            if abs(max_yellow - self.target_pixel) > self.target_threshold:
                # 曲がるので減速
                if self.throttle > self.throttle_min:
                    self.throttle -= self.delta_th
                if self.throttle < self.throttle_min:
                    self.throttle = self.throttle_min
            else:
                # 直進時は加速
                if self.throttle < self.throttle_max:
                    self.throttle += self.delta_th
                if self.throttle > self.throttle_max:
                    self.throttle = self.throttle_max
        else:
            logger.info(
                f"ラインを検出できません: 信頼度 {confidence} < {self.confidence_threshold}"
            )

        # 診断情報を表示
        if self.overlay_image:
            cam_img = self.overlay_display(cam_img, mask, max_yellow, confidence)

        return self.steering, self.throttle, cam_img

    def overlay_display(self, cam_img, mask, max_yellow, confidense):
        """マスクを元画像に合成し、制御値を表示する。

        Args:
            cam_img: RGB の NumPy 配列。
            mask: 検出したマスク。
            max_yellow: 黄色が最も多いインデックス。
            confidense: 信頼度。

        Returns:
            np.ndarray: オーバーレイ後の画像。
        """

        mask_exp = np.stack((mask,) * 3, axis=-1)
        iSlice = self.scan_y
        img = np.copy(cam_img)
        img[iSlice : iSlice + self.scan_height, :, :] = mask_exp
        # 画像を BGR に変換する場合の例

        display_str = []
        display_str.append("操舵:{:.1f}".format(self.steering))
        display_str.append("スロットル:{:.2f}".format(self.throttle))
        display_str.append("黄位置:{:d}".format(max_yellow))
        display_str.append("信頼度:{:.2f}".format(confidense))

        y = 10
        x = 10

        for s in display_str:
            cv2.putText(img, s, color=(0, 0, 0), org=(x ,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
            y += 10

        return img

