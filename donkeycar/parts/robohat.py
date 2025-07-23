#!/usr/bin/env python3
"""
Robotics Masters 製 RoboHAT MM1 を Donkeycar で操作するためのスクリプト。

author: @wallarug (Cian Byrne) 2019
contrib: @peterpanstechland 2019
contrib: @sctse999 2020

注: このリポジトリに同梱されている code.py と併せて使用すること。
    ``donkeycar/contrib/robohat/code.py`` を参照。
"""

import time
import logging
import donkeycar as dk

try:
    import serial
except ImportError:
    print("PySerial が見つかりません。 'pip install pyserial' を実行してください")

logger = logging.getLogger(__name__)

class RoboHATController:
    """SAMD51 からの信号を読み取りステアリングとスロットルへ変換するドライバ。

    入力信号の範囲は ``1000`` ～ ``2000``。
    出力範囲は ``-1.00`` ～ ``1.00``。
    """

    def __init__(self, cfg, debug=False):
        # 基本となる変数
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.recording_latch = None
        self.auto_record_on_throttle = cfg.AUTO_RECORD_ON_THROTTLE
        self.STEERING_MID = cfg.MM1_STEERING_MID
        self.MAX_FORWARD = cfg.MM1_MAX_FORWARD
        self.STOPPED_PWM = cfg.MM1_STOPPED_PWM
        self.MAX_REVERSE = cfg.MM1_MAX_REVERSE
        self.SHOW_STEERING_VALUE = cfg.MM1_SHOW_STEERING_VALUE
        self.DEAD_ZONE = cfg.JOYSTICK_DEADZONE
        self.debug = debug

        try:
            self.serial = serial.Serial(cfg.MM1_SERIAL_PORT, 115200, timeout=1)
        except serial.SerialException:
            print("シリアルポートが見つかりません。 'sudo raspi-config' で有効化してください")
        except serial.SerialTimeoutException:
            print("シリアル接続がタイムアウトしました")

    def shutdown(self):
        try:
            self.serial.close()
        except:
            pass

    def read_serial(self):
        """シリアルポートから RC コントローラの値を読み取り、ステアリングとスロットルへ変換する。

        フォーマットは ``####,####`` で、最初の数値がステアリング、2 番目がスロットル。
        """
        line = str(self.serial.readline().decode()).strip('\n').strip('\r')

        output = line.split(", ")
        if len(output) == 2:
            if self.SHOW_STEERING_VALUE:
                print("MM1: steering={}".format(output[0]))

            if output[0].isnumeric() and output[1].isnumeric():
                angle_pwm = float(output[0])
                throttle_pwm = float(output[1])

                if self.debug:
                    print("angle_pwm = {}, throttle_pwm= {}".format(angle_pwm, throttle_pwm))


                if throttle_pwm >= self.STOPPED_PWM:
                    # 入力 PWM (1500 - 2000) を最大前進値に合わせてスケーリング
                    throttle_pwm = dk.utils.map_range_float(throttle_pwm,
                                                                1500, 2000,
                                                                self.STOPPED_PWM,
                                                                self.MAX_FORWARD )
                    # print("throttle_pwm を {} から {} に再マッピング".format(output[1], throttle_pwm))

                    # 前進する
                    self.throttle = dk.utils.map_range_float(throttle_pwm,
                                                             self.STOPPED_PWM,
                                                             self.MAX_FORWARD,
                                                             0, 1.0)
                else:
                    throttle_pwm = dk.utils.map_range_float(throttle_pwm,
                                                                1000, 1500,
                                                                self.MAX_REVERSE,
                                                                self.STOPPED_PWM)


                    # 後退する
                    self.throttle = dk.utils.map_range_float(throttle_pwm,
                                                             self.MAX_REVERSE,
                                                             self.STOPPED_PWM,
                                                             -1.0, 0)

                if angle_pwm >= self.STEERING_MID:
                    # 左折
                    self.angle = dk.utils.map_range_float(angle_pwm,
                                                          2000, self.STEERING_MID,
                                                          -1, 0)
                else:
                    # 右折
                    self.angle = dk.utils.map_range_float(angle_pwm,
                                                          self.STEERING_MID, 1000,
                                                          0, 1)

                if self.debug:
                    print("angle = {}, throttle = {}".format(self.angle, self.throttle))

                if self.auto_record_on_throttle:
                    was_recording = self.recording
                    self.recording = self.throttle > self.DEAD_ZONE
                    if was_recording != self.recording:
                        self.recording_latch = self.recording
                        logger.debug(f"JoystickController::on_throttle_changes() 記録状態を {self.recording} に設定")

                time.sleep(0.01)

    def update(self):
        # 起動直後のクラッシュを防ぐため少し待つ
        print("シリアルポートをウォームアップ中...")
        time.sleep(3)

        while True:
            try:
                self.read_serial()
            except:
                print("MM1: シリアル入力の読み取りエラー")
                break

    def run(self, img_arr=None, mode=None, recording=None):
        """スレッド実行用メソッドを呼び出す。

        Args:
            img_arr: 現在のカメラ画像または ``None``。
            mode: デフォルトの動作モード。
            recording: デフォルトの録画モード。
        """
        return self.run_threaded(img_arr, mode, recording)

    def run_threaded(self, img_arr=None, mode=None, recording=None):
        """RC 入力を処理して現在の状態を返す。

        Args:
            img_arr: 現在のカメラ画像。
            mode: デフォルトの動作モード。
            recording: デフォルトの録画モード。

        Returns:
            現在のステアリング値、スロットル値、モード、録画フラグ。
        """
        self.img_arr = img_arr

        #
        # 引数が ``None`` でなければ既定値を上書きする
        #
        #
        # 引数が ``None`` でなければ既定値を上書きする
        #
        if mode is not None:
            self.mode = mode
        if recording is not None and recording != self.recording:
            logger.debug(f"RoboHATController::run_threaded() デフォルト値から記録状態を {recording} に設定")
            self.recording = recording
        if self.recording_latch is not None:
            logger.debug(f"RoboHATController::run_threaded() ラッチ値から記録状態を {self.recording_latch} に設定")
            self.recording = self.recording_latch
            self.recording_latch = None

        return self.angle, self.throttle, self.mode, self.recording


class RoboHATDriver:
    """Robo HAT MM1 ボードを利用した PWM モーターコントローラ。

    Robotics Masters によって開発された。
    """

    def __init__(self, cfg, debug=False):
        # シリアルポートを使って Robo HAT を初期化
        self.pwm = serial.Serial(cfg.MM1_SERIAL_PORT, 115200, timeout=1)
        self.MAX_FORWARD = cfg.MM1_MAX_FORWARD
        self.MAX_REVERSE = cfg.MM1_MAX_REVERSE
        self.STOPPED_PWM = cfg.MM1_STOPPED_PWM
        self.STEERING_MID = cfg.MM1_STEERING_MID
        self.debug = debug

    """ステアリングとスロットルは -1.0～1.0 の範囲でなければならない。
    この関数は 1.0 を超える値や -1.0 未満の値を補正する。
    """

    def trim_out_of_bound_value(self, value):
        if value > 1:
            print("MM1: 警告 値が範囲外です: {}".format(value))
            return 1.0
        elif value < -1:
            print("MM1: 警告 値が範囲外です: {}".format(value))
            return -1.0
        else:
            return value

    def set_pulse(self, steering, throttle):
        try:
            steering = self.trim_out_of_bound_value(steering)
            throttle = self.trim_out_of_bound_value(throttle)

            if throttle > 0:
                output_throttle = dk.utils.map_range(throttle,
                                                     0, 1.0,
                                                     self.STOPPED_PWM, self.MAX_FORWARD)
            else:
                output_throttle = dk.utils.map_range(throttle,
                                                     -1, 0,
                                                     self.MAX_REVERSE, self.STOPPED_PWM)

            if steering > 0:
                output_steering = dk.utils.map_range(steering,
                                                     0, 1.0,
                                                     self.STEERING_MID, 1000)
            else:
                output_steering = dk.utils.map_range(steering,
                                                     -1, 0,
                                                     2000, self.STEERING_MID)

            if self.is_valid_pwm_value(output_steering) and self.is_valid_pwm_value(output_throttle):
                if self.debug:
                    print("output_steering=%d, output_throttle=%d" % (output_steering, output_throttle))
                self.write_pwm(output_steering, output_throttle)
            else:
                print(f"警告: steering = {output_steering}, STEERING_MID = {self.STEERING_MID}")
                print(f"警告: throttle = {output_throttle}, MAX_FORWARD = {self.MAX_FORWARD}, STOPPED_PWM = {self.STOPPED_PWM}, MAX_REVERSE = {self.MAX_REVERSE}")
                print("PWM 値を MM1 に送信しません")

        except OSError as err:
            print(
                "PWM 設定中に予期しない問題が発生しました (モーターボードへの配線を確認してください): {0}".format(err))

    def is_valid_pwm_value(self, value):
        if 1000 <= value <= 2000:
            return True
        else:
            return False

    def write_pwm(self, steering, throttle):
        self.pwm.write(b"%d, %d\r" % (steering, throttle))

    def run(self, steering, throttle):
        self.set_pulse(steering, throttle)

    def shutdown(self):
        try:
            self.serial.close()
        except:
            pass
