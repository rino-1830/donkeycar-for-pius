import os
import array
import time
import struct
import random
from threading import Thread
import logging

from prettytable import PrettyTable

# 書きやすくするためのインポート
from donkeycar.parts.web_controller.web import LocalWebController
from donkeycar.parts.web_controller.web import WebFpv

logger = logging.getLogger(__name__)

class Joystick(object):
    """物理ジョイスティックへのインターフェース。

    利用可能なボタンと軸の名前および値を保持し、状態変化を取得できる。
    """
    def __init__(self, dev_fn='/dev/input/js0'):
        self.axis_states = {}
        self.button_states = {}
        self.axis_names = {}
        self.button_names = {}
        self.axis_map = []
        self.button_map = []
        self.jsdev = None
        self.dev_fn = dev_fn


    def init(self):
        """Linux デバイスツリー上のパスから利用可能なボタンと軸を取得する。"""
        try:
            from fcntl import ioctl
        except ModuleNotFoundError:
            self.num_axes = 0
            self.num_buttons = 0
            logger.warn("fnctl モジュールがサポートされていないため、ジョイスティックは使用できません。")
            return False

        if not os.path.exists(self.dev_fn):
            logger.warn(f"{self.dev_fn} が見つかりません")
            return False

        # デバイスとの接続を確立してボタンをマッピングする初期化処理
        # ジョイスティックデバイスを開く
        logger.info(f'{self.dev_fn} を開いています...')
        self.jsdev = open(self.dev_fn, 'rb')

        # デバイス名を取得
        buf = array.array('B', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf) # JSIOCGNAME(len)
        self.js_name = buf.tobytes().decode('utf-8')
        logger.info('デバイス名: %s' % self.js_name)

        # 軸とボタンの数を取得
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf) # JSIOCGAXES
        self.num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf) # JSIOCGBUTTONS
        self.num_buttons = buf[0]

        # 軸マップを取得
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf) # JSIOCGAXMAP

        for axis in buf[:self.num_axes]:
            axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # ボタンマップを取得
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf) # JSIOCGBTNMAP

        for btn in buf[:self.num_buttons]:
            btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0
            #print('btn', '0x%03x' % btn, 'name', btn_name)

        return True


    def show_map(self):
        """このジョイスティックで検出されたボタンと軸を一覧表示する。"""
        print ('%d axes found: %s' % (self.num_axes, ', '.join(self.axis_map)))
        print ('%d buttons found: %s' % (self.num_buttons, ', '.join(self.button_map)))


    def poll(self):
        """ジョイスティックの状態を取得する。

        戻り値は押されたボタンと軸の名前、状態、軸の値を含む。
        ボタン状態は変更なしの場合 ``None``、押下 ``1``、離す ``0`` を返す。
        軸値は ``-1`` から ``+1`` の範囲の浮動小数点数。
        """
        button = None
        button_state = None
        axis = None
        axis_val = None

        if self.jsdev is None:
            return button, button_state, axis, axis_val

        # メインイベントループ
        evbuf = self.jsdev.read(8)

        if evbuf:
            tval, value, typev, number = struct.unpack('IhBB', evbuf)

            if typev & 0x80:
                # 初期化イベントを無視
                return button, button_state, axis, axis_val

            if typev & 0x01:
                button = self.button_map[number]
                #print(tval, value, typev, number, button, 'pressed')
                if button:
                    self.button_states[button] = value
                    button_state = value
                    logger.info("ボタン: %s 状態: %d" % (button, value))

            if typev & 0x02:
                axis = self.axis_map[number]
                if axis:
                    fvalue = value / 32767.0
                    self.axis_states[axis] = fvalue
                    axis_val = fvalue
                    logger.debug("axis: %s val: %f" % (axis, fvalue))

        return button, button_state, axis, axis_val


class PyGameJoystick(object):
    """pygame を利用したジョイスティック入力用の基本クラス。"""
    def __init__( self,
                  poll_delay=0.0,
                  throttle_scale=1.0,
                  steering_scale=1.0,
                  throttle_dir=-1.0,
                  dev_fn='/dev/input/js0',
                  auto_record_on_throttle=True,
                  which_js=0):

        import pygame

        pygame.init()

        # ジョイスティックを初期化
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(which_js)
        self.joystick.init()
        name = self.joystick.get_name()
        logger.info(f"ジョイスティックデバイスを検出: {name}")

        self.axis_states = [ 0.0 for i in range(self.joystick.get_numaxes())]
        self.button_states = [ 0 for i in range(self.joystick.get_numbuttons() + self.joystick.get_numhats() * 4)]
        self.axis_names = {}
        self.button_names = {}
        self.dead_zone = 0.07
        for i in range(self.joystick.get_numaxes()):
            self.axis_names[i] = i
        for i in range(self.joystick.get_numbuttons() + self.joystick.get_numhats() * 4):
            self.button_names[i] = i

    def poll(self):
        import pygame

        button = None
        button_state = None
        axis = None
        axis_val = None

        pygame.event.get()


        for i in range( self.joystick.get_numaxes() ):
            val = self.joystick.get_axis( i )
            if abs(val) < self.dead_zone:
                val = 0.0
            if self.axis_states[i] != val and i in self.axis_names:
                axis = self.axis_names[i]
                axis_val = val
                self.axis_states[i] = val
                logging.debug("axis: %s val: %f" % (axis, val))
                #print("axis: %s val: %f" % (axis, val))


        for i in range( self.joystick.get_numbuttons() ):
            state = self.joystick.get_button( i )
            if self.button_states[i] != state:
                if not i in self.button_names:
                    logger.info(f'ボタン: {i}')
                    continue
                button = self.button_names[i]
                button_state = state
                self.button_states[i] = state
                logging.info("button: %s state: %d" % (button, state))
                #print("button: %s state: %d" % (button, state))

        for i in range( self.joystick.get_numhats() ):
            hat = self.joystick.get_hat( i )
            horz, vert = hat
            iBtn = self.joystick.get_numbuttons() + (i * 4)
            states = (horz == -1, horz == 1, vert == -1, vert == 1)
            for state in states:
                state = int(state)
                if self.button_states[iBtn] != state:
                    if not iBtn in self.button_names:
                        logger.info(f"ボタン: {iBtn}")
                        continue
                    button = self.button_names[iBtn]
                    button_state = state
                    self.button_states[iBtn] = state
                    logging.info("button: %s state: %d" % (button, state))
                    #print("button: %s state: %d" % (button, state))

                iBtn += 1

        return button, button_state, axis, axis_val
        
    def set_deadzone(self, val):
        self.dead_zone = val

# このクラスは RCReceiver 用のヘルパー
class Channel:
    """RC 受信の各チャンネルを表すクラス。"""
    def __init__(self, pin):
        self.pin = pin
        self.tick = None
        self.high_tick = None

class RCReceiver:
    """PWM 信号を読み取るためのレシーバー。"""
    MIN_OUT = -1
    MAX_OUT = 1
    def __init__(self, cfg, debug=False):
        """RCReceiver の初期化処理。

        Args:
            cfg: 設定オブジェクト。
            debug (bool): デバッグログを出力するかどうか。
        """
        import pigpio
        self.pi = pigpio.pi()

        # standard variables
        self.channels = [Channel(cfg.STEERING_RC_GPIO), Channel(cfg.THROTTLE_RC_GPIO), Channel(cfg.DATA_WIPER_RC_GPIO)]
        self.min_pwm = 1000
        self.max_pwm = 2000
        self.oldtime = 0
        self.STEERING_MID = cfg.PIGPIO_STEERING_MID
        self.MAX_FORWARD = cfg.PIGPIO_MAX_FORWARD
        self.STOPPED_PWM = cfg.PIGPIO_STOPPED_PWM
        self.MAX_REVERSE = cfg.PIGPIO_MAX_REVERSE
        self.RECORD = cfg.AUTO_RECORD_ON_THROTTLE
        self.debug = debug
        self.mode = 'user'
        self.is_action = False
        self.invert = cfg.PIGPIO_INVERT
        self.jitter = cfg.PIGPIO_JITTER
        self.factor = (self.MAX_OUT - self.MIN_OUT) / (self.max_pwm - self.min_pwm)
        self.cbs = []
        self.signals = [0,0,0]
        for channel in self.channels:
            self.pi.set_mode(channel.pin, pigpio.INPUT)
            self.cbs.append(self.pi.callback(channel.pin, pigpio.EITHER_EDGE, self.cbf))
            if self.debug:
                logger.info(f'RCReceiver GPIO {channel.pin} を作成しました')
    

    def cbf(self, gpio, level, tick):
        import pigpio
        """pigpio の割り込みコールバック。

        Args:
            gpio: 監視対象の GPIO 番号。
            level: 立ち上がり/立ち下がりエッジ。
            tick: 起動からのマイクロ秒カウンタ。
        """
        for channel in self.channels:
            if gpio == channel.pin:            
                if level == 1:                
                    channel.high_tick = tick            
                elif level == 0:                
                    if channel.high_tick is not None:                    
                        channel.tick = pigpio.tickDiff(channel.high_tick, tick)

    def pulse_width(self, high):
        """PWM パルス幅をマイクロ秒で返す。

        Args:
            high: パルス幅。

        Returns:
            float: パルス幅。値が ``None`` の場合は ``0.0``。
        """
        if high is not None:
            return high
        else:
            return 0.0

    def run(self, mode=None, recording=None):
        """RC 信号を読み取り現在のステアリングとスロットルを返す。

        Args:
            mode: デフォルトのモード文字列。
            recording: 録画状態のデフォルト値。

        Returns:
            tuple: ステアリング値、スロットル値、モード、録画フラグ。
        """

        i = 0
        for channel in self.channels:
            # signal is a value in [0, (MAX_OUT-MIN_OUT)]
            self.signals[i] = (self.pulse_width(channel.tick) - self.min_pwm) * self.factor
            # convert into min max interval
            if self.invert:
                self.signals[i] = -self.signals[i] + self.MAX_OUT
            else:
                self.signals[i] += self.MIN_OUT
            i += 1
        if self.debug:
            logger.info(f'RC CH1 信号:{round(self.signals[0], 3)}, RC CH2 信号:{round(self.signals[1], 3)}, RC CH3 信号:{round(self.signals[2], 3)}')

        # check mode channel if present
        if (self.signals[2] - self.jitter) > 0:  
            self.mode = 'local'
        else:
            # pass though value if provided
            self.mode = mode if mode is not None else 'user'

        # check throttle channel
        if ((self.signals[1] - self.jitter) > 0) and self.RECORD: # is throttle above jitter level? If so, turn on auto-record 
            is_action = True
        else:
            # pass through default value
            is_action = recording if recording is not None else False
        return self.signals[0], self.signals[1], self.mode, is_action

    def shutdown(self):
        """終了時にすべてのコールバックを解除する。"""
        for channel in self.channels:
            self.cbs[channel].cancel()



class JoystickCreator(Joystick):
    """新しいジョイスティックマッピングを作成するためのヘルパークラス。"""
    def __init__(self, *args, **kwargs):
        super(JoystickCreator, self).__init__(*args, **kwargs)

        self.axis_names = {}
        self.button_names = {}

    def poll(self):

        button, button_state, axis, axis_val = super(JoystickCreator, self).poll()

        return button, button_state, axis, axis_val

class PS3JoystickSixAd(Joystick):
    """/dev/input/js0 で利用可能な PS3 ジョイスティック用インターフェース。

    Jetson Nano で sixad を使用した際のマッピングを含む。
    """
    def __init__(self, *args, **kwargs):
        super(PS3JoystickSixAd, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00 : 'left_stick_horz',
            0x01 : 'left_stick_vert',
            0x02 : 'right_stick_horz',
            0x03 : 'right_stick_vert',
        }

        self.button_names = {
            0x120 : 'select',
            0x123 : 'start',
            0x130 : 'PS',

            0x12a : 'L1',
            0x12b : 'R1',
            0x128 : 'L2',
            0x129 : 'R2',
            0x121 : 'L3',
            0x122 : 'R3',

            0x12c : "triangle", 
            0x12d : "circle",
            0x12e : "cross",
            0x12f : 'square',

            0x124 : 'dpad_up',
            0x126 : 'dpad_down',
            0x127 : 'dpad_left',
            0x125 : 'dpad_right',
        }


class PS3JoystickOld(Joystick):
    """/dev/input/js0 用 PS3 ジョイスティックの古いドライバー向けインターフェース。"""
    def __init__(self, *args, **kwargs):
        super(PS3JoystickOld, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00 : 'left_stick_horz',
            0x01 : 'left_stick_vert',
            0x02 : 'right_stick_horz',
            0x05 : 'right_stick_vert',

            0x1a : 'tilt_x',
            0x1b : 'tilt_y',
            0x3d : 'tilt_a',
            0x3c : 'tilt_b',

            0x32 : 'L1_pressure',
            0x33 : 'R1_pressure',
            0x31 : 'R2_pressure',
            0x30 : 'L2_pressure',

            0x36 : 'cross_pressure',
            0x35 : 'circle_pressure',
            0x37 : 'square_pressure',
            0x34 : 'triangle_pressure',

            0x2d : 'dpad_r_pressure',
            0x2e : 'dpad_d_pressure',
            0x2c : 'dpad_u_pressure',
        }

        self.button_names = {
            0x120 : 'select',
            0x123 : 'start',
            0x2c0 : 'PS',

            0x12a : 'L1',
            0x12b : 'R1',
            0x128 : 'L2',
            0x129 : 'R2',
            0x121 : 'L3',
            0x122 : 'R3',

            0x12c : "triangle", 
            0x12d : "circle",
            0x12e : "cross",
            0x12f : 'square',

            0x124 : 'dpad_up',
            0x126 : 'dpad_down',
            0x127 : 'dpad_left',
            0x125 : 'dpad_right',
        }


class PS3Joystick(Joystick):
    """/dev/input/js0 用 PS3 ジョイスティックインターフェース。"""
    def __init__(self, *args, **kwargs):
        super(PS3Joystick, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00 : 'left_stick_horz',
            0x01 : 'left_stick_vert',
            0x03 : 'right_stick_horz',
            0x04 : 'right_stick_vert',

            0x02 : 'L2_pressure',
            0x05 : 'R2_pressure',
        }

        self.button_names = {
           0x13a : 'select', #8 314
           0x13b : 'start', #9 315
           0x13c : 'PS', #a  316

           0x136 : 'L1', #4 310
           0x137 : 'R1', #5 311
           0x138 : 'L2', #6 312
           0x139 : 'R2', #7 313
           0x13d : 'L3', #b 317
           0x13e : 'R3', #c 318

           0x133 : "triangle",  #2 307
           0x131 : "circle",    #1 305
           0x130 : "cross",    #0 304
           0x134 : 'square',    #3 308

           0x220 : 'dpad_up', #d 544
           0x221 : 'dpad_down', #e 545
           0x222 : 'dpad_left', #f 546
           0x223 : 'dpad_right', #10 547
       }


class PS4Joystick(Joystick):
    """/dev/input/js0 で利用可能な PS4 ジョイスティックのインターフェース。"""
    def __init__(self, *args, **kwargs):
        super(PS4Joystick, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00 : 'left_stick_horz',
            0x01 : 'left_stick_vert',
            0x03 : 'right_stick_horz',
            0x04 : 'right_stick_vert',

            0x02 : 'left_trigger_axis',
            0x05 : 'right_trigger_axis',

            0x10 : 'dpad_leftright',
            0x11 : 'dpad_updown',

            0x19 : 'tilt_a',
            0x1a : 'tilt_b',
            0x1b : 'tilt_c',

            0x06 : 'motion_a',
            0x07 : 'motion_b',
            0x08 : 'motion_c',
        }

        self.button_names = {

            0x134 : 'square',
            0x130 : 'cross',
            0x131 : 'circle',
            0x133 : 'triangle',

            0x138 : 'L1',
            0x139 : 'R1',
            0x136 : 'L2',
            0x137 : 'R2',
            0x13a : 'L3',
            0x13b : 'R3',

            0x13d : 'pad',
            0x13a : 'share',
            0x13b : 'options',
            0x13c : 'PS',
        }


class PS3JoystickPC(Joystick):
    """Ubuntu などで利用する PS3 ジョイスティックのインターフェース。"""
    def __init__(self, *args, **kwargs):
        super(PS3JoystickPC, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00 : 'left_stick_horz',
            0x01 : 'left_stick_vert',
            0x03 : 'right_stick_horz',
            0x04 : 'right_stick_vert',

            0x1a : 'tilt_x',
            0x1b : 'tilt_y',
            0x3d : 'tilt_a',
            0x3c : 'tilt_b',

            0x32 : 'L1_pressure',
            0x33 : 'R1_pressure',
            0x05 : 'R2_pressure',
            0x02 : 'L2_pressure',

            0x36 : 'cross_pressure',
            0x35 : 'circle_pressure',
            0x37 : 'square_pressure',
            0x34 : 'triangle_pressure',

            0x2d : 'dpad_r_pressure',
            0x2e : 'dpad_d_pressure',
            0x2c : 'dpad_u_pressure',
        }

        self.button_names = {
            0x13a : 'select',
            0x13b : 'start',
            0x13c : 'PS',

            0x136 : 'L1',
            0x137 : 'R1',
            0x138 : 'L2',
            0x139 : 'R2',
            0x13d : 'L3',
            0x13e : 'R3',

            0x133 : "triangle",
            0x131 : "circle",
            0x130 : "cross",
            0x134 : 'square',

            0x220 : 'dpad_up',
            0x221 : 'dpad_down',
            0x222 : 'dpad_left',
            0x223 : 'dpad_right',
        }


class PyGamePS4Joystick(PyGameJoystick):
    """pygame 経由で PS4 ジョイスティックを扱うクラス。"""
    def __init__(self, *args, **kwargs):
        super(PyGamePS4Joystick, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00 : 'left_stick_horz',
            0x01 : 'left_stick_vert',
            0x03 : 'right_stick_vert',
            0x02 : 'right_stick_horz',
        }

        self.button_names = {
            2 : "circle",
            1 : "cross",
            0 : 'square',
            3 : "triangle",

            8 : 'share',
            9 : 'options',
            13 : 'pad',

            4 : 'L1',
            5 : 'R1',
            6 : 'L2',
            7 : 'R2',
            10 : 'L3',
            11 : 'R3',
            14 : 'dpad_left',
            15 : 'dpad_right',
            16 : 'dpad_down',
            17 : 'dpad_up',
        }


class XboxOneJoystick(Joystick):
    """Xbox Wireless Controller 用のインターフェース。"""
    def __init__(self, *args, **kwargs):
        super(XboxOneJoystick, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00 : 'left_stick_horz',
            0x01 : 'left_stick_vert',
            0x05 : 'right_stick_vert',
            0x02 : 'right_stick_horz',
            0x0a : 'left_trigger',
            0x09 : 'right_trigger',
            0x10 : 'dpad_horiz',
            0x11 : 'dpad_vert'
        }

        self.button_names = {
            0x130: 'a_button',
            0x131: 'b_button',
            0x133: 'x_button',
            0x134: 'y_button',
            0x13b: 'options',
            0x136: 'left_shoulder',
            0x137: 'right_shoulder',
        }

class LogitechJoystick(Joystick):
    """Logitech 製ジョイスティック用インターフェース。"""
    def __init__(self, *args, **kwargs):
        super(LogitechJoystick, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00: 'left_stick_horz',
            0x01: 'left_stick_vert',
            0x03: 'right_stick_horz',
            0x04: 'right_stick_vert',

            0x02: 'L2_pressure',
            0x05: 'R2_pressure',

            0x10: 'dpad_leftright', # 1 is right, -1 is left
            0x11: 'dpad_up_down', # 1 is down, -1 is up
        }

        self.button_names = {
            0x13a: 'back',  # 8 314
            0x13b: 'start',  # 9 315
            0x13c: 'Logitech',  # a  316

            0x130: 'A',
            0x131: 'B',
            0x133: 'X',
            0x134: 'Y',

            0x136: 'L1',
            0x137: 'R1',

            0x13d: 'left_stick_press',
            0x13e: 'right_stick_press',
        }


class Nimbus(Joystick):
    """SteelNimbus ジョイスティック向けインターフェース。"""
    def __init__(self, *args, **kwargs):
        super(Nimbus, self).__init__(*args, **kwargs)

        self.button_names = {
            0x130 : 'a',
            0x131 : 'b',
            0x132 : 'x',
            0x133 : 'y',
            0x135 : 'R1',
            0x137 : 'R2',
            0x134 : 'L1',
            0x136 : 'L2',
        }

        self.axis_names = {
            0x0 : 'lx',
            0x1 : 'ly',
            0x2 : 'rx',
            0x5 : 'ry',
            0x11 : 'hmm',
            0x10 : 'what',
        }


class WiiU(Joystick):
    """WiiUPro ジョイスティック向け設定。"""
    def __init__(self, *args, **kwargs):
        super(WiiU, self).__init__(*args, **kwargs)

        self.button_names = {
            305: 'A',
            304: 'B',
            307: 'X',
            308: 'Y',
            312: 'LEFT_BOTTOM_TRIGGER',
            310: 'LEFT_TOP_TRIGGER',
            313: 'RIGHT_BOTTOM_TRIGGER',
            311: 'RIGHT_TOP_TRIGGER',
            317: 'LEFT_STICK_PRESS',
            318: 'RIGHT_STICK_PRESS',
            314: 'SELECT',
            315: 'START',
            547: 'PAD_RIGHT',
            546: 'PAD_LEFT',
            544: 'PAD_UP',
            548: 'PAD_DOWN,',
        }

        self.axis_names = {
            0: 'LEFT_STICK_X',
            1: 'LEFT_STICK_Y',
            3: 'RIGHT_STICK_X',
            4: 'RIGHT_STICK_Y',
        }


class RC3ChanJoystick(Joystick):
    """3 チャンネル RC 送信機からの入力を扱う。"""
    def __init__(self, *args, **kwargs):
        super(RC3ChanJoystick, self).__init__(*args, **kwargs)


        self.button_names = {
            0x120 : 'Switch-up',
            0x121 : 'Switch-down',
        }


        self.axis_names = {
            0x1 : 'Throttle',
            0x0 : 'Steering',
        }


class JoystickController(object):
    """ジョイスティック入力を各種操作に割り当てるための基底クラス。"""

    ES_IDLE = -1
    ES_START = 0
    ES_THROTTLE_NEG_ONE = 1
    ES_THROTTLE_POS_ONE = 2
    ES_THROTTLE_NEG_TWO = 3


    def __init__(self, poll_delay=0.0,
                 throttle_scale=1.0,
                 steering_scale=1.0,
                 throttle_dir=-1.0,
                 dev_fn='/dev/input/js0',
                 auto_record_on_throttle=True):

        self.img_arr = None
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.mode_latch = None
        self.poll_delay = poll_delay
        self.running = True
        self.last_throttle_axis_val = 0
        self.throttle_scale = throttle_scale
        self.steering_scale = steering_scale
        self.throttle_dir = throttle_dir
        self.recording = False
        self.recording_latch = None
        self.constant_throttle = False
        self.auto_record_on_throttle = auto_record_on_throttle
        self.dev_fn = dev_fn
        self.js = None
        self.tub = None
        self.num_records_to_erase = 100
        self.estop_state = self.ES_IDLE
        self.chaos_monkey_steering = None
        self.dead_zone = 0.0

        self.button_down_trigger_map = {}
        self.button_up_trigger_map = {}
        self.axis_trigger_map = {}
        self.init_trigger_maps()


    def init_js(self):
        """派生クラスで実装すべきジョイスティック初期化処理。"""
        raise(Exception("Subclass needs to define init_js"))


    def init_trigger_maps(self):
        """ボタンと関数の対応表を作成する。派生クラスで実装すること。"""
        raise(Exception("init_trigger_maps"))


    def set_deadzone(self, val):
        """記録を開始する最小スロットル値を設定する。

        Args:
            val (float): スロットルのデッドゾーン値。
        """
        self.dead_zone = val


    def print_controls(self):
        """ボタンおよび軸のマッピングを表示する。"""
        pt = PrettyTable()
        pt.field_names = ["control", "action"]
        for button, control in self.button_down_trigger_map.items():
            pt.add_row([button, control.__name__])
        for axis, control in self.axis_trigger_map.items():
            pt.add_row([axis, control.__name__])
        print("ジョイスティック操作一覧:")
        print(pt)

        # print("Joystick Controls:")
        # print("On Button Down:")
        # print(self.button_down_trigger_map)
        # print("On Button Up:")
        # print(self.button_up_trigger_map)
        # print("On Axis Move:")
        # print(self.axis_trigger_map)


    def set_button_down_trigger(self, button, func):
        """ボタン押下時のトリガーを設定する。

        Args:
            button (str): ボタン名称。
            func (Callable): 実行する関数。
        """
        self.button_down_trigger_map[button] = func


    def set_button_up_trigger(self, button, func):
        """ボタン解放時のトリガーを設定する。

        Args:
            button (str): ボタン名称。
            func (Callable): 実行する関数。
        """
        self.button_up_trigger_map[button] = func


    def set_axis_trigger(self, axis, func):
        """軸操作時のトリガーを設定する。

        Args:
            axis (str): 軸名称。
            func (Callable): 実行する関数。
        """
        self.axis_trigger_map[axis] = func


    def set_tub(self, tub):
        self.tub = tub


    def erase_last_N_records(self):
        """直近の記録を削除する。"""
        if self.tub is not None:
            try:
                self.tub.delete_last_n_records(self.num_records_to_erase)
                logger.info('直近 %d 件の記録を削除しました。' % self.num_records_to_erase)
            except:
                logger.info('削除に失敗しました')


    def on_throttle_changes(self):
        """ユーザーモードでスロットルがゼロ以外になった場合に録画を開始する。"""
        if self.auto_record_on_throttle:
            recording = (abs(self.throttle) > self.dead_zone and self.mode == 'user')
            if recording != self.recording:
                self.recording = recording
                self.recording_latch = self.recording
                logger.debug(f"JoystickController::on_throttle_changes() setting recording = {self.recording}")


    def emergency_stop(self):
        """緊急停止シーケンスを開始する。"""
        logger.warn('緊急停止！')
        self.mode = "user"
        self.recording = False
        self.constant_throttle = False
        self.estop_state = self.ES_START
        self.throttle = 0.0


    def update(self):
        """ジョイスティック入力をポーリングする。"""

        # ジョイスティックが使用可能になるまで待機
        while self.running and self.js is None and not self.init_js():
            time.sleep(3)

        while self.running:
            button, button_state, axis, axis_val = self.js.poll()

            if axis is not None and axis in self.axis_trigger_map:
                # 対応する関数を呼び出す
                self.axis_trigger_map[axis](axis_val)

            if button and button_state >= 1 and button in self.button_down_trigger_map:
                # ボタン押下時の関数を呼び出す
                self.button_down_trigger_map[button]()

            if button and button_state == 0 and button in self.button_up_trigger_map:
                # ボタン解放時の関数を呼び出す
                self.button_up_trigger_map[button]()

            time.sleep(self.poll_delay)

    def do_nothing(self, param):
        """何もしないダミー関数。

        軸の割り当てを解除する際に使用する。
        """
        pass



    def set_steering(self, axis_val):
        self.angle = self.steering_scale * axis_val
        #print("angle", self.angle)


    def set_throttle(self, axis_val):
        #this value is often reversed, with positive value when pulling down
        self.last_throttle_axis_val = axis_val
        self.throttle = (self.throttle_dir * axis_val * self.throttle_scale)
        #print("throttle", self.throttle)
        self.on_throttle_changes()


    def toggle_manual_recording(self):
        """手動で録画の開始と停止を切り替える。"""
        if self.auto_record_on_throttle:
            logger.info('スロットルによる自動録画が有効なため、手動モードの切り替えを無視します。')
        elif self.recording:
            self.recording = False
            self.recording_latch = self.recording
            logger.debug(f"JoystickController::toggle_manual_recording() setting recording and recording_latch = {self.recording}")
        else:
            self.recording = True
            self.recording_latch = self.recording
            logger.debug(f"JoystickController::toggle_manual_recording() setting recording and recording_latch = {self.recording}")

        logger.info(f'録画状態: {self.recording}')


    def increase_max_throttle(self):
        """最大スロットル値を増加させる。"""
        self.throttle_scale = round(min(1.0, self.throttle_scale + 0.01), 2)
        if self.constant_throttle:
            self.throttle = self.throttle_scale
            self.on_throttle_changes()
        else:
            self.throttle = (self.throttle_dir * self.last_throttle_axis_val * self.throttle_scale)

        logger.info(f'スロットル倍率: {self.throttle_scale}')


    def decrease_max_throttle(self):
        """最大スロットル値を減少させる。"""
        self.throttle_scale = round(max(0.0, self.throttle_scale - 0.01), 2)
        if self.constant_throttle:
            self.throttle = self.throttle_scale
            self.on_throttle_changes()
        else:
            self.throttle = (self.throttle_dir * self.last_throttle_axis_val * self.throttle_scale)

        logger.info(f'スロットル倍率: {self.throttle_scale}')


    def toggle_constant_throttle(self):
        """定速走行のオン・オフを切り替える。"""
        if self.constant_throttle:
            self.constant_throttle = False
            self.throttle = 0
            self.on_throttle_changes()
        else:
            self.constant_throttle = True
            self.throttle = self.throttle_scale
            self.on_throttle_changes()
        logger.info(f'定速走行: {self.constant_throttle}')


    def toggle_mode(self):
        """走行モードを順に切り替える。"""
        if self.mode == 'user':
            self.mode = 'local_angle'
        elif self.mode == 'local_angle':
            self.mode = 'local'
        else:
            self.mode = 'user'
        self.mode_latch = self.mode
        logger.info(f'新しいモード: {self.mode}')


    def chaos_monkey_on_left(self):
        self.chaos_monkey_steering = -0.2


    def chaos_monkey_on_right(self):
        self.chaos_monkey_steering = 0.2


    def chaos_monkey_off(self):
        self.chaos_monkey_steering = None


    def run_threaded(self, img_arr=None, mode=None, recording=None):
        """スレッドモードでコントローラーを実行する。

        Args:
            img_arr: カメラ画像。
            mode: デフォルトのモード。
            recording: 録画状態の初期値。
        """
        self.img_arr = img_arr

        #
        # enforce defaults if they are not none.
        #
        if mode is not None:
            self.mode = mode
        if self.mode_latch is not None:
            self.mode = self.mode_latch
            self.mode_latch = None
        if recording is not None and recording != self.recording:
            logger.debug(f"JoystickController::run_threaded() setting recording from default = {recording}")
            self.recording = recording
        if self.recording_latch is not None:
            logger.debug(f"JoystickController::run_threaded() setting recording from latch = {self.recording_latch}")
            self.recording = self.recording_latch
            self.recording_latch = None

        # E-Stop のステートマシン処理
        if self.estop_state > self.ES_IDLE:
            if self.estop_state == self.ES_START:
                self.estop_state = self.ES_THROTTLE_NEG_ONE
                return 0.0, -1.0 * self.throttle_scale, self.mode, False
            elif self.estop_state == self.ES_THROTTLE_NEG_ONE:
                self.estop_state = self.ES_THROTTLE_POS_ONE
                return 0.0, 0.01, self.mode, False
            elif self.estop_state == self.ES_THROTTLE_POS_ONE:
                self.estop_state = self.ES_THROTTLE_NEG_TWO
                self.throttle = -1.0 * self.throttle_scale
                return 0.0, self.throttle, self.mode, False
            elif self.estop_state == self.ES_THROTTLE_NEG_TWO:
                self.throttle += 0.05
                if self.throttle >= 0.0:
                    self.throttle = 0.0
                    self.estop_state = self.ES_IDLE
                return 0.0, self.throttle, self.mode, False

        if self.chaos_monkey_steering is not None:
            return self.chaos_monkey_steering, self.throttle, self.mode, False

        return self.angle, self.throttle, self.mode, self.recording


    def run(self, img_arr=None, mode=None, recording=None):
        """スレッドを利用しない実行関数。"""
        return self.run_threaded(img_arr, mode, recording)


    def shutdown(self):
        # ポーリングスレッドを終了させるフラグを立て、少し待機
        self.running = False
        time.sleep(0.5)


class JoystickCreatorController(JoystickController):
    """新しいコントローラーとマッピング作成を支援するクラス。"""
    def __init__(self, *args, **kwargs):
        super(JoystickCreatorController, self).__init__(*args, **kwargs)


    def init_js(self):
        """ジョイスティックの初期化を試みる。"""
        try:
            self.js = JoystickCreator(self.dev_fn)
            if not self.js.init():
                self.js = None
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None

        return self.js is not None


    def init_trigger_maps(self):
        """ボタンから関数へのマッピングを初期化する。"""
        pass


class PS3JoystickController(JoystickController):
    """PS3 ジョイスティック用のコントローラークラス。"""
    def __init__(self, *args, **kwargs):
        super(PS3JoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        """ジョイスティックの初期化を試みる。"""
        try:
            self.js = PS3Joystick(self.dev_fn)
            if not self.js.init():
                self.js = None
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        """ボタンから関数へのマッピングを初期化する。"""

        self.button_down_trigger_map = {
            'select' : self.toggle_mode,
            'circle' : self.toggle_manual_recording,
            'triangle' : self.erase_last_N_records,
            'cross' : self.emergency_stop,
            'dpad_up' : self.increase_max_throttle,
            'dpad_down' : self.decrease_max_throttle,
            'start' : self.toggle_constant_throttle,
            "R1" : self.chaos_monkey_on_right,
            "L1" : self.chaos_monkey_on_left,
        }

        self.button_up_trigger_map = {
            "R1" : self.chaos_monkey_off,
            "L1" : self.chaos_monkey_off,
        }

        self.axis_trigger_map = {
            'left_stick_horz' : self.set_steering,
            'right_stick_vert' : self.set_throttle,
        }


class PS3JoystickSixAdController(PS3JoystickController):
    """sixad 経由で接続された PS3 コントローラー用クラス。"""
    def init_js(self):
        """ジョイスティックの初期化を試みる。"""
        try:
            self.js = PS3JoystickSixAd(self.dev_fn)
            if not self.js.init():
                self.js = None
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None

    def init_trigger_maps(self):
        """ボタンから関数へのマッピングを初期化する。"""
        super(PS3JoystickSixAdController, self).init_trigger_maps()

        self.axis_trigger_map = {
            'right_stick_horz' : self.set_steering,
            'left_stick_vert' : self.set_throttle,
        }

class PS4JoystickController(JoystickController):
    """PS4 ジョイスティック用のコントローラークラス。"""
    def __init__(self, *args, **kwargs):
        super(PS4JoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        """ジョイスティックの初期化を試みる。"""
        try:
            self.js = PS4Joystick(self.dev_fn)
            if not self.js.init():
                self.js = None
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        """PS4 用のボタンマッピングを初期化する。"""

        self.button_down_trigger_map = {
            'share' : self.toggle_mode,
            'circle' : self.toggle_manual_recording,
            'triangle' : self.erase_last_N_records,
            'cross' : self.emergency_stop,
            'L1' : self.increase_max_throttle,
            'R1' : self.decrease_max_throttle,
            'options' : self.toggle_constant_throttle,
        }

        self.axis_trigger_map = {
            'left_stick_horz' : self.set_steering,
            'right_stick_vert' : self.set_throttle,
        }


class PyGamePS4JoystickController(PS4JoystickController):
    """pygame を利用した PS4 コントローラー用クラス。"""
    def __init__(self, which_js=0, *args, **kwargs):
        super(PyGamePS4JoystickController, self).__init__(*args, **kwargs)
        self.which_js=which_js


    def init_js(self):
        """ジョイスティックの初期化を試みる。"""
        try:
            self.js = PyGamePS4Joystick(which_js=self.which_js)
        except Exception as e:
            logger.error(e)
            self.js = None
        return self.js is not None



class XboxOneJoystickController(JoystickController):
    """Xbox One コントローラー用のコントローラークラス。"""
    def __init__(self, *args, **kwargs):
        super(XboxOneJoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        """ジョイスティックの初期化を試みる。"""
        try:
            self.js = XboxOneJoystick(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None


    def magnitude(self, reversed = False):
        def set_magnitude(axis_val):
            """軸の値をスロットルの強さに変換する。"""
            # Axis values range from -1. to 1.
            minimum = -1.
            maximum = 1.
            # Magnitude is now normalized in the range of 0 - 1.
            magnitude = (axis_val - minimum) / (maximum - minimum)
            if reversed:
                magnitude *= -1
            self.set_throttle(magnitude)
        return set_magnitude


    def init_trigger_maps(self):
        """ボタンから関数へのマッピングを初期化する。"""

        self.button_down_trigger_map = {
            'a_button': self.toggle_mode,
            'b_button': self.toggle_manual_recording,
            'x_button': self.erase_last_N_records,
            'y_button': self.emergency_stop,
            'right_shoulder': self.increase_max_throttle,
            'left_shoulder': self.decrease_max_throttle,
            'options': self.toggle_constant_throttle,
        }

        self.axis_trigger_map = {
            'left_stick_horz': self.set_steering,
            'right_stick_vert': self.set_throttle,
            # Forza Mode
            'right_trigger': self.magnitude(),
            'left_trigger': self.magnitude(reversed = True),
        }

class XboxOneSwappedJoystickController(XboxOneJoystickController):
    """左右のスティック操作を入れ替えた Xbox One 用コントローラー。"""
    def __init__(self, *args, **kwargs):
        super(XboxOneSwappedJoystickController, self).__init__(*args, **kwargs)

    def init_trigger_maps(self):
        """ボタンから関数へのマッピングを初期化する。"""
        super(XboxOneSwappedJoystickController, self).init_trigger_maps()

        # make the actual swap of the sticks
        self.set_axis_trigger('right_stick_horz', self.set_steering)
        self.set_axis_trigger('left_stick_vert', self.set_throttle)

        # unmap default assinments to the axes
        self.set_axis_trigger('left_stick_horz', self.do_nothing)
        self.set_axis_trigger('right_stick_vert', self.do_nothing)


class LogitechJoystickController(JoystickController):
    """Logitech コントローラー用のクラス。"""
    def __init__(self, *args, **kwargs):
        super(LogitechJoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        """ジョイスティックの初期化を試みる。"""
        try:
            self.js = LogitechJoystick(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        """ボタンから関数へのマッピングを初期化する。"""

        self.button_down_trigger_map = {
            'start': self.toggle_mode,
            'B': self.toggle_manual_recording,
            'Y': self.erase_last_N_records,
            'A': self.emergency_stop,
            'back': self.toggle_constant_throttle,
            "R1" : self.chaos_monkey_on_right,
            "L1" : self.chaos_monkey_on_left,
        }

        self.button_up_trigger_map = {
            "R1" : self.chaos_monkey_off,
            "L1" : self.chaos_monkey_off,
        }

        self.axis_trigger_map = {
            'left_stick_horz': self.set_steering,
            'right_stick_vert': self.set_throttle,
            'dpad_leftright' : self.on_axis_dpad_LR,
            'dpad_up_down' : self.on_axis_dpad_UD,
        }

    def on_axis_dpad_LR(self, val):
        if val == -1.0:
            self.on_dpad_left()
        elif val == 1.0:
            self.on_dpad_right()

    def on_axis_dpad_UD(self, val):
        if val == -1.0:
            self.on_dpad_up()
        elif val == 1.0:
            self.on_dpad_down()

    def on_dpad_up(self):
        self.increase_max_throttle()

    def on_dpad_down(self):
        self.decrease_max_throttle()

    def on_dpad_left(self):
        logger.error("dpad 左は未割り当てです")

    def on_dpad_right(self):
        logger.error("dpad 右は未割り当てです")


class NimbusController(JoystickController):
    """Nimbus コントローラー用のクラス。"""
    def __init__(self, *args, **kwargs):
        super(NimbusController, self).__init__(*args, **kwargs)


    def init_js(self):
        # ジョイスティックを初期化
        try:
            self.js = Nimbus(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        # ボタンと関数のマッピングを初期化

        self.button_down_trigger_map = {
            'y' : self.erase_last_N_records,
            'b' : self.toggle_mode,
            'a' : self.emergency_stop,
        }

        self.axis_trigger_map = {
            'lx' : self.set_steering,
            'ry' : self.set_throttle,
        }


class WiiUController(JoystickController):
    """WiiU コントローラー用のクラス。"""
    def __init__(self, *args, **kwargs):
        super(WiiUController, self).__init__(*args, **kwargs)


    def init_js(self):
        # ジョイスティックを初期化
        try:
            self.js = WiiU(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        # ボタンと関数のマッピングを初期化

        self.button_down_trigger_map = {
            'Y' : self.erase_last_N_records,
            'B' : self.toggle_mode,
            'A' : self.emergency_stop,
        }

        self.axis_trigger_map = {
            'LEFT_STICK_X' : self.set_steering,
            'RIGHT_STICK_Y' : self.set_throttle,
        }



class RC3ChanJoystickController(JoystickController):
    """3 チャンネル RC 用コントローラー。"""
    def __init__(self, *args, **kwargs):
        super(RC3ChanJoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        # ジョイスティックを初期化
        try:
            self.js = RC3ChanJoystick(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            logger.error(f"{self.dev_fn} が見つかりません")
            self.js = None
        return self.js is not None

    def on_steering(self, val, reverse = True):
        if reversed:
            val *= -1
        self.set_steering(val)

    def on_throttle(self, val, reverse = True):
        if reversed:
            val *= -1
        self.set_throttle(val)

    def on_switch_up(self):
        if self.mode == 'user':
            self.erase_last_N_records()
        else:
            self.emergency_stop()

    def on_switch_down(self):
        self.toggle_mode()

    def init_trigger_maps(self):
        # ボタンと関数のマッピングを初期化

        self.button_down_trigger_map = {
            'Switch-down' : self.on_switch_down,
            'Switch-up' : self.on_switch_up,
        }


        self.axis_trigger_map = {
            'Steering' : self.on_steering,
            'Throttle' : self.on_throttle,
        }


class JoyStickPub(object):
    """ZeroMQ を利用してローカルジョイスティックの入力を配信するクラス。"""
    def __init__(self, port = 5556, dev_fn='/dev/input/js1'):
        import zmq
        self.dev_fn = dev_fn
        self.js = PS3JoystickPC(self.dev_fn)
        self.js.init()
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%d" % port)


    def run(self):
        while True:
            button, button_state, axis, axis_val = self.js.poll()
            if axis is not None or button is not None:
                if button is None:
                    button  = "0"
                    button_state = 0
                if axis is None:
                    axis = "0"
                    axis_val = 0
                message_data = (button, button_state, axis, axis_val)
                self.socket.send_string( "%s %d %s %f" % message_data)
                logger.info(f"送信: {message_data}")


class JoyStickSub(object):
    """ZeroMQ を利用してリモートジョイスティックの入力を購読するクラス。"""
    def __init__(self, ip, port = 5556):
        import zmq
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect("tcp://%s:%d" % (ip, port))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        self.button = None
        self.button_state = 0
        self.axis = None
        self.axis_val = 0.0
        self.running = True


    def shutdown(self):
        self.running = False
        time.sleep(0.1)


    def update(self):
        while self.running:
            payload = self.socket.recv().decode("utf-8")
            # 受信データを確認
            button, button_state, axis, axis_val = payload.split(' ')
            self.button = button
            self.button_state = (int)(button_state)
            self.axis = axis
            self.axis_val = (float)(axis_val)
            if self.button == "0":
                self.button = None
            if self.axis == "0":
                self.axis = None


    def run_threaded(self):
        pass


    def poll(self):
        ret = (self.button, self.button_state, self.axis, self.axis_val)
        self.button = None
        self.axis = None
        return ret


def get_js_controller(cfg):
    cont_class = None
    if cfg.CONTROLLER_TYPE == "ps3":
        cont_class = PS3JoystickController
    elif cfg.CONTROLLER_TYPE == "ps3sixad":
        cont_class = PS3JoystickSixAdController
    elif cfg.CONTROLLER_TYPE == "ps4":
        cont_class = PS4JoystickController
    elif cfg.CONTROLLER_TYPE == "nimbus":
        cont_class = NimbusController
    elif cfg.CONTROLLER_TYPE == "xbox":
        cont_class = XboxOneJoystickController
    elif cfg.CONTROLLER_TYPE == "xboxswapped":
        cont_class = XboxOneSwappedJoystickController
    elif cfg.CONTROLLER_TYPE == "wiiu":
        cont_class = WiiUController
    elif cfg.CONTROLLER_TYPE == "F710":
        cont_class = LogitechJoystickController
    elif cfg.CONTROLLER_TYPE == "rc3":
        cont_class = RC3ChanJoystickController
    elif cfg.CONTROLLER_TYPE == "pygame":
        cont_class = PyGamePS4JoystickController
    else:
        raise(Exception("未知のコントローラータイプ: " + cfg.CONTROLLER_TYPE))

    ctr = cont_class(throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                                throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                                steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE,
                                dev_fn=cfg.JOYSTICK_DEVICE_FILE)

    ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
    return ctr


if __name__ == "__main__":
    # XboxOneJoystickController のテスト
    js = XboxOneJoystick('/dev/input/js0')
    js.init()

    while True:
        button, button_state, axis, axis_val = js.poll()
        if button is not None or axis is not None:
            print(button, button_state, axis, axis_val)
        time.sleep(0.1)
