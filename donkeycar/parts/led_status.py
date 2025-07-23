"""LEDステータスを制御するためのGPIO操作モジュール。"""

import time
import RPi.GPIO as GPIO

class LED:
    """GPIOピンを利用してLEDを制御するクラス。"""
    def __init__(self, pin):
        """LEDに使用するGPIOピンを設定する。

        Args:
            pin (int): LEDに割り当てるGPIOピン番号。
        """

        self.pin = pin

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT)
        self.blink_changed = 0
        self.on = False

    def toggle(self, condition):
        """条件に応じてLEDをオンまたはオフにする。

        Args:
            condition (bool): Trueで点灯、Falseで消灯。
        """

        if condition:
            GPIO.output(self.pin, GPIO.HIGH)
            self.on = True
        else:
            GPIO.output(self.pin, GPIO.LOW)
            self.on = False

    def blink(self, rate):
        """指定した周期でLEDを点滅させる。

        Args:
            rate (float): 点滅周期(秒)。
        """

        if time.time() - self.blink_changed > rate:
            self.toggle(not self.on)
            self.blink_changed = time.time()

    def run(self, blink_rate):
        """点滅レートに基づいてLEDを制御する。

        Args:
            blink_rate (float): 0で消灯、正で点滅、負で点灯。
        """

        if blink_rate == 0:
            self.toggle(False)
        elif blink_rate > 0:
            self.blink(blink_rate)
        else:
            self.toggle(True)

    def shutdown(self):
        """LEDを消灯してGPIO設定を解除する。"""

        self.toggle(False)
        GPIO.cleanup()


class RGB_LED:
    """RGB LEDをPWM制御するクラス。"""
    def __init__(self, pin_r, pin_g, pin_b, invert_flag=False):
        """RGB LEDに使うGPIOピンを設定する。

        Args:
            pin_r (int): 赤色LEDのGPIOピン番号。
            pin_g (int): 緑色LEDのGPIOピン番号。
            pin_b (int): 青色LEDのGPIOピン番号。
            invert_flag (bool, optional): 信号を反転させるかどうか。デフォルトはFalse。
        """

        self.pin_r = pin_r
        self.pin_g = pin_g
        self.pin_b = pin_b
        self.invert = invert_flag
        print('GPIOをボードモードで設定します')
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin_r, GPIO.OUT)
        GPIO.setup(self.pin_g, GPIO.OUT)
        GPIO.setup(self.pin_b, GPIO.OUT)
        freq = 50
        self.pwm_r = GPIO.PWM(self.pin_r, freq)
        self.pwm_g = GPIO.PWM(self.pin_g, freq)
        self.pwm_b = GPIO.PWM(self.pin_b, freq)
        self.pwm_r.start(0)
        self.pwm_g.start(0)
        self.pwm_b.start(0)
        self.zero = 0
        if( self.invert ):
            self.zero = 100

        self.rgb = (50, self.zero, self.zero)

        self.blink_changed = 0
        self.on = False

    def toggle(self, condition):
        """条件に応じてRGB LEDを点灯または消灯する。

        Args:
            condition (bool): Trueで点灯、Falseで消灯。
        """

        if condition:
            r, g, b = self.rgb
            self.set_rgb_duty(r, g, b)
            self.on = True
        else:
            self.set_rgb_duty(self.zero, self.zero, self.zero)
            self.on = False

    def blink(self, rate):
        """指定した周期でRGB LEDを点滅させる。

        Args:
            rate (float): 点滅周期(秒)。
        """

        if time.time() - self.blink_changed > rate:
            self.toggle(not self.on)
            self.blink_changed = time.time()

    def run(self, blink_rate):
        """点滅レートに基づいてRGB LEDを制御する。

        Args:
            blink_rate (float): 0で消灯、正で点滅、負で点灯。
        """

        if blink_rate == 0:
            self.toggle(False)
        elif blink_rate > 0:
            self.blink(blink_rate)
        else:
            self.toggle(True)

    def set_rgb(self, r, g, b):
        """RGB LEDの色を設定する。

        Args:
            r (int): 赤のデューティ(0-100)。
            g (int): 緑のデューティ(0-100)。
            b (int): 青のデューティ(0-100)。
        """

        r = r if not self.invert else 100 - r
        g = g if not self.invert else 100 - g
        b = b if not self.invert else 100 - b
        self.rgb = (r, g, b)
        self.set_rgb_duty(r, g, b)

    def set_rgb_duty(self, r, g, b):
        """PWMのデューティ比を直接設定する。"""

        self.pwm_r.ChangeDutyCycle(r)
        self.pwm_g.ChangeDutyCycle(g)
        self.pwm_b.ChangeDutyCycle(b)

    def shutdown(self):
        """RGB LEDを消灯してGPIO設定を解除する。"""

        self.toggle(False)
        GPIO.cleanup()


if __name__ == "__main__":
    import time
    import sys
    pin_r = int(sys.argv[1])
    pin_g = int(sys.argv[2])
    pin_b = int(sys.argv[3])
    rate = float(sys.argv[4])
    print('出力ピン', pin_r, pin_g, pin_b, 'レート', rate)

    p = RGB_LED(pin_r, pin_g, pin_b)
    
    iter = 0
    while iter < 50:
        p.run(rate)
        time.sleep(0.1)
        iter += 1
    
    delay = 0.1

    iter = 0
    while iter < 100:
        p.set_rgb(iter, 100-iter, 0)
        time.sleep(delay)
        iter += 1
    
    iter = 0
    while iter < 100:
        p.set_rgb(100 - iter, 0, iter)
        time.sleep(delay)
        iter += 1

    p.shutdown()

