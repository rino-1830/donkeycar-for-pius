class Sombrero:
    """Adam Conwayが開発した、Donkeycarの電源管理とPWMを制御するPi Hat。

    GPIO 26をLOWにしてPWM出力を有効化する必要があります。
    GPIOのモードはコード全体で統一するため、物理ピン37に対応するBOARDモードを使用します。
    """

    def __init__(self):
        try:
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(37, GPIO.OUT)
            GPIO.output(37, GPIO.LOW)
            print("ソンブレロを有効化しました")
        except:
            pass

    def __del__(self):
        try:
            import RPi.GPIO as GPIO

            GPIO.cleanup()
            print("ソンブレロを無効化しました")
        except:
            pass
