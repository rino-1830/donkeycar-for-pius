# -*- coding: utf-8 -*-

import time

class Lambda:
    """関数をDonkeyパーツとしてラップするクラス。"""

    def __init__(self, f):
        """使用する関数を受け取る。

        Args:
            f: 実行する関数。
        """
        self.f = f
        
    def run(self, *args, **kwargs):
        return self.f(*args, **kwargs)
    
    def shutdown(self):
        return

class TriggeredCallback:
    """トリガされるとコールバックを実行するパーツ。"""

    def __init__(self, args, func_cb):
        """コールバック実行に必要な引数と関数を渡す。

        Args:
            args: コールバック関数へ渡す引数。
            func_cb: 実行するコールバック関数。
        """
        self.args = args
        self.func_cb = func_cb

    def run(self, trigger):
        if trigger:
            self.func_cb(self.args)

    def shutdown(self):
        return

class DelayedTrigger:
    """一定時間後に真を返すトリガ。"""

    def __init__(self, delay):
        """遅延時間を設定する。

        Args:
            delay: トリガが有効になるまでのループ回数。
        """
        self.ticks = 0
        self.delay = delay

    def run(self, trigger):
        if self.ticks > 0:
            self.ticks -= 1
            if self.ticks == 0:
                return True

        if trigger:
            self.ticks = self.delay

        return False

    def shutdown(self):
        return


class PIDController:
    """PID 計算を行い制御値を返すクラス。

    dt(経過時間)とプロセス変数の現在値に基づき計算を行う。
    詳細は https://github.com/chrisspen/pid_controller/blob/master/pid_controller/pid.py を参照。
    """

    def __init__(self, p=0, i=0, d=0, debug=False):
        """PID コントローラを初期化する。

        Args:
            p: 比例ゲイン。
            i: 積分ゲイン。
            d: 微分ゲイン。
            debug: True の場合デバッグ出力を行う。
        """

        # ゲインを初期化する
        self.Kp = p
        self.Ki = i
        self.Kd = d

        # コントローラが目標とする値
        self.target = 0

        # Δt用の変数を初期化する
        self.prev_tm = time.time()
        self.prev_err = 0
        self.error = None
        self.totalError = 0

        # 出力値を初期化する
        self.alpha = 0

        # デバッグフラグ (True でコンソール出力)
        self.debug = debug

    def run(self, err):
        """誤差値から制御値を計算する。"""

        curr_tm = time.time()

        self.difError = err - self.prev_err

        # 経過時間を計算する
        dt = curr_tm - self.prev_tm

        # 出力変数を初期化する
        curr_alpha = 0

        # 比例項を加算する
        curr_alpha += -self.Kp * err

        # 積分項を加算する
        curr_alpha += -self.Ki * (self.totalError * dt)

        # 微分項を加算する (0除算を避ける)
        if dt > 0:
            curr_alpha += -self.Kd * (self.difError / float(dt))

        # 次回ループ用の値を保持する
        self.prev_tm = curr_tm
        self.prev_err = err
        self.totalError += err

        # 出力を更新する
        self.alpha = curr_alpha

        if self.debug:
            print('PIDの誤差値:', round(err, 4))
            print('PID出力:', round(curr_alpha, 4))

        return curr_alpha


def twiddle(evaluator, tol=0.001, params=3, error_cmp=None, initial_guess=None):
    """座標降下法によるパラメータチューニングアルゴリズム。

    詳細は https://github.com/chrisspen/pid_controller/blob/master/pid_controller/pid.py
    および https://en.wikipedia.org/wiki/Coordinate_descent を参照。

    Args:
        evaluator: 数値パラメータ群を受け取り誤差を返す呼び出し可能オブジェクト。
        tol: 許容誤差。値が小さいほどチューニングが厳密になる。
        params: 調整するパラメータの数。
        error_cmp: 2つの誤差を受け取り、最初の値がより良い場合に True を返す関数。
        initial_guess: チューニング開始時のパラメータ。

    Returns:
        list: チューニングされたパラメータ。
    """

    def _error_cmp(a, b):
        # a が b より 0 に近ければ True を返す
        return abs(a) < abs(b)
        
    if error_cmp is None:
        error_cmp = _error_cmp

    if initial_guess is None:
        p = [0]*params
    else:
        p = list(initial_guess)
    dp = [1]*params
    best_err = evaluator(*p)
    steps = 0
    while sum(dp) > tol:
        steps += 1
        print('ステップ:', steps, '許容誤差:', tol, '最良誤差:', best_err)
        for i, _ in enumerate(p):

            # まずはパラメータを増加させてみる
            p[i] += dp[i]
            err = evaluator(*p)

            if error_cmp(err, best_err):
                # 増加により誤差が減ったので記録し dp の範囲を広げる
                best_err = err
                dp[i] *= 1.1
            else:
                # それ以外では誤差が増えたので元に戻して dp を減少させる
                p[i] -= 2.*dp[i]
                err = evaluator(*p)

                if error_cmp(err, best_err):
                    # 減少により誤差が減ったので記録し dp の範囲を広げる
                    best_err = err
                    dp[i] *= 1.1

                else:
                    # それでも改善しない場合はパラメータを戻して dp の範囲を狭める
                    p[i] += dp[i]
                    dp[i] *= 0.9
                
    return p
