#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2017年6月25日(日) 10:44:24 作成

@author: wroscoe
"""

import logging
import time
import traceback
from threading import Thread

import numpy as np
from prettytable import PrettyTable

from .memory import Memory

logger = logging.getLogger(__name__)


class PartProfiler:
    def __init__(self):
        self.records = {}

    def profile_part(self, p):
        self.records[p] = {"times": []}

    def on_part_start(self, p):
        self.records[p]["times"].append(time.time())

    def on_part_finished(self, p):
        now = time.time()
        prev = self.records[p]["times"][-1]
        delta = now - prev
        thresh = 0.000001
        if delta < thresh or delta > 100000.0:
            delta = thresh
        self.records[p]["times"][-1] = delta

    def report(self):
        logger.info("Part Profile Summary: (times in ms)")
        pt = PrettyTable()
        field_names = ["part", "max", "min", "avg"]
        pctile = [50, 90, 99, 99.9]
        pt.field_names = field_names + [str(p) + "%" for p in pctile]
        for p, val in self.records.items():
            # 初期化にかかった時間や割り込みによる不完全な値を除くため
            # 最初と最後の計測結果を取り除く
            arr = val["times"][1:-1]
            if len(arr) == 0:
                continue
            row = [
                p.__class__.__name__,
                "%.2f" % (max(arr) * 1000),
                "%.2f" % (min(arr) * 1000),
                "%.2f" % (sum(arr) / len(arr) * 1000),
            ]
            row += ["%.2f" % (np.percentile(arr, p) * 1000) for p in pctile]
            pt.add_row(row)
        logger.info("\n" + str(pt))


class Vehicle:
    def __init__(self, mem=None):
        if not mem:
            mem = Memory()
        self.mem = mem
        self.parts = []
        self.on = True
        self.threads = []
        self.profiler = PartProfiler()

    def add(self, part, inputs=[], outputs=[], threaded=False, run_condition=None):
        """
        車両のドライブループにパーツを追加するメソッド。

        Parameters
        ----------
            part: class
                run() メソッドを持つ Donkey のパーツ
            inputs : list
                メモリから取得するチャンネル名
            outputs : list
                メモリへ保存するチャンネル名
            threaded : boolean
                別スレッドで実行する場合は True
            run_condition : str
                実行するかどうかを判定するメモリキー
        """
        assert type(inputs) is list, "inputs is not a list: %r" % inputs
        assert type(outputs) is list, "outputs is not a list: %r" % outputs
        assert type(threaded) is bool, "threaded is not a boolean: %r" % threaded

        p = part
        logger.info("Adding part {}.".format(p.__class__.__name__))
        entry = {}
        entry["part"] = p
        entry["inputs"] = inputs
        entry["outputs"] = outputs
        entry["run_condition"] = run_condition

        if threaded:
            t = Thread(target=part.update, args=())
            t.daemon = True
            entry["thread"] = t

        self.parts.append(entry)
        self.profiler.profile_part(part)

    def remove(self, part):
        """
        登録済みのパーツをリストから取り除く
        """
        self.parts.remove(part)

    def start(self, rate_hz=10, max_loop_count=None, verbose=False):
        """
        車両のメインドライブループを開始する。

        これは車両のメインスレッドで、スレッド化されたパーツを起動した後、
        各パーツを実行してメモリを更新し続ける無限ループに入る。

        Parameters
        ----------

        rate_hz : int
            ドライブループが目指す周波数。処理が重いと実際の周波数は
            これより低くなる場合がある。
        max_loop_count : int
            ループの実行回数上限。すべてのパーツが動作するかテストする際に使用。
        verbose: bool
            デバッグ出力をシェルに表示するかどうか
        """

        try:
            self.on = True

            for entry in self.parts:
                if entry.get("thread"):
                    # 更新用スレッドを開始
                    entry.get("thread").start()

            # パーツが温まるまで待機
            logger.info("Starting vehicle at {} Hz".format(rate_hz))

            loop_start_time = time.time()
            loop_count = 0
            while self.on:
                start_time = time.time()
                loop_count += 1

                self.update_parts()

                # max_loop_count を超えたらループを終了
                if max_loop_count and loop_count >= max_loop_count:
                    self.on = False
                else:
                    sleep_time = 1.0 / rate_hz - (time.time() - start_time)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    else:
                        # ループ周波数を維持できなかった場合に警告を表示
                        if verbose:
                            logger.info(
                                "WARN::Vehicle: jitter violation in vehicle loop "
                                "with {0:4.0f}ms".format(abs(1000 * sleep_time))
                            )

                    if verbose and loop_count % 200 == 0:
                        self.profiler.report()

            loop_total_time = time.time() - loop_start_time
            logger.info(
                f"Vehicle executed {loop_count} steps in {loop_total_time} seconds."
            )

            return loop_count, loop_total_time

        except KeyboardInterrupt:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            self.stop()

    def update_parts(self):
        """
        すべてのパーツを順に実行する
        """
        for entry in self.parts:
            run = True
            # 実行条件があれば確認
            if entry.get("run_condition"):
                run_condition = entry.get("run_condition")
                run = self.mem.get([run_condition])[0]

            if run:
                # パーツを取得
                p = entry["part"]
                # パーツの実行時間計測開始
                self.profiler.on_part_start(p)
                # メモリから入力を取得
                inputs = self.mem.get(entry["inputs"])
                # パーツを実行
                if entry.get("thread"):
                    outputs = p.run_threaded(*inputs)
                else:
                    outputs = p.run(*inputs)

                # 出力をメモリに保存
                if outputs is not None:
                    self.mem.put(entry["outputs"], outputs)
                # パーツの実行時間計測終了
                self.profiler.on_part_finished(p)

    def stop(self):
        logger.info("車両とそのパーツをシャットダウンします...")
        for entry in self.parts:
            try:
                entry["part"].shutdown()
            except AttributeError:
                # shutdown メソッドが無い場合は無視する
                pass
            except Exception as e:
                logger.error(e)

        self.profiler.report()
