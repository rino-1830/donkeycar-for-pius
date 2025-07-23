#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""リアルタイムのCPU・メモリおよび実行周波数を解析するパフォーマンスモニター。

作者:
    @miro (Meir Tseitlin) 2020

備考:
"""
import time
import psutil


class PerfMonitor:
    """パフォーマンス計測を行うクラス。"""

    def __init__(self, cfg):
        """インスタンスを初期化する。

        Args:
            cfg: 設定オブジェクト。``DRIVE_LOOP_HZ``を参照する。
        """

        self.STATS_BUFFER_SIZE = 10
        self._calc_buffer = [cfg.DRIVE_LOOP_HZ for i in range(self.STATS_BUFFER_SIZE)]
        self._runs_counter = 0
        self._last_calc_time = time.time()
        self._on = True
        self._update_metrics()
        print("パフォーマンスモニターを開始しました。")

    def _update_metrics(self):
        """CPU とメモリの使用率を更新する。"""
        self._mem_percent = psutil.virtual_memory().percent
        self._cpu_percent = psutil.cpu_percent()

    def update(self):
        """メトリクスを定期的に更新し続ける。"""
        while self._on:
            self._update_metrics()
            time.sleep(2)

    def shutdown(self):
        """スレッドの停止を指示する。"""
        # スレッドを停止させることを指示する
        self._on = False
        print('パフォーマンスモニターを停止します')
        time.sleep(.2)

    def run_threaded(self):
        """実行時に呼び出され、実際のループ周波数を計算する。

        Returns:
            tuple: ``(cpu_percent, mem_percent, vehicle_frequency)`` を返す。
        """

        # 実際の周波数を計算する
        curr_time = time.time()
        if curr_time - self._last_calc_time > 1:
            self._calc_buffer[int(curr_time) % self.STATS_BUFFER_SIZE] = self._runs_counter
            self._runs_counter = 0
            self._last_calc_time = curr_time

        self._runs_counter += 1

        vehicle_frequency = float(sum(self._calc_buffer)) / self.STATS_BUFFER_SIZE

        return self._cpu_percent, self._mem_percent, vehicle_frequency
