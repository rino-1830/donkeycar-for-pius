# -*- coding: utf-8 -*-
"""リアルタイムのメトリクスを外部サーバへ配信する ``Telemetry`` クラス。

作者: @miro (Meir Tseitlin) 2020

注意:
"""
import os
import queue
import time
import json
import logging
import numpy as np
from logging import StreamHandler
from paho.mqtt.client import Client as MQTTClient

logger = logging.getLogger()

LOG_MQTT_KEY = 'log/default'


class MqttTelemetry(StreamHandler):
    """MQTT を用いてテレメトリーを送信するクラス。

    システム各部からテレメトリーを収集し、一定周期でサーバへ送信する。
    テレメトリーのレポートはタイムスタンプ付きでメモリに保持され、サーバへ
    プッシュされるまで保存される。
    """

    def __init__(self, cfg):
        """インスタンスを初期化する。

        Args:
            cfg: Telemetry 用の設定を含むオブジェクト。
        """

        StreamHandler.__init__(self)

        self.PUBLISH_PERIOD = cfg.TELEMETRY_PUBLISH_PERIOD
        self._last_publish = time.time()
        self._telem_q = queue.Queue()
        self._step_inputs = cfg.TELEMETRY_DEFAULT_INPUTS.split(',')
        self._step_types = cfg.TELEMETRY_DEFAULT_TYPES.split(',')
        self._total_updates = 0
        self._donkey_name = os.environ.get('DONKEY_NAME', cfg.TELEMETRY_DONKEY_NAME)
        # 既定値として 'iot.eclipse.org' などの MQTT ブローカーを使用
        self._mqtt_broker = os.environ.get('DONKEY_MQTT_BROKER', cfg.TELEMETRY_MQTT_BROKER_HOST)
        self._topic = cfg.TELEMETRY_MQTT_TOPIC_TEMPLATE % self._donkey_name
        self._use_json_format = cfg.TELEMETRY_MQTT_JSON_ENABLE
        self._mqtt_client = MQTTClient()
        self._mqtt_client.connect(self._mqtt_broker, cfg.TELEMETRY_MQTT_BROKER_PORT)
        self._mqtt_client.loop_start()
        self._on = True
        if cfg.TELEMETRY_LOGGING_ENABLE:
            self.setLevel(logging.getLevelName(cfg.TELEMETRY_LOGGING_LEVEL))
            self.setFormatter(logging.Formatter(cfg.TELEMETRY_LOGGING_FORMAT))
            logger.addHandler(self)

    def add_step_inputs(self, inputs, types):
        """サポートされている入力を登録する。

        すでに登録済みの入力は無視される。

        Args:
            inputs (list[str]): 追加する入力名のリスト。
            types (list[str]): 各入力に対応する型。

        Returns:
            Tuple[list[str], list[str]]: 更新後の入力名と型のリスト。
        """

        for ind in range(0, len(inputs or [])):
            if types[ind] in ['float', 'str', 'int'] and inputs[ind] not in self._step_inputs:
                self._step_inputs.append(inputs[ind])
                self._step_types.append(types[ind])

        return self._step_inputs, self._step_types

    @staticmethod
    def filter_supported_metrics(inputs, types):
        """サポートされているメトリクスだけを抽出する。

        Args:
            inputs (list[str]): 入力名のリスト。
            types (list[str]): 各入力に対応する型。

        Returns:
            Tuple[list[str], list[str]]: サポートされる入力名と型のリスト。
        """

        supported_inputs = []
        supported_types = []
        for ind in range(0, len(inputs or [])):
            if types[ind] in ['float', 'str', 'int']:
                supported_inputs.append(inputs[ind])
                supported_types.append(types[ind])
        return supported_inputs, supported_types

    def report(self, metrics):
        """任意の値を持つ辞書を受け取りレポートを行う。

        Args:
            metrics (dict): 送信するメトリクス。

        Returns:
            int: 秒単位のタイムスタンプ。
        """
        curr_time = int(time.time())

        # 秒単位に丸めた時刻でサンプルを保存
        try:
            self._telem_q.put((curr_time, metrics), block=False)
        except queue.Full:
            pass

        return curr_time

    def emit(self, record):
        """logging モジュールから直接テレメトリーへ出力するためのインターフェース。

        Args:
            record (logging.LogRecord): ログレコード。
        """
        msg = {LOG_MQTT_KEY: self.format(record)}
        self.report(msg)

    @property
    def qsize(self):
        """キューに格納されたテレメトリーデータの数を返す。"""
        return self._telem_q.qsize()

    def publish(self):
        """キュー内のデータを MQTT ブローカーへ送信する。"""

        # パケットを作成
        packet = {}
        while not self._telem_q.empty():
            next_item = self._telem_q.get()
            packet.setdefault(next_item[0], {}).update(next_item[1])

        if not packet:
            return
            
        if self._use_json_format:
            packet = [{'ts': k, 'values': v} for k, v in packet.items()]
            payload = json.dumps(packet)

            try:
                self._mqtt_client.publish(self._topic, payload)
            except Exception as e:
                logger.error(f'ログ {self._topic} の送信エラー: {e}')
        else:
            # ステップごとのメトリクスは最後のタイムスタンプのみ送信
            last_sample = packet[list(packet)[-1]]
            for k, v in last_sample.items():
                if k in self._step_inputs:
                    topic = f'{self._topic}/{k}'
                    
                    try:
                        # 非対応の numpy 型を標準の Python 型に変換
                        if isinstance(v, np.generic):
                            v = np.asscalar(v)

                        self._mqtt_client.publish(topic, v)
                    except TypeError:
                        logger.error(f'値の型 {type(v)} ではトピック "{topic}" を送信できません')
                    except Exception as e:
                        logger.error(f'ログ {topic} の送信エラー: {e}')

            # すべてのログを送信
            for tm, sample in packet.items():
                if LOG_MQTT_KEY in sample:
                    topic = f'{self._topic}/{LOG_MQTT_KEY}'
                    try:
                        self._mqtt_client.publish(topic, sample[LOG_MQTT_KEY])
                    except Exception as e:
                        logger.error(f'ログ {topic} の送信エラー: {e}')

        self._total_updates += 1
        return

    def run(self, *args):
        """Donkey パーツとして値を受け取り、入力キーと組み合わせて保存する。

        Args:
            *args: 各入力に対応する値。

        Returns:
            int: キューに格納されたデータ数。
        """
        assert len(self._step_inputs) == len(args)
        
        # キューへ追加
        record = dict(zip(self._step_inputs, args))
        self.report(record)

        # 一定間隔で送信
        curr_time = time.time()
        if curr_time - self._last_publish > self.PUBLISH_PERIOD and self.qsize > 0:

            self.publish()
            self._last_publish = curr_time
        return self.qsize

    def run_threaded(self, *args):
        """スレッド実行時に値を受け取るエントリポイント。

        Args:
            *args: 各入力に対応する値。

        Returns:
            int: キューに格納されたデータ数。
        """

        assert len(self._step_inputs) == len(args)

        # キューへ追加
        record = dict(zip(self._step_inputs, args))
        self.report(record)
        return self.qsize

    def update(self):
        """バックグラウンドで ``publish`` を周期的に実行する。"""

        logger.info(
            f"テレメトリ MQTT サーバーに接続しました (送信対象: {', '.join(self._step_inputs)})"
        )
        while self._on:
            self.publish()
            time.sleep(self.PUBLISH_PERIOD)

    def shutdown(self):
        """スレッドを停止し MQTT クライアントを終了する。"""

        # スレッドを停止することを示す
        self._on = False
        logger.debug('MQTT テレメトリーを停止します')
        time.sleep(.2)
