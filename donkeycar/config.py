# -*- coding: utf-8 -*-
"""
2017年9月13日水曜日 21:27:44 作成

@author: wroscoe
"""

import os
import types
from logging import getLogger

logger = getLogger(__name__)


class Config:
    def from_pyfile(self, filename):
        d = types.ModuleType("config")
        d.__file__ = filename
        try:
            with open(filename, mode="rb") as config_file:
                exec(compile(config_file.read(), filename, "exec"), d.__dict__)
        except IOError as e:
            e.strerror = "設定ファイルを読み込めません (%s)" % e.strerror
            raise
        self.from_object(d)
        return True

    def from_object(self, obj):
        for key in dir(obj):
            if key.isupper():
                setattr(self, key, getattr(obj, key))

    def __str__(self):
        result = []
        for key in dir(self):
            if key.isupper():
                result.append((key, getattr(self, key)))
        return str(result)

    def show(self):
        for attr in dir(self):
            if attr.isupper():
                print(attr, ":", getattr(self, attr))


def load_config(config_path=None, myconfig="myconfig.py"):
    if config_path is None:
        import __main__ as main

        main_path = os.path.dirname(os.path.realpath(main.__file__))
        config_path = os.path.join(main_path, "config.py")
        if not os.path.exists(config_path):
            local_config = os.path.join(os.path.curdir, "config.py")
            if os.path.exists(local_config):
                config_path = local_config

    logger.info(f"設定ファイルを読み込み中: {config_path}")
    cfg = Config()
    cfg.from_pyfile(config_path)

    # 同じパスに存在する任意の myconfig.py を探す。
    personal_cfg_path = config_path.replace("config.py", myconfig)
    if os.path.exists(personal_cfg_path):
        logger.info(f"個人用設定 {myconfig} から上書きを読み込みます")
        personal_cfg = Config()
        personal_cfg.from_pyfile(personal_cfg_path)
        cfg.from_object(personal_cfg)
    else:
        logger.warning(f"個人用設定が見つかりません: {personal_cfg_path}")

    return cfg
