#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2017年6月25日(日) 11:07:48 作成

@author: wroscoe
"""


class Memory:
    """
    キーと値のペアを保存するための簡易クラス。
    """

    def __init__(self, *args, **kw):
        self.d = {}

    def __setitem__(self, key, value):
        if type(key) is str:
            self.d[key] = value
        else:
            if type(key) is not tuple:
                key = tuple(key)
                value = tuple(key)
            for i, k in enumerate(key):
                self.d[k] = value[i]

    def __getitem__(self, key):
        if type(key) is tuple:
            return [self.d[k] for k in key]
        else:
            return self.d[key]

    def update(self, new_d):
        self.d.update(new_d)

    def put(self, keys, inputs):
        if len(keys) > 1:
            for i, key in enumerate(keys):
                try:
                    self.d[key] = inputs[i]
                except IndexError as e:
                    error = str(e) + " キーに問題があります: " + str(key)
                    raise IndexError(error)

        else:
            self.d[keys[0]] = inputs

    def get(self, keys):
        result = [self.d.get(k) for k in keys]
        return result

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()

    def items(self):
        return self.d.items()
