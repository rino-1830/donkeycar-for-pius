"""プラットフォーム依存の情報を取得するユーティリティ群。"""

import os
import platform

#
# ハードウェアやOSを判定するための関数群
#
def is_mac():
    """macOS かどうかを判定する。"""
    return "Darwin" == platform.system()

#
# Tegra チップIDが存在する場合に読み取る
#
def read_chip_id() -> str:
    """Tegra チップIDを読み取る。

    Tegra 以外のプラットフォームでは空文字を返す。
    """
    try:
        with open("/sys/module/tegra_fuse/parameters/tegra_chip_id", "r") as f:
            return next(f)
    except FileNotFoundError:
        pass
    return ""

_chip_id = None

def is_jetson() -> bool:
    """このプラットフォームが Jetson かどうかを判定する。"""
    global _chip_id
    if _chip_id is None:
        _chip_id = read_chip_id()
    return _chip_id != ""
