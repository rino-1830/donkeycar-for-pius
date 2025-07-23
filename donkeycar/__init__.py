import logging
import os
import sys

from pkg_resources import get_distribution
from pyfiglet import Figlet

__version__ = get_distribution("donkeycar").version

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())

f = Figlet(font="speed")


print(f.renderText("DonkeyCar"))
print(f"donkey v{__version__} を使用しています...")

if sys.version_info.major < 3 or sys.version_info.minor < 6:
    msg = f"Donkeyは Python 3.6 以上が必要です。現在のバージョンは {sys.version} です"
    raise ValueError(msg)

# CPython のデフォルトの再帰制限は小さすぎるため拡張する。
sys.setrecursionlimit(10**5)

from . import config, contrib, utils
from .config import load_config
from .memory import Memory
from .vehicle import Vehicle
