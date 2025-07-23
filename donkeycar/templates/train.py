#!/usr/bin/env python3
"""TensorFlow を使用して Keras モデルを学習するスクリプトです。

基本的な使い方は ``train.py --tubs data/ --model models/mypilot.h5`` です。

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]

Options:
    -h --help              この画面を表示します。
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    train(cfg, tubs, model, model_type, comment)


if __name__ == "__main__":
    main()
