# Donkeycar: Python製自動運転ライブラリ

![ビルド状況](https://github.com/autorope/donkeycar/actions/workflows/python-package-conda.yml/badge.svg?branch=main)
![リント状況](https://github.com/autorope/donkeycar/actions/workflows/superlinter.yml/badge.svg?branch=main)
![リリース](https://img.shields.io/github/v/release/autorope/donkeycar)

[![貢献者一覧](https://img.shields.io/github/contributors/autorope/donkeycar)](#contributors-)
![Issue数](https://img.shields.io/github/issues/autorope/donkeycar)
![Pull Request数](https://img.shields.io/github/issues-pr/autorope/donkeycar?)
![フォーク数](https://img.shields.io/github/forks/autorope/donkeycar)
![スター数](https://img.shields.io/github/stars/autorope/donkeycar)
![ライセンス](https://img.shields.io/github/license/autorope/donkeycar)

![Discord](https://img.shields.io/discord/662098530411741184.svg?logo=discord&colorB=7289DA)

DonkeycarはPython向けのミニマルでモジュール式の自動運転ライブラリです。趣味で取り組む人や学生が素早く実験でき、コミュニティからの貢献を受け入れやすいように開発されています。

#### クイックリンク
* [Donkeycar 更新情報と例](http://donkeycar.com)
* [組み立て手順とソフトウェアのドキュメント](http://docs.donkeycar.com)
* [Discord / チャット](https://discord.gg/PN6kFeA)

![donkeycar](https://github.com/autorope/donkeydocs/blob/master/docs/assets/build_hardware/donkey2.png)

#### Donkey を使いたい場面
* RCカーを自動運転させたい
* [DIY Robocars](http://diyrobocars.com) のような自動運転レースに参加したい
* オートパイロットやマッピング、コンピュータービジョン、ニューラルネットワークを試したい
* センサーのデータ（画像、入力、センサー値）を記録したい
* Web やゲームコントローラー、またはRCコントローラーで車を操作したい
* コミュニティが提供する走行データを活用したい
* 既存のCADモデルを使って改造したい

### 走らせてみよう
Donkey2を組み立てたら車の電源を入れ、http://localhost:8887 にアクセスして運転を開始できます。

### 車の挙動を変更する
Donkeycarは一連のイベントを実行することで制御されます。

```python
#1秒間に10回画像を取得して記録する車両を定義

import time
from donkeycar import Vehicle
from donkeycar.parts.cv import CvCam
from donkeycar.parts.tub_v2 import TubWriter
V = Vehicle()

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

#カメラパーツを追加
cam = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
V.add(cam, outputs=['image'], threaded=True)

#カメラのウォームアップ
while cam.run() is None:
    time.sleep(1)

#画像を記録するためのTubパーツを追加
tub = TubWriter(path='./dat', inputs=['image'], types=['image_array'])
V.add(tub, inputs=['image'], outputs=['num_records'])

#10Hzでループを開始
V.start(rate_hz=10)
```

詳しくは[ホームページ](http://donkeycar.com)、[ドキュメント](http://docs.donkeycar.com)を参照するか、[Discordサーバー](http://www.donkeycar.com/community.html)に参加してください。
