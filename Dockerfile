FROM python:3.6

WORKDIR /app

# tensorflow（CPUのみ版）で donkey をインストール
ADD ./setup.py /app/setup.py
ADD ./README.md /app/README.md
RUN pip install -e .[tf]

# テスト用の依存関係を取得
RUN pip install -e .[dev]

# パスワードなしで実行するために Jupyter Notebook を設定
RUN pip install jupyter notebook
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password = ''">>/root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''">>/root/.jupyter/jupyter_notebook_config.py

# コードが変更されても pip のインストールが更新されないように、インストール後にアプリディレクトリ全体を追加
ADD . /app

# Jupyter Notebook を起動
CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888 --allow-root  --notebook-dir=/app/notebooks

# donkeycar のポート
EXPOSE 8887

# Jupyter Notebook のポート
EXPOSE 8888