# ReinforcementLearning-Prac
OpenAI Gymで強化学習を試すためのリポジトリ

---
## 初期設定

pipenvを用いているので先にそれをインストール

`pip install pipenv`

matplotlibのpyplotを使いたいのでtkinterもインストールする

`pipenv install`でパッケージをインストール

atariの環境で学習を行いたい場合はOSに応じて

Linux：`pip install gym[atari]`

Windows：`pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py`

を更に実行する

その後、`pipenv shell`でshellに入る

## 作成したファイル

- policy_right.py

  mountain-carで常に右を選択するという方策を選択

- policy_qlearn.py

  mountain-carをQ学習によって学習を進める

- show_graph.py

  出力されたrewards.csvからグラフを作成する。ファイルパスを引数に渡して実行できる