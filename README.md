# gym-test
OpenAI Gymを試すためのリポジトリ

---
## 初期設定

pipenvを用いているので先にそれをインストール

`pip install pipenv`

matplotlibのpyplotを使いたいのでtkinterもインストールする

## 作成したファイル

- policy_right.py

  mountain-carで常に右を選択するという方策を選択
  
- policy_qlearn.py

  mountain-carをQ学習によって学習を進めるテスト
