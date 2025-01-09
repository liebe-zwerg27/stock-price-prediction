# Stock Price Prediction

yfinanceライブラリを用いて任意の株価データを取得し、LSTM（Long Short-Term Memory）を用いて将来の株価を予測するプロジェクトです。

## ディレクトリ構成

```
stock-price-prediction/
├── README.md                # このファイル
├── config.py                # パラメータ設定
├── dataset/                 # ダウンロードした株価データ (分割されたCSVファイル)
│   ├── dataset_part_1.csv
│   ├── ...
│   └── dataset_part_10.csv
├── eval.py                  # 評価用スクリプト
├── input_data.py            # データセットのロードと前処理
├── models.py                # LSTMモデル定義
├── parameters.json          # パラメータ設定ファイル
├── requirements.txt         # 必要なPythonパッケージ
├── result/                  # 結果出力ディレクトリ
├── train.py                 # メインのスクリプト
├── training_module.py       # 学習処理、評価、可視化
├── utils.py                 # ユーティリティ関数群
└── visualise.py             # 可視化関数
```

## ブランチ構成について

本プロジェクトでは、開発用とgithubへの公開用で異なるブランチを使用しています。

*   **公開用ブランチ (`pub`):** このブランチはGitHubで公開するためのブランチです。開発用ブランチ内の個人情報や機密情報を含まないように構成されています。
*   **開発用ブランチ (非公開):** 実際の開発作業はこのブランチで行っています。所属する研究室の方針に基づき、研究に深く関連する情報が含まれる可能性があるこのブランチはGitHubには公開していません。

## 概要

このプロジェクトでは、以下の手順で株価予測を行います。

1.  **データ取得:** `input_data.py`でyfinanceを用いて指定されたティッカーシンボルの株価データをダウンロードし、`dataset/`ディレクトリに10分割されたCSVファイルとして保存します。これにより、データの前処理や分割を効率的に行えるようにしています。
2.  **データ前処理:** ダウンロードしたデータは`input_data.py`内で標準化（StandardScalerを使用）が行われます。これにより、データのスケールを揃え、モデルの学習を安定させます。
3.  **モデル学習:** `training_module.py`でLSTMモデルの学習を行います。Early Stoppingを導入しており、過学習を抑制します。
4.  **予測と評価:** 学習済みモデルを用いて訓練データとテストデータに対する予測を行い、結果を可視化します。

## 実行方法

### 結果出力用ディレクトリを作成
```bash
mkdir result
```

### 必要なパッケージのインストール

以下のコマンドで必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

`requirements.txt`の内容は以下の通りです。

```
cycler==0.11.0
fonttools==4.29.1
joblib==1.1.0
kiwisolver==1.3.2
matplotlib==3.5.1
numpy==1.21.4
packaging==21.3
Pillow==9.0.1
pyparsing==3.0.7
python-dateutil==2.8.2
scikit-learn==0.24.1
six==1.16.0
torch==1.10.2
typing-extensions==3.10.0.2
```

### パラメータ設定

`parameters.json`で各種パラメータを設定します。以下は使用例です。

```json
{
    "dataset": "Yfinance-AAPL",
    "history_window_length": 30,
    "prediction_distance": 1,
    "target_data_column":["Open", "Close", "High", "Low"],
    "target_label_column":["Close"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "seed": 5,
    "batch_size": 128,
    "lstm_lr": 0.0001,
    "num_layers": 2,
    "dropout": 0,
    "lstm_epochs_num": 20000,
    "hidden_size": 128,
    "cross_validation": true,
    "n_splits": 1,
    "earlystop": true,
    "patience": 30,
    "delta": 0,
    "approach": "lstm"
}
```

主要なパラメータの説明：

*   `dataset`: 取得する株価のティッカーシンボル (例: `Yfinance-AAPL`)。`Yfinance-`に続けてティッカーシンボルを指定します。
*   `history_window_length`: 部分時系列を設定します。
*   `prediction_distance`: 何日後の株価を予測するか (例: 1は翌日の株価を予測)
*   `target_data_column`: 入力に使用するカラム名。複数選択可能。 (例: `["Close"]`は終値のみを使用)
*   `target_label_column`: 予測対象のカラム名。複数選択不可。 (例: `["Close"]`は終値を予測)
*   `start_date`, `end_date`: データ取得期間 (YYYY-MM-DD形式)
*   `batch_size`: バッチサイズ。学習時のミニバッチのサイズを指定します。
*   `lstm_lr`: LSTMの学習率。
*   `num_layers`: LSTMのレイヤー数。
*   `dropout`: ドロップアウト率。過学習を抑制するために使用します。
*   `lstm_epochs_num`: 学習エポック数。
*   `hidden_size`: LSTMの隠れ層のサイズ。
*   `cross_validation`: クロスバリデーションを行うかどうか。現在は`n_splits=1`のみ対応しています。
*   `n_splits`: 分割数。現在は1のみに対応。
*   `earlystop`: Early Stoppingを行うかどうか。
*   `patience`: Early Stoppingの patience。損失が改善しないエポック数を指定します。
*   `delta`: Early Stoppingにおける損失の改善幅の閾値を指定します。
*   `approach`: 使用する手法 (現在は`lstm`のみ)。

### 学習の実行

以下のコマンドで学習を実行します。

```bash
python train.py -p parameters.json
```

`-p`オプションでパラメータファイルを指定します。

### 結果の確認

学習結果は`result/{実行日時}`ディレクトリに保存されます。

*   `output_data_png/`: 学習結果のグラフ。
    *   `train_result.png`: 訓練データに対する予測結果と正解値の比較グラフ。
    *   `test_result.png`: テストデータに対する予測結果と正解値の比較グラフ。
    *   `lstm_train_loss.png`: 訓練損失の推移グラフ。
    *   `lstm_valid_loss.png`: バリデーション損失の推移グラフ。
    *   `lstm_train_valid_loss.png`: 訓練損失とバリデーション損失の比較グラフ。
*   `scalers/`: 標準化に使用したスケーラー (`scalers_data.pkl`)。

## 各モジュールの説明

*   `train.py`: メインスクリプト。
*   `config.py`: パラメータを定義する。
*   `input_data.py`: データセットのロードと前処理を行う。yfinanceからのデータ取得、データ分割、標準化等。
*   `models.py`: LSTMモデルの定義。
*   `training_module.py`: 学習、評価、結果の可視化。Early Stoppingを含む。
*   `utils.py`: ユーティリティ関数群。
*   `visualise.py`: グラフ描画等。

## 今後の課題

*   他のモデルの実装
*   クロスバリデーションの拡張 (n_splits > 1への対応)による汎化性能の評価
*   ハイパーパラメータチューニングの自動化
*   任意の期間における株価予測の可視化
*   複数の株価データの同時予測


## 実行環境

*   OS: macOS Sequoia
*   Python: 3.9.9
