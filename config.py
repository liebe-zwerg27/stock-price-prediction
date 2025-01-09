"""
プロジェクト内のパラメータを管理するためのモジュール．
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=False)
class Parameters:
    """
    プログラム全体を通して共通のパラメータを保持するクラス．
    """
    args: dict = field(default_factory=lambda: {})  # コマンドライン引数
    run_date: str = ''  # 実行時の時刻
    git_revision: str = ''  # 実行時のプログラムのGitのバージョン
    
    dataset: str = 'Yfinance-AAPL'  # 使用するデータセットのID 
    prediction_distance: int = 1  # どれだけ先の品質を推定するかを指定
    scaler_load_path: str = None  # ロードするscalerのパス
    start_date: str = '2020-01-01' # ロードする株式データの始点
    end_date: str = '2024-01-01' # ロードする株式データの終点
    device: str = ''  # デバイス
    seed: int = 42  # 乱数シード

    # dataとlabelの対象column名
    target_data_column: List[str] = field(default_factory=lambda: ["Open", "Close", "High", "Low"])  # 説明変数のカラム名
    target_label_column: List[str] = field(default_factory=lambda:["Close"])  # 目的変数のカラム名
    
    # 訓練データパラメータ
    batch_size: int = 1  # ミニバッチ作成のためのバッチサイズ(1,2,4,8,16,・・・,1024,2048,4096）

    # LSTMパラメータ
    lstm_lr: float = 0.01  # 0以上の浮動小数点数．学習率．
    num_layers: int = 4  # 繰り返しレイヤーの数
    dropout:  float = 0  # 非ゼロの場合、最後の層を除く各LSTM層の出力にドロップアウト層を導入
    lstm_epochs_num: int = 10000  # LSTMエポック数
    hidden_size: int = 128  # 隠れ層のサイズ

    # cross validation パラメータ
    cross_validation: bool = True  # Trueならクロスバリデーション。基本的にTrue。
    n_splits: int = 1  # 分割数。将来的に拡大可能。現在は1のみに対応。

    # early_stopping パラメータ
    earlystop: bool = True
    patience: int = 50  # 指定回数のlossの最小値の更新ができなかった時ストップ
    delta: float = -0.1  # 更新検出量。閾値。マイナスはstopをかかりにくくする方向に作用。

    # data_load設定
    history_window_length: int = 10  # スライディングウィンドウサイズ
    
    # 前処理設定
    norm_mode: str = 'Standard'  # 標準化モード。"MinMax"で正規化も選択可能

    approach: str = 'lstm'  # 使用する手法

def common_args(parser):
    """
    コマンドライン引数を定義する関数．
    """
    parser.add_argument("-p", "--parameters", help="パラメータ設定ファイルのパスを指定．デフォルトはNone", type=str, default=None)
    parser.add_argument('-s', '--save', type=str, default='./result/', help='モデルを保存するディレクトリ')
    return parser

if __name__ == "__main__":
    pass
