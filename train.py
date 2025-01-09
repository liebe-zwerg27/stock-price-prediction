import os
import argparse
import logging
from config import common_args, Parameters
from utils import dump_params, setup_params, set_logging
from training_module import train_main

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser = common_args(parser)  # コマンドライン引数引数を読み込み
    args = parser.parse_args()
    params = Parameters(**setup_params(vars(args), args.parameters))  # args，run_date，git_revisionなどを追加した辞書を取得
    
    # 結果出力用ファイルの作成
    result_dir = f'result/{params.run_date}'  # 結果出力ディレクトリ
    os.mkdir(result_dir)  # 実行日時を名前とするディレクトリを作成
    dump_params(params, f'{result_dir}')  # パラメータを出力
    
    # matplotlibのログを抑制
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    set_logging(result_dir)  # ログを標準出力とファイルに出力するよう設定

    # trainとtestの評価
    if params.approach == 'lstm':  # LSTM Regression
        train_main(params)
    else:
        raise ValueError(f'Invalid approach: {params.approach}')

if __name__ == "__main__":
    main()