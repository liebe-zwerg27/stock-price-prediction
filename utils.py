"""便利な関数群"""
import torch
import subprocess
import logging
import json
from datetime import datetime
import os
from dataclasses import asdict
import numpy as np

logger = logging.getLogger(__name__)


def setup_params(args_dict, path=None):
    """
    コマンドライン引数などの辞書を受け取り，実行時刻，Gitのリビジョン，jsonファイルからの引数と結合した辞書を返す．
    """
    run_date = datetime.now()

    param_dict = {}
    if path:
        param_dict = json.load(open(path, 'r'))  # jsonからパラメータを取得
    param_dict.update({'args': args_dict})  # コマンドライン引数を上書き
    param_dict.update({'run_date': run_date.strftime('%Y%m%d_%H%M%S')})  # 実行時刻を上書き
    return param_dict


def dump_params(params, outdir, partial=False):
    """
    データクラスで定義されたパラメータをjson出力する関数
    """
    params_dict = asdict(params)  # デフォルトパラメータを取得
    if os.path.exists(f'{outdir}/parameters.json'):
        raise Exception('"parameters.json" is already exist. ')
    if partial:
        del params_dict['args']  # jsonからし指定しないキーを削除
        del params_dict['run_date']  # jsonからし指定しないキーを削除
        del params_dict['device']  # jsonからし指定しないキーを削除
    with open(f'{outdir}/parameters.json', 'w') as f:
        json.dump(params_dict, f, indent=4)  # デフォルト設定をファイル出力


def set_logging(result_dir):
    """
    ログを標準出力とファイルに書き出すよう設定する関数
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # ログレベル
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # ログのフォーマット
    # 標準出力へのログ出力設定
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # 出力ログレベル
    handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(handler)
    # ファイル出力へのログ出力設定
    file_handler = logging.FileHandler(f'{result_dir}/log.log', 'w')  # ログ出力ファイル
    file_handler.setLevel(logging.DEBUG)  # 出力ログレベル
    file_handler.setFormatter(formatter)  # フォーマットを指定
    logger.addHandler(file_handler)
    return logger


def get_gpu_info(nvidia_smi_path='nvidia-smi', no_units=True):
    """
    空いているGPUの番号を取得します。
    """
    try:
        # GPUが利用可能かチェック
        if torch.cuda.is_available():
            nu_opt = '' if not no_units else ',nounits'
            cmd = f'{nvidia_smi_path} --query-gpu=index,memory.used --format=csv,noheader{nu_opt}'
            output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
            gpu_info = [line.split(', ') for line in output.strip().split('\n')]
            
            # メモリ使用率が最も低いGPUを選択
            min_gpu_index = min(gpu_info, key=lambda x: int(x[1]))[0]

            return f'cuda:{min_gpu_index}'
        else:
            return 'cpu'
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # エラーハンドリング
        print(f"Error: {e}")
        return 'cpu'


def get_device():
    """
    実行環境のデバイス(GPU or CPU) を取得
    """
    device = torch.device(get_gpu_info())

    return device


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='lstm.nn', trace_func=logger.info):
        """
        Early stops the training if validation loss doesn't improve after a given patience.

        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decrease.
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
