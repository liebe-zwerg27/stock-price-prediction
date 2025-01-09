import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Any
from collections.abc import Iterable
import pandas as pd

logger = logging.getLogger(__name__)

def visualize(sample_list: np.ndarray,
              result_dir: str,
              file_name: str,
              dim: int,
              point: bool = True,
              pdf: bool = True) -> None:

    if pdf:
        out_format = '.pdf'
    else:
        out_format = '.png'

    if point:
        marker = 'o'
    else:
        marker = None
    if dim == 1:
        plt.figure()
        plt.plot(range(len(sample_list)), sample_list, label=["Train_loss","Valid_loss"], marker=marker, markersize=2)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        if "train_valid" in file_name:
            plt.legend()
        plt.savefig(f'{result_dir}/{file_name}{out_format}')
        plt.close()

def visualize_close(sample_list: str,
                   result_dir: str,
                   file_name: str,
                   dim: int,
                   point: bool = True,
                   pdf: bool = False) -> None:
    if pdf:
        out_format = '.pdf'
    else:
        out_format = '.png'

    if point:
        marker = 'o'
    else:
        marker = None

    if dim == 1:
        plt.figure()
        plt.rcParams["xtick.direction"] = "in"  # x軸の目盛線を内向きへ
        plt.rcParams["ytick.direction"] = "in"  # y軸の目盛線を内向きへ

        sample_list = sample_list.squeeze()
        for i in range(sample_list.shape[1]):
            plt.plot(range(len(sample_list)), sample_list[:, i], marker=marker, markersize=2,
                    color='red' if i == 0 else 'black',  # 最初の系列は赤色(予測値)、2番目の系列は黒色(理論値)
                    label='Predicted value' if i == 0 else 'Theoretical value')

        plt.xlabel("Date")
        plt.ylabel("Price")
        # 凡例を追加
        plt.legend()
        plt.tight_layout()
        # 横軸の範囲を設定
        plt.xlim(0, 200)
        plt.savefig(f'{result_dir}/{file_name}{out_format}')
        plt.close()
        
        # csv出力
        df = pd.DataFrame(sample_list)
        df.to_csv(f'{result_dir}/{file_name}.csv', header=False, index=False)

if __name__ == '__main__':
    pass
