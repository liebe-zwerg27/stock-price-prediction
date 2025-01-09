"""
データセットの準備など，データ入力関連のモジュール
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import logging
import os
from config import Parameters
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from itertools import chain
import random
import yfinance as yf

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    """
    ベースとなるデータセットクラス．
    すべてのデータセットに共通の関数などはここに記述する．
    """

    def extract_data_and_labels_pandas(self, list_split_df_source: List[pd.DataFrame], window_size: int, prediction_distance: int, target_data: List[str], target_label: List[str]) -> Union[List[List], List[List]]:
        """
        Pandas DataFrameから学習用dataとlabelのListを作成する関数
        """

        data_list = []
        label_list = []

        for df in list_split_df_source:
            # Add missing columns in target_data and target_label if they don't exist
            for column in target_data:
                if column not in df.columns:
                    df[column] = 0  # Add missing column and fill with 0s
            for column in target_label:
                if column not in df.columns:
                    df[column] = 0  # Add missing column and fill with 0s

            # Calculate the number of segments
            num_segments = len(df) - window_size - prediction_distance + 1

            # Extract data and labels
            for i in range(num_segments):
                data_segment = df.iloc[i:i+window_size][target_data].values.astype(np.float64)
                label_segment = df.iloc[i+window_size:i+window_size+prediction_distance][target_label].values.astype(np.float64)[-1]
                label_segment = np.array([label_segment])  # 2次元配列に変換
                data_list.append(data_segment)
                label_list.append(label_segment)

        return data_list, label_list
    

    def standardize_data(self, data_list_concatenated: np.array, label_list_concatenated: np.array, data_list: List[np.array], label_list: List[np.array], params: 'Parameters', mode: str = 'train', target_data_column: List[str] = None, target_label_column: List[str] = None) -> Tuple[List[np.array], List[np.array], Dict[str, Any]]:
        """
        data, labelのデータを標準化する
        """

        # 各次元のデータの分割
        data_dimensions = [data_list_concatenated[:, i] for i in range(len(target_data_column))]
        label_dimensions = [label_list_concatenated[:, i] for i in range(len(target_label_column))]

        scalers = {}

        # スケーラーの選択
        if params.norm_mode == "Standard":
            scaler_class = StandardScaler
        elif params.norm_mode == "MinMax":
            scaler_class = MinMaxScaler
        else:
            pass

        if mode == 'train':
            # 標準化スケーラーを辞書に保存
            for i, column in enumerate(target_data_column):
                scaler = scaler_class()
                scaler.fit(data_dimensions[i].reshape(-1, 1))
                scalers[column] = scaler
                globals()[f"sc_{column}"] = scaler  # デバッグ用に値を格納

            for i, column in enumerate(target_label_column):
                scaler = scaler_class()
                scaler.fit(label_dimensions[i].reshape(-1, 1))
                scalers[column+"_label"] = scaler
                globals()[f"sc_{column}_label"] = scaler  # デバッグ用に値を格納

            scalers.pop('null', None)

            # スケーラーの情報をファイルに保存
            if os.path.exists(f'{params.scaler_load_path}'):
                with open(f'{params.scaler_load_path}', 'rb') as f:
                    scalers_old = pickle.load(f)
                scalers.update(scalers_old)

            with open(f'result/{params.run_date}/scalers/scalers_data.pkl', 'wb') as f:
                pickle.dump(scalers, f)

        else:
            # スケーラーの情報をファイルから読み込む
            with open(f'result/{params.run_date}/scalers/scalers_data.pkl', 'rb') as f:
                scalers = pickle.load(f)

        # 標準化の適用
        for i, column in enumerate(target_data_column):
            if column in scalers.keys():  # columnがscalersのキーに含まれている場合
                scaler = scalers[column]
                for x in data_list:
                    x[:, i] = scaler.transform(x[:, i].reshape(-1, 1)).flatten()

        for i, column in enumerate(target_label_column):
            if column in scalers.keys():  # columnがscalersのキーに含まれている場合
                scaler = scalers[column+"_label"]
                for x in label_list:
                    x[:, i] = scaler.transform(x[:, i].reshape(-1, 1)).flatten()

        logger.info('Standardization complete.')
        return data_list, label_list

class YfinanceTraceDataset(BaseDataset):

    def __init__(self,
                 params,
                 target: str,
                 split_ids: List[int] = None,
                 mode: str = None) -> None:
            
        num_target_data_column = len(params.target_data_column)
        self.params = params
        self.file_ids = split_ids  # 使用するファイルのID
        self.target = target
        self.in_dim = num_target_data_column  # 入力次元
        self.out_dim = 1  # 出力次元
        self.dim = self.out_dim  # 1要素の次元 出力
        self.all_dim = self.in_dim  # 1要素の次元 (condition分含む)
        self.mode = mode
        target_data_column = params.target_data_column
        target_label_column = params.target_label_column
        target_label = target_label_column[0]
        seed_number = params.seed
        start_date = params.start_date
        end_date = params.end_date
        
        if mode == "train":
            # 株価データの取得
            data = yf.download(target, start=start_date, end=end_date)

            # データを10分割
            num_splits = 10
            split_size = len(data) // num_splits

            # datasetディレクトリを作成
            output_dir = 'dataset'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # データを分割してCSVファイルとして保存
            for i in range(num_splits):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < num_splits - 1 else len(data)

                split_data = data.iloc[start_idx:end_idx]

                # 一時的にCSV形式でデータを文字列として保存（インデックスを含む）
                csv_content = split_data.to_csv(index=True)

                # 各行をリストとして扱い、2行目と3行目を削除
                csv_lines = csv_content.splitlines()
                cleaned_csv_lines = [csv_lines[0]] + csv_lines[3:]  # 1行目（列名） + 4行目以降（データ部分）

                # 再構築したCSV内容
                cleaned_csv_content = "\n".join(cleaned_csv_lines)

                # 保存するファイル名
                file_name = f"{output_dir}/dataset_part_{i + 1}.csv"

                # ファイルに書き込む
                with open(file_name, 'w') as f:
                    f.write(cleaned_csv_content)

                print(f"Saved {file_name}")
        else:
            pass

        #  結果出力用ファイルの作成
        result_dir = f'result/{params.run_date}'  # 結果出力ディレクトリ
        if os.path.exists(result_dir):
            pass
        else:
            os.mkdir(result_dir)  # 実行日時を名前とするディレクトリを作成
        
        file_list = sorted(os.listdir(path=f'dataset/')[:])  # ファイル一覧を取得
        
        # random.seed(seed_number)  # 乱数シードを固定
        # random.shuffle(file_list)  # ファイルリストをシャッフル
        file_list = np.array(file_list)

        id_name_dict = {}
        df_Yfinance = pd.DataFrame()

        for s_id in split_ids:
            f = file_list[s_id]  
            df_source = pd.read_csv(f'dataset/{f}')
            id_name_dict[s_id] = f
            csv_id_value = f  #　追加
            df_source['csv_id'] = csv_id_value  #　追加
            df_Yfinance = pd.concat([df_Yfinance, df_source])

        #  前処理 Close列のstr型を適切な型に
        df_Yfinance[target_label] = df_Yfinance[target_label].astype(float)
        data_list = []
        label_list = []

        for s_id in split_ids: 
            f = id_name_dict[s_id]  
            df_sorted_id = df_Yfinance[df_Yfinance['csv_id'] == f].reset_index()

            df_Yfinance_selection = df_sorted_id[target_data_column] 
            _data_list = np.array(df_Yfinance_selection)
            data_list.append(_data_list)

            df_label = df_sorted_id[[target_label]]  # 目的変数のみを抽出
            _label_list = np.array(df_label)
            label_list.append(_label_list)

        # 標準化関数を用いた標準化
        data_list_concatenated = np.concatenate(data_list, axis=0)
        label_list_concatenated = np.concatenate(label_list, axis=0)
        
        data_list, label_list = self.standardize_data(data_list_concatenated,label_list_concatenated,data_list,label_list,params,self.mode, target_data_column=params.target_data_column, target_label_column=params.target_label_column) 

        # リストをフラットにする
        flat_data_list = list(chain(*data_list))

        # カラムを削除してからデータを追加
        df_Yfinance = df_Yfinance.drop(target_data_column, axis=1)

        # # 各配列をデータフレームに変換し、df_Yfinanceに連結
        df_new = pd.DataFrame(flat_data_list, columns=target_data_column)  # 新しいデータフレーム作成
        # インデックスを合わせる
        df_new.index = df_Yfinance.index
        df_Yfinance = pd.concat([df_Yfinance, df_new], axis=1)  # df_Yfinanceに新しいデータを追加

        # 最終的に df_Yfinance をリストに追加
        df_concatenated = [df_Yfinance]

        data_list, label_list = self.extract_data_and_labels_pandas(df_concatenated, window_size=params.history_window_length, prediction_distance=params.prediction_distance, target_data=params.target_data_column, target_label=params.target_label_column)
       
        self.data_list = data_list
        label_list = [arr.reshape(-1) for arr in label_list]
        self.label_list = label_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = torch.Tensor(self.data_list[idx]).float()
        label = torch.Tensor(self.label_list[idx]).float()
        return data, label

def load_dataset(params):
    logger.info('Loading dataset...')
    if params.dataset.startswith('Yfinance-'):
        target = params.dataset.split('-')[-1]
        train_dataset = YfinanceTraceDataset(params,
                                        target=target,
                                        split_ids=[0, 1, 2, 3, 4, 5, 6],
                                        mode='train')
        valid_dataset = YfinanceTraceDataset(params,
                                        target=target,
                                        split_ids=[7, 8],
                                        mode='valid')
        test_dataset = YfinanceTraceDataset(params,
                                        target=target,
                                        split_ids=[9],
                                        mode='test')
    else:
        raise Exception(f'Unknown dataset ID "{params.dataset}". ')

    return train_dataset, valid_dataset, test_dataset

def padding_collate_fn(batch):
    """異なる長さのデータをパッディング(0で埋める処理)"""
    out_data_list = []
    out_label_list = []
    for sample in batch:
        out_data, out_label = sample
        out_data_list.append(torch.Tensor(out_data))
        out_label_list.append(torch.Tensor(out_label))
    padding_data = pad_sequence(out_data_list)
    padding_label = pad_sequence(out_label_list)

    return padding_data, padding_label
  
if __name__ == "__main__":
    pass
