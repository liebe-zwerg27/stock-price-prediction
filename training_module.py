"""LSTM学習モジュール

   LSTMの訓練の準備と実施を行うモジュールである.

"""

import os
import logging
from input_data import load_dataset, padding_collate_fn
from visualise import visualize, visualize_close
from eval import lstm_check, calculate_mse_rmse
from models import LSTM
from utils import get_device, EarlyStopping
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Any
import itertools
import numpy as np
import pickle


logger = logging.getLogger(__name__)

import random

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_main(params) -> None:
    """
    データセットのロードと可視化、LSTMの学習及び予測結果の可視化
    """

    vars(params).update({'device': str(get_device())})
    
    # シード値固定
    torch_fix_seed(seed=params.seed)

    # 結果出力用ファイルの作成
    result_dir = f'result/{params.run_date}'  # 結果出力ディレクトリ

    os.makedirs(result_dir + '/output_data_png', exist_ok=True)  # 出力データのpngファイルを保存するディレクトリ
    os.makedirs(result_dir + '/scalers', exist_ok=True)  # 標準化データの保存ディレクトリ
    
    # データセットをロード
    train_dataset, valid_dataset, test_dataset = load_dataset(params)

    # LSTMの学習
    model= lstm_train(train_dataset, valid_dataset, test_dataset, params, result_dir)
    
    if params.dataset.startswith("Yfinance-"):
        if train_dataset.dim == 1:
            # 標準化データの読み込み
            if os.path.exists(f'result/{params.run_date}/scalers/scalers_data.pkl'):
                scalers_file_path = f'result/{params.run_date}/scalers/scalers_data.pkl'
                with open(scalers_file_path, 'rb') as f:
                        scalers = pickle.load(f)
            
            # LSTMによる予測結果の確認
            _predicted_list = lstm_check(model, train_dataset, params)
            predicted_list = _predicted_list[:len(train_dataset)]  # LSTMの学習結果をチェック
            
            # Train正解データの確認
            # データスライス【先頭】から
            answer_all_list = [x[1] for x in train_dataset]  # 全ての正解データ xにdetalistとlabellistが入っている
            answer_list = answer_all_list[:len(train_dataset)]  # 正解データのスライス
            
            # 標準化逆変換:予測値Close
            predicted_list = predicted_list.detach().cpu().numpy().tolist()
            predicted_list = np.array(predicted_list).reshape(-1, train_dataset.dim)
            
            # Single
            predicted_list = list(itertools.chain.from_iterable(scalers[params.target_label_column[0]].inverse_transform(predicted_list[:, ]).reshape(-1, 1)))  # Close ERP
            addup_predicted_list = [[[v] for v in predicted_list]]  
            addup_predicted_list = np.round(addup_predicted_list).tolist()  # 逆標準化により発生した元の値との誤差を修正
            
            # 標準化:正解値
            answer_list = np.array([x.to('cpu').detach().numpy().copy().tolist() for x in answer_list])
            
            # Single
            answer_list = list(itertools.chain.from_iterable(scalers[params.target_label_column[0]].inverse_transform(answer_list[:, ]).reshape(-1, 1)))  # Close ERP
            addup_answer_list = [[[v] for v in answer_list]]  
            addup_answer_list = np.round(addup_answer_list).tolist()  # 逆標準化により発生した元の値との誤差を修正

            # 予測値+正解値　結合リスト
            mix_predicted_list = list(map(list.__add__, addup_predicted_list[0], addup_answer_list[0]))

        visualize_close(torch.tensor([mix_predicted_list]),
                    result_dir+'/output_data_png',
                    'train_result',
                    train_dataset.dim,
                    point=False,
                    pdf=False)  # LSTMの学習結果と正解データを折れ線グラフに描画
        
        
        if test_dataset.dim == 1:
            # LSTMによる予測結果の確認
            _predicted_list = lstm_check(model, test_dataset, params)
            predicted_list = _predicted_list[:len(test_dataset)]  # LSTMの学習結果をチェック

            # Test正解データの確認
            # データスライス【先頭】から
            answer_all_list = [x[1] for x in test_dataset]  # 全ての正解データ
            answer_list = answer_all_list[:len(test_dataset)]

            # 標準化逆変換:予測値Close
            predicted_list = predicted_list.detach().cpu().numpy().tolist()
            predicted_list = np.array(predicted_list).reshape(-1, test_dataset.dim)

            # Single
            predicted_list = list(itertools.chain.from_iterable(scalers[params.target_label_column[0]].inverse_transform(predicted_list[:, ]).reshape(-1, 1)))  # Close ERP
            addup_predicted_list = [[[v] for v in predicted_list]]  
            addup_predicted_list = np.round(addup_predicted_list).tolist()  # 逆標準化により発生した元の値との誤差を修正

            # 標準化逆変換:正解値
            answer_list = np.array([x.to('cpu').detach().numpy().copy().tolist() for x in answer_list])

            # Single
            answer_list = list(itertools.chain.from_iterable(scalers[params.target_label_column[0]].inverse_transform(answer_list[:, ]).reshape(-1, 1)))  # Close ERP 標準化
            addup_answer_list = [[[v] for v in answer_list]]  
            addup_answer_list = np.round(addup_answer_list).tolist()  # 逆標準化により発生した元の値との誤差を修正

            # 予測値+正解値の結合リスト作成
            mix_predicted_list = list(map(list.__add__, addup_predicted_list[0], addup_answer_list[0]))

        visualize_close(torch.tensor([mix_predicted_list]),
                    result_dir+'/output_data_png',
                    'test_result',
                    test_dataset.dim,
                    point=False,
                    pdf=False)  # LSTMの学習結果と正解データを折れ線グラフに描画

def lstm_train(train_dataset, valid_dataset, test_dataset, params, result_dir) -> list[Any]:
    """LSTMの訓練を実施する関数．
    訓練用、テスト用のデータセットを渡し、LSTMによる訓練を行う関数である.
    """

    logger.info('Training LSTM...')

    vars(params).update({'device': str(get_device())})
    device = params.device
    lstm_lr = params.lstm_lr
    num_layers = params.num_layers
    dropout = params.dropout
    batch_size = params.batch_size
    lstm_epochs_num = params.lstm_epochs_num
    hidden_size = params.hidden_size
    cross_validation = params.cross_validation
    patience = params.patience
    delta = params.delta
    dim = train_dataset.out_dim
    all_dim = train_dataset.in_dim

    if cross_validation and params.n_splits == 1:
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   collate_fn=padding_collate_fn,
                                                   shuffle=False)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size=batch_size,
                                                    collate_fn=padding_collate_fn,
                                                    shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=batch_size,
                                                    collate_fn=padding_collate_fn,
                                                    shuffle=False)
        lstm_loss_data = []
        lstm_valid_loss_data = []
    else:
        pass

    if params.dataset.startswith('Yfinance-'):
        model = LSTM(all_dim, hidden_size, dim, num_layers, dropout).to(device)
    else:
        model = LSTM(all_dim, hidden_size, dim, num_layers, dropout).to(device)
    criterion = nn.MSELoss()  # MSEロスはl2ノルムではなく，各成分の2乗の平均をとったものなので注意
    optimizer = Adam(model.parameters(), lr=lstm_lr)
    
    if cross_validation and params.n_splits == 1:
        early_stop = params.earlystop
        epoch = 0  
        if early_stop is True:
            early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta, path=result_dir + '/lstm.nn')
            
        for epoch in range(lstm_epochs_num):
            # 学習
            running_loss = 0.0
            train_loss = 0

            for i, (data, label) in enumerate(train_dataloader):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                if dim == 1:
                    output, _ = model(data)
                    loss = criterion(output, label)
                loss.backward()  # 誤差逆伝播
                optimizer.step()  # 最適化
                running_loss += loss.item()

            num_batches = len(train_dataloader)
            train_loss = running_loss / num_batches

            # バリデーション
            running_valid_loss = 0.0
            valid_loss = 0
            
            with torch.no_grad():
                for j, (data, label) in enumerate(valid_dataloader):
                    data = data.to(device)
                    label = label.to(device)
                    if dim == 1:    
                        output, _ = model(data)
                        loss = criterion(output, label)
                    running_valid_loss += loss.item()

                num_batches = len(valid_dataloader)
                valid_loss = running_valid_loss / num_batches

            logger.info('epoch: %d, traning loss: %.5f, valid loss: %.5f' % (epoch, train_loss, valid_loss))
            
            lstm_loss_data.append(train_loss)  # 各foldごとのlossを追加
            lstm_valid_loss_data.append(valid_loss)  # 各foldごとのvalid lossを追加

            if early_stop is True:
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break
            else:
                pass

            
        model.load_state_dict(torch.load(result_dir + '/lstm.nn'))
        
        mse, rmse, mae = calculate_mse_rmse(model, train_dataloader, dim ,device)
        logger.info("Train MSE: %.5f, RMSE: %.5f, MAE: %.5f" % (mse, rmse, mae))
        mse, rmse, mae = calculate_mse_rmse(model, valid_dataloader, dim ,device)
        logger.info("Valid MSE: %.5f, RMSE: %.5f, MAE: %.5f" % (mse, rmse, mae))
        mse, rmse, mae = calculate_mse_rmse(model, test_dataloader, dim ,device)
        logger.info("Test MSE: %.5f, RMSE: %.5f, MAE: %.5f" % (mse, rmse, mae))                        
    else:
        pass
    
    if lstm_loss_data:
        visualize(lstm_loss_data,
                result_dir,
                'lstm_train_loss',
                1,
                pdf=False)  # lossをグラフ化

        if cross_validation:
            visualize(lstm_valid_loss_data,
                      result_dir,
                      'lstm_valid_loss',
                      1,
                      pdf=False)  # valid lossをグラフ化
            
            lstm_loss_cf = [[x, y] for x, y in zip(lstm_loss_data, lstm_valid_loss_data)] # torch.tensor
            visualize(lstm_loss_cf,
                      result_dir,
                      'lstm_train_valid_loss',
                      1,
                      pdf=False)  # train,validのlossをグラフ化

    return model

if __name__ == "__main__":
    train_main()
