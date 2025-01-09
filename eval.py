import torch
import numpy as np
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def lstm_check(model, dataset, params):

    device = params.device

    from input_data import padding_collate_fn
    dataloader = DataLoader(dataset,
                            batch_size=params.batch_size,
                            collate_fn=padding_collate_fn,
                            shuffle=False)
    
    model.eval()  # モデルを評価モードに設定
    
    sample_list = []  # 予測された時系列のリスト

    for _, (data, _) in enumerate(dataloader):
        data = data.to(device)  # 最初の一部だけ取り出す
        output, _ = model(data)

        series = np.empty((0, dataset.dim), float)  # 一つの時系列 (初期化)
        item = output.T.to('cpu').detach().numpy().copy()
        series = np.append(series, item, axis=0)
        sample_list.append(series)

    sample_list = flatten_and_convert_to_tensor(sample_list)
    return sample_list     
            
def flatten_and_convert_to_tensor(list_of_lists):
    """
    リストのリストをフラット化して，torch.Tensorに変換する関数．
    """
    flattened_array = np.array([item for sublist in list_of_lists for item in sublist])
    return torch.Tensor(flattened_array)

def calculate_mse_rmse(model, test_dataloader, dim, device):
    """
    テストデータセットのMSE（Mean Squared Error）、RMSE（Root Mean Squared Error）、
    そしてMAE（Mean Absolute Error）を計算する関数。
    """
    model.eval()  # モデルを評価モードに設定

    mse_loss = torch.nn.MSELoss()  # MSE loss関数のインスタンス化
    mae_loss = torch.nn.L1Loss()   # MAE loss関数のインスタンス化
    total_mse = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for data, label in test_dataloader:
            data = data.to(device)
            label = label.to(device)

            if dim == 1:
                output, _ = model(data)
                mse = mse_loss(output, label)
                mae = mae_loss(output, label)

            total_mse += mse.item()
            total_mae += mae.item()

        batch_size = len(test_dataloader)

    mse = total_mse / batch_size
    rmse = np.sqrt(mse)
    mae = total_mae / batch_size

    return mse, rmse, mae

if __name__ == '__main__':
    pass

