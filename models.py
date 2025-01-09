"""LSTMニューラルネットモジュール

    ニューラルネットワーク(NeuralNet)の作成

"""

import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM用のニューラルネット
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5, sigmoid_out=False) -> None:
        """
        ニューラルネットワークの各層の定義
        Args:
            input_dim(int): 入力層の次元
            hidden_dim(int): 隠れ層の次元
            output_dim(int):　出力層の次元
            num_layers(int): LSTMを重ねる層の数
            dropout(float): dropoutのrate
            sigmoid_out(bool): 出力する際にSigmoidを適用するかどうか
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 入力層
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)  # 隠れ層
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 出力層
        
        if sigmoid_out is True:  # outputする前にsoftmax関数を適用するかどうか
            self.output_layer = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Sigmoid())
        else:  # 処理
            self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, inputs, hidden_cell=None) -> tuple[float]:
        """
        ニューラルネットワークの各層の定義
        入力層、隠れ層のLinerクラスのインスタンス生成、出力層の定義

        Args:
            inputs(int): 入力層の次元
            hidden_cell(int): 隠れ層の次元

        Returns:
            output: 出力層の結果
            (hidden, cell): 隠れ層とセル状態の結果

        """
        # 入力に線形層を追加
        x = self.fc1(inputs) 
        
        # 時系列と隠れ層初期値から，出力，隠れ層，セル状態を計算
        x, hidden_cell = self.lstm(x, hidden_cell) 
        
        # 出力層
        output = self.output_layer(x[-1, :, :]).T

        return output, hidden_cell
