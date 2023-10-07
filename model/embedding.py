import math

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.pe[:, : x.size(1), :]

        return self.norm(x)


class TimeEmbedding(nn.Module):
    def __init__(self, model_dim, time_num):
        super(TimeEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels=time_num, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return self.norm(x)


class DataEmbedding(nn.Module):
    def __init__(self, model_dim, feature_num):
        super(DataEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels=feature_num, out_channels=model_dim, kernel_size=(1,))
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x
