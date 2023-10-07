import numpy as np
import torch
from torch import nn


class OrdAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(OrdAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K, V):
        residual = Q.clone()

        Q = self.W_Q(Q).view(Q.size(0), Q.size(1), self.head_num, self.atten_dim)
        K = self.W_K(K).view(K.size(0), K.size(1), self.head_num, self.atten_dim)
        V = self.W_V(V).view(V.size(0), V.size(1), self.head_num, self.atten_dim)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)


class MixAttention(nn.Module):
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual):
        super(MixAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        self.W_Q_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_Q_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q_data, Q_time, K_data, K_time, V_time):
        residual = Q_data.clone()

        Q_data = self.W_Q_data(Q_data).view(Q_data.size(0), Q_data.size(1), self.head_num, self.atten_dim)
        Q_time = self.W_Q_time(Q_time).view(Q_time.size(0), Q_time.size(1), self.head_num, self.atten_dim)
        K_data = self.W_K_data(K_data).view(K_data.size(0), K_data.size(1), self.head_num, self.atten_dim)
        K_time = self.W_K_time(K_time).view(K_time.size(0), K_time.size(1), self.head_num, self.atten_dim)
        V_time = self.W_V_time(V_time).view(V_time.size(0), V_time.size(1), self.head_num, self.atten_dim)

        Q_data, Q_time = Q_data.transpose(1, 2), Q_time.transpose(1, 2)
        K_data, K_time = K_data.transpose(1, 2), K_time.transpose(1, 2)
        V_time = V_time.transpose(1, 2)

        scores_data = torch.matmul(Q_data, K_data.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_time = torch.matmul(Q_time, K_time.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores_data + scores_time)
        context = torch.matmul(attn, V_time)

        context = context.transpose(1, 2)
        context = context.reshape(residual.size(0), residual.size(1), -1)
        output = self.dropout(self.fc(context))

        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)
