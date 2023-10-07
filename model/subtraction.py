import torch
from torch import nn


class OffsetSubtraction(nn.Module):
    def __init__(self, window_size, feature_num, d):
        super(OffsetSubtraction, self).__init__()
        init_index = (torch.arange(window_size) + window_size).unsqueeze(-1).unsqueeze(-1)
        init_index = init_index.repeat(1, feature_num, 2 * d + 1)
        delay = torch.Tensor([0] + [i for i in range(1, d + 1)] + [-i for i in range(1, d + 1)]).int()
        delay = delay.unsqueeze(0).unsqueeze(0).repeat(window_size, feature_num, 1)
        self.index = init_index + delay
        self.d = d

    def forward(self, subed, sub):
        batch_size = subed.shape[0]
        index = self.index.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(sub.device)

        front = sub[:, 0:1, :].repeat(1, sub.shape[1], 1)
        end = sub[:, -1:, :].repeat(1, sub.shape[1], 1)
        sub = torch.cat([front, sub, end], dim=1)
        sub = torch.gather(sub.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1), dim=1, index=index)

        res = subed.unsqueeze(-1).repeat(1, 1, 1, 2 * self.d + 1) - sub
        res = torch.gather(res, dim=-1, index=torch.argmin(torch.abs(res), dim=-1).unsqueeze(-1))

        return res.reshape(subed.shape)
