import torch
from torch import nn

from model.decomposition import DynamicDecomposition
from model.diffusion import Diffusion
from model.reconstruction import Reconstruction


class DDDR(nn.Module):
    def __init__(self, time_steps, beta_start, beta_end, window_size, model_dim, ff_dim, atten_dim, feature_num,
                 time_num, block_num, head_num, dropout, device, d, t):
        super(DDDR, self).__init__()

        self.device = device
        self.window_size = window_size
        self.t = t

        self.dynamic_decomposition = DynamicDecomposition(
            window_size=window_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=block_num,
            head_num=head_num,
            dropout=dropout,
            d=d
        )

        self.diffusion = Diffusion(
            time_steps=time_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )

        self.reconstruction = Reconstruction(
            window_size=window_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=block_num,
            head_num=head_num,
            dropout=dropout
        )

    def forward(self, data, time, p=0):
        disturb = torch.rand(data.shape[0], data.shape[2]) * p
        disturb = disturb.unsqueeze(1).repeat(1, self.window_size, 1).float().to(self.device)
        data = data + disturb

        stable, trend = self.dynamic_decomposition(data, time)

        bt = torch.full((data.shape[0],), self.t).to(self.device)
        sample_noise = torch.randn_like(data).float().to(self.device)
        noise_data = self.diffusion.q_sample(data, trend, bt, sample_noise)

        recon = self.reconstruction(noise_data, trend, time)

        return stable, trend - disturb, recon - disturb
