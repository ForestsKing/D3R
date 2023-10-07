from torch import nn

from model.block import SpatialTemporalTransformerBlock
from model.embedding import TimeEmbedding, DataEmbedding, PositionEmbedding


class Reconstruction(nn.Module):
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num,
                 dropout):
        super(Reconstruction, self).__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.position_embedding = PositionEmbedding(model_dim)

        self.decoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.decoder_blocks.append(
                SpatialTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dp)
            )

        self.fc1 = nn.Linear(model_dim, feature_num, bias=True)

    def forward(self, noise, trend, time):
        trend = self.data_embedding(trend)
        x = self.data_embedding(noise) - trend
        x = x + self.position_embedding(noise) + self.time_embed(time)

        for block in self.decoder_blocks:
            x = block(x)

        out = self.fc1(x + trend)

        return out
