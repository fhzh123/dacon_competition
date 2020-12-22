import math

import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):

    def __init__(self, d_embedding, max_len=512, segment_embedding=False):
        super().__init__()

        # # If segment_embedding max length is full
        # if segment_embedding:
        #     max_len += 1

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_embedding).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_embedding, 2).float() * -(math.log(10000.0) / d_embedding)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]