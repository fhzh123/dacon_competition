import torch
import torch.nn as nn
from torch.nn import functional as F

from .token import TokenEmbedding
from .positional import PositionalEmbedding
from .segment import SegmentEmbedding

class TotalEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information using sin, cos
    sum of all these features are output of Embedding
    """

    def __init__(self, vocab_size, d_model, d_embedding, pad_idx=0, max_len=512, 
                 embedding_dropout=0.1, segment_embedding=False):
        """
        :param vocab_size: total vocab size
        :param d_embedding: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.segment_embedding = segment_embedding

        self.token = TokenEmbedding(vocab_size=vocab_size, d_embedding=d_embedding, pad_idx=pad_idx)
        self.position = PositionalEmbedding(d_embedding=d_embedding, max_len=max_len, 
                                            segment_embedding=segment_embedding)
        if segment_embedding:
            self.segment = SegmentEmbedding(d_embedding=d_embedding)
        self.linear_layer = nn.Linear(d_embedding, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, sequence):
        if self.segment_embedding:
            x = self.token(sequence) + self.position(sequence) + self.segment(sequence)
        else:
            x = self.token(sequence) + self.position(sequence)
        x = self.embedding_dropout(x)
        return self.norm(self.linear_layer(x))