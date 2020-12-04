import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_embedding=512, pad_idx=0):
        super().__init__(vocab_size, d_embedding, padding_idx=pad_idx)