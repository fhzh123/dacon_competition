# Import PyTorch
import torch
import torch.nn.functional as F
from torch import nn

class rnn_GRU(nn.Module):
    def __init__(self, src_vocab_num, trg_num=5, d_embedding=256, d_model=512,
                 n_layers=1, pad_idx=0, dropout=0.0, embedding_dropout=0.0):
        super(rnn_GRU, self).__init__()

        self.src_vocab_num = src_vocab_num
        self.d_embedding = d_embedding
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout

        self.embedding = nn.Embedding(src_vocab_num, d_embedding, padding_idx=pad_idx)
        self.src_linear = nn.Linear(d_embedding, d_model)

        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.trg_linear = nn.Linear(d_model*2, d_model)
        self.trg_linear2 = nn.Linear(d_model*2, d_model)
        
    def forward(self, src, hidden=None, cell=None):
        # Source sentence embedding
        outputs = src.transpose(0, 1)
        outputs = self.embedding(outputs)  # (max_caption_length, batch_size, embed_dim)
        outputs = self.src_linear(F.dropout(outputs, p=self.embedding_dropout, inplace=True)) # (max_caption_length, batch_size, embed_dim)
        # Bidirectional SRU
        outputs, hidden = self.gru(outputs, hidden)
        # sum bidirectional outputs
        outputs = torch.tanh(self.trg_linear(outputs)) # (max_caption_length, batch_size, embed_dim)
        outputs = self.trg_linear2(torch.cat((outputs[-2,:,:], outputs[-1,:,:]), dim=1))
        return outputs, hidden.transpose(0, 1)