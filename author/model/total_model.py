import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .embedding.transformer_embedding import TransformerEmbedding
from .modules.transformer_encoder import Transformer_Encoder
from .modules.transformer_sublayers import PositionalEncoding
from .modules.transformer_utils import get_pad_mask, get_subsequent_mask

class Total_model(nn.Module):
    def __init__(self, src_vocab_num, author_num=5, pad_idx=0, bos_idx=1, eos_idx=2, 
                 max_len=300, d_model=512, d_embedding=256, n_head=8, d_k=64, d_v=64,
                 dim_feedforward=2048, dropout=0.1, num_encoder_layer=8, bilinear=False,
                 device=None):

        super(Total_model, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.bilinear = bilinear
        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        # self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding, 
        #                                           pad_idx=self.pad_idx, max_len=self.max_len)
        self.src_embedding = nn.Embedding(src_vocab_num, d_embedding, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_embedding, n_position=max_len)
        self.embedding_linear = nn.Linear(d_embedding, d_model)
        self.embedding_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output part
        self.trg_output_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.trg_output_norm = nn.LayerNorm(d_embedding)
        self.trg_output_linear2 = nn.Linear(d_embedding, author_num, bias=False)
        self.trg_output_bilinear = nn.Bilinear(author_num, author_num, author_num)
        self.trg_softmax = nn.Softmax(dim=1)

        # Transformer
        self.transformer_encoder = Transformer_Encoder(d_model, dim_feedforward, 
                                                       n_layers=num_encoder_layer,
                                                       n_head=n_head, d_k=d_k, d_v=d_v, 
                                                       pad_idx=pad_idx, dropout=dropout)

    def forward(self, src_input_sentence):
        src_mask = get_pad_mask(src_input_sentence, self.pad_idx)

        # encoder_out = self.src_embedding(src_input_sentence)#.transpose(0, 1)
        encoder_out = self.embedding_norm(self.embedding_linear(self.position_enc(self.src_embedding(src_input_sentence))))
        encoder_out, *_ = self.transformer_encoder(encoder_out, src_mask)
        # encoder_out = encoder_out.transpose(0, 1).contiguous()

        encoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(encoder_out))))
        encoder_out = self.trg_output_linear2(encoder_out)
        if self.bilinear:
            logit = self.trg_softmax(self.trg_output_bilinear(encoder_out[:,0,:], encoder_out[:,-1,:]))
        else:
            logit = self.trg_softmax(encoder_out[:,0,:])
        return logit