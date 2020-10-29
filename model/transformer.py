import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .embedding.transformer_embedding import TransformerEmbedding, 


class Transformer(nn.Module):
    def __init__(self, src_vocab_num, author_num=4, pad_idx=0, bos_idx=1, eos_idx=2, 
                 max_len=300, d_model=512, d_embedding=256, n_head=8, 
                 dim_feedforward=2048, dropout=0.1, num_encoder_layer=8,
                 device=None):

        super(Transformer, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = src_max_len

        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding, 
                                                  pad_idx=self.pad_idx, max_len=self.src_max_len)

        self.src_output_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.src_output_norm = nn.LayerNorm(d_embedding)
        self.src_output_linear2 = nn.Linear(d_embedding, author_num, bias=False)
        
        # Transformer
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward , dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layer)

    def forward(self, src_input_sentence, trg_input_sentence, king_id, tgt_mask, non_pad_position=None):
        src_key_padding_mask = (src_input_sentence == self.pad_idx)

        encoder_out = self.src_embedding(src_input_sentence).transpose(0, 1)
        encoder_out = self.transformer_encoder(encoder_out, 
                                               src_key_padding_mask=src_key_padding_mask)
        encoder_out = encoder_out.transpose(0, 1).contiguous()

        encoder_out = self.src_output_norm(self.dropout(F.gelu(self.src_output_linear(encoder_out))))
        encoder_out = self.src_output_linear2(encoder_out)
        return encoder_out

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device='cuda')) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask