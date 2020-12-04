import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .embedding.total_embedding import TotalEmbedding
from .modules.transformer_encoder import Transformer_Encoder
from .modules.rnn_layers import rnn_GRU
# from .modules.transformer_sublayers import PositionalEncoding
from .modules.transformer_utils import get_pad_mask

class Total_model(nn.Module):
    def __init__(self, src_vocab_num, trg_num=2, pad_idx=0, bos_idx=1, eos_idx=2, 
                 max_len=300, d_model=512, d_embedding=256, n_head=8, d_k=64, d_v=64,
                 dim_feedforward=2048, dropout=0.1, bilinear=False,
                 num_transformer_layer=8, num_rnn_layer=6, device=None):

        super(Total_model, self).__init__()

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.bilinear = bilinear
        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        self.src_total_embedding = TotalEmbedding(src_vocab_num, d_model, d_embedding, 
                                                  pad_idx=self.pad_idx, max_len=self.max_len,
                                                  segment_embedding=True)
        self.src_title_embedding = TotalEmbedding(src_vocab_num, d_model, d_embedding, 
                                                  pad_idx=self.pad_idx, max_len=self.max_len,
                                                  segment_embedding=False)
        self.src_content_embedding = TotalEmbedding(src_vocab_num, d_model, d_embedding, 
                                                    pad_idx=self.pad_idx, max_len=self.max_len,
                                                    segment_embedding=False) 
        # self.src_embedding_trs = nn.Embedding(src_vocab_num, d_embedding, padding_idx=pad_idx)
        # self.position_enc = PositionalEncoding(d_embedding, n_position=max_len)
        # self.embedding_linear = nn.Linear(d_embedding, d_model)
        # self.embedding_norm = nn.LayerNorm(d_model, eps=1e-6)

        #===================================#
        #==========GAP with linear==========#
        #===================================#

        self.src_embedding_gap = nn.Embedding(src_vocab_num, d_embedding, padding_idx=pad_idx)
        self.gap_linear1 = nn.Linear(d_embedding, d_model)
        self.gap_linear2 = nn.Linear(d_model, d_embedding)
        self.gap_linear3 = nn.Linear(d_embedding, trg_num)

        #===================================#
        #============Transformer============#
        #===================================#

        # Transformer
        self.transformer_encoder = Transformer_Encoder(d_model, dim_feedforward, 
                                                       n_layers=num_transformer_layer,
                                                       n_head=n_head, d_k=d_k, d_v=d_v, 
                                                       pad_idx=pad_idx, dropout=dropout)

        # Transformer output part
        self.trs_trg_output_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.trs_trg_output_norm = nn.LayerNorm(d_embedding)
        self.trs_trg_output_linear2 = nn.Linear(d_embedding, trg_num, bias=False)
        self.trs_trg_softmax = nn.Softmax(dim=1)

        #===================================#
        #=============RNN(GRU)==============#
        #===================================#

        # GRU
        self.rnn_gru = rnn_GRU(src_vocab_num, trg_num=trg_num, d_embedding=d_embedding, 
                               d_model=d_model, n_layers=num_rnn_layer, pad_idx=pad_idx,
                               dropout=dropout, embedding_dropout=dropout*0.5)

        # RNN output part
        self.rnn_trg_output_linear = nn.Linear(d_model, d_embedding, bias=False)
        self.rnn_trg_output_norm = nn.LayerNorm(d_embedding)
        self.rnn_trg_output_linear2 = nn.Linear(d_embedding, trg_num, bias=False)
        self.rnn_trg_softmax = nn.Softmax(dim=1)

        #===================================#
        #============Concatenate============#
        #===================================#

        # Concat
        if bilinear:
            self.output_linear = nn.Bilinear(trg_num, trg_num, trg_num)
        else:
            self.output_linear = nn.Linear(trg_num*3, trg_num)

    def forward(self, src_total, src_title, src_content):

        #===================================#
        #==========GAP with linear==========#
        #===================================#

        gap_out = self.src_embedding_gap(src_content).mean(dim=1)
        gap_out = self.gap_linear3(self.gap_linear2(self.gap_linear1(gap_out)))

        #===================================#
        #=============RNN(GRU)==============#
        #===================================#

        rnn_out, *_ = self.rnn_gru(src_content)
        rnn_out = self.rnn_trg_output_norm(self.dropout(F.gelu(self.rnn_trg_output_linear(rnn_out))))
        rnn_out = self.rnn_trg_output_linear2(rnn_out)
        
        #===================================#
        #============Transformer============#
        #===================================#

        src_mask = get_pad_mask(src_total, self.pad_idx)

        encoder_out = self.src_total_embedding(src_total)#.transpose(0, 1)
        # encoder_out = self.position_enc(self.src_embedding_trs(src_input_sentence))
        # encoder_out = self.embedding_norm(self.embedding_linear(encoder_out))
        encoder_out, *_ = self.transformer_encoder(encoder_out, src_mask)
        # encoder_out = encoder_out.transpose(0, 1).contiguous()

        encoder_out = self.trs_trg_output_norm(self.dropout(F.gelu(self.trs_trg_output_linear(encoder_out))))
        encoder_out = self.trs_trg_output_linear2(encoder_out)[:,0,:]

        #===================================#
        #============Concatenate============#
        #===================================#

        if self.bilinear:
            logit = self.output_linear(rnn_out, encoder_out)
        else:
            logit = self.output_linear(torch.cat((rnn_out, encoder_out, gap_out), dim=1))

        return logit