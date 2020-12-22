import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

from .embedding.total_embedding import TotalEmbedding
from .modules.transformer_encoder import Transformer_Encoder
from .modules.rnn_layers import rnn_GRU, rnn_LSTM
# from .modules.transformer_sublayers import PositionalEncoding
from .modules.transformer_utils import get_pad_mask

class Total_model(nn.Module):
    def __init__(self, model_type, src_vocab_num_dict, trg_num=2, pad_idx=0, bos_idx=1, eos_idx=2, 
                 max_len=300, d_model=512, d_embedding=256, n_head=8, d_k=64, d_v=64,
                 dim_feedforward=2048, dropout=0.1, bilinear=False,
                 num_transformer_layer=8, num_rnn_layer=6, device=None):

        super(Total_model, self).__init__()

        self.model_type = model_type
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_len = max_len

        self.bilinear = bilinear
        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        self.src_total_spm_embedding = TotalEmbedding(src_vocab_num_dict['spm'], d_model, d_embedding, 
                                                      pad_idx=self.pad_idx, max_len=self.max_len,
                                                      segment_embedding=True)
        self.src_total_khaiii_embedding = TotalEmbedding(src_vocab_num_dict['khaiii'], d_model, d_embedding, 
                                                         pad_idx=self.pad_idx, max_len=self.max_len,
                                                         segment_embedding=True)
        self.src_total_konlpy_embedding = TotalEmbedding(src_vocab_num_dict['konlpy'], d_model, d_embedding, 
                                                         pad_idx=self.pad_idx, max_len=self.max_len,
                                                         segment_embedding=True)
        self.src_cont_spm_embedding = TotalEmbedding(src_vocab_num_dict['spm'], d_model, d_embedding, 
                                                     pad_idx=self.pad_idx, max_len=self.max_len,
                                                     segment_embedding=False)
        self.src_cont_khaiii_embedding = TotalEmbedding(src_vocab_num_dict['khaiii'], d_model, d_embedding, 
                                                        pad_idx=self.pad_idx, max_len=self.max_len,
                                                        segment_embedding=False)
        self.src_cont_konlpy_embedding = TotalEmbedding(src_vocab_num_dict['konlpy'], d_model, d_embedding, 
                                                        pad_idx=self.pad_idx, max_len=self.max_len,
                                                        segment_embedding=False) 
        # self.src_embedding_trs = nn.Embedding(src_vocab_num, d_embedding, padding_idx=pad_idx)
        # self.position_enc = PositionalEncoding(d_embedding, n_position=max_len)
        # self.embedding_linear = nn.Linear(d_embedding, d_model)
        # self.embedding_norm = nn.LayerNorm(d_model, eps=1e-6)

        #===================================#
        #==========GAP with linear==========#
        #===================================#

        if model_type in ['total', 'gap']:
            # Global average pooling
            self.src_embedding_gap = nn.Embedding(src_vocab_num_dict['spm'], d_embedding, padding_idx=pad_idx)
            self.gap_linear1 = nn.Linear(d_embedding, d_model)
            self.gap_linear2 = nn.Linear(d_model, d_embedding)
            self.gap_linear3 = nn.Linear(d_embedding, trg_num)

        #===================================#
        #=============RNN(GRU)==============#
        #===================================#

        if model_type in ['total', 'rnn']:
            # RNN
            self.rnn_gru = rnn_GRU(src_vocab_num_dict['spm'], trg_num=trg_num, d_embedding=d_embedding, 
                                   d_model=d_model, n_layers=num_rnn_layer, pad_idx=pad_idx,
                                   dropout=dropout, embedding_dropout=dropout*0.5)

            # LSTM
            self.rnn_lstm = rnn_LSTM(src_vocab_num_dict['spm'], trg_num=trg_num, d_embedding=d_embedding, 
                                     d_model=d_model, n_layers=num_rnn_layer, pad_idx=pad_idx,
                                     dropout=dropout, embedding_dropout=dropout*0.5)

            # RNN output part
            self.gru_trg_output_linear = nn.Linear(d_model, d_embedding, bias=True)
            self.gru_trg_output_norm = nn.LayerNorm(d_embedding)
            self.gru_trg_output_linear2 = nn.Linear(d_embedding, trg_num, bias=True)
            self.gru_trg_softmax = nn.Softmax(dim=1)

            self.lstm_trg_output_linear = nn.Linear(d_model, d_embedding, bias=True)
            self.lstm_trg_output_norm = nn.LayerNorm(d_embedding)
            self.lstm_trg_output_linear2 = nn.Linear(d_embedding, trg_num, bias=True)
            self.lstm_trg_softmax = nn.Softmax(dim=1)

        #===================================#
        #============Transformer============#
        #===================================#

        if model_type in ['total', 'transformer']:
            # Transformer
            self.trs_encoder_spm_total = Transformer_Encoder(d_model, dim_feedforward, 
                                                             n_layers=num_transformer_layer,
                                                             n_head=n_head, d_k=d_k, d_v=d_v, 
                                                             pad_idx=pad_idx, dropout=dropout)
            self.trs_encoder_khaiii_total = Transformer_Encoder(d_model, dim_feedforward, 
                                                                n_layers=num_transformer_layer,
                                                                n_head=n_head, d_k=d_k, d_v=d_v, 
                                                                pad_idx=pad_idx, dropout=dropout)
            self.trs_encoder_konlpy_total = Transformer_Encoder(d_model, dim_feedforward, 
                                                                n_layers=num_transformer_layer,
                                                                n_head=n_head, d_k=d_k, d_v=d_v, 
                                                                pad_idx=pad_idx, dropout=dropout)

            self.trs_encoder_spm_cont = Transformer_Encoder(d_model, dim_feedforward, 
                                                            n_layers=num_transformer_layer,
                                                            n_head=n_head, d_k=d_k, d_v=d_v, 
                                                            pad_idx=pad_idx, dropout=dropout)
            self.trs_encoder_khaiii_cont = Transformer_Encoder(d_model, dim_feedforward, 
                                                               n_layers=num_transformer_layer,
                                                               n_head=n_head, d_k=d_k, d_v=d_v, 
                                                               pad_idx=pad_idx, dropout=dropout)
            self.trs_encoder_konlpy_cont = Transformer_Encoder(d_model, dim_feedforward, 
                                                               n_layers=num_transformer_layer,
                                                               n_head=n_head, d_k=d_k, d_v=d_v, 
                                                               pad_idx=pad_idx, dropout=dropout)

            # Transformer output part
            # 1) Total
            self.trs_trg_total_output_linear_spm = nn.Linear(d_model, d_embedding, bias=True)
            self.trs_trg_total_output_norm_spm = nn.LayerNorm(d_embedding)
            self.trs_trg_total_output_linear2_spm = nn.Linear(d_embedding, trg_num, bias=True)

            self.trs_trg_total_output_linear_khaiii = nn.Linear(d_model, d_embedding, bias=True)
            self.trs_trg_total_output_norm_khaiii = nn.LayerNorm(d_embedding)
            self.trs_trg_total_output_linear2_khaiii = nn.Linear(d_embedding, trg_num, bias=True)

            self.trs_trg_total_output_linear_konlpy = nn.Linear(d_model, d_embedding, bias=True)
            self.trs_trg_total_output_norm_konlpy = nn.LayerNorm(d_embedding)
            self.trs_trg_total_output_linear2_konlpy = nn.Linear(d_embedding, trg_num, bias=True)

            # 2) Content only
            self.trs_trg_output_linear_spm = nn.Linear(d_model, d_embedding, bias=True)
            self.trs_trg_output_norm_spm = nn.LayerNorm(d_embedding)
            self.trs_trg_output_linear2_spm = nn.Linear(d_embedding, trg_num, bias=True)

            self.trs_trg_output_linear_khaiii = nn.Linear(d_model, d_embedding, bias=True)
            self.trs_trg_output_norm_khaiii = nn.LayerNorm(d_embedding)
            self.trs_trg_output_linear2_khaiii = nn.Linear(d_embedding, trg_num, bias=True)

            self.trs_trg_output_linear_konlpy = nn.Linear(d_model, d_embedding, bias=True)
            self.trs_trg_output_norm_konlpy = nn.LayerNorm(d_embedding)
            self.trs_trg_output_linear2_konlpy = nn.Linear(d_embedding, trg_num, bias=True)

        #===================================#
        #============Concatenate============#
        #===================================#

        # Embedding concat
        self.embedding_concat_total_linear = nn.Linear(d_embedding*3, d_embedding)
        self.trg_output_total_linear = nn.Linear(d_embedding, trg_num)
        self.embedding_concat_cont_linear = nn.Linear(d_embedding*3, d_embedding)
        self.trg_output_cont_linear = nn.Linear(d_embedding, trg_num)

        # Logit concat
        if bilinear:
            self.output_linear = nn.Bilinear(trg_num, trg_num, trg_num)
        else:
            self.output_linear1 = nn.Linear(trg_num*5, trg_num*3)
            self.output_linear2 = nn.Linear(trg_num*3, trg_num)

    def forward(self, spm_total_src, khaiii_total_src, konlpy_total_src,
                spm_src, khaiii_src, konlpy_src):

        #===================================#
        #==========GAP with linear==========#
        #===================================#

        if self.model_type in ['total', 'gap']:
            gap_out = self.src_embedding_gap(spm_src).mean(dim=1)
            gap_out = self.gap_linear3(self.gap_linear2(self.gap_linear1(gap_out)))

        #===================================#
        #=============RNN(GRU)==============#
        #===================================#

        if self.model_type in ['total', 'rnn']:
            # RNN modules
            gru_out, *_ = self.rnn_gru(spm_src)
            lstm_out, *_ = self.rnn_lstm(spm_src)

            # Output linear
            gru_out = self.gru_trg_output_norm(self.dropout(F.gelu(self.gru_trg_output_linear(gru_out))))
            gru_out = self.gru_trg_output_linear2(gru_out)

            lstm_out = self.lstm_trg_output_norm(self.dropout(F.gelu(self.lstm_trg_output_linear(lstm_out))))
            lstm_out = self.lstm_trg_output_linear2(lstm_out)
        
        #===================================#
        #============Transformer============#
        #===================================#

        if self.model_type in ['total', 'transformer']:
            spm_src_mask = get_pad_mask(spm_src, self.pad_idx)
            khaiii_src_mask = get_pad_mask(khaiii_src, self.pad_idx)
            konlpy_src_mask = get_pad_mask(konlpy_src, self.pad_idx)

            spm_src_mask_total = get_pad_mask(spm_total_src, self.pad_idx)
            khaiii_src_mask_total = get_pad_mask(khaiii_total_src, self.pad_idx)
            konlpy_src_mask_total = get_pad_mask(konlpy_total_src, self.pad_idx)

            #=====Total=====#
            # SentencePiece input
            spm_encoder_out_total = self.src_total_spm_embedding(spm_total_src)
            spm_encoder_out_total, *_ = self.trs_encoder_spm_total(spm_encoder_out_total, spm_src_mask_total)
            spm_encoder_out_total = self.trs_trg_total_output_norm_spm(self.dropout(F.gelu(self.trs_trg_total_output_linear_spm(spm_encoder_out_total))))
            spm_encoder_out_total = F.max_pool1d(spm_encoder_out_total.permute(0,2,1), spm_encoder_out_total.size(1)).squeeze(2)
            # spm_encoder_out = self.trs_trg_output_linear2_spm(spm_encoder_out)[:,0,:]

            # Khaiii input
            khaiii_encoder_out_total = self.src_total_khaiii_embedding(khaiii_total_src)
            khaiii_encoder_out_total, *_ = self.trs_encoder_khaiii_total(khaiii_encoder_out_total, khaiii_src_mask_total)
            khaiii_encoder_out_total = self.trs_trg_total_output_norm_khaiii(self.dropout(F.gelu(self.trs_trg_total_output_linear_khaiii(khaiii_encoder_out_total))))
            khaiii_encoder_out_total = F.max_pool1d(khaiii_encoder_out_total.permute(0,2,1), khaiii_encoder_out_total.size(1)).squeeze(2)
            # khaiii_encoder_out = self.trs_trg_output_linear2_khaiii(khaiii_encoder_out)[:,0,:]

            # KoNLPy input
            konlpy_encoder_out_total = self.src_total_konlpy_embedding(konlpy_total_src)
            konlpy_encoder_out_total, *_ = self.trs_encoder_konlpy_total(konlpy_encoder_out_total, konlpy_src_mask_total)
            konlpy_encoder_out_total = self.trs_trg_total_output_norm_konlpy(self.dropout(F.gelu(self.trs_trg_total_output_linear_konlpy(konlpy_encoder_out_total))))
            konlpy_encoder_out_total = F.max_pool1d(konlpy_encoder_out_total.permute(0,2,1), konlpy_encoder_out_total.size(1)).squeeze(2)
            # konlpy_encoder_out = self.trs_trg_output_linear2_konlpy(konlpy_encoder_out)[:,0,:]

            # Concat
            encoder_out_total = self.embedding_concat_total_linear(torch.cat((spm_encoder_out_total, 
                                                                              khaiii_encoder_out_total, 
                                                                              konlpy_encoder_out_total), dim=1))
            encoder_out_total = self.trg_output_total_linear(encoder_out_total)

            #=====Content only=====#
            # SentencePiece input
            spm_encoder_out = self.src_cont_spm_embedding(spm_src)
            spm_encoder_out, *_ = self.trs_encoder_spm_cont(spm_encoder_out, spm_src_mask)
            spm_encoder_out = self.trs_trg_output_norm_spm(self.dropout(F.gelu(self.trs_trg_output_linear_spm(spm_encoder_out))))
            spm_encoder_out = F.max_pool1d(spm_encoder_out.permute(0,2,1), spm_encoder_out.size(1)).squeeze(2)
            # spm_encoder_out = self.trs_trg_output_linear2_spm(spm_encoder_out)[:,0,:]

            # Khaiii input
            khaiii_encoder_out = self.src_cont_khaiii_embedding(khaiii_src)
            khaiii_encoder_out, *_ = self.trs_encoder_khaiii_cont(khaiii_encoder_out, khaiii_src_mask)
            khaiii_encoder_out = self.trs_trg_output_norm_khaiii(self.dropout(F.gelu(self.trs_trg_output_linear_khaiii(khaiii_encoder_out))))
            khaiii_encoder_out = F.max_pool1d(khaiii_encoder_out.permute(0,2,1), khaiii_encoder_out.size(1)).squeeze(2)
            # khaiii_encoder_out = self.trs_trg_output_linear2_khaiii(khaiii_encoder_out)[:,0,:]

            # KoNLPy input
            konlpy_encoder_out = self.src_cont_konlpy_embedding(konlpy_src)
            konlpy_encoder_out, *_ = self.trs_encoder_konlpy_cont(konlpy_encoder_out, konlpy_src_mask)
            konlpy_encoder_out = self.trs_trg_output_norm_konlpy(self.dropout(F.gelu(self.trs_trg_output_linear_konlpy(konlpy_encoder_out))))
            konlpy_encoder_out = F.max_pool1d(konlpy_encoder_out.permute(0,2,1), konlpy_encoder_out.size(1)).squeeze(2)
            # konlpy_encoder_out = self.trs_trg_output_linear2_konlpy(konlpy_encoder_out)[:,0,:]

            # Concat
            encoder_out = self.embedding_concat_cont_linear(torch.cat((spm_encoder_out, khaiii_encoder_out, konlpy_encoder_out), dim=1))
            encoder_out = self.trg_output_cont_linear(encoder_out)

            # Model forward
            # encoder_out, *_ = self.transformer_encoder(encoder_out, src_mask)

            # encoder_out = self.src_total_embedding(spm_src)#.transpose(0, 1)
            # encoder_out = self.position_enc(self.src_embedding_trs(src_input_sentence))
            # encoder_out = self.embedding_norm(self.embedding_linear(encoder_out))
            # encoder_out, *_ = self.transformer_encoder(encoder_out, src_mask)
            # encoder_out = encoder_out.transpose(0, 1).contiguous()

            # encoder_out = self.trs_trg_output_norm(self.dropout(F.gelu(self.trs_trg_output_linear(encoder_out))))
            # encoder_out = self.trs_trg_output_linear2(encoder_out)[:,0,:]

        #===================================#
        #============Concatenate============#
        #===================================#

        if self.model_type == 'total':
            if self.bilinear:
                logit = self.output_linear(rnn_out, encoder_out)
            else:
                total_concat = torch.cat((gru_out, lstm_out, encoder_out, gap_out, encoder_out_total), dim=1)
                logit = self.output_linear2(F.gelu(self.output_linear1(total_concat)))

        elif self.model_type == 'gap':
            logit = gap_out
        elif self.model_type == 'rnn':
            logit = rnn_out
        elif self.model_type == 'transformer':
            logit = encoder_out

        return logit