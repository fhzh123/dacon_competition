# Import PyTorch
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from .layer import TransformerEncoderLayer

class Transformer(nn.Module):
    def __init__(self, model_type: str = 'sep', input_size: int = 3500,
                 d_model: int = 512, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 10, dropout: float = 0.3):
    
        super(Transformer, self).__init__()

        self.model_type = model_type
        self.dropout = nn.Dropout(dropout)
        if model_type == 'sep':
            n_classes = 1
        elif model_type == 'total':
            n_classes = 2
        else:
            raise NameError(f'{model_type} is not defined')

        # Image embedding part
        self.src_input_linear = nn.Embedding(input_size, d_embedding)
        self.src_input_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.src_input_linear2 = nn.Linear(d_embedding, d_model)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])
        
        # Transformer Encoder part 2: Seperate model type (sep)
        self_attn2 = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders2 = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn2, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Target linear part (Not averaging)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, n_classes)

        if model_type == 'sep':
            self.trg_output_linear_sep = nn.Linear(d_model, d_embedding)
            self.trg_output_norm_sep = nn.LayerNorm(d_embedding, eps=1e-12)
            self.trg_output_linear2_sep = nn.Linear(d_embedding, n_classes)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p) 

    @autocast()
    def forward(self, src_menu: Tensor) -> Tensor:
        # Image embedding
        emb_out = self.src_input_norm(self.dropout(F.gelu(self.src_input_linear(src_menu))))
        emb_out = self.src_input_linear2(emb_out)
        encoder_out = emb_out.transpose(0, 1)

        # Transformer Encoder
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out)

        # Target linear
        encoder_out = encoder_out.transpose(0, 1)
        encoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(encoder_out))))
        encoder_out = self.trg_output_linear2(encoder_out)

        # If model_type is 'sep'
        if self.model_type == 'sep':
            encoder_out2 = emb_out.transpose(0, 1)
            for encoder in self.encoders2:
                encoder_out2 = encoder(encoder_out2)

            encoder_out2 = encoder_out2.transpose(0, 1)
            encoder_out2 = self.trg_output_norm_sep(self.dropout(F.gelu(self.trg_output_linear_sep(encoder_out2))))
            encoder_out2 = self.trg_output_linear2_sep(encoder_out2)

        if self.model_type == 'total':
            logit = encoder_out[:,0,:]
        elif self.model_type == 'sep':
            logit = torch.cat([encoder_out[:,0,:], encoder_out2[:,0,:]], dim=1)

        return logit