# Import PyTorch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.modules.activation import MultiheadAttention
# Import custom modules
from ..model.layer import TransformerEncoderLayer
from ..transformer.embedding import PatchEmbedding

class Transformer(nn.Module):
    def __init__(self, model_type: str = 'sep', d_model: int = 512, d_embedding: int = 256, 
                 n_head: int = 8, dim_feedforward: int = 2048,
                 num_encoder_layer: int = 10, dropout: float = 0.3):
    
        super(Transformer, self).__init__()

        self.model_type = model_type
        self.dropout = nn.Dropout(dropout)

        # Image embedding part
        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size,
            d_model=d_model, img_size=img_size, triple_patch=triple_patch)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Target linear part (Not averaging)
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, n_classes)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p) 

    @autocast
    def forward(self, src_img: Tensor) -> Tensor:
        # Image embedding
        encoder_out = self.patch_embedding(src_img).transpose(0, 1)
        
        # Transformer Encoder
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out)

        # Target linear
        encoder_out = encoder_out.transpose(0, 1)
        encoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(encoder_out))))
        encoder_out = self.trg_output_linear2(encoder_out)
        return encoder_out
