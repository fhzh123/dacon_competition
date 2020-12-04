import torch
import torch.nn as nn

class SegmentEmbedding(nn.Module):

    def __init__(self, d_embedding, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.seg_embedding = nn.Embedding(3, d_embedding)

    def forward(self, x):
        # Segment token setting
        sep_ix_list = (x==4).nonzero().transpose(0,1)[1]
        eos_ix_list = (x==2).nonzero().transpose(0,1)[1] - sep_ix_list

        sep_seg_list = [torch.LongTensor([1]).repeat(ix.item()) for ix in sep_ix_list]
        eos_seg_list = [torch.LongTensor([2]).repeat(ix.item()) for ix in eos_ix_list]

        # Segment token 
        for i, (sep_seg, eos_seg) in enumerate(zip(sep_seg_list, eos_seg_list)):
            sep_ix_len = len(sep_seg)
            eos_ix_lne = len(eos_seg)
            end_seg = torch.LongTensor([0]).repeat(x.size(1) - sep_ix_len - eos_ix_lne)
            segment_ = torch.cat((sep_seg, eos_seg, end_seg), dim=0).unsqueeze(0)
            if i == 0:
                segment_sequence = segment_
            else:
                segment_sequence = torch.cat((segment_sequence, segment_), dim=0)

        # Segment embedding
        segment_sequence = self.seg_embedding(segment_sequence.cuda())
        return segment_sequence