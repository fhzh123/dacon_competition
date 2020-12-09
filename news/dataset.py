# Import Module

import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, spm_src_list, khaiii_src_list, konlpy_src_list, 
                 date_list, ord_list, trg_or_id_list, min_len=4, max_len=999):
        data = list()
        for spm_src, khaiii_src, konlpy_src, date, order, trg_or_id in zip(
            spm_src_list, khaiii_src_list, konlpy_src_list, date_list, ord_list, trg_or_id_list
        ):
            if min_len <= len(khaiii_src) <= max_len:
                data.append((spm_src, khaiii_src, konlpy_src, date, order, trg_or_id))
        
        self.data = data
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        spm_src, khaiii_src, konlpy_src, date, order, trg_or_id = self.data[index]
        return spm_src, khaiii_src, konlpy_src, date, order, trg_or_id
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0, isTrain=True):
        self.dim = dim
        self.pad_index = pad_index
        self.isTrain = isTrain

    def pad_collate(self, batch):
        def pad_tensor(vec, max_len, dim):
            pad_size = list(vec.shape)
            pad_size[dim] = max_len - vec.size(dim)
            return torch.cat([vec, torch.LongTensor(*pad_size).fill_(self.pad_index)], dim=dim)

        def pack_sentence(sentences):
            sentences_len = max(map(lambda x: len(x), sentences))
            sentences = [pad_tensor(torch.LongTensor(seq), sentences_len, self.dim) for seq in sentences]
            sentences = torch.cat(sentences)
            sentences = sentences.view(-1, sentences_len)
            return sentences

        if self.isTrain:
            spm_src, khaiii_src, konlpy_src, date, order, trg = zip(*batch)
            return pack_sentence(spm_src), pack_sentence(khaiii_src), pack_sentence(konlpy_src), \
                torch.LongTensor(date), torch.LongTensor(order), torch.LongTensor(trg)
        else:
            spm_src, khaiii_src, konlpy_src, date, order, id_ = zip(*batch)
            return pack_sentence(spm_src), pack_sentence(khaiii_src), pack_sentence(konlpy_src), \
                torch.LongTensor(date), torch.LongTensor(order), id_

    def __call__(self, batch):
        return self.pad_collate(batch)