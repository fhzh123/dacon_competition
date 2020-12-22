# Import Module

# Import PyTorch
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, spm_src_total_list, khaiii_src_total_list, konlpy_src_total_list,
                 spm_src_list, khaiii_src_list, konlpy_src_list, 
                 date_list, ord_list, id_list, trg_list=None, isTrain=True,
                 min_len=4, max_len=999):
        data = list()
        if isTrain:
            for spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_, trg in zip(
                spm_src_total_list, khaiii_src_total_list, konlpy_src_total_list, 
                spm_src_list, khaiii_src_list, konlpy_src_list, date_list, 
                ord_list, id_list, trg_list
            ):
                if min_len <= len(spm_total_src) <= max_len:
                    if min_len <= len(khaiii_total_src) <= max_len:
                        if min_len <= len(konlpy_total_src) <= max_len:
                            data.append((spm_total_src, khaiii_total_src, konlpy_total_src,
                                         spm_src, khaiii_src, konlpy_src, date, order, id_, trg))
        else:
            for spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_ in zip(
                spm_src_total_list, khaiii_src_total_list, konlpy_src_total_list,
                spm_src_list, khaiii_src_list, konlpy_src_list, date_list, 
                ord_list, id_list
            ):
                if min_len <= len(spm_total_src) <= max_len:
                    if min_len <= len(khaiii_total_src) <= max_len:
                        if min_len <= len(konlpy_total_src) <= max_len:
                            data.append((spm_total_src, khaiii_total_src, konlpy_total_src, 
                                         spm_src, khaiii_src, konlpy_src, date, order, id_))
        
        self.data = data
        self.isTrain = isTrain
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        if self.isTrain:
            spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_, trg = self.data[index]
            return spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_, trg
        else:
            spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_ = self.data[index]
            return spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_
    
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
            spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_, trg = zip(*batch)
            return pack_sentence(spm_total_src), pack_sentence(khaiii_total_src), pack_sentence(konlpy_total_src), \
                pack_sentence(spm_src), pack_sentence(khaiii_src), pack_sentence(konlpy_src), \
                torch.LongTensor(date), torch.LongTensor(order), id_, torch.LongTensor(trg)
        else:
            spm_total_src, khaiii_total_src, konlpy_total_src, spm_src, khaiii_src, konlpy_src, date, order, id_ = zip(*batch)
            return pack_sentence(spm_total_src), pack_sentence(khaiii_total_src), pack_sentence(konlpy_total_src), \
                pack_sentence(spm_src), pack_sentence(khaiii_src), pack_sentence(konlpy_src), \
                torch.LongTensor(date), torch.LongTensor(order), id_

    def __call__(self, batch):
        return self.pad_collate(batch)