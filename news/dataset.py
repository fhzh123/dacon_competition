# Import Module

import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, total_src_list, title_src_list, content_src_list, 
                 date_list, ord_list, trg_list=None, min_len=4, max_len=999):
        data = list()
        for total_src, title_src, content_src, date, order, trg in zip(
            total_src_list, title_src_list, content_src_list, date_list, ord_list, trg_list
        ):
            if min_len <= len(total_src) <= max_len:
                data.append((total_src, title_src, content_src, date, order, trg))
            else:
                print(len(total_src))
        
        self.data = data
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        total_src, title_src, content_src, date, order, trg = self.data[index]
        return total_src, title_src, content_src, date, order, trg
    
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
            total_src, title_src, content_src, date, order, trg = zip(*batch)
            return pack_sentence(total_src), pack_sentence(title_src), pack_sentence(content_src), \
                torch.LongTensor(date), torch.LongTensor(order), torch.LongTensor(trg)
        else:
            total_src, title_src, content_src, date, order, id_ = zip(*batch)
            return pack_sentence(total_src), pack_sentence(title_src), pack_sentence(content_src), \
                torch.LongTensor(date), torch.LongTensor(order), id_

    def __call__(self, batch):
        return self.pad_collate(batch)