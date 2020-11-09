# Import Module
import sentencepiece as spm
from itertools import chain
from random import random, randrange

import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, src_list, trg_list, idx_list, min_len=4, max_len=500):
        data = list()
        for src, trg, idx in zip(src_list, trg_list, idx_list):
            if min_len <= len(src) <= max_len:
                data.append((src, trg, idx))
        
        self.data = data
        self.num_data = len(self.data)
        
    def __getitem__(self, index):
        src, trg, idx = self.data[index]
        return src, trg, idx
    
    def __len__(self):
        return self.num_data

class PadCollate:
    def __init__(self, pad_index=0, dim=0):
        self.dim = dim
        self.pad_index = pad_index

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

        src_sentences, output, idx = zip(*batch)
        return pack_sentence(src_sentences), torch.LongTensor(output), torch.LongTensor(idx)

    def __call__(self, batch):
        return self.pad_collate(batch)