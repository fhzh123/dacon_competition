import os
import ast
import pickle
import pandas as pd
# Import PyTorch
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path: str, phase: str = 'train'):
        self.phase = phase
        # Data load
        if phase == 'train':
            df = pd.read_csv(os.path.join(data_path, 'train_processed.csv'))
        elif phase == 'valid':
            df = pd.read_csv(os.path.join(data_path, 'valid_processed.csv'))
        elif phase == 'test':
            df = pd.read_csv(os.path.join(data_path, 'test_processed.csv'))
        with open(os.path.join(data_path, 'menu_dict.pkl'), 'rb')as f:
            menu_dict = pickle.load(f)
        cls_token = len(menu_dict) + 1
        sep_token = len(menu_dict) + 2

        # Append parsed data
        data = list()
        lunch_label, supper_label = list(), list()
        test_date = list()
        for i in range(len(df)):
            m1 = ast.literal_eval(df.iloc[i]['breakfast_parsing'])
            m2 = ast.literal_eval(df.iloc[i]['lunch_parsing'])
            m3 = ast.literal_eval(df.iloc[i]['supper_parsing'])
            data.append([cls_token] + m1 + [sep_token] + m2 + [sep_token] + m3)
            if phase in ['train', 'valid']:
                lunch_label.append(df.iloc[i]['target_lunch'])
                supper_label.append(df.iloc[i]['target_supper'])
            elif phase == 'test':
                test_date.append(df.iloc[i]['datetime'])

        # Self setting
        self.data = tuple(data)
        self.lunch_label = tuple(lunch_label)
        self.supper_label = tuple(supper_label)
        self.test_date = tuple(test_date)
        self.num_data = len(self.data)

    def __getitem__(self, index):
        total = self.data[index]
        if self.phase == 'test':
            date = self.test_date[index]
            return total, date
        else:
            lunch_target, supper_target = self.lunch_label[index], self.supper_label[index]
            return total, lunch_target, supper_target
    
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

        output_ = zip(*batch)
        if len(list(output_)) == 3:
            total, lunch_target, supper_target = zip(*batch)
            return pack_sentence(total), torch.DoubleTensor(lunch_target), torch.DoubleTensor(supper_target)
        else:
            total, date = zip(*batch)
            return pack_sentence(total), date

    def __call__(self, batch):
        return self.pad_collate(batch)