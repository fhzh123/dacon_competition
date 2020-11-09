import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
# from PIL import Image

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_path, data_path, transform=None, isTest=False):
        if not isTest:
            self.isTest = False
            self.img_path = img_path
            # self.img_path = glob(os.path.join(data_path, 'train/*/*/*.JPG'))

            label_data = pd.read_csv(os.path.join(data_path, 'public/train.csv'))
            img_path_label = pd.DataFrame({'id': [x.split('/')[-1][:-4] for x in self.img_path]})
            self.label = pd.merge(label_data, img_path_label, on='id')['landmark_id'].tolist()
        
        else:
            self.isTest = True
            sefl.img_path = img_path
            # self.img_path = glob(os.path.join(data_path, 'test/*/*.JPG'))
            self.img_id = [x.split('/')[-1][:-4] for x in img_path]

        self.num_data = len(self.img_path)
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.img_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Image Augmentation
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # Return Value
        if not self.isTest:
            label = self.label[index]
            return torch.tensor(image, dtype=torch.float), label
        else:
            img_id = self.img_id[index]
            return torch.tensor(image, dtype=torch.float), img_id

    def __len__(self):
        return self.num_data