import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_path, isTrain=True, transform=None):
        self.img_path = img_path
        self.isTrain = isTrain

        if isTrain:
            self.label = [img.split('train/')[1][0] for img in self.img_path]
        else:
            self.id = [int(img.split('test/')[1].split('_')[0]) for img in self.img_path]
        self.letter = [img[-5] for img in self.img_path]

        self.num_data = len(self.img_path)
        self.transform = transform

    def __getitem__(self, index):
        letter = self.letter[index]
        image = Image.open(self.img_path[index])
        image = image.convert('RGB')
        # Image Augmentation
        if self.transform is not None:
            image = self.transform(image)
        # Return Value
        if self.isTrain:
            label = self.label[index]
            return image, letter, label
        else:
            id_ = self.id[index]
            return image, letter, id_

    def __len__(self):
        return self.num_data