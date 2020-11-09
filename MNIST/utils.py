# Import Modules
import os
import fcntl
import struct
import termios
import numpy as np
from glob import glob

import torch

def terminal_size():
    th, tw, hp, wp = struct.unpack('HHHH',
        fcntl.ioctl(0, termios.TIOCGWINSZ,
        struct.pack('HHHH', 0, 0, 0, 0)))
    return tw

def train_valid_split(random_seed, data_path, valid_ratio):
    # Image load
    np.random.seed(random_seed)
    total_train_img_list = glob(os.path.join(data_path, 'train/*/*.jpg'))
    # Data split
    valid_size = int(len(total_train_img_list)*valid_ratio)
    valid_img_list = list(np.random.choice(total_train_img_list, size=valid_size))
    train_img_list = list(set(total_train_img_list) - set(valid_img_list))
    return train_img_list, valid_img_list

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        xavier(m.bias.data)

# model.apply(weights_init)