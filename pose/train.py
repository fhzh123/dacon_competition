import os
from typing import Tuple, List, Sequence, Callable, Dict

import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import KeypointDataset, collate_fn

def training(args):

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data pre-setting
    dat = pd.read_csv(os.path.join(args.data_path,'train_df.csv'))
    index_list = list(range(len(dat)))
    random.shuffle(index_list)
    valid_count = int(len(index_list) * args.split)
    train_df = dat.iloc[index_list[:-valid_count]]
    valid_df = dat.iloc[index_list[-valid_count:]]

    # Transform setting
    transforms_dict = {
        'train': A.Compose([
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2()
        ], 
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True)),
        'valid': A.Compose([
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True))
    }

    # PyTorch dataloader setting
    dataset_dict = {
        'train': KeypointDataset(os.path.join(args.data_path, 'train_imgs/'), 
                                train_df, transforms_dict['train']), 
        'valid': KeypointDataset(os.path.join(args.data_path, 'train_imgs/'), 
                                valid_df, transforms_dict['valid']), 
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=collate_fn),
        'valid': DataLoader(dataset_dict['valid'], batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=collate_fn),
    }

    # Model setting
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )
    model = KeypointRCNN(
        backbone, 
        num_classes=2,
        num_keypoints=24,
        box_roi_pool=roi_pooler,
        keypoint_roi_pool=keypoint_roi_pooler
    )
    model = model.to(device)

    # Optimizer setting
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                    patience=len(dataloader_dict['train'])/1.5)

    # Resume
    start_epoch = 0
    if args.resume:
        print('resume!')
        checkpoint = torch.load(args.file_name, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.to(device)

    # Train start
    for epoch in range(start_epoch, args.num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
            for i, (images, targets) in enumerate(tqdm(dataloader_dict[phase])):
                # Optimizer setting
                optimizer.zero_grad()

                # Input, output setting
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.set_grad_enabled(phase == 'train'):
                    losses = model(images, targets)
                    loss = sum(loss for loss in losses.values())
                    if phase == 'train':
                        loss.backward()
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()

                    if (i+1) % 100 == 0:
                        print(f'| epoch: {epoch} | loss: {loss.item():.4f}', end=' | ')
                        for k, v in losses.items():
                            print(f'{k[5:]}: {v.item():.4f}', end=' | ')
                        print()
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, args.file_name)