# Import Modules
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

# Import PyTorch
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Import Custom Module
from dataset import CustomDataset
from model import conv_model

def main(args):
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Information Load
    with open(os.path.join(args.model_path, 'hyperparameter.json'), 'r') as f:
        info_ = json.load(f)
    resize_pixel = info_['resize_pixel']

    # Model Load & Setting
    model = conv_model(efficientnet_not_use=False,
                       efficient_model_number=6,
                       letter_model_path=args.letter_model_path)
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'model.pt')))
    model.eval()
    model.to(device)

    # Data Setting
    test_img_list = glob(os.path.join(args.data_path, 'test/*.jpg'))
    test_transforms = transforms.Compose([
        transforms.Resize((resize_pixel, resize_pixel)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_dataset = CustomDataset(test_img_list, isTrain=False, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Evaluation
    id_list = list()
    pred_list = list()

    for inputs, letters, id_ in tqdm(test_dataloader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            id_list.extend(id_.tolist())
            pred_list.extend(preds.tolist())

    # Submission CSV Setting
    submission = pd.DataFrame({
        'id': id_list,
        'digit': pred_list
    })
    submission = submission.sort_values(by=['id'])
    submission.to_csv(os.path.join(args.submission_save_path, 'submission.csv'), index=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Order_net argparser')
    # Path Setting
    parser.add_argument('--data_path', type=str, default='./data', help='Data path setting')
    parser.add_argument('--model_path', type=str, default='./data', help='Data path setting')
    parser.add_argument('--submission_save_path', type=str, default='./KH/save')
    parser.add_argument('--letter_model_path', type=str, default='./KH/save/letter/2020-08-28_06:29:07.02/')
    # Image Setting
    parser.add_argument('--batch_size', type=int, default=32, help='Test batch size')
    args = parser.parse_args()

    total_start_time = time.time()
    main(args)
    print('Done! {:.4f}min spend!'.format((time.time() - total_start_time)/60))