# Import modules
import os
import time
import argparse
import datetime
import numpy as np

# Import PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import custom modules
from dataset import CustomDataset
from model.efficientDet.loss import FocalLoss
from model.backbone import EfficientDetBackbone
from Transforms import Compose, Resize, Normalize, ToTensor
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string


def collate_fn(batch):

    return tuple(zip(*batch))


def train(args):

    #===================================#
    #============Pre-setting============#
    #===================================#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = Compose([
        Resize((args.resize_pixel, args.resize_pixel)),
        ToTensor(),
        Normalize()
    ])

    #===================================#
    #=========Dataloader setting========#
    #===================================#

    dataset_dict = {
        'train': CustomDataset(data_path=args.folder_path, json_path=args.json_path,
                               transforms=data_transform),
        'valid': CustomDataset(data_path=args.folder_path, json_path=args.json_path,
                               transforms=data_transform),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=True, collate_fn=collate_fn),
        'valid': DataLoader(dataset_dict['valid'], batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False, collate_fn=collate_fn)
    }

    #===================================#
    #===========Model setting===========#
    #===================================#

    model = EfficientDetBackbone(num_classes=16, compound_coef=0)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=len(dataloader_dict['train'])/1.5, 
                                                           verbose=True)
    criterion = FocalLoss()
    model.to(device)


    #===================================#
    #=============Training==============#
    #===================================#

    best_val_loss = None
    freq = 0

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    for e in range(args.num_epochs):
        start_time_e = time.time()
        print(f'Model Fitting: [{e+1}/{args.num_epochs}]')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
                val_cls_loss = 0
                val_reg_loss = 0

            for i, (images, targets) in enumerate(dataloader_dict[phase]):

                # Optimizer setting
                optimizer.zero_grad()

                # Source, target device setting
                imgs = [image.to(device) for image in images]
                targets = [{ k: v.to(device) for k, v in t.items()}  for t in targets ]
                #annotation_ = data['annot'].to(device)
            
                for img, target in zip(imgs, targets):

                    labels=target['labels'].unsqueeze(0)

                    if target['labels'].shape[0]>1 :
                        
                         labels=target['labels'][0].reshape(1,1)
                         bbox = target['boxes'][0].reshape(1,4)
                         annotation_=torch.cat((bbox,labels), axis=1)
                         annotation_=annotation_.type(torch.FloatTensor).to(device)
                      

                    else:

                        labels=target['labels'].reshape(1,1)
                        bbox = target['boxes'].reshape(1,4)
                        annotation_=torch.cat((bbox,labels), axis=1)
                        annotation_=annotation_.type(torch.FloatTensor).to(device)

                    #annotation_ =  torch.cat(imgs,targets).to(device)

                        # Model / Calculate loss
                    with torch.set_grad_enabled(phase == 'train'):
                        _, regression, classification, anchors = model(img.unsqueeze(0).to(device))
                        cls_loss, reg_loss = criterion(classification, regression,
                                                        anchors, annotation_)
                        cls_loss, reg_loss = cls_loss.mean(), reg_loss.mean()
                    
                        # If phase train, then backward loss and step optimizer and scheduler
                    if phase == 'train':
                        loss = cls_loss + reg_loss 
                        loss.backward()
                        optimizer.step()
                        scheduler.step(loss)

                        # Print loss value only training
        if freq == args.print_freq or i == 0 or i == len(dataloader_dict['train']):
            cls_loss_print = cls_loss.item()
            reg_loss_print = reg_loss.item()
            print("[Epoch:%d][%d/%d] cls_loss_print:%3.3f | reg_loss_print:%3.3f | lr:%1.6f | spend_time:%5.2fmin"
                    % (e+1, i, len(dataloader_dict['train']), cls_loss_print, reg_loss_print, 
                      optimizer.param_groups[0]['lr'], (time.time() - start_time_e) / 60))
            freq = 0
            freq += 1
                                
        if phase == 'valid':
            val_cls_loss += cls_loss.item()
            val_reg_loss += reg_loss.item()

            # Finishing iteration
        if phase == 'valid':
            val_cls_loss /= len(dataloader_dict['valid'])
            val_reg_loss /= len(dataloader_dict['valid'])
            print("[Epoch:%d] val_cls_loss:%3.3f | val_reg_loss:%3.3f | spend_time:%5.2fmin"
                % (e+1, val_cls_loss, val_reg_loss, (time.time() - start_time_e) / 60))
            if not best_val_loss or val_reg_loss < best_val_loss:
                print("[!] saving model...")
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                    torch.save(model.state_dict(), 
                               os.path.join(args.model_path, f'model_saved.pt'))
                    best_val_loss = val_reg_loss

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='parser')
    
    # Image setting
    parser.add_argument('--folder_path', type=str, default='./Dataset',
                        help='Dataset saved path setting')
    parser.add_argument('--resize_pixel', type=int, default=256, help='Resize pixel')
    # Train setting
    parser.add_argument('--num_epochs',type=int, default=1, help='The number of epoch')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--optim', type=str, default='adamW', help='learning_rate')
    parser.add_argument('--log_path',type=str, default='./logs', help='log_path')
    parser.add_argument('--num_workers', type=int, default=4, help='CPU Worker setting')
    parser.add_argument('--saved_path',type=str, default='./checkpoint', help='save_path')
    parser.add_argument('--load_weights', type=str, default="./pretrained_model/efficientdet-d0.pth",
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--head_only', type=boolean_string, default=True,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--print_freq', type=int, default=100, help='Print frequency')
    # Save path
    #parser.add_argument('--model_path', type=str, default='/HDD/sujincho/IPIU_fashionretrieval/model')
    parser.add_argument('--model_path', type=str, default='../model')
    args=parser.parse_args()
    train(args)
