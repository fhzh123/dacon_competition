# Import modules
import os
import copy
import time
import albumentations as A
from efficientnet_pytorch import EfficientNet

# Import PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# Import custom modules
from dataset import CustomDataset
from optimizer import RAdam, WarmupLinearSchedule
from utils import train_valid_split

def train(args):
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformation & augmentation
    data_transforms = {
        'train': A.Compose([
            A.Resize(args.resize_pixel, args.resize_pixel, always_apply=True),
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            # A.pytorch.transforms.ToTensor(),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
        'valid': A.Compose([
            A.Resize(args.resize_pixel, args.resize_pixel, always_apply=True),
            # A.pytorch.ToTensor(),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }

    # Train, valid data split
    train_img_list, valid_img_list = train_valid_split(random_seed=args.random_seed, 
                                                       data_path=args.data_path,
                                                       valid_ratio=args.valid_ratio)

    # Custom dataset & dataloader setting
    image_datasets = {
        'train': CustomDataset(train_img_list, args.data_path, transform=data_transforms['train'],
                               isTest=False),
        'valid': CustomDataset(valid_img_list, args.data_path, transform=data_transforms['valid'],
                               isTest=False)
    }
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers),
        'valid': DataLoader(image_datasets['valid'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers),
    }

    # Training setting
    model = EfficientNet.from_pretrained(f'efficientnet-b{args.efficientnet_model_number}', 
                                         num_classes=1049)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloaders['train'])*3, 
                                     t_total=len(dataloaders['train'])*args.num_epochs)
    model.to(device)

    # Initialize
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9999999999
    early_stop = False

    # Train start
    for epoch in range(args.num_epochs):
        start_time_e = time.time()
        print('[Epoch {}/{}]'.format(epoch + 1, args.num_epochs))

        if early_stop:
            print('Early Stopping!!!')
            break

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()

            # Iterate over data
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = torch.tensor([int(x) for x in labels]).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # If phase train, then backward loss and step optimizer and scheduler
                    if phase == 'train':
                        loss.backward()
                        torch_utils.clip_grad_norm_(model.parameters(), 
                                                    args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()

                        # Print loss value only training
                        if i == 0 or freq == args.print_freq or i==len(dataloaders['train']):
                            total_loss = loss.item()
                            print("[Epoch:%d][%d/%d] train_loss:%5.3f | learning_rate:%3.8f | spend_time:%5.2fmin"
                                    % (epoch+1, i, len(dataloaders['train']), 
                                    total_loss, optimizer.param_groups[0]['lr'], 
                                    (time.time() - start_time_e) / 60))
                            freq = 0
                        freq += 1

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                if phase=='valid':
                    val_loss = running_loss
                running_corrects += torch.sum(preds == labels.data)

            # Epoch loss calculate
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            if phase == 'valid' and epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train' and epoch_loss < 0.001:
                early_stop = True

            spend_time = (time.time() - start_time) / 60
            print('{} Loss: {:.4f} Acc: {:.4f} Time: {:.3f}min'.format(phase, epoch_loss, epoch_acc, spend_time))

    # Model Saving
    model.load_state_dict(best_model_wts)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    best_loss = round(best_loss, 4)
    save_path_ = os.path.join(args.save_path, str(datetime.datetime.now())[:10] + f'_{best_loss}')
    os.mkdir(save_path_)
    print('Best validation loss: {:.4f}'.format(best_loss))
    with open(os.path.join(save_path_, 'hyperparameter.json'), 'w') as f:
        json.dump({
            'efficientnet_not_use': args.efficientnet_not_use,
            'efficientnet_model_number': args.efficientnet_model_number,
            'num_epochs': args.num_epochs,
            'resize_pixel': args.resize_pixel,
            'random_affine': args.random_affine,
            'lr': args.lr,
            'random_seed': args.random_seed,
            'best_loss': best_loss
        }, f)
    torch.save(model.state_dict(), os.path.join(save_path_, 'model.pt'))