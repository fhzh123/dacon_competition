# Import modules
import os
import gc
import time
import logging
# Import PyTorch
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler
# Import custom modules
from dataset import CustomDataset, PadCollate
from model.transformer import Transformer
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, scaler, logger, device):

    # Train setting
    start_time_e = time.time()
    model = model.train()

    for i, (img, label) in enumerate(dataloader):

        # Optimizer setting
        optimizer.zero_grad()

        # Input, output setting
        img = img.to(device, non_blocking=True)
        label = label.long().to(device, non_blocking=True)

        # Model
        logit = model(img)
        first_token = logit[:,0,:]
        # first_token = logit

        # Loss calculate
        loss = label_smoothing_loss(first_token, label, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if args.scheduler in ['constant', 'warmup']:
            scheduler.step()
        if args.scheduler == 'reduce_train':
            scheduler.step(mlm_loss)

        # Print loss value only training
        acc = (((first_token.argmax(dim=1) == label).sum()) / label.size(0)) * 100
        if i == 0 or freq == args.print_freq or i==len(dataloader):
            batch_log = "[Epoch:%d][%d/%d] train_loss:%2.3f  | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                    % (epoch+1, i, len(dataloader), 
                    loss.item(), acc.item(), optimizer.param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60)
            write_log(logger, batch_log)
            freq = 0
        freq += 1

def valid_epoch(args, model, dataloader, device):

    # Validation setting
    model = model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(dataloader):

            # Input, output setting
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # Model
            logit = model(img)
            first_token = logit[:,0,:]
            # first_token = logit

            # Loss calculate
            loss = F.cross_entropy(first_token, label)

            # Print loss value only training
            acc = (((first_token.argmax(dim=1) == label).sum()) / label.size(0)) * 100
            val_loss += loss.item()
            val_acc += acc.item()

    return val_loss, val_acc

def vit_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Dataloader setting
    write_log(logger, "Load data...")
    gc.disable()
    dataset_dict = {
        'train': CustomDataset(data_path=args.preprocessed_path, phase='train'),
        'valid': CustomDataset(data_path=args.preprocessed_path, phase='valid'),
        'test': CustomDataset(data_path=args.preprocessed_path, phase='test')
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers, collate_fn=PadCollate()),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers, collate_fn=PadCollate()),
        'test': DataLoader(dataset_dict['test'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers, collate_fn=PadCollate())
    }
    gc.enable()
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")
    model = Transformer(n_classes=1000, d_model=args.d_model, d_embedding=args.d_embedding, 
                               n_head=args.n_head, dim_feedforward=args.dim_feedforward,
                               num_encoder_layer=args.num_encoder_layer, img_size=args.img_size, 
                               patch_size=args.patch_size, dropout=args.dropout,
                               triple_patch=args.triple_patch)
    model = model.train()
    model = model.to(device)

    # 2) Optimizer setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 2) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint.pth.tar'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.train()
        model = model.to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):

        train_epoch(args, epoch, model, dataloader_dict['train'], optimizer, scheduler, scaler, logger, device)
        val_loss, val_acc = valid_epoch(args, model, dataloader_dict['valid'], device)

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)
        if val_acc > best_val_acc:
            write_log(logger, 'Checkpoint saving...')
            # Checkpoint path setting
            if not os.path.exists(args.vit_save_path):
                os.mkdir(args.vit_save_path)
            # Save
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(args.vit_save_path, f'checkpoint.pth.tar'))
            best_val_acc = val_acc
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc, 2)})% is better...'
            write_log(logger, else_log)

    # 3)
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Accuracy: {round(best_val_acc, 2)}')