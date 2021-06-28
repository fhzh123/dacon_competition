# Import modules
import os
import gc
import time
import logging
from apex import amp
# Import PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from dataset import CustomDataset, PadCollate
from model.transformer import Transformer
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log

def training(args):
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
    unique_menu_count = dataset_dict['train'].unique_count()
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
    model = Transformer(model_type=args.model_type, input_size=unique_menu_count,
                        d_model=args.d_model, d_embedding=args.d_embedding,
                        n_head=args.n_head, dim_feedforward=args.dim_feedforward,
                        num_encoder_layer=args.num_encoder_layer,
                        dropout=args.dropout)
    model = model.train()
    model = model.to(device)

    # 2) Optimizer setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=True)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

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

    best_val_rmse = 9999999

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                train_start_time = time.time()
                freq = 0
            elif phase == 'valid':
                model.eval()
                val_loss = 0
                val_rmse = 0

            for i, (src_menu, label_lunch, label_supper) in enumerate(dataloader_dict[phase]):

                # Optimizer setting
                optimizer.zero_grad()

                # Input, output setting
                src_menu = src_menu.to(device, non_blocking=True)
                label_lunch = label_lunch.float().to(device, non_blocking=True)
                label_supper = label_supper.float().to(device, non_blocking=True)

                # Model
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(enabled=True):
                        if args.model_type == 'sep':
                            logit = model(src_menu)
                            logit_lunch = logit[:,0]
                            logit_supper = logit[:,0]
                        elif args.model_type == 'total':
                            logit = model(src_menu)
                            logit_lunch = logit[:,0]
                            logit_supper = logit[:,1]

                    # Loss calculate
                    loss_lunch = criterion(logit_lunch, label_lunch)
                    loss_supper = criterion(logit_supper, label_supper)
                    loss = loss_lunch + loss_supper

                # Back-propagation
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    # Scheduler setting
                    if args.scheduler in ['constant', 'warmup']:
                        scheduler.step()
                    if args.scheduler == 'reduce_train':
                        scheduler.step(loss)

                # Print loss value
                rmse_loss = torch.sqrt(loss)
                if phase == 'train':
                    if i == 0 or freq == args.print_freq or i==len(dataloader_dict['train']):
                        batch_log = "[Epoch:%d][%d/%d] train_MSE_loss:%2.3f  | train_RMSE_loss:%2.3f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                                % (epoch+1, i, len(dataloader_dict['train']), 
                                loss.item(), rmse_loss.item(), optimizer.param_groups[0]['lr'], 
                                (time.time() - train_start_time) / 60)
                        write_log(logger, batch_log)
                        freq = 0
                    freq += 1
                elif phase == 'valid':
                    val_loss += loss.item()
                    val_rmse += rmse_loss.item()

        if phase == 'valid':
            val_loss /= len(dataloader_dict['valid'])
            val_rmse /= len(dataloader_dict['valid'])
            write_log(logger, 'Validation Loss: %3.3f' % val_loss)
            write_log(logger, 'Validation RMSE: %3.3f' % val_rmse)

            if val_rmse < best_val_rmse:
                write_log(logger, 'Checkpoint saving...')
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict()
                }, os.path.join(args.save_path, f'checkpoint_cap.pth.tar'))
                best_val_rmse = val_rmse
                best_epoch = epoch
            else:
                else_log = f'Still {best_epoch} epoch RMSE({round(best_val_rmse, 3)}) is better...'
                write_log(logger, else_log)

    # 3)
    write_log(logger, f'Best Epoch: {best_epoch+1}')
    write_log(logger, f'Best Accuracy: {round(best_val_rmse, 3)}')