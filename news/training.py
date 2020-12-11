# Import modules
import os
import time
import pickle
import datetime
import pandas as pd
import torch_optimizer as optim_lib

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import custom modules
from dataset import CustomDataset, PadCollate
from model.total_model import Total_model
from utils import WarmupLinearSchedule, LabelSmoothingLoss
# from model.optimizer import Ralamb, WarmupLinearSchedule

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'preprocessed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_vocab_num_dict = dict()

        total_train_text_indices_spm = data_['total_train_text_indices_spm']
        total_valid_text_indices_spm = data_['total_valid_text_indices_spm']
        total_train_text_indices_khaiii = data_['total_train_text_indices_khaiii']
        total_valid_text_indices_khaiii = data_['total_valid_text_indices_khaiii']
        total_train_text_indices_konlpy = data_['total_train_text_indices_konlpy']
        total_valid_text_indices_konlpy = data_['total_valid_text_indices_konlpy']
        train_date_list = data_['train_date_list']
        valid_date_list = data_['valid_date_list']
        train_ord_list = data_['train_ord_list']
        valid_ord_list = data_['valid_ord_list']
        train_info_list = data_['train_info_list']
        valid_info_list = data_['valid_info_list']
        word2id_spm = data_['word2id_spm']
        word2id_khaiii = data_['word2id_khaiii']
        word2id_konlpy = data_['word2id_konlpy']

        src_vocab_num_dict['spm'] = len(word2id_spm.keys())
        src_vocab_num_dict['khaiii'] = len(word2id_khaiii.keys())
        src_vocab_num_dict['konlpy'] = len(word2id_konlpy.keys())
        del data_

    dataset_dict = {
        'train': CustomDataset(total_train_text_indices_spm, total_train_text_indices_khaiii, 
                               total_train_text_indices_konlpy,
                               train_date_list, train_ord_list, train_info_list,
                               min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(total_valid_text_indices_spm, total_valid_text_indices_khaiii, 
                               total_valid_text_indices_konlpy,
                               valid_date_list, valid_ord_list, valid_info_list,
                               min_len=args.min_len, max_len=args.max_len),
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True),
        'valid': DataLoader(dataset_dict['valid'], collate_fn=PadCollate(), drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True)
    }
    print(f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model Setting===========#
    #===================================#

    print("Build model")
    model = Total_model(args.model_type, src_vocab_num_dict, trg_num=2, pad_idx=args.pad_idx, bos_idx=args.bos_idx,
                        eos_idx=args.eos_idx, max_len=args.max_len, d_model=args.d_model,
                        d_embedding=args.d_embedding, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                        bilinear=args.bilinear, num_transformer_layer=args.num_transformer_layer,
                        num_rnn_layer=args.num_rnn_layer, device=device)
    # optimizer = Ralamb(params=filter(lambda p: p.requires_grad, model.parameters()), 
    #                    lr=args.max_lr, weight_decay=args.w_decay)
    # optimizer = optim_lib.Lamb(params=model.parameters(), 
    #                        lr=args.max_lr, weight_decay=args.w_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.max_lr, momentum=args.momentum,
                          weight_decay=args.w_decay)
    if args.n_warmup_epochs != 0:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.n_warmup_epochs*len(dataloader_dict['train']), 
                                        t_total=len(dataloader_dict['train'])*args.num_epoch)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                      patience=len(dataloader_dict['train'])/1.5)
    criterion = LabelSmoothingLoss(classes=2, smoothing=args.label_smoothing)
    model.to(device)

    #===================================#
    #===========Model Training==========#
    #===================================#

    best_val_loss = None
    freq = 0

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    for e in range(args.num_epoch):
        start_time_e = time.time()
        print(f'Model Fitting: [{e+1}/{args.num_epoch}]')
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            if phase == 'valid':
                model.eval()
                val_loss = 0
                val_acc = 0
            for i, (src_spm, src_khaiii, src_konlpy, date, order, trg) in enumerate(dataloader_dict[phase]):

                # Optimizer setting
                optimizer.zero_grad()

                # Source, Target sentence setting
                src_spm = src_spm.to(device)
                src_khaiii = src_khaiii.to(device)
                src_konlpy = src_konlpy.to(device)
                trg = trg.to(device)

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    predicted_logit = model(src_spm, src_khaiii, src_konlpy)

                    # If phase train, then backward loss and step optimizer and scheduler
                    if phase == 'train':
                        loss = criterion(predicted_logit, trg)
                        loss.backward()
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                        if args.n_warmup_epochs != 0:
                            scheduler.step()
                        else:
                            scheduler.step(loss)
                        # Print loss value only training
                        if freq == args.print_freq or i == 0 or i == len(dataloader_dict['train']):
                            total_loss = loss.item()
                            _, predicted = predicted_logit.max(dim=1)
                            accuracy = sum(predicted == trg).item() / predicted.size(0)
                            print("[Epoch:%d][%d/%d] train_loss:%5.3f | Accuracy:%2.3f | lr:%1.6f | spend_time:%5.2fmin"
                                    % (e+1, i, len(dataloader_dict['train']), total_loss, accuracy, 
                                    optimizer.param_groups[0]['lr'], (time.time() - start_time_e) / 60))
                            freq = 0
                        freq += 1
                    if phase == 'valid':
                        loss = F.cross_entropy(predicted_logit, trg)
                        val_loss += loss.item()
                        _, predicted = predicted_logit.max(dim=1)
                        accuracy = sum(predicted == trg).item() / predicted.size(0)
                        val_acc += accuracy

            # Finishing iteration
            if phase == 'valid':
                val_loss /= len(dataloader_dict['valid'])
                val_acc /= len(dataloader_dict['valid'])
                print("[Epoch:%d] val_loss:%5.3f | Accuracy:%5.2f | spend_time:%5.2fmin"
                        % (e+1, val_loss, val_acc, (time.time() - start_time_e) / 60))
                if not best_val_loss or val_loss < best_val_loss:
                    print("[!] saving model...")
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    torch.save(model.state_dict(), 
                               os.path.join(args.model_path, f'model_saved.pt'))
                    best_val_loss = val_loss

    #===================================#
    #============Result save============#
    #===================================#

    if not os.path.isfile(os.path.join(args.save_path, 'results.csv')):
        column_list = ['date_time', 'best_val_loss', 'tokenizer', 'valid_percent', 'vocab_size',
                       'num_epoch', 'batch_size', 'max_len', 'n_warmup_epochs', 'max_lr',
                       'momentum', 'w_decay', 'dropout', 'grad_clip', 'model_type', 'bilinear',
                       'num_transformer_layer', 'num_rnn_layer', 'd_model', 'd_embedding',
                       'd_k', 'd_v', 'n_head', 'dim_feedforward']
        pd.DataFrame(columns=column_list).to_csv(os.path.join(args.save_path, 'results.csv'), index=False)
    
    results_dat = pd.read_csv(os.path.join(args.save_path, 'results.csv'))
    new_row = {
        'date_time':datetime.datetime.today().strftime('%m/%d/%H:%M'),
        'best_val_loss': best_val_loss,
        'tokenizer': args.sentencepiece_tokenizer,
        'valid_percent': args.valid_percent,
        'vocab_size': args.vocab_size,
        'num_epoch': args.num_epoch,
        'batch_size': args.batch_size,
        'max_len': args.max_len,
        'n_warmup_epochs': args.n_warmup_epochs,
        'max_lr': args.max_lr,
        'momentum': args.momentum,
        'w_decay': args.w_decay,
        'dropout': args.dropout,
        'grad_clip': args.grad_clip,
        'model_type': args.model_type,
        'bilinear': args.bilinear,
        'num_transformer_layer': args.num_transformer_layer,
        'num_rnn_layer': args.num_rnn_layer,
        'd_model': args.d_model,
        'd_embedding': args.d_embedding,
        'd_k': args.d_k,
        'd_v': args.d_v,
        'n_head': args.n_head,
        'dim_feedforward': args.dim_feedforward,
        'label_smoothing': args.label_smoothing
    }
    results_dat = results_dat.append(new_row, ignore_index=True)
    results_dat.to_csv(os.path.join(args.save_path, 'results.csv'), index=False)