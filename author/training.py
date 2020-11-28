# Import modules
import os
import time
import pickle

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# Import custom modules
from dataset import CustomDataset, PadCollate
from model.transformer import Transformer
from model.optimizer import Ralamb, WarmupLinearSchedule

def training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'preprocessed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        train_text_indices = data_['train_text_indices']
        valid_text_indices = data_['valid_text_indices']
        train_author_indices = data_['train_author_indices']
        valid_author_indices = data_['valid_author_indices']
        train_index_indices = data_['train_index_indices']
        valid_index_indices = data_['valid_index_indices']
        vocab_list = data_['vocab_list']
        vocab_num = len(vocab_list)
        word2id = data_['word2id']
        del data_

    dataset_dict = {
        'train': CustomDataset(train_text_indices, train_author_indices, train_index_indices,
                            min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(valid_text_indices, valid_author_indices, valid_index_indices,
                            min_len=args.min_len, max_len=args.max_len)
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
    model = Transformer(vocab_num, author_num=5, pad_idx=args.pad_idx, bos_idx=args.bos_idx,
                        eos_idx=args.eos_idx, max_len=args.max_len, d_model=args.d_model,
                        d_embedding=args.d_embedding, n_head=args.n_head,
                        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                        num_encoder_layer=args.num_encoder_layer, device=device)
    optimizer = Ralamb(params=filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=args.max_lr, weight_decay=args.w_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.n_warmup_epochs*len(dataloader_dict['train']), 
                                     t_total=len(dataloader_dict['train'])*args.num_epoch)
    model.to(device)

    #===================================#
    #===========Model Training==========#
    #===================================#

    best_val_loss = None
    freq = 0

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
            for i, (src, trg, index_) in enumerate(dataloader_dict[phase]):

                # Optimizer setting
                optimizer.zero_grad()

                # Source, Target sentence setting
                src = src.to(device)
                trg = trg.to(device)

                # Model / Calculate loss
                with torch.set_grad_enabled(phase == 'train'):
                    predicted_logit = model(src)
                    loss = F.cross_entropy(predicted_logit, trg)

                    # If phase train, then backward loss and step optimizer and scheduler
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        clip_grad_norm_(model.parameters(), args.grad_clip)
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
                               os.path.join(args.save_path, f'model_saved.pt'))
                    best_val_loss = val_loss