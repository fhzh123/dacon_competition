# Import modules
import os
import time
import pickle
import pandas as pd

# Import PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import custom modules
from dataset import CustomDataset, PadCollate
from model.total_model import Total_model

def testing(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #============Data Load==============#
    #===================================#

    print('Data Load & Setting!')
    with open(os.path.join(args.save_path, 'test_preprocessed.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        test_text_indices = data_['test_text_indices']
        test_index_indices = data_['test_index_indices']
        vocab_list = data_['vocab_list']
        vocab_num = len(vocab_list)
        word2id = data_['word2id']
        del data_

    test_dataset = CustomDataset(test_text_indices, test_index_indices, test_index_indices,
                                 min_len=args.min_len, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, collate_fn=PadCollate(), drop_last=False,
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
    print(f"Total number of testsets iterations - {len(test_dataset)}, {len(test_dataloader)}")

    #===================================#
    #===========Model Setting===========#
    #===================================#

    print("Build model")
    model = Total_model(vocab_num, author_num=5, pad_idx=args.pad_idx, bos_idx=args.bos_idx,
                        eos_idx=args.eos_idx, max_len=args.max_len, d_model=args.d_model,
                        d_embedding=args.d_embedding, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                        bilinear=args.bilinear, num_transformer_layer=args.num_transformer_layer,
                        num_rnn_layer=args.num_rnn_layer, device=device)
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model_saved2.pt')))
    model = model.to(device)
    model = model.eval()

    freq = 0
    start_time = time.time()

    for i, (src, _, index_) in enumerate(test_dataloader):
        src = src.to(device)
        trg_softmax = nn.Softmax(dim=1)

        with torch.no_grad():
            predicted_logit = model(src)
            predicted_logit_clone = trg_softmax(predicted_logit.clone().detach())
            index_clone = index_.clone().detach()
            if i == 0:
                predicted_total = torch.cat((index_clone.type('torch.FloatTensor').unsqueeze(1), 
                                       predicted_logit_clone.cpu()), dim=1)
            else:
                predicted = torch.cat((index_clone.type('torch.FloatTensor').unsqueeze(1), 
                                       predicted_logit_clone.cpu()), dim=1)
                predicted_total = torch.cat((predicted_total, predicted), dim=0)

        if freq == 100 or i == 0 or i == len(test_dataloader):
            spend_time = time.time() - start_time
            print('testing...[%d/%d] %2.2fmin spend' % 
                  (i, len(test_dataloader), spend_time / 60))
            freq = 0
        freq += 1

    #===================================#
    #======Submission csv setting=======#
    #===================================#

    submission_dat = pd.DataFrame(predicted_total.numpy())
    submission_dat[0] = submission_dat[0].astype(int)
    submission_dat.columns = ['index', 0, 1, 2, 3, 4]
    submission_dat = submission_dat.sort_values(by=['index'], ascending=True)
    submission_dat.to_csv(os.path.join(args.save_path, 'submission.csv'), index=False, encoding='utf-8')