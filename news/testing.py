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
        src_vocab_num_dict = dict()
        
        total_test_text_indices_spm = data_['total_test_text_indices_spm']
        test_title_indices_spm = data_['test_title_indices_spm']
        test_content_indices_spm = data_['test_content_indices_spm']
        total_test_text_indices_khaiii = data_['total_test_text_indices_khaiii']
        test_title_indices_khaiii = data_['test_title_indices_khaiii']
        test_content_indices_khaiii = data_['test_content_indices_khaiii']
        total_test_text_indices_konlpy = data_['total_test_text_indices_konlpy']
        test_title_indices_konlpy = data_['test_title_indices_konlpy']
        test_content_indices_konlpy = data_['test_content_indices_konlpy']
        test_date_list = data_['test_date_list']
        test_ord_list = data_['test_ord_list']
        test_id_list = data_['test_id_list']
        word2id_spm = data_['word2id_spm']
        word2id_khaiii = data_['word2id_khaiii']
        word2id_konlpy = data_['word2id_konlpy']
        src_vocab_num_dict['spm'] = len(word2id_spm.keys())
        src_vocab_num_dict['khaiii'] = len(word2id_khaiii.keys())
        src_vocab_num_dict['konlpy'] = len(word2id_konlpy.keys())
        del data_

    test_dataset = CustomDataset(total_test_text_indices_spm, total_test_text_indices_khaiii, 
                                 total_test_text_indices_konlpy,
                                 test_date_list, test_ord_list, test_id_list,
                                 isTrain=False, min_len=args.min_len, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, collate_fn=PadCollate(isTrain=False), drop_last=False,
                                 batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    print(f"Total number of testsets iterations - {len(test_dataset)}, {len(test_dataloader)}")

    #===================================#
    #============Model load=============#
    #===================================#

    print("Load model")
    model = Total_model(args.model_type, src_vocab_num_dict, trg_num=2, pad_idx=args.pad_idx, bos_idx=args.bos_idx,
                        eos_idx=args.eos_idx, max_len=args.max_len, d_model=args.d_model,
                        d_embedding=args.d_embedding, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
                        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                        bilinear=args.bilinear, num_transformer_layer=args.num_transformer_layer,
                        num_rnn_layer=args.num_rnn_layer, device=device)
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'model_saved.pt')))
    model = model.to(device)
    model = model.eval()

    #===================================#
    #=============Testing===============#
    #===================================#

    freq = 0
    start_time = time.time()

    for i, (total_src, title_src, content_src, date, order, id_) in enumerate(test_dataloader):
        total_src = total_src.to(device)
        title_src = title_src.to(device)
        content_src = content_src.to(device)

        with torch.no_grad():
            predicted_logit = model(total_src, title_src, content_src)
            predicted = predicted_logit.max(dim=1)[1].clone().tolist()
            if i == 0:
                id_list = id_
                info_list = predicted
            else:
                id_list = id_list + id_
                info_list = info_list + predicted

        if freq == args.test_print_freq or i == 0 or i == len(test_dataloader):
            spend_time = time.time() - start_time
            print('testing...[%d/%d] %2.2fmin spend' % 
                  (i, len(test_dataloader), spend_time / 60))
            freq = 0
        freq += 1

    #===================================#
    #============Rule-base==============#
    #===================================#

    submission_id = pd.read_csv(os.path.join(args.data_path, 'sample_submission.csv'))['id']
    submission_pre = pd.DataFrame({
        'id': id_list,
        'info': info_list
    })
    submission_dat = pd.merge(pd.DataFrame(submission_id), submission_pre, on='id', how='left')
    
    test_dat = pd.read_csv(os.path.join(args.data_path, 'news_test.csv'))
    nan_content = pd.merge(test_dat[['id', 'content']], submission_dat.loc[submission_dat['info'].isnull()], 
                           on='id', how='right')
    submission_dat = submission_dat.dropna()

    rule_base_list = ['무료', '증권방송', '바로가기']
    for i, content in enumerate(nan_content['content']):
        if any([rule in content for rule in rule_base_list]):
            nan_content['info'].iloc[i] = 1
        else:
            nan_content['info'].iloc[i] = 0

    submission_dat = pd.concat([submission_dat, nan_content[['id', 'info']]])
    submission_dat = pd.merge(pd.DataFrame(submission_id), submission_dat, on='id', how='left') # Sorting
    submission_dat['info'] = submission_dat['info'].apply(int)

    #===================================#
    #======Submission csv setting=======#
    #===================================#

    submission_dat.to_csv(os.path.join(args.results_path, 'submission.csv'), index=False, encoding='utf-8')