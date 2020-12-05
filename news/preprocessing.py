# Import modules
import os
import re
import pickle
import numpy as np
import pandas as pd
import sentencepiece as spm

# Import custom modules
from utils import spm_encoding

def preprocessing(args):

    #===================================#
    #========Data load & Setting========#
    #===================================#

    # 1) Train data load
    train_dat = pd.read_csv(os.path.join(args.data_path, 'news_train.csv'))

    # 2) Train valid split
    total_date_list = train_dat['date'].tolist()
    total_title_list = train_dat['title'].tolist()
    total_content_list = train_dat['content'].tolist()
    total_ord_list = train_dat['ord'].tolist()
    total_info_list = train_dat['info'].tolist()

    data_len = len(train_dat)
    valid_index = np.random.choice(range(data_len), int(data_len*args.valid_percent), replace=False)
    train_index = list(set(range(data_len)) - set(valid_index))

    train_date_list = [total_date_list[i] for i in train_index]
    train_title_list = [total_title_list[i] for i in train_index]
    train_content_list = [total_content_list[i] for i in train_index]
    train_ord_list = [total_ord_list[i] for i in train_index]
    train_info_list = [total_info_list[i] for i in train_index]

    valid_date_list = [total_date_list[i] for i in valid_index]
    valid_title_list = [total_title_list[i] for i in valid_index]
    valid_content_list = [total_content_list[i] for i in valid_index]
    valid_ord_list = [total_ord_list[i] for i in valid_index]
    valid_info_list = [total_info_list[i] for i in valid_index]

    # 3) Preprocess path setting
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    #===================================#
    #===========SentencePiece===========#
    #===================================#

    # 1) Make Korean text to train vocab
    with open(f'{args.save_path}/text.txt', 'w') as f:
        for text in train_title_list + train_content_list:
            f.write(f'{text}\n')

    # 2) SentencePiece model training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.save_path}/text.txt --model_prefix={args.save_path}/m_text '
        f'--vocab_size={args.vocab_size} --character_coverage=0.9995 --model_type={args.tokenizer}'
        f'--split_by_whitespace=true --pad_id={args.pad_idx} --unk_id={args.unk_idx} '
        f'--bos_id={args.bos_idx} --eos_id={args.eos_idx} --user_defined_symbols=[SEP]')

    # 3) Korean vocabulrary setting
    vocab_list = list()
    with open(f'{args.save_path}/m_text.vocab') as f:
        for line in f:
            vocab_list.append(line[:-1].split('\t')[0])
    word2id = {w: i for i, w in enumerate(vocab_list)}

    # 4) SentencePiece model load
    spm_ = spm.SentencePieceProcessor()
    spm_.Load(f"{args.save_path}/m_text.model")

    # 5) Korean parsing by SentencePiece model
    total_train_text_indices = spm_encoding(train_title_list, train_content_list, 
                                      spm_, args, total=True)
    total_valid_text_indices = spm_encoding(valid_title_list, valid_content_list, 
                                      spm_, args, total=True)
    train_title_indices, train_content_indices = spm_encoding(train_title_list, train_content_list, 
                                                              spm_, args, total=False)
    valid_title_indices, valid_content_indices = spm_encoding(valid_title_list, valid_content_list, 
                                                              spm_, args, total=False)

    #===================================#
    #========Test data processing=======#
    #===================================#

    test_dat = pd.read_csv(os.path.join(args.data_path, 'news_test.csv'))
    total_test_text_indices = spm_encoding(test_dat['title'], test_dat['content'], 
                                     spm_, args, total=True)
    test_title_indices, test_content_indices = spm_encoding(test_dat['title'], test_dat['content'], 
                                                            spm_, args, total=False)

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence saving...')
    with open(os.path.join(args.save_path, 'preprocessed.pkl'), 'wb') as f:
        pickle.dump({
            'total_train_text_indices': total_train_text_indices,
            'total_valid_text_indices': total_valid_text_indices,
            'train_title_indices': train_title_indices,
            'valid_title_indices': valid_title_indices,
            'train_content_indices': train_content_indices,
            'valid_content_indices': valid_content_indices,
            'train_date_list': train_date_list,
            'valid_date_list': valid_date_list,
            'train_ord_list': train_ord_list,
            'valid_ord_list': valid_ord_list,
            'train_info_list': train_info_list,
            'valid_info_list': valid_info_list,
            'vocab_list': vocab_list,
            'word2id': word2id
        }, f)

    with open(os.path.join(args.save_path, 'test_preprocessed.pkl'), 'wb') as f:
        pickle.dump({
            'total_test_text_indices': total_test_text_indices,
            'test_title_indices': test_title_indices,
            'test_content_indices': test_content_indices,
            'test_date_list': test_dat['date'],
            'test_ord_list': test_dat['ord'],
            'test_id_list': test_dat['id'],
            'vocab_list': vocab_list,
            'word2id': word2id
        }, f)