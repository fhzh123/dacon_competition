# Import modules
import os
import re
import pickle
import numpy as np
import pandas as pd

# Import custom modules
from model.preprocessing.khaiii import khaiii_encoder
from model.preprocessing.konlpy import konlpy_encoder
from model.preprocessing.sentencepiece import sentencepiece_encoder

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

    # 1) Initiating
    sentencepiece_encoder_module = sentencepiece_encoder(args)

    # 2) Training
    word2id_spm = sentencepiece_encoder_module.training(train_title_list, train_content_list)

    # 3) Encoding
    total_train_text_indices_spm, train_title_indices_spm, train_content_indices_spm = \
        sentencepiece_encoder_module.encoding(train_title_list, train_content_list)
    total_valid_text_indices_spm, valid_title_indices_spm, valid_content_indices_spm = \
        sentencepiece_encoder_module.encoding(valid_title_list, valid_content_list)

    #===================================#
    #==============Khaiii===============#
    #===================================#

    print('Khaiii processing...')

    # 1) Initiating
    khaiii_encoder_module = khaiii_encoder(args)

    # 2) Parsing
    train_title_list_khaiii = khaiii_encoder_module.parsing_sentence(train_title_list)
    train_content_list_khaiii = khaiii_encoder_module.parsing_sentence(train_content_list)
    valid_title_list_khaiii = khaiii_encoder_module.parsing_sentence(valid_title_list)
    valid_content_list_khaiii = khaiii_encoder_module.parsing_sentence(valid_content_list)

    # 3) Counter(word2id) setting
    word2id_khaiii = khaiii_encoder_module.counter_update(train_title_list_khaiii, train_content_list_khaiii)

    # 4) Encoding
    total_train_text_indices_khaiii, train_title_indices_khaiii, train_content_indices_khaiii = \
        khaiii_encoder_module.encode_sentence(train_title_list_khaiii, train_content_list_khaiii)
    total_valid_text_indices_khaiii, valid_title_indices_khaiii, valid_content_indices_khaiii = \
        khaiii_encoder_module.encode_sentence(valid_title_list_khaiii, valid_content_list_khaiii)

    #===================================#
    #==============KoNLPy===============#
    #===================================#

    print('KoNLPy processing...')

    # 1) Initiating
    konlpy_encoder_module = konlpy_encoder(args, type=args.konlpy_parser)

    # 2) Parsing
    train_title_list_konlpy = konlpy_encoder_module.parsing_sentence(train_title_list)
    train_content_list_konlpy = konlpy_encoder_module.parsing_sentence(train_content_list)
    valid_title_list_konlpy = konlpy_encoder_module.parsing_sentence(valid_title_list)
    valid_content_list_konlpy = konlpy_encoder_module.parsing_sentence(valid_content_list)

    # 3) Counter(word2id) setting
    word2id_konlpy = konlpy_encoder_module.counter_update(train_title_list_konlpy, train_content_list_konlpy)

    # 4) Encoding
    total_train_text_indices_konlpy, train_title_indices_konlpy, train_content_indices_konlpy = \
        konlpy_encoder_module.encode_sentence(train_title_list_konlpy, train_content_list_konlpy)
    total_valid_text_indices_konlpy, valid_title_indices_konlpy, valid_content_indices_konlpy = \
        konlpy_encoder_module.encode_sentence(valid_title_list_konlpy, valid_content_list_konlpy)


    #===================================#
    #========Test data processing=======#
    #===================================#

    print('Test data processing...')

    test_dat = pd.read_csv(os.path.join(args.data_path, 'news_test.csv'))
    total_test_text_indices_spm, test_title_indices_spm, test_content_indices_spm = \
        sentencepiece_encoder_module.encoding(test_dat['title'], test_dat['content'])
    total_test_text_indices_khaiii, test_title_indices_khaiii, test_content_indices_khaiii = \
        khaiii_encoder_module.encode_sentence(test_dat['title'], test_dat['content'])
    total_test_text_indices_konlpy, test_title_indices_konlpy, test_content_indices_konlpy = \
        konlpy_encoder_module.encode_sentence(test_dat['title'], test_dat['content'])

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence saving...')
    with open(os.path.join(args.save_path, 'preprocessed.pkl'), 'wb') as f:
        pickle.dump({
            'total_train_text_indices_spm': total_train_text_indices_spm,
            'total_valid_text_indices_spm': total_valid_text_indices_spm,
            'train_title_indices_spm': train_title_indices_spm,
            'valid_title_indices_spm': valid_title_indices_spm,
            'train_content_indices_spm': train_content_indices_spm,
            'valid_content_indices_spm': valid_content_indices_spm,
            'total_train_text_indices_khaiii': total_train_text_indices_khaiii,
            'total_valid_text_indices_khaiii': total_valid_text_indices_khaiii,
            'train_title_indices_khaiii': train_title_indices_khaiii,
            'valid_title_indices_khaiii': valid_title_indices_khaiii,
            'train_content_indices_khaiii': train_content_indices_khaiii,
            'valid_content_indices_khaiii': valid_content_indices_khaiii,
            'total_train_text_indices_konlpy': total_train_text_indices_konlpy,
            'total_valid_text_indices_konlpy': total_valid_text_indices_konlpy,
            'train_title_indices_konlpy': train_title_indices_konlpy,
            'valid_title_indices_konlpy': valid_title_indices_konlpy,
            'train_content_indices_konlpy': train_content_indices_konlpy,
            'valid_content_indices_konlpy': valid_content_indices_konlpy,
            'train_date_list': train_date_list,
            'valid_date_list': valid_date_list,
            'train_ord_list': train_ord_list,
            'valid_ord_list': valid_ord_list,
            'train_info_list': train_info_list,
            'valid_info_list': valid_info_list,
            'word2id_spm': word2id_spm,
            'word2id_khaiii': word2id_khaiii,
            'word2id_konlpy': word2id_konlpy,
            'konlpy_type': args.konlpy_type
        }, f)

    with open(os.path.join(args.save_path, 'test_preprocessed.pkl'), 'wb') as f:
        pickle.dump({
            'total_test_text_indices_spm': total_test_text_indices_spm,
            'test_title_indices_spm': test_title_indices_spm,
            'test_content_indices_spm': test_content_indices_spm,
            'total_test_text_indices_khaiii': total_test_text_indices_khaiii,
            'test_title_indices_khaiii': test_title_indices_khaiii,
            'test_content_indices_khaiii': test_content_indices_khaiii,
            'total_test_text_indices_konlpy': total_test_text_indices_konlpy,
            'test_title_indices_konlpy': test_title_indices_konlpy,
            'test_content_indices_konlpy': test_content_indices_konlpy,
            'test_date_list': test_dat['date'],
            'test_ord_list': test_dat['ord'],
            'test_id_list': test_dat['id'],
            'word2id_spm': word2id_spm,
            'word2id_khaiii': word2id_khaiii,
            'word2id_konlpy': word2id_konlpy,
            'konlpy_type': args.konlpy_type
        }, f)