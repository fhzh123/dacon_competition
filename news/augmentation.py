# Import modules
import os
import re
import pickle
import pandas as pd

# Import PyTorch
import torch

# Import custom modules
from model.preprocessing.sentencepiece import sentencepiece_encoder

def augmentation(args):

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
    total_id_list = (train_dat['n_id'] + '_' + train_dat['ord'].astype(str)).tolist()
    total_info_list = train_dat['info'].tolist()

    data_len = len(train_dat)
    valid_index = np.random.choice(range(data_len), int(data_len*args.valid_percent), replace=False)
    train_index = list(set(range(data_len)) - set(valid_index))

    train_date_list = [total_date_list[i] for i in train_index]
    train_title_list = [total_title_list[i] for i in train_index]
    train_content_list = [total_content_list[i] for i in train_index]
    train_ord_list = [total_ord_list[i] for i in train_index]
    train_id_list = [total_id_list[i] for i in train_index]
    train_info_list = [total_info_list[i] for i in train_index]

    valid_date_list = [total_date_list[i] for i in valid_index]
    valid_title_list = [total_title_list[i] for i in valid_index]
    valid_content_list = [total_content_list[i] for i in valid_index]
    valid_ord_list = [total_ord_list[i] for i in valid_index]
    valid_id_list = [total_id_list[i] for i in valid_index]
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