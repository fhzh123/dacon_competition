# Import modules
import os
import pandas as pd

# Import PyTorch
import torch

# Import custom modules

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

    # 3)