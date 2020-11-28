# Import modules
import os
import pickle
import pandas as pd
import sentencepiece as spm

# Import custom modules
from utils import train_valid_split

def preprocessing(args):

    #===================================#
    #========Data load & Setting========#
    #===================================#

    # 1) Train data load
    train_dat = pd.read_csv(os.path.join(args.data_path, 'train.csv'))

    # 2) Train valid split
    index_list = train_dat['index']
    text_list = train_dat['text']
    author_list = train_dat['author']
    text_, author_, index_ = train_valid_split(text_list, author_list, index_list, split_percent=args.valid_percent)

    # 3) Preprocess path setting
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    #===================================#
    #===========SentencePiece===========#
    #===================================#

    # 1) Make Korean text to train vocab
    with open(f'{args.save_path}/text.txt', 'w') as f:
        for text in text_['train']:
            f.write(f'{text}\n')

    # 2) SentencePiece model training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.save_path}/text.txt --model_prefix={args.save_path}/m_text '
        f'--vocab_size={args.vocab_size} --character_coverage=0.995 '
        f'--split_by_whitespace=true --pad_id={args.pad_idx} --unk_id={args.unk_idx} '
        f'--bos_id={args.bos_idx} --eos_id={args.eos_idx}')

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
    train_text_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in text_['train']]
    valid_text_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in text_['valid']]

    #===================================#
    #========Test data processing=======#
    #===================================#

    test_dat = pd.read_csv(os.path.join(args.data_path, 'test_x.csv'))
    test_text_indices = [[args.bos_idx] + spm_.EncodeAsIds(text) + [args.eos_idx] for text in test_dat['text']]
    test_index_indices = test_dat['index'].tolist()

    #===================================#
    #==============Saving===============#
    #===================================#

    print('Parsed sentence saving...')
    with open(os.path.join(args.save_path, 'preprocessed.pkl'), 'wb') as f:
        pickle.dump({
            'train_text_indices': train_text_indices,
            'valid_text_indices': valid_text_indices,
            'train_author_indices': author_['train'],
            'valid_author_indices': author_['valid'],
            'train_index_indices': index_['train'],
            'valid_index_indices': index_['valid'],
            'vocab_list': vocab_list,
            'word2id': word2id
        },f)

    with open(os.path.join(args.save_path, 'test_preprocessed.pkl'), 'wb') as f:
        pickle.dump({
            'test_text_indices': test_text_indices,
            'test_index_indices': test_index_indices,
            'vocab_list': vocab_list,
            'word2id': word2id
        },f)