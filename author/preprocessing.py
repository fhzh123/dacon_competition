# Import modules
import os
import re
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

    # 2) Preprocessing
    def alpha_num(text):
        return re.sub(r'[^a-zA-z0-9\s]', '', text)

    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stopwords:
                final_text.append(i.strip())
        return " ".join(final_text)

    stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", 
                "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", 
                "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", 
                "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", 
                "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", 
                "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", 
                "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", 
                "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", 
                "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", 
                "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
                "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

    # 2) Train valid split
    index_list = train_dat['index']
    text_list = train_dat['text'].str.lower().apply(alpha_num).apply(remove_stopwords)
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
        f'--vocab_size={args.vocab_size} --character_coverage=0.995 --model_type=bpe '
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
    test_dat['text'] = test_dat['text'].str.lower().apply(alpha_num).apply(remove_stopwords)
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