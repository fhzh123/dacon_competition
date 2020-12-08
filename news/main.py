import os
import time
import argparse

# Import custom modules
from preprocessing import preprocessing
from training import training
from testing import testing

def main(args):
    start_time = time.time()

    if args.preprocessing:
        preprocessing(args)

    if args.training:
        training(args)
    
    if args.testing:
        testing(args)

    end_time = round((time.time() - start_time) / 60, 4)
    print(f'Done!; {end_time}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Joseon Dynasty Crawling')
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    # Preprocessing setting
    parser.add_argument('--data_path', type=str, default='/HDD/kyohoon/dacon/news', help='Data path')
    parser.add_argument('--save_path', type=str, default='./preprocessing', help='Preprocessed data save path')
    parser.add_argument('--model_path', type=str, default='/HDD/kyohoon/dacon/news/model', help='Model path')
    parser.add_argument('--tokenizer', type=str, default='unigram', 
                        choices=['unigram', 'bpe', 'chr'], help='SentencePiece library model type; default is unigram')
    parser.add_argument('--valid_percent', type=float, default=0.2, help='Train / valid ratio setting; default is 0.2')
    parser.add_argument('--pad_idx', default=0, type=int, help='Pad index')
    parser.add_argument('--bos_idx', default=1, type=int, help='Index of bos token')
    parser.add_argument('--eos_idx', default=2, type=int, help='Index of eos token')
    parser.add_argument('--unk_idx', default=3, type=int, help='Index of unk token')
    parser.add_argument('--sep_idx', default=4, type=int, help='Index of SEP token')
    parser.add_argument('--vocab_size', default=24000, type=int, help='Vocabulary size; default is 24000')
    # Training setting
    parser.add_argument('--num_epoch', default=30, type=int, help='Epoch count; default is 30')
    parser.add_argument('--batch_size', default=48,  type=int, help='Batch size; default is 48')
    parser.add_argument('--min_len', default=4, type=int, help='Minumum length of sentences; default is 4')
    parser.add_argument('--max_len', default=550, type=int, help='Maximum lenghth of sentences; default is 550')
    parser.add_argument('--label_smoothing', default=0, type=float, help='Label smoothing; default is 0.')
    parser.add_argument('--n_warmup_epochs', default=0, type=int, help='Learning rate warmup epoch; default is 0')
    parser.add_argument('--max_lr', default=1e-4, type=float, help='Maximum learning rate of warmup; default is 1e-4')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum for SGD optimizer; default is 0.9')
    parser.add_argument('--w_decay', default=1e-5, type=float, help='Weight decay of optimizer; dafault is 1e-5')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout ratio; default is 0.2')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; default is 5')
    # Model setting
    parser.add_argument('--model_type', type=str, default='total', choices=['total', 'transformer', 'rnn', 'gap'],
                        help='Model type; default is total')
    parser.add_argument('--bilinear', action='store_true')
    parser.add_argument('--num_transformer_layer', type=int, default=6, help='Number of transformer layer; default is 6')
    parser.add_argument('--num_rnn_layer', type=int, default=6, help='Number of rnn layer; default is 6')
    parser.add_argument('--d_model', type=int, default=512, help='Hidden state vector dimension; default is 512')
    parser.add_argument('--d_embedding', type=int, default=512, help='Embedding vector dimension; default is 256')
    parser.add_argument('--d_k', type=int, default=64, help='Key vector dimension; default is 64')
    parser.add_argument('--d_v', type=int, default=64, help='Value vector dimension; default is 64')
    parser.add_argument('--n_head', type=int, default=8, help='Multihead count; default is 8')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Feedforward layer Dimension; default is 2048')
    # Print frequency
    parser.add_argument('--print_freq', type=int, default=300, help='Print train loss frequency; default is 300')
    parser.add_argument('--test_print_freq', type=int, default=300, help='Print test frequency; default is 300')
    args = parser.parse_args()

    main(args)