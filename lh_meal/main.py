# Import modules
import time
import argparse
# Training
from preprocessing import preprocessing

def main(args):
    # Time setting
    total_start_time = time.time()

    if args.preprocessing:
        preprocessing(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--preprocessed_path', default='./preprocessing', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--data_path', default='/HDD/kyohoon/lh_meal', type=str,
                        help='Original data path')
    parser.add_argument('--vit_save_path', default='/HDD/kyohoon/model_checkpoint/vit/', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--captioning_preprocess_path', default='./preprocessing', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--captioning_data_path', default='/HDD/dataset/coco', type=str,
                        help='Original data path')
    parser.add_argument('--captioning_save_path', default='/HDD/kyohoon/model_checkpoint/captioning/', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--transgan_preprocess_path', default='./preprocessing', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--transgan_data_path', default='/HDD/dataset/celeba', type=str,
                        help='Original data path')
    parser.add_argument('--transgan_save_path', default='./testing_img', type=str,
                        help='Model checkpoint file path')
    # Data setting
    parser.add_argument('--img_size', default=64, type=int,
                        help='Image resize size; Default is 256')
    parser.add_argument('--vocab_size', default=8000, type=int,
                        help='Caption vocabulary size; Default is 8000')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_id', default=3, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_id', default=1, type=int,
                        help='Start token index; Default is 1')
    parser.add_argument('--eos_id', default=2, type=int,
                        help='End token index; Default is 2')
    parser.add_argument('--min_count', default=1, type=int,
                        help='Minimum length of caption; Default is 1')
    parser.add_argument('--max_len', default=300, type=int,
                        help='Maximum length of caption; Default is 300')
    # Model setting
    parser.add_argument('--patch_size', default=32, type=int, 
                        help='ViT patch size; Default is 32')
    parser.add_argument('--d_model', default=1024, type=int, 
                        help='Transformer model dimension; Default is 768')
    parser.add_argument('--d_embedding', default=256, type=int, 
                        help='Transformer embedding word token dimension; Default is 256')
    parser.add_argument('--n_head', default=16, type=int, 
                        help="Multihead Attention's head count; Default is 16")
    parser.add_argument('--dim_feedforward', default=2048, type=int, 
                        help="Feedforward network's dimension; Default is 2048")
    parser.add_argument('--dropout', default=0.1, type=float, 
                        help="Dropout ration; Default is 0.1")
    parser.add_argument('--embedding_dropout', default=0.1, type=float, 
                        help="Embedding dropout ration; Default is 0.1")
    parser.add_argument('--num_encoder_layer', default=12, type=int, 
                        help="Number of encoder layers; Default is 12")
    parser.add_argument('--num_decoder_layer', default=12, type=int, 
                        help="Number of decoder layers; Default is 12")
    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD'; Default is AdamW")
    parser.add_argument('--scheduler', default='constant', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is constant")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Training setting
    parser.add_argument('--num_epochs', default=10, type=int, 
                        help='Training epochs; Default is 10')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size; Default is 16')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    # Print frequency
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')
    args = parser.parse_args()

    main(args)