import time
import argparse

from train import train

def main(args):
    train(args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Dacon landmark argparser')
    # Path setting
    parser.add_argument('--data_path', type=str, 
                        default='/HDD/kyohoon/dacon/landmark', help='Data path setting')
    # Model setting
    parser.add_argument('--efficientnet_model_number', type=str, default=7, help='EfficientNet model number')
    parser.add_argument('--save_path', type=str, default='./save')
    # Training setting
    parser.add_argument('--num_epochs', type=int, default=30, help='The number of epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--resize_pixel', type=int, default=360)
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate setting')
    parser.add_argument('--w_decay', type=float, default=1e-6)
    parser.add_argument('--max_grad_norm', type=int, default=5, help='Gradient clipping max norm')
    # Utils setting
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='Train / Valid split ratio')
    parser.add_argument('--random_seed', type=int, default=42, help='Random state setting')
    parser.add_argument('--num_workers', type=int, default=8, help='CPU worker setting')
    parser.add_argument('--print_freq', type=int, default=300)
    args = parser.parse_args()

    total_start_time = time.time()
    main(args)
    print('Done! {:.4f}min spend!'.format((time.time() - total_start_time)/60))