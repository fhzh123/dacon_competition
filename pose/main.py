import time
import argparse

from train import training

def main(args):
    start_time = time.time()

    if args.training:
        training(args)
    
    # if args.testing:
    #     testing(args)

    end_time = round((time.time() - start_time) / 60, 4)
    print(f'Done!; {end_time}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Joseon Dynasty Crawling')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Preprocessing setting
    parser.add_argument('--data_path', type=str, default='/HDD/dacon/pose', help='Data path')
    parser.add_argument('--split', type=float, default=0.2, help='Train, valid split percent')
    # Training setting
    parser.add_argument('--num_workers', default=4 ,type=int, help='CPU worker count; default is 4')
    parser.add_argument('--num_epochs', default=30, type=int, help='Epoch count; default is 30')
    parser.add_argument('--batch_size', default=8,  type=int, help='Batch size; default is 8')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate of warmup; default is 1e-4')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum for SGD optimizer; default is 0.9')
    parser.add_argument('--w_decay', default=5e-4, type=float, help='Weight decay of optimizer; dafault is 5e-4')
    parser.add_argument('--grad_clip', type=int, default=5, help='Set gradient clipping; default is 5')
    # Print frequency
    parser.add_argument('--print_freq', type=int, default=1000, help='Print train loss frequency; default is 1000')
    args = parser.parse_args()

    main(args)