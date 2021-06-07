import os
import pandas as pd

def preprocessing(args):
    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))