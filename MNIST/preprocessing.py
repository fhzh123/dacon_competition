# Import Module
import os
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

def main(args):
    # Data Load
    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    train_ = train.drop(columns = ['id', 'digit', 'letter'])

    test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    sample = pd.read_csv(os.path.join(args.data_path, 'submission.csv'))

    # Train Image Save
    print('Train data saving...')
    img_save(train, isTrain=True)

    # Test Image Save
    print('Test data saving...')
    img_save(test, isTrain=False)

    # Show Sample Submission
    print(sample.head())


def img_save(dataset, isTrain=True):
    if isTrain:
        for_image_dataset = dataset.drop(columns=['id', 'letter', 'digit'])
    else:
        for_image_dataset = dataset.drop(columns=['id', 'letter'])
    for i in tqdm(range(len(dataset))):
        id_ = dataset.iloc[i]['id']
        letter_ = dataset.iloc[i]['letter']
        img_array = np.array(for_image_dataset.iloc[i]).reshape((28, 28)).astype('uint8')
        img = Image.fromarray(img_array)
        if isTrain:
            digit_ = str(dataset.iloc[i]['digit'])
            if not os.path.exists(os.path.join('../data/train/', digit_)):
                os.mkdir(os.path.join('../data/train/', digit_))
            img.save(os.path.join(f'../data/train/{digit_}/', f'{id_}_{letter_}.jpg'))
        else:
            img.save(os.path.join(f'../data/test/', f'{id_}_{letter_}.jpg'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    args = parser.parse_args()
    parser.add_argument('--data_path', default='../data/', type=str, help='Default Data Path')
    args = parser.parse_args()

    main(args)
    print('Done!')