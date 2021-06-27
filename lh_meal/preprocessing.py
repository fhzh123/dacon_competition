import os
import re
import pickle
import random
import pandas as pd
from datetime import datetime
from collections import Counter

def preprocessing(args):

    # Path setting
    if not os.path.exists(args.preprocessed_path):
        os.mkdir(args.preprocessed_path)

    # Data load
    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

    # Train & Valid split
    split_index = random.sample(range(train.shape[0]), 
                                int(train.shape[0] * args.valid_split_ratio))
    valid = train.iloc[split_index]
    valid.index = range(len(valid))
    train = train.iloc[list(set(range(train.shape[0])) - set(split_index))]
    train.index = range(len(train))

    # Menu split
    train['breakfast'] = train['조식메뉴'].apply(lambda x: x.split())
    train['lunch'] = train['중식메뉴'].apply(lambda x: x.split())
    train['supper'] = train['석식메뉴'].apply(lambda x: x.split())
    valid['breakfast'] = valid['조식메뉴'].apply(lambda x: x.split())
    valid['lunch'] = valid['중식메뉴'].apply(lambda x: x.split())
    valid['supper'] = valid['석식메뉴'].apply(lambda x: x.split())
    test['breakfast'] = test['조식메뉴'].apply(lambda x: x.split())
    test['lunch'] = test['중식메뉴'].apply(lambda x: x.split())
    test['supper'] = test['석식메뉴'].apply(lambda x: x.split())

    # Date processing
    train['datetime'] = train['일자'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    train['year'] = train['datetime'].apply(lambda x: x.year)
    train['month'] = train['datetime'].apply(lambda x: x.month)
    train['day'] = train['datetime'].apply(lambda x: x.day)
    train['weekday'] = train['datetime'].apply(lambda x: x.weekday())
    valid['datetime'] = valid['일자'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    valid['year'] = valid['datetime'].apply(lambda x: x.year)
    valid['month'] = valid['datetime'].apply(lambda x: x.month)
    valid['day'] = valid['datetime'].apply(lambda x: x.day)
    valid['weekday'] = valid['datetime'].apply(lambda x: x.weekday())
    test['datetime'] = test['일자'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    test['year'] = test['datetime'].apply(lambda x: x.year)
    test['month'] = test['datetime'].apply(lambda x: x.month)
    test['day'] = test['datetime'].apply(lambda x: x.day)
    test['weekday'] = test['datetime'].apply(lambda x: x.weekday())

    # Target processing
    train['total'] = train['본사정원수'] - train['본사휴가자수'] - \
                     train['본사출장자수'] - train['현본사소속재택근무자수']
    train['target_lunch'] = train['중식계'] / train['total']
    train['target_supper'] = train['석식계'] / train['total']
    valid['total'] = valid['본사정원수'] - valid['본사휴가자수'] - \
                     valid['본사출장자수'] - valid['현본사소속재택근무자수']
    valid['target_lunch'] = valid['중식계'] / valid['total']
    valid['target_supper'] = valid['석식계'] / valid['total']
    test['total'] = test['본사정원수'] - test['본사휴가자수'] - \
                    test['본사출장자수'] - test['현본사소속재택근무자수']

    # Processing by Counter
    menu_counter = Counter()
    for i in range(len(train)):
        menu_counter.update(train['breakfast'].iloc[i])
        menu_counter.update(train['lunch'].iloc[i])
        menu_counter.update(train['supper'].iloc[i])
    
    menu_dict = dict()
    menu_dict['<pad>'] = 0
    for i, (k, v) in enumerate(menu_counter.items()):
        if v >= args.min_count:
            menu_dict[k] = i + 1

    # Encoding
    train['breakfast_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in train['breakfast'][i]] for i in range(len(train))]
    train['lunch_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in train['lunch'][i]] for i in range(len(train))]
    train['supper_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in train['supper'][i]] for i in range(len(train))]
    valid['breakfast_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in valid['breakfast'][i]] for i in range(len(valid))]
    valid['lunch_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in valid['lunch'][i]] for i in range(len(valid))]
    valid['supper_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in valid['supper'][i]] for i in range(len(valid))]
    test['breakfast_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in test['breakfast'][i]] for i in range(len(test))]
    test['lunch_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in test['lunch'][i]] for i in range(len(test))]
    test['supper_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in test['supper'][i]] for i in range(len(test))]

    # DataFrame setting
    reg = re.compile(r'[a-zA-Z]')
    remain_column_train = list()

    for col in train.columns:
        if reg.match(col):
            remain_column_train.append(col)

    remain_column_test = [x for x in remain_column_train if x not in ['target_lunch', 'target_supper']]

    train_dat = train[remain_column_train]
    valid_dat = valid[remain_column_train]
    test_dat = test[remain_column_test]

    # Saving
    train_dat.to_csv(os.path.join(args.preprocessed_path, 'train_processed.csv'), index=False)
    valid_dat.to_csv(os.path.join(args.preprocessed_path, 'valid_processed.csv'), index=False)
    test_dat.to_csv(os.path.join(args.preprocessed_path, 'test_processed.csv'), index=False)
    with open(os.path.join(args.preprocessed_path, 'menu_dict.pkl'), 'wb') as f:
        pickle.dump(menu_dict, f)