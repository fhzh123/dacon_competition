import os
import pandas as pd
from collections import Counter

def preprocessing(args):

    # Path setting
    if not os.path.exists(args.preprocessed_path):
        os.mkdir(args.preprocessed_path)

    # Data load
    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))

    # Menu split
    train['breakfast'] = train['조식메뉴'].apply(lambda x: x.split())
    train['lunch'] = train['중식메뉴'].apply(lambda x: x.split())
    train['supper'] = train['석식메뉴'].apply(lambda x: x.split())
    test['breakfast'] = test['조식메뉴'].apply(lambda x: x.split())
    test['lunch'] = test['중식메뉴'].apply(lambda x: x.split())
    test['supper'] = test['석식메뉴'].apply(lambda x: x.split())

    # Target processing
    train['total'] = train['본사정원수'] - train['본사휴가자수'] - \
                     train['본사출장자수'] - train['현본사소속재택근무자수']
    train['target_lunch'] = train['중식계'] / train['total']
    train['target_supper'] = train['석식계'] / train['total']
    test['total'] = test['본사정원수'] - test['본사휴가자수'] - \
                    test['본사출장자수'] - test['현본사소속재택근무자수']

    # Processing by Counter
    menu_counter = Counter()
    for i in range(len(train)):
        menu_counter.update(train['breakfast'].iloc[i])
        menu_counter.update(train['lunch'].iloc[i])
        menu_counter.update(train['supper'].iloc[i])
    
    menu_dict = dict()
    for i, (k, v) in enumerate(menu_counter.items()):
        if v >= args.min_count:
            menu_dict[i] = k

    # Encoding
    train['breakfast_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in train['breakfast'][i]] for i in range(len(train))]
    train['lunch_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in train['lunch'][i]] for i in range(len(train))]
    train['supper_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in train['supper'][i]] for i in range(len(train))]
    test['breakfast_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in test['breakfast'][i]] for i in range(len(test))]
    test['lunch_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in test['lunch'][i]] for i in range(len(test))]
    test['supper_parsing'] = \
        [[menu_dict.get(x, len(menu_counter.items())) for x in test['supper'][i]] for i in range(len(test))]

    # Saving
    word.encode().isalpha():