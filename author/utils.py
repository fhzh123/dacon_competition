import random
import numpy as np

def train_valid_split(record_list1, record_list2, record_list3, split_percent=0.2):
    # Check paired data
    assert len(record_list1) == len(record_list2) 
    assert len(record_list2) == len(record_list3)
    assert len(record_list1) == len(record_list3)

    # 
    data_len = len(record_list1)
    valid_num = int(data_len * split_percent)

    # Index setting
    valid_index = np.random.choice(data_len, valid_num, replace=False)
    train_index = list(set(range(data_len)) - set(valid_index))

    # Split
    train_record_list1 = [record_list1[i] for i in train_index]
    train_record_list2 = [record_list2[i] for i in train_index]
    train_record_list3 = [record_list3[i] for i in train_index]
    valid_record_list1 = [record_list1[i] for i in valid_index]
    valid_record_list2 = [record_list2[i] for i in valid_index]
    valid_record_list3 = [record_list3[i] for i in valid_index]

    # Dictionary setting
    split_record1 = {'train': train_record_list1,
                     'valid': valid_record_list1}

    split_record2 = {'train': train_record_list2,
                     'valid': valid_record_list2}

    split_record3 = {'train': train_record_list3,
                     'valid': valid_record_list3}

    return split_record1, split_record2, split_record3