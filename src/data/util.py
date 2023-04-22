"""
auxiliary methods for data io and splitting
"""

# load packages
import os 
import pickle
from typing import List, Tuple
import numpy as np

# ============ methods ============
def read_data(index_name: str, path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """read from preprocessed path, return data, targets, and dates""" 
    index_path = os.path.join(path, 'preprocessed', index_name)
    data = np.load(os.path.join(index_path, 'X.npy'))
    targets = np.load(os.path.join(index_path, 'y.npy'))
    with open(os.path.join(index_path, 'dates.pkl'), 'rb') as f:
        dates = pickle.load(f)
    
    return data, targets, dates

def ts_split(
    data: np.ndarray,
    targets: np.ndarray,
    dates: List[str],
    train: int = 240,
    val: int = 40,
    test: int = 40,
    K: int=5,
    make_tabular=False,
) -> Tuple:
    """
    time series train test split
    :param data: the feature data of shape (length, 100, 3)
    :param targets: the target data of shape (length, )
    :param dates: the corresponding dates of shape (length, )
    :param train, val, test: the number of days for train, val, and test data
    :param K: the number of folds for validation 
    :param make_tabular: True to make dataset tabular

    :return generated slices of data, target, and dates packed in dict, with the following pairs:
        - "train": list of data/targets/dates of length K 
        - "val": list of data/targets/dates of length K 
        - "test": a single period test
    """
    # flatten dataset to be tabular
    if make_tabular:
        data = data.reshape(len(data), -1)
    
    total_length = data.shape[0] 
    period_length = train + val * K + test
    period_start_idx = 0 
    
    while period_start_idx + period_length < total_length:
        period_dict, target_dict, dates_dict = [{'train': [], 'val': []} for _ in range(3)]
        cur_idx = period_start_idx
        
        # gather K-fold validations
        for _ in range(K):
            # slice
            # train
            train_data = data[cur_idx:cur_idx + train]
            train_targets = targets[cur_idx:cur_idx + train]
            train_dates = dates[cur_idx:cur_idx + train]

            # validation
            val_data = data[cur_idx + train:cur_idx + train + val]
            val_targets = targets[cur_idx + train:cur_idx + train + val]
            val_dates = dates[cur_idx + train:cur_idx + train + val]
            
            # append to dict
            period_dict['train'].append(train_data)
            target_dict['train'].append(train_targets)
            dates_dict['train'].append(train_dates)
            period_dict['val'].append(val_data)
            target_dict['val'].append(val_targets)
            dates_dict['val'].append(val_dates)

            # update start index of each fold
            cur_idx += val

        # pack test
        test_data = data[cur_idx + train + val: cur_idx + train + val + test]
        test_targets = targets[cur_idx + train + val: cur_idx + train + val + test]
        test_dates = dates[cur_idx + train + val: cur_idx + train + val + test]

        period_dict['test'] = test_data
        target_dict['test'] = test_targets
        dates_dict['test'] = test_dates

        # also feed in "train_all": from start of this period to end of val
        period_dict['train_all'] = data[period_start_idx:cur_idx + train + val]
        target_dict['train_all'] = targets[period_start_idx:cur_idx + train + val]

        yield period_dict, target_dict, dates_dict

        # update period start (move by test)
        period_start_idx += test