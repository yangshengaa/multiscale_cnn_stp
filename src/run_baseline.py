"""
use multilogisitc regression as baseline
"""

# load packages 
import os 
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

# load file 
from data import read_data, ts_split
from utils import load_config

# ===== arguments =====
parser = argparse.ArgumentParser()

# data 
parser.add_argument("--data", type=list, 
    default='000016', 
    choices=["000016", "000300", "000852", "000903", "000905"],
    help='name of index data to read'
)

# path specification
parser.add_argument("--tag", type=str, default='exp', help='tag for config paths')

# length of data period
parser.add_argument("--train", type=int, default=240, help='number of days to train')
parser.add_argument("--val", type=int, default=40, help='the number of days to validate')
parser.add_argument("--test", type=int, default=40, help='the number of days to test')
parser.add_argument("--K", type=int, default=5, help='the number of folds for validation')

# model specific hyperparameters
parser.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1, 10], help='penalty strength')

args = parser.parse_args()

# load path 
paths = load_config(args.tag)
model_name = f'mlr_{args.data}'
os.makedirs(os.path.join(paths['model_dir'], model_name), exist_ok=True)
os.makedirs(os.path.join(paths['result_dir'], model_name), exist_ok=True)

# load data 
data, targets, dates = read_data(args.data, paths['data_dir'])

def train():
    """train loop"""
    for period_dict, target_dict, dates_dict in ts_split(data, targets, dates, args.train, args.val, args.test, args.K, make_tabular=True):
        # use average validation accuracy to pick the best hyperparameter
        val_avg_acc = {}
        for c in args.C:
            cur_val_acc = []
            for X_train, X_val, y_train, y_val in zip(
                    period_dict['train'], period_dict['val'], target_dict['train'], target_dict['val']
                ):
                # train 
                mlr = LogisticRegression(penalty='l2', C=c).fit(X_train)
                cur_acc = mlr.score(X_val, y_val)
                cur_val_acc.append(cur_acc)
            avg_acc = np.mean(cur_val_acc)
            val_avg_acc[c] = avg_acc
        
        # find best hyperparameter
        pass 
        # TODO:


def main():
    pass 

if __name__ == '__main__':
    main()
