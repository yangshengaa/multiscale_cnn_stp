"""
data preprocessing: convert each index csv into three files: 
- X.npy: # shape: (num_samples, 100, 3)
- y.npy: # shape: (num_sampes,)
- dates.npy: # a serialized file 
"""

# load packages
import os 
import sys 
import argparse
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd 

# load file
sys.path.append('../')
from utils import load_config

# ===== arguments =====
parser = argparse.ArgumentParser()

# data 
parser.add_argument("--data", type=list, default=["000016", "000300", "000852", "000903", "000905"], help='name of index data to read')

# hypereparams
parser.add_argument("--t", type=int, default=100, help='the number of days to retain for each sample')
parser.add_argument("--k", type=int, default=5, help='the lookforward period')
parser.add_argument('--thr', type=float, default=0.007, help='the threshold to assign labels')
parser.add_argument("--standardize", action='store_true', default=False, help='True to standardize data')
parser.add_argument('--winsorize', action='store_true', default=False, help='True to clamp data of more than 2sigma to be exactly 2sigma')

# path specification
parser.add_argument("--save-dir", type=str, default='preprocessed', help='the name of directory under which preprocessed data is stored')
parser.add_argument("--tag", type=str, default='exp', help='tag for config paths')

args = parser.parse_args()

# load paths
paths = load_config(tag=args.tag)
store_path = os.path.join('../../', paths['data_dir'], args.save_dir)
os.makedirs(store_path, exist_ok=True)

# ===== aux =====
def read_index_data(name: str) -> pd.DataFrame:
    """read csv"""
    csv_path = os.path.join('../../', paths['data_dir'], 'raw', f'sh{name}.csv')
    df = pd.read_csv(csv_path, index_col=0, usecols=['date', 'close', 'volume', 'money'])
    # change nmae 
    df.columns = ['close', 'volume', 'turnover']
    return df 

def get_features_and_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List]:
    """create rolling window and readoff dates"""
    # dates 
    dates = df.index.tolist()[args.t - 1: - args.k] # chop off look-forward and backward ates
    # features 
    X = np.array(list(df.iloc[:-args.k].rolling(args.t))[args.t - 1:])  # shape:  (# num features, args.t, 3)
    # targets 
    close_price = df['close']
    close_change = ((close_price.rolling(args.k).mean().shift(-args.k) - close_price) / close_price).dropna()[dates] 
    y = (close_change.abs() > args.thr) * ((close_change > 0) * 2 - 1) # (assign labels -1, 0, 1)
    y = y.to_numpy()

    return X, y, dates

def standardize_data(X: np.ndarray) -> np.ndarray:
    """standardize each features within respective rolling window"""
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_standardized = (X - X_mean) / X_std
    return X_standardized

def winsorize_data(X: np.ndarray) -> np.ndarray:
    """winsorize each features to be within """
    X_winsorized = np.where(X.abs() > 2, 2 * np.sign(X), X)
    return X_winsorized

def save_data(X: np.ndarray, y: np.ndarray, dates: List, name: str):
    """save to dir"""
    cur_dir = os.path.join(store_path, name)
    os.makedirs(cur_dir, exist_ok=True)
    np.save(os.path.join(cur_dir, "X.npy"), X)
    np.save(os.path.join(cur_dir, 'y.npy'), y)
    with open(os.path.join(cur_dir, 'dates.pkl'), 'wb') as f:
        pickle.dump(dates, f)
    

# ==== main ====
def main():
    for index_name in args.data:
        df = read_index_data(index_name)
        X, y, dates = get_features_and_targets(df)
        if args.standardize: X = standardize_data(X)
        if args.winsorize: X = winsorize_data(X)

        # save 
        save_data(X, y, dates, index_name)


if __name__ == '__main__':
    main()
