"""
use multilogisitc regression as baseline
"""

# load packages 
import os 
import argparse
import warnings 
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# load file 
from data import read_data, ts_split
from utils import load_config, log_predictions, log_baseline_params

# ===== arguments =====
parser = argparse.ArgumentParser()

# data 
parser.add_argument("--data", type=str, 
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
parser.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100], help='penalty strength')

# print all?
parser.add_argument("--verbose", action='store_true', default=False, help='turn on to print all training information')

args = parser.parse_args()

# load path 
paths = load_config(args.tag)
model_name = f'mlr_{args.data}'
model_path = os.path.join(paths['model_dir'], model_name)
os.makedirs(model_path, exist_ok=True)
result_path = os.path.join(paths['result_dir'], model_name)
os.makedirs(result_path, exist_ok=True)

# load data 
data, targets, dates = read_data(args.data, paths['data_dir'])

def train():
    """train loop"""
    for period_dict, target_dict, dates_dict in ts_split(data, targets, dates, args.train, args.val, args.test, args.K, make_tabular=True):
        period_start_date = dates_dict['train'][0][0]
        period_end_date = dates_dict['test'][-1]
        print(f'====== {period_start_date} to {period_end_date} =======')

        # use average validation accuracy to pick the best hyperparameter
        val_avg_acc = []
        for c in args.C:
            cur_val_acc = []
            for fold, (X_train, X_val, y_train, y_val) in enumerate(zip(
                    period_dict['train'], period_dict['val'], target_dict['train'], target_dict['val']
                )):
                # train 
                mlr = LogisticRegression(penalty='l2', C=c).fit(X_train, y_train)
                cur_acc = mlr.score(X_val, y_val)

                # print 
                if args.verbose:
                    print(f"c = {c}: fold = {fold}, val acc = {cur_acc}")
                
                cur_val_acc.append(cur_acc)
            avg_acc = np.mean(cur_val_acc)
            val_avg_acc.append(avg_acc) 
        
        # find best hyperparameter
        best_param_idx = np.argmax(val_avg_acc)
        best_c, bets_val_avg_acc = args.C[best_param_idx], val_avg_acc[best_param_idx]
        print(f"selected param: c = {best_c}, val avg acc: {bets_val_avg_acc:.4f}")
        
        # retrain 
        mlr = LogisticRegression(penalty='l2', C=best_c).fit(period_dict['train_all'], target_dict['train_all'])
        test_pred = mlr.predict(period_dict['test'])
        test_y = target_dict['test']
        test_acc = accuracy_score(test_y, test_pred)
        test_f1 = f1_score(test_y, test_pred, average='macro')  # TODO: maybe weighted? need discussion
        print(f"test metrics: acc = {test_acc:.4f}, f1 = {test_f1:.4f}")

        # log 
        # save model hyperparam to 
        log_baseline_params(best_c, period_start_date, period_end_date, model_path)
        log_predictions(test_pred, dates_dict['test'], result_path)
        print()
        

def main():
    train()

if __name__ == '__main__':
    main()
