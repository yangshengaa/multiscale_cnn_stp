"""
train neural network models
- FCNN / MLP
- CNN

training using float32 throughout
"""

# load packages 
import os 
import argparse
# import warnings 
# warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch 
import torch.nn as nn
from torchsummary import summary

# load file 
from data import read_data, ts_split
from model import MLP, CNN
from utils import load_config, log_predictions, log_nn_params, log_nn_weights

# ===== arguments =====
parser = argparse.ArgumentParser()

# data 
parser.add_argument("--data", type=str, 
    default='000016', 
    choices=["000016", "000300", "000852", "000903", "000905"],
    help='name of index data to read'
)

# model 
parser.add_argument("--model", type=str, default='CNN', choices=['MLP', "CNN"], help='the choice of neural network models')
parser.add_argument("--hidden-dims", nargs="+", type=int, default=[1000], help='hidden layer dimensions')
parser.add_argument("--nl", type=str, default="ReLU", help='the nonlinearity')
parser.add_argument("--scale", type=int, default=1, help='downsampling scale for CNN')
parser.add_argument("--num-filters", type=int, default=32, help='number of filters for CNN')
parser.add_argument("--gru-hidden", type=int, default=32, help='dimension of gru hidden state')

# train
parser.add_argument("--opt", type=str, default='Adam', help='type of optimizer')
parser.add_argument("--lr", type=float, default=0.01, help='the learning rate')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train')
parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')

# path specification
parser.add_argument("--tag", type=str, default='exp', help='tag for config paths')

# length of data period
parser.add_argument("--train", type=int, default=240, help='number of days to train')
parser.add_argument("--val", type=int, default=40, help='the number of days to validate')
parser.add_argument("--test", type=int, default=40, help='the number of days to test')
parser.add_argument("--K", type=int, default=5, help='the number of folds for validation')

# model specific hyperparameters
parser.add_argument('--seed', type=int, default=404, help='the seed for reproducibility')
parser.add_argument('--no-gpu', action='store_true', default=False, help='turn off gpu usage')

# print all?
parser.add_argument("--verbose", action='store_true', default=False, help='turn on to print all training information')
parser.add_argument("--log-weights", action='store_true', default=False, help='log model weights if true')

args = parser.parse_args()

# specify device
device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')

# specify seed
torch.manual_seed(args.seed)

# load path 
paths = load_config(args.tag)
model_name = f'{args.model}_{args.data}_tr{args.train}_va{args.val}_te{args.test}'
model_path = os.path.join(paths['model_dir'], model_name)
os.makedirs(model_path, exist_ok=True)
result_path = os.path.join(paths['result_dir'], model_name)
os.makedirs(result_path, exist_ok=True)

# load data 
data, targets, dates = read_data(args.data, paths['data_dir'])
loss_func = nn.CrossEntropyLoss()


def get_model(verbose=False) -> nn.Module:
    """retrieve model according to parameters"""
    # get nonlinearity 
    nl = getattr(nn, args.nl)()

    # get model
    if args.model == 'MLP':
        input_dim, output_dim = 300, 3
        model = MLP(input_dim, args.hidden_dims, output_dim, nl).to(device)
    elif args.model == "CNN":
        window_size, num_features, output_dim = 100, 3, 3
        model = CNN(window_size, num_features, args.num_filters, args.gru_hidden, output_dim, args.scale, nl).to(device)
        if verbose:
            print(summary(model, (100, 3)))
    else:
        raise NotImplementedError()
    
    return model


def train():
    """train loop for neural net"""
    make_tabular = args.model == 'MLP'
    for period_dict, target_dict, dates_dict in ts_split(
            data, targets, dates, args.train, args.val, args.test, args.K,
            make_tabular=make_tabular, return_tensor=True
    ):
        period_start_date = dates_dict['train'][0][0]
        period_end_date = dates_dict['test'][-1]
        print(f'====== {period_start_date} to {period_end_date} =======')

        # ================= cross validation ===============
        # use average validation accuracy to pick the best hyperparameter
        # ! now it is just a fake cross validation
        best_val_avg_acc = 0
        best_scale, best_gru_hidden = None, None
        scale_list = [1, 2, 3, 4, 5, 6, 7, 8]
        gru_hidden_list = [32, 64, 128, 256]
        for s in scale_list:
            for h in gru_hidden_list:
                args.scale = s
                args.gru_hidden = h
                cur_val_acc = []
                for fold, (X_train, X_val, y_train, y_val) in enumerate(zip(
                        period_dict['train'], period_dict['val'], target_dict['train'], target_dict['val']
                )):
                    # send to device (full batch training)
                    X_train, X_val, y_train, y_val = X_train.to(device), X_val.to(device), y_train.to(device), y_val.to(
                        device)

                    # get model
                    model = get_model()

                    # initialize optimizer
                    opt = getattr(torch.optim, args.opt)(model.parameters(), lr=args.lr, weight_decay=args.wd)

                    # train
                    loop = range(args.epochs)
                    if args.verbose:
                        loop = tqdm(loop)
                    model.train()
                    for e in loop:
                        opt.zero_grad()
                        outputs = model(X_train)
                        loss = loss_func(outputs, y_train)
                        loss.backward()
                        opt.step()

                        if args.verbose:
                            loop.set_description(f"scale = {s}, gru_hidden = {h}, fold = {fold}")
                            loop.set_postfix_str(f"train loss = {loss.item():.4f}")

                    # validate
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_pred = val_outputs.argmax(dim=1)
                        cur_acc = accuracy_score(y_val.detach().cpu().numpy(), val_pred.detach().cpu().numpy())
                        cur_val_acc.append(cur_acc)

                    # print
                    if args.verbose:
                        print(f"scale = {s}, gru_hidden = {h}: fold = {fold}, val acc = {cur_acc}")

                    cur_val_acc.append(cur_acc)
                avg_acc = np.mean(cur_val_acc)
                if args.verbose:
                    print(f"scale = {s}, gru_hidden = {h}: avg val acc = {avg_acc}")
                if avg_acc > best_val_avg_acc:
                    if args.verbose:
                        print(f"beat current best val acc {best_val_avg_acc}")
                    best_val_avg_acc = avg_acc
                    best_scale = s
                    best_gru_hidden = h

        # find best hyperparameter
        args.scale, args.gru_hidden = best_scale, best_gru_hidden
        print(f"selected param: scale = {args.scale}, gru_hidden = {args.gru_hidden}, val avg acc: {best_val_avg_acc:.4f}")

        # =========== retrain ===========
        # send to device
        X_train_all, y_train_all = period_dict['train_all'].to(device), target_dict['train_all'].to(device)
        X_test, y_test = period_dict['test'].to(device), target_dict['test'].to(device)

        # get model
        model = get_model(verbose=True)

        # initialize optimizer
        opt = getattr(torch.optim, args.opt)(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # train
        loop = range(args.epochs)
        if args.verbose: loop = tqdm(loop)
        model.train()
        for e in loop:
            opt.zero_grad()
            outputs = model(X_train_all)
            loss = loss_func(outputs, y_train_all)
            loss.backward()
            opt.step()

            if args.verbose:
                loop.set_description(f"testing loop: ")
                loop.set_postfix_str(f"train loss = {loss.item():.4f}")

        # validate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_pred = test_outputs.argmax(dim=1)
            test_acc = accuracy_score(y_test.detach().cpu().numpy(), test_pred.detach().cpu().numpy())
            test_f1 = f1_score(y_test.detach().cpu().numpy(), test_pred.detach().cpu().numpy(), average='macro')
            #calculate ROC AUC score
            test_prob = torch.softmax(test_outputs,dim=1).detach().cpu().numpy()
            test_roc_auc = roc_auc_score(y_test.detach().cpu().numpy(),test_prob,multi_class = "ovr")
            print(f"test metrics: acc = {test_acc:.4f}, f1 = {test_f1:.4f}, roc_auc= {test_roc_auc:.4f}")

        # log 
        # save model hyperparam (this is also fake now, need to decide what hyperparameters are to be tested)
        log_nn_params(args.scale, args.gru_hidden, period_start_date, period_end_date, model_path)
        # the following line is memory intensive
        if args.log_weights: log_nn_weights(model, period_start_date, period_end_date, model_path)
        log_predictions(test_pred.detach().cpu().numpy(), dates_dict['test'], result_path)
        print()

def main():
    train()

if __name__ == '__main__':
    main()
