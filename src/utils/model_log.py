"""
log parameters and predictions
"""

# load packages
import os 
from typing import List
import numpy as np
import pandas as pd 

# log predictions
def log_predictions(predictions: np.ndarray, dates: List[str], result_path: str):
    """log into a pandas series"""
    save_name = os.path.join(result_path, 'predictions.csv')
    new_series = pd.Series(predictions, index=dates, name='prediction')
    new_series.index.name = 'date'

    # build on previous
    if os.path.exists(save_name):
        log_series = pd.read_csv(save_name, index_col=0).squeeze()
        # append new dates
        log_series = pd.concat((log_series, new_series))
    else:
        log_series = new_series
    # save 
    log_series.to_csv(save_name)

# =========== baseline ============
def log_baseline_params(best_c: float, start_date: str, end_date: str, model_path: str):
    """store best_c by dates"""
    save_name = os.path.join(model_path, 'hyperparam.csv')
    new_df = pd.DataFrame([[start_date, end_date, best_c]], columns=['start_date', 'end_date', 'best_c'], )

    # build on previous
    if os.path.exists(save_name):
        log_df = pd.read_csv(save_name)
        # append new dates
        log_df = pd.concat((log_df, new_df))
    else:
        log_df = new_df
    # save 
    log_df.to_csv(save_name, index=False)

# ============ nn ===============
# TODO:
