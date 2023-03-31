# multiscale_cnn_stp

AC 209B Final Project: Stock Trend Prediction using Convolutional Neural Network

Team Member: Xiaohan Zhao, Xiaochen Wang, Ziqing Luo, Sheng Yang, Chao Wang

## Environment

At root, run:

```bash
conda create -n stp
conda activate stp
conda install python==3.8
conda install pytorch torchvision torchaudio -c pytorch
pip install --no-cache-dir autopep8 jupyterlab toml timebudget tensorboard torch-tb-profiler
pip install --no-cache-dir statsmodels seaborn scipy pillow xgboost tqdm
pip install -e .
```

## Preprocess

At root, run:

```bash
cd src/data
python preprocess.py --standardize
```

See other arguments in [preprocess.py](src/data/preprocess.py).

## Referecnes

- [MTDNN](https://www.ijcai.org/proceedings/2020/0628.pdf)
