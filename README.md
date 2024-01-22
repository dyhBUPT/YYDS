# YYDS
YYDS: Visible-Infrared Person Re-Identification with Coarse Description

# Abstract
TBD

# Experiments
TBD

# Data Preparation

1. Download datasets [SYSU-MM01](https://github.com/InnovArul/rgb_IR_personreid), 
[RegDB](http://dm.dongguk.edu/link.html) and [LLCM](https://github.com/ZYK100/LLCM/tree/main).
Then, please change correlated variables to your data path, i.e., `data_path` in `train.py`/`test.py`.
The preprocessed SYSU-MM01 is also provided in baidu disk (TBD).

2. Download textual model [RoBERTa](https://huggingface.co/roberta-base).
Then, please change the variable `RoBERTa_path` in `utils_new.py` to your model path.

# Requirements

- python=3.8
- torch==2.0.1
- torchvision==0.15.2
- tensorboard=2.12.1
- numpy==1.24.3
- transformers==4.32.1
- tqdm==4.65.0
- einops==0.7.0
- nltk==3.8.1
- datetime==5.2

# Test
For direct testing, please download our prepared checkpoints and extracted features from baidu disk (TBD).

#### 1) Baseline on SYSU-MM01

##### All search mode
```shell script
python test.py --gpu 0 --resume path_to_model/SYSU_Baseline/sysu_deen_p4_n4_lr_0.1_seed_0_best.t --mode all
```
```shell script
Rank-1: 72.36% | Rank-5: 92.15% | Rank-10: 96.56%| Rank-20: 99.03%| mAP: 68.24%| mINP: 54.03%
```

##### Indoor search mode
```shell script
python test.py --gpu 0 --resume path_to_model/SYSU_Baseline/sysu_deen_p4_n4_lr_0.1_seed_0_best.t --mode indoor
```
```shell script
Rank-1: 78.50% | Rank-5: 96.00% | Rank-10: 98.45%| Rank-20: 99.50%| mAP: 82.06%| mINP: 78.21%
```

#### 2) YYDS (w/o Joint Relation Module) on SYSU-MM01

##### All search mode
```shell script
python test.py --gpu 0 --text_mode v1 --resume path_to_model/SYSU_YYDS_woJoint/sysu_deen_p4_n4_lr_0.1_seed_0_best.t --mode all
```
```shell script
Rank-1: 84.54% | Rank-5: 96.78% | Rank-10: 98.84%| Rank-20: 99.63%| mAP: 80.01%| mINP: 67.89%
```

##### Indoor search mode
```shell script
python test.py --gpu 0 --text_mode v1 --resume path_to_model/SYSU_YYDS_woJoint/sysu_deen_p4_n4_lr_0.1_seed_0_best.t --mode indoor
```
```shell script
Rank-1: 89.47% | Rank-5: 98.53% | Rank-10: 99.37%| Rank-20: 99.78%| mAP: 90.64%| mINP: 87.84%
```

#### 3) YYDS on SYSU-MM01

##### All search mode
```shell script
python test.py --gpu 0 --text_mode v2 --resume path_to_model/SYSU_YYDS/sysu_deen_p4_n4_lr_0.1_seed_0_best.t --mode all
```
```shell script
Rank-1: 85.54% | Rank-5: 97.72% | Rank-10: 99.30%| Rank-20: 99.78%| mAP: 81.64%| mINP: 70.51%
```

##### Indoor search mode
```shell script
python test.py --gpu 0 --text_mode v2 --resume path_to_model/SYSU_YYDS/sysu_deen_p4_n4_lr_0.1_seed_0_best.t --mode indoor
```
```shell script
Rank-1: 89.13% | Rank-5: 98.99% | Rank-10: 99.66%| Rank-20: 99.96%| mAP: 91.00%| mINP: 88.55%
```

#### 4) Baseline on RegDB
```shell script
python test.py --gpu 0 --dataset regdb --resume path_to_model/RegDB_Baseline
```
```shell script
Rank-1: 89.13% | Rank-5: 94.67% | Rank-10: 96.81%| Rank-20: 98.54%| mAP: 81.76%| mINP: 66.91%
```

#### 5) YYDS on RegDB
```shell script
python test.py --gpu 0 --dataset regdb --text_mode v2 --resume path_to_model/RegDB_YYDS
```
```shell script
Rank-1: 90.16% | Rank-5: 95.29% | Rank-10: 97.29%| Rank-20: 98.80%| mAP: 83.53%| mINP: 69.41%
```

#### 6) Baseline on LLCM
```shell script
python test.py --gpu 0 --dataset llcm --resume path_to_model/LLCM_Baseline/llcm_deen_p4_n4_lr_0.1_seed_0_best.t
```
```shell script
Rank-1: 56.51% | Rank-5: 78.09% | Rank-10: 85.30%| Rank-20: 91.31%| mAP: 63.21%| mINP: 59.87%
```

#### 7) YYDS on LLCM
```shell script
python test.py --gpu 0 --dataset llcm --text_mode v2 --resume path_to_model/LLCM_YYDS/llcm_deen_p4_n4_lr_0.1_seed_0_best.t 
```
```shell script
Rank-1: 58.22% | Rank-5: 80.62% | Rank-10: 87.24%| Rank-20: 92.64%| mAP: 65.09%| mINP: 61.72%
```

# Train

First, please change the variable `SAVE_DIR` in `train.py` to your root for saving output files.

#### 1) Trian on SYSU-MM01 and LLCM
Here is an example for training baseline:
```shell script
python -m torch.distributed.run --nproc_per_node 2 --master_port 10000 train.py --gpu 0,1 --dataset sysu --log_path tmp
```
here:
- `nproc_per_node` is the gpu numbers.
- `dataset` can be `sysu` or `llcm`.
- `log_path` is the directory name for saving output files.

To train YYDS, please set `--text_mode v2`. Or you can set `--text_mode v1` to train "YYDS w/o Joint Relation Module".

#### 2) Train on RegDB
For RegDB, we should train 10 models with `trail` 1-10 respectively.
Please directly use our provided scripts:
```shell script
bash train_regdb.bash
```

# Re-Ranking
You can perform both original and our proposed k-reciprocal re-ranking algorithm by running `re_ranking.py`.
Please change the variable `FEAT_DIR` to your model path.

Then, you can choose the method and dataset with the following variables:
- `dataset`: `sysu`, `regdb`, `llcm`.
- `method`: `baseline`, `constrained`, `extended`, `divided`.

Please note that, for these four methods, only parameters `k1`, `k2` and `lam` are needed.
If you want the MA-LQE method, `k3` is also needed (it is not applied for `constrained` method in our codes).

The selected best parameters (`[k1, k2, lam]` or `[k1, k2, k3, lam]`) 
for different methods and datasets are listed here:

|Dataset|Method|Parameters|
|:----:|:-----:|:-----:|
| SYSU-MM01  | k-reciprocal      | [40, 35, 0.1]    |
|            | constrained       | [30, 30, 0.2]    |
|            | divided           | [40, 30, 0.1]    |
|            | extended          | [35, 35, 0]      |
|            | extended + MA-LQE | [35, 35, 4, 0.1] |
| RegDB      | k-reciprocal      | [15, 5, 0.3]     |
|            | constrained       | [10, 1, 0.1]     |
|            | divided           | [10, 10, 0.3]    |
|            | extended          | [10, 5, 0]       |
|            | extended + MA-LQE | [10, 10, 3, 0]   |
| SYSU-MM01  | k-reciprocal      | [25, 10, 0.3]    |
|            | constrained       | [15, 15, 0.4]    |
|            | divided           | [25, 10, 0.5]    |
|            | extended          | [25, 10, 0.3]    |
|            | extended + MA-LQE | [25, 10, 0, 0.4] |

# Citation
TBD

# Acknowledgement
Our codes are based on [DEEN](https://github.com/ZYK100/LLCM). Thanks for their excellent work!