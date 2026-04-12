# 运行指南：Irvin et al. 2021 论文复现

所有命令在项目根目录（`methane-gapfill-ml/`）下执行。

---

## 环境安装

```bash
pip install fire tqdm ngboost xgboost scikit-learn python-dateutil pdfplumber
```

---

## 站点列表（论文 17 个湿地站点）

```
US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC
```

原始数据来源：`E:\微云\data\环境\CH4\CH4\`（FLUXNET-CH4 v1.0）

---

## 步骤 0：提取原始数据

从 FLUXNET-CH4 zip 包提取 HH（半小时）CSV，生成各站点 `data/{SiteID}/raw.csv`。

```bash
python prepare_data.py
```

已完成，可跳过。

---

## 步骤 1：数据预处理（Preprocess）

估计各站点真实缺测长度分布，生成人工缺测训练/验证/测试集。

**耗时估计：每站点约 5~15 分钟，共约 2~4 小时。**

```bash
python -m fluxgapfill.main preprocess \
  --data_dir data \
  --sites "US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC" \
  --n_train 10 \
  --eval_frac 0.1 \
  --n_mc 100
```

输出（每个站点目录下）：
- `data/{SiteID}/training/train{1-10}.csv`
- `data/{SiteID}/training/valid{1-10}.csv`
- `data/{SiteID}/test.csv`
- `data/{SiteID}/gap.csv`

---

## 步骤 2：模型训练（Train）

训练 4 种模型 × 4 种预测变量集 × 17 站点 × 10 折 = 2720 个模型。

**耗时估计：数小时到 1~2 天（取决于机器性能）。**

```bash
python -m fluxgapfill.main train \
  --data_dir data \
  --sites "US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC" \
  --models "lasso,ann,rf,xgb" \
  --predictors_paths "fluxgapfill/predictors/temporal.txt,fluxgapfill/predictors/meteorological.txt,fluxgapfill/predictors/knox.txt,fluxgapfill/predictors/all.txt" \
  --overwrite_existing_models True
```

4 种预测变量集（来自论文 Table 3）：

| 文件 | 子集名 | 包含变量 |
|------|--------|---------|
| `temporal.txt` | temporal | sin/cos 季节 + DOY |
| `meteorological.txt` | meteorological | TA_F, SW_IN_F, WS_F, PA_F |
| `knox.txt` | knox | temporal + meteorological（论文 baseline）|
| `all.txt` | all | temporal + 全部可用变量 |

输出：`data/{SiteID}/models/{model}/{predictor}/model{1-10}.pkl`

---

## 步骤 3：模型评估（Test）

在 test 集上评估，并生成不确定性标定系数 `scale.json`（gapfill 的前置条件）。

**耗时估计：1~3 小时。**

```bash
python -m fluxgapfill.main test \
  --data_dir data \
  --sites "US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC" \
  --models "lasso,ann,rf,xgb" \
  --predictors_paths "fluxgapfill/predictors/temporal.txt,fluxgapfill/predictors/meteorological.txt,fluxgapfill/predictors/knox.txt,fluxgapfill/predictors/all.txt" \
  --split test \
  --distribution laplace \
  --overwrite_results True
```

输出：
- `data/{SiteID}/models/{model}/{predictor}/test_results.csv`
- `data/{SiteID}/models/{model}/{predictor}/scale.json`
- `data/results/test.csv`（汇总）

---

## 步骤 4：缺测填补（Gapfill）

用训练好的模型填补真实缺测（`gap.csv`），并计算年度碳预算。

**耗时估计：30 分钟~1 小时。**

```bash
python -m fluxgapfill.main gapfill \
  --data_dir data \
  --sites "US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC" \
  --models "lasso,ann,rf,xgb" \
  --predictors_paths "fluxgapfill/predictors/temporal.txt,fluxgapfill/predictors/meteorological.txt,fluxgapfill/predictors/knox.txt,fluxgapfill/predictors/all.txt" \
  --distribution laplace
```

输出：
- `data/{SiteID}/gapfilled/{model}_{predictor}_laplace.csv`（填补后完整时间序列）
- `data/{SiteID}/gapfilled/{model}_{predictor}_laplace_budget.csv`（年度碳预算）

---

## 一键运行全部步骤

**注意：总耗时可能超过 24 小时，建议在 tmux/screen 中运行。**

```bash
python -m fluxgapfill.main run_all \
  --data_dir data \
  --sites "US-Uaf,US-Los,SE-Deg,FI-Sii,US-Twt,FI-Si2,CA-SCB,NZ-Kop,FI-Lom,JP-Mse,JP-BBY,BR-Npw,US-Tw4,US-WPT,US-Myb,US-Tw1,US-OWC" \
  --models "lasso,ann,rf,xgb" \
  --predictors_paths "fluxgapfill/predictors/temporal.txt,fluxgapfill/predictors/meteorological.txt,fluxgapfill/predictors/knox.txt,fluxgapfill/predictors/all.txt" \
  --n_train 10 \
  --eval_frac 0.1 \
  --n_mc 100 \
  --split test \
  --distribution laplace \
  --overwrite_existing_models True \
  --overwrite_results True
```

---

## 快速测试（单站点单模型）

先验证流程可以跑通再全量运行：

```bash
# 预处理
python -m fluxgapfill.main preprocess \
  --data_dir data --sites "NZ-Kop" --n_train 10 --eval_frac 0.1 --n_mc 100

# 训练（只跑 rf + all）
python -m fluxgapfill.main train \
  --data_dir data --sites "NZ-Kop" --models "rf" \
  --predictors_paths "fluxgapfill/predictors/all.txt" \
  --overwrite_existing_models True

# 评估
python -m fluxgapfill.main test \
  --data_dir data --sites "NZ-Kop" --models "rf" \
  --predictors_paths "fluxgapfill/predictors/all.txt" \
  --split test --distribution laplace --overwrite_results True

# 填补
python -m fluxgapfill.main gapfill \
  --data_dir data --sites "NZ-Kop" --models "rf" \
  --predictors_paths "fluxgapfill/predictors/all.txt" \
  --distribution laplace
```

---

## 数据目录结构

```
methane-gapfill-ml/
├── prepare_data.py          # 步骤 0：提取 raw.csv
├── RUN.md                   # 本文件
├── data/
│   ├── {SiteID}/
│   │   ├── raw.csv          # 输入数据
│   │   ├── gap.csv          # 真实缺测行
│   │   ├── test.csv         # 人工缺测测试集
│   │   ├── training/
│   │   │   ├── train{1-10}.csv
│   │   │   └── valid{1-10}.csv
│   │   ├── models/
│   │   │   └── {model}/{predictor}/
│   │   │       ├── model{1-10}.pkl
│   │   │       ├── scale.json
│   │   │       └── test_results.csv
│   │   └── gapfilled/
│   │       ├── {model}_{predictor}_laplace.csv
│   │       └── {model}_{predictor}_laplace_budget.csv
│   └── results/
│       └── test.csv         # 所有站点汇总结果
└── fluxgapfill/
    └── predictors/
        ├── temporal.txt
        ├── meteorological.txt
        ├── knox.txt          # baseline（论文中）
        └── all.txt
```
