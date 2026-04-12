# methane-gapfill-ml / FluxGapfill — 项目说明（供 AI 与开发者）

本文档概括本仓库的目标、架构、数据流、模块职责与运行方式，便于快速理解与修改代码。

---

## 1. 项目概述

- **仓库名**：`methane-gapfill-ml`（GitHub：`stanfordmlgroup/methane-gapfill-ml`）
- **PyPI 包名**：`fluxgapfill`（`setup.py` 中 `NAME = 'fluxgapfill'`）
- **版本**：`fluxgapfill/__version__.py` 与 `setup.py` 均为 **0.2.1**
- **用途**：用机器学习对 **涡度相关法（eddy covariance）湿地甲烷通量 `FCH4`** 进行 **缺测填补（gap-filling）**，并估计 **预测不确定性**、做 **后处理标定** 与 **年度碳预算** 汇总。
- **对应论文**：*Gap-filling eddy covariance methane fluxes: Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands*（Agricultural and Forest Meteorology, 2021）。README 中含 Elsevier 作者链接与 Zenodo 工具引用说明。

---

## 2. 技术栈

| 类别 | 依赖 |
|------|------|
| 数据与数值 | `pandas`, `scipy`, `numpy`（随上述引入） |
| 机器学习 | `scikit-learn`（LassoCV、MLP、RandomForest、SimpleImputer、RandomizedSearchCV 等） |
| 梯度提升 | `xgboost` |
| 概率分布 / NGBoost | `ngboost`（`Normal`、`Laplace` 分布，用于集成不确定度） |
| CLI | `fire` |
| 进度条 | `tqdm` |
| 其它 | `python-dateutil`（`gapfill.py` 中 `relativedelta`）；`environment.yml` 为 Conda 环境示例 |

**Python 要求**：`setup.py` 标明 `>=3.6.0`。

---

## 3. 仓库目录结构

```
methane-gapfill-ml/
├── README.md                 # 用户文档、引用格式
├── LICENSE                   # Apache 2.0（README）；setup.py 中 license 字段为 MIT（以 LICENSE 文件为准）
├── setup.py                  # 打包与 install_requires
├── manifest.in               # include README.md LICENSE
├── fluxgapfill_tutorial.ipynb
├── CLAUDE.md                 # 本文件
└── fluxgapfill/              # 主包
    ├── __init__.py
    ├── __version__.py
    ├── main.py               # Fire CLI 入口
    ├── environment.yml       # conda 环境名 ch4-gap-ml
    ├── preprocess/           # 数据加载、人工缺测划分
    │   ├── preprocess.py
    │   ├── load.py
    │   ├── artificial.py     # 缺测长度分布、采样、learn_gap_dist
    │   └── distances.py      # 分布间距离（Cramér–von Mises 等）
    ├── train/                # train.py
    ├── test/                 # test.py（评估与不确定性指标）
    ├── gapfill/              # gapfill.py、budget_date_ranges.json
    ├── models/               # base.py, models.py, ensemble.py
    ├── metrics/              # metrics.py
    └── predictors/           # parse, check, process；*.txt 预测变量列表示例
```

**注意**：仓库根目录 **没有** 独立的 `main.py`；README 中的 `python main.py` 通常指在 **`fluxgapfill` 目录下** 执行该目录内的 `main.py`，或使用 `python -m fluxgapfill.main`（需在包可导入路径下）。

---

## 4. 数据约定

### 4.1 原始 CSV（每站点）

- 路径：`{data_dir}/{SiteID}/raw.csv`
- **必填列**：
  - `TIMESTAMP_END`：格式 **`YYYYMMDDHHmm`**（例如 `201312060030`）
  - `FCH4`：甲烷通量，单位 **nmol m⁻² s⁻¹**；缺测为 NaN（另可将 `-9999` 等通过 `na_values` 标为 NA）
- **其余列**：均视为候选 **输入预测变量**（训练/预测时使用 `predictors` 或 `predictors_paths` 指定子集）。
- **特殊关键字预测变量**（在 `predictors` 列表中）：
  - `temporal`：由 `TIMESTAMP_END` 派生年周期 sin/cos 与时间增量（见 `predictors/process.py`）
  - `all`：在 `add_all_predictors` 中展开为除 `FCH4*` 外的所有列

### 4.2 风温风向

- 若预测变量含 `WD`（风向），`BaseModel.preprocess` 会调用 `process_wind_direction_predictor`：转为弧度并拆成 `sin/cos`，避免不连续角。

### 4.3 测试站点 `TEST`

- `preprocess(..., sites=TEST)` 时从固定 GitHub URL 拉取示例 `raw.csv` 并写入本地 `data/TEST/`，便于无本地数据时试用（需网络）。

---

## 5. 端到端流水线

推荐顺序：**preprocess → train → test → gapfill**（`main.py` 中 `run_all` 即按此顺序调用）。

### 5.1 Preprocess（`fluxgapfill/preprocess/preprocess.py`）

1. 读取 `raw.csv`，识别真实缺测：`FCH4.isna()` → **`gap.csv`**（保留原始缺测行供最后填补）。
2. **划分方式** `split_method`：
   - **`artificial`（默认）**：从真实 `FCH4` 序列学习 **连续缺测长度** 的经验分布，再与几何分布凸组合，用 Monte Carlo + 距离度量（`dist`：`CramerVonMises`、`KolmogorovSmirnoff` 等）搜索参数，得到 **人工缺测采样 PMF**（`learn_gap_dist` / `artificial.py`）。
   - 用该 PMF 在 **非缺测** 观测上随机挖洞，生成 **test 集**（被挖洞的行）与 **`n_train` 对** train/valid（每对一次独立挖洞）。
   - **`random`**：在无非缺测子集上按 `eval_frac` 做 `train_test_split`，重复 `n_train` 次得到多对 train/valid；test 为一次性划分。
3. 输出（每站点目录下）：
   - `training/train{i}.csv`, `training/valid{i}.csv`
   - `test.csv`
   - `gap.csv`
   - `args.json`（记录预处理参数）

### 5.2 Train（`fluxgapfill/train/train.py`）

- 对每个 `site` × `model` × **预测子集名**（来自文件 stem 或 `predictors`）：
  - 校验 CSV 中含所需列（`check_predictors_present`）。
  - 对每一对 `train_k` / `valid_k` 训练一个模型实例，保存为  
    `{data_dir}/{site}/models/{model}/{predictor_subset}/model{k}.pkl`
  - 在验证集上计算 `log_metrics`（默认 `pr2`, `nmae`），写入 `training_results.csv`。
  - 训练参数写入同目录 `args.json`；若已有模型且未 `overwrite_existing_models`，会比对新旧非通用参数并可能报错。

**模型注册**（`models/models.py` → `get_model_class`）：

| CLI 名 | 类 | 要点 |
|--------|-----|------|
| `lasso` | `Lasso` | `LassoCV` + `StandardScaler` |
| `ann` | `ANN` | `MLPRegressor` + `RandomizedSearchCV`（`BaseModel.fit`） |
| `rf` | `RandomForest` | `RandomForestRegressor` |
| `xgb` | `XGBoost` | `XGBRegressor` |

**共同逻辑**（`models/base.py`）：`preprocess` → `impute`（列均值）→（可选）标准化 → `RandomizedSearchCV(..., scoring="neg_mean_squared_error")`。

### 5.3 Test（`fluxgapfill/test/test.py`）

- **`split=test`**：读取 `test.csv`，用 **`EnsembleModel`** 加载该配置下全部 `model*.pkl`，对每点：
  - 集成均值预测、基于成员方差的 **Laplace/Normal 分布**（`ngboost`）
  - `uncertainty_scale`：**Platt 式标定**（`EnsembleModel.uncertainty_scale`：Laplace 用 \|y−μ\|/scale 的均值；Normal 用标准化平方误差均值）
  - 写入 `scale.json`（**gapfill 前置条件**）
  - 输出点预测、`test_predictions.csv`、`test_results.csv`（含 `metric_dict` 与 `uncertainty_metric_dict`：calibration、sharpness 等）
- **`split=train` 或 `valid`**：逐 fold 读 `train{k}.csv` 或 `valid{k}.csv`，加载对应 `model{k}.pkl` 做点预测（无分布部分）；聚合均值写入 `*_results.csv`。
- 汇总路径：`{data_dir}/results/{split}.csv`（若已存在且未 `overwrite_results` 会报错）。

**实现注意**：`split` 为 `train`/`valid` 的分支里，源码在构造 `mean_scores` 的顺序上存在可疑之处（`mean_scores` 在赋值前被使用）；若运行该分支报错，需本地修复后再用。

### 5.4 Gapfill（`fluxgapfill/gapfill/gapfill.py`）

1. 读取 **`gap.csv`**（仅真实缺测时段）与 **`raw.csv`**。
2. 必须存在对应模型的 **`scale.json`**（由 `test` 且 **`--distribution` 与 gapfill 一致** 生成）。
3. `EnsembleModel.predict_individual` 得到各成员预测；均值 → `FCH4_F`；用标定后的分布算 **95% 区间半宽** → `FCH4_F_UNCERTAINTY`；成员扩散列 `FCH4_F1` … `FCH4_FN`。
4. 与 `raw.csv` 按 `TIMESTAMP_END` 合并；**有观测** 的行用观测 `FCH4` 覆盖 `FCH4_F` 与 `FCH4_F*`，不确定度置 0。
5. 输出：`{site}/gapfilled/{model}_{predictor_subset}_{distribution}.csv`
6. **年度预算**：通量先换算为 **g C m⁻² half-hour⁻¹**（`ch4_conversion`），再按 `budget_date_ranges.json` 或默认按年汇总；`*_budget.csv` 含各区间 mean ± 1.96×std（成员间）。

---

## 6. 核心概念小结

- **人工缺测（artificial gaps）**：使训练/验证/测试上的“被挖洞”在统计上接近真实缺测长度分布，减轻因随机单点 hold-out 带来的评估偏差。
- **集成（ensemble）**：同一配置下 **多折模型** 的算术平均为点预测；**成员间方差** 参数化 Laplace/Normal，得到预测区间。
- **不确定性标定**：在 hold-out test 上估计尺度因子，使区间覆盖率与宽度更合理；gapfill 时复用 `scale.json`。

---

## 7. 命令行（Fire）

入口：`fluxgapfill/main.py` 中 `fire.Fire()` 暴露：

- `preprocess`, `train`, `test`, `gapfill`, `run_all`

**重要**：`preprocess` / `train` 等均需要 **`data_dir`**（数据根目录，其下为各 `SiteID` 子文件夹）。README 部分示例未写出 `data_dir`，实际调用需补上。

**多值参数**：逗号分隔字符串或列表（见 README）。

**预测变量文件**：`predictors_paths` 指向文本文件，**每行一个变量名**；子集名称默认为 **文件名不含扩展名**（如 `meteorological.txt` → 子集名 `meteorological`）。README 示例中的 `predictors/meteorlogical.txt` 为拼写错误，仓库内文件名为 **`meteorological.txt`**。

---

## 8. 维护与开发时需注意的问题

1. **`fluxgapfill/preprocess/load.py`**：若数据无 `year` 列，代码使用 `dt.year`，但文件 **未导入 `datetime` 也未定义 `dt`**，可能运行时报错。更稳妥写法应使用 `datetimes.dt.year` 或显式 `datetime`。
2. **`gapfill.get_site_budget_ranges` 默认分支**使用 `gap_df['Year']`，而 `load_raw_data` 添加的是 **`year`（小写）**。若 CSV 无 `Year` 列，可能 KeyError；可在数据中加 `Year` 或与 `year` 统一。
3. **`test.py`**：`split` 为 `train`/`valid` 时 `mean_scores` 赋值顺序问题（见 5.3）。
4. **`gapfill.py` 主循环**：源码在 `for site` 内对聚合结果 `eval_scores_df.to_csv` 的缩进可能导致 **每个站点重复覆盖** `results/{split}.csv`；若与预期不符需审阅循环结构。
5. **依赖 `dateutil`**：`gapfill.py` 需要 `python-dateutil`；若仅用 `setup.py` 安装且环境极简，需确认已安装。

---

## 9. 引用

使用本工具或方法时请按 **README.md** 中的论文与 Zenodo 条目引用（算法论文 + 工具包版本说明）。

---

## 10. 快速索引（模块 → 职责）

| 模块 | 职责 |
|------|------|
| `preprocess/preprocess.py` | 站点级划分与写 CSV |
| `preprocess/artificial.py` | 缺测长度 PMF、人工挖洞、`learn_gap_dist` |
| `preprocess/distances.py` | 两样本缺测长度分布的距离度量 |
| `preprocess/load.py` | 读 `raw.csv`、校验时间格式 |
| `train/train.py` | 多折训练、保存 pkl、验证指标 |
| `test/test.py` | 集成评估、不确定性指标、`scale.json` |
| `gapfill/gapfill.py` | 真实缺测填补、预算 CSV |
| `models/base.py` | 预处理、插补、搜索、保存 |
| `models/ensemble.py` | 多模型集成均值、分布、标定因子 |
| `metrics/metrics.py` | MSE/MAE/NMAE/R²/pr²/bias；calibration/sharpness |
| `predictors/parse.py` | `predictors_paths` / `predictors` 解析为字典 |
| `predictors/process.py` | `temporal`、`all`、风向 sin/cos |

---

*文档根据当前仓库源码整理；若上游更新，请以代码为准并同步修订本文件。*
