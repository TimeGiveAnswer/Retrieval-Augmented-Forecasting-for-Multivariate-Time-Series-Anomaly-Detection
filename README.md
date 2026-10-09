# Retrieval-Augmented Forecasting for Multivariate Time-Series Anomaly Detection

## Abstract

New equipment typically has only limited normal data at the early deployment stage, while predictive detectors are constrained by a fixed input window and cannot access similar historical patterns beyond it. To address these two issues, this paper proposes a retrieval-augmented prediction method for frozen prediction models that, without using any anomalous samples, turns normal history beyond the window into usable detection evidence. The method consists of four components: normal memory construction, per-variable history retrieval, selective reference fusion, and multi-signal anomaly scoring. Specifically, a normal prototype memory bank is built from normal sequences, retaining boundary samples and subsequent observations; candidates are screened by statistical features and waveform grouping, then re-ranked by Dynamic Time Warping (DTW), and similar histories together with their subsequent observations are returned per variable; a Selective Reference Fusion Network (SRF-Net) limits the influence of unreliable references on the frozen base prediction model through candidate selection and continuous gating, with the normalization parameters, combination weights, and alarm threshold determined on a validation set. Low-Rank Adaptation (LoRA) may optionally be applied for cross-device use. On five datasets—PSM, MSL, SMAP, SMD, and SWaT—using PatchTST, TimesNet, and MambaSSM as base prediction models, point-wise evaluation is performed with AUPRC and AUROC without point adjustment (PA). Results show that SRF-Only raises the mean AUPRC and AUROC from 0.3014 and 0.6578 to 0.3643 and 0.7426, relative improvements of 20.89% and 12.90%, respectively, while SRF+LoRA achieves 0.3815 and 0.7644. Ablation indicates that retrieval distance and per-variable retrieval are the primary sources of the performance gains. The method can invoke normal history beyond the input window without updating the base prediction model's parameters, and the gains are more pronounced on data with evident periodic or repetitive structure.

## 代码范围与实现边界

- `core/`：数据读取与处理、正常记忆库、SRF、权重级 LoRA、训练与 raw/no-PA 评分。
- `experiments/main.py`：3 种骨干 × Native / LoRA-Only / SRF-Only / SRF+LoRA，默认 3 个种子。
- `experiments/comparison.py`：SRF-Only、SRF+LoRA 与 RAFT-style 1NN/UniformK/SoftK、kNN-IDW、MemAE-style SparseAttn、TS-RAG-style ARM、ProtoMem-style Coreset、MP-style NN-Score；所有结果现场计算。
- `experiments/ablation.py`：边界样本、逐变量检索、候选选择、连续门控、记忆距离、参考分歧度的六项消融。
- `experiments/statistics.py`：均值±样本标准差、Friedman、Nemenyi、配对 Wilcoxon 及 CD 图。
- `experiments/sensitivity.py`：K、记忆库规模、L 与评分权重；`experiments/plots.py` 输出参数曲线。
- `experiments/efficiency.py` 与 `experiments/mechanism.py`：资源开销与成功/失败机制案例。

当前骨干是 PatchTST/TimesNet/MambaSSM 机制对应的实现，命令名分别为 `patch_transformer`、`period_conv`、`selective_ssm`。`style` 插件是机制级控制组。

## 安装与数据

Python ≥ 3.10。建议使用独立环境，先按硬件安装 PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

数据须由使用者自行获取并置于仓库外，通过 `--data-root /path/to/data` 指定。本代码不自动下载、上传或打包数据。支持：

- PSM：`PSM/train.csv`、`test.csv`、`test_label.csv`，时间戳列不作为特征。
- SMD：`SMD/{train,test,test_label}/machine-X-Y.txt`（或同名 `.npy`），亦支持 `machine-X-Y_train.npy` 等常见处理后布局。输入 `SMD` 自动发现全部机器，不再限定 3 台代表机器。
- MSL/SMAP：NASA 原始 `train/{chan_id}.npy`、`test/{chan_id}.npy` 与 `labeled_anomalies.csv`；自动逐实体运行，窗口不跨实体。也支持完整的 `MSL/{MSL_train,MSL_test,MSL_test_label}.npy`、SMAP 对应数组；聚合数组须由使用者保证没有跨实体拼接伪边界。
- SWaT：正常/攻击 CSV，包含原始 `Normal/Attack` 标签字段，亦支持常见 `SWaT/train.csv`、`test.csv` 布局。原始 Excel 应由使用者在仓库外转换为 CSV

## 运行

以下命令在仓库根目录执行，默认全量。不同数据规模下结果可能与原缩减实验不同。

```bash
python experiments/main.py --data-root /path/to/data --output-dir outputs/main --seeds 42 52 62

python experiments/comparison.py --data-root /path/to/data --output-dir outputs/comparison --seeds 

python experiments/ablation.py plan --data-root /path/to/data --output-dir outputs/ablation --suites module
python experiments/ablation.py run --data-root /path/to/data --output-dir outputs/ablation --suites module

python experiments/statistics.py --results outputs/main/results.jsonl --output-dir outputs/statistics
python experiments/statistics.py --results outputs/comparison/results.jsonl --output-dir outputs/plugin_statistics

python experiments/sensitivity.py run --data-root /path/to/data --output-dir outputs/sensitivity
python experiments/plots.py --input outputs/sensitivity/family_summary.csv --output-dir outputs/sensitivity_figures

```
