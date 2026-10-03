# ctn-malff-sex-differences
# MRI 静息态 fMRI 分析代码

本目录包含一套从静息态 BOLD 影像预处理、mALFF 计算、AAL3 ROI 汇总、组间 ROI 筛选，到 ROI 级广义线性模型（OLS/GLM）和结果绘图的脚本。代码主要面向 Windows 环境，部分数据路径和资源文件路径直接写在脚本中，运行前请按本机目录结构修改。

## 1. 分析流程

建议按以下顺序执行：

```text
BIDS/OpenNeuro 数据
        │
        ├─ data preprocessing/spm_bold_to_mni.m
        │      SPM12：头动校正、BOLD-T1w 配准、T1w 分割并归一化到 MNI 空间
        │
        ├─ data preprocessing/compute_malff.m
        │      去混杂回归后计算 0.01–0.08 Hz ALFF，并进行全局均值标准化
        │
        ├─ data preprocessing/平均激活值提取.py
        │      将 mALFF NIfTI 重采样到 MNI/AAL3 网格并提取每个 ROI 均值
        │
        ├─ z ROI selection/compare_patient_healthy_malff.py
        │      患者与健康对照比较、Grubbs 异常值替换和大差异 ROI 筛选
        │
        ├─ z ROI selection/compare_patient_sex_malff.py
        │      患者男性与女性比较，并导出筛选 ROI 及全 ROI 校正值
        │
        ├─ ols model/build_model_inputs.py
        │      整合临床表、mALFF、Fisher-z 连接矩阵和图论指标
        │
        ├─ ols model/all_malff_connectivity_prediction_LR*.py
        │      ROI 级全样本 GLM、FDR 校正和拟合图
        │
        └─ plot/*.py
               Figure S1–S3、脑表面/切片和指定 ROI 可视化
```

SPM 归一化、mALFF 计算和 ROI 均值提取脚本支持跳过已有的有效输出，并记录批处理状态；组间比较、建模和绘图脚本通常会重新写出结果。

## 2. 目录结构

```text
代码/
├─ data preprocessing/
│  ├─ spm_bold_to_mni.m           # BOLD 到 MNI 空间的 SPM12 批处理
│  ├─ compute_malff.m             # 去混杂 mALFF 计算
│  └─ 平均激活值提取.py             # AAL3 ROI 均值提取
├─ z ROI selection/
│  ├─ compare_patient_healthy_malff.py
│  └─ compare_patient_sex_malff.py
├─ ols model/
│  ├─ build_model_inputs.py
│  ├─ all_malff_connectivity_prediction_LR.py
│  ├─ all_malff_connectivity_prediction_LR year.py
│  └─ all_malff_connectivity_prediction_FD.py
└─ plot/
   ├─ brain_3d.py
   ├─ brain_3d_ROI.py
   ├─ Figure_S1.py
   ├─ Figure_S2.py
   └─ Figure_S3.py
```

## 3. 环境依赖

### MATLAB

- MATLAB（需支持脚本末尾的局部函数），以及 SPM12。
- SPM12，并在 `spm_bold_to_mni.m` 和 `compute_malff.m` 中将 `SPM_PATH` 改为实际安装目录。
- SPM12 的 `tpm/TPM.nii`、MNI 模板及本项目使用的 AAL3 图谱。

### Python

建议使用 Python 3.10 或更新版本，并安装：

```powershell
python -m pip install numpy pandas scipy matplotlib seaborn nibabel nilearn openpyxl pillow
```

主要用途：

- `numpy`、`pandas`：表格和数值计算；
- `scipy`：统计检验、t 分布和图像变换；
- `nibabel`、`nilearn`：NIfTI 读写和重采样；
- `matplotlib`、`seaborn`、`pillow`：绘图和图像导出；
- `openpyxl`：读取 Excel 文件。

中文文件名脚本建议在 UTF-8 环境下运行。脚本中的 Windows 路径使用原始字符串或 MATLAB 路径格式，修改时请保留反斜杠规则。

## 4. 数据和路径约定

### 4.1 原始 BIDS 数据

`spm_bold_to_mni.m` 默认从 `D:\openneuro data` 读取类似以下结构的数据：

```text
D:\openneuro data\sub-009\anat\sub-009_run-01_T1w.nii.gz
D:\openneuro data\sub-009\func\sub-009_task-rest_bold.nii.gz
```

脚本默认匹配：

- BOLD：`*task-rest*bold.nii*`；
- T1w：`*T1w.nii*`。

它会在 `D:\output` 下生成 MNI 空间 4D BOLD，文件名以 `_space-MNI152_spm.nii` 结尾；SPM 中间文件保存到 `D:\output\spm_work_batch_to_mni`，批处理状态写入 `batch_status.csv`。默认输出体素大小为 `[2 2 2]` mm。

### 4.2 mALFF 计算输入

`compute_malff.m` 读取 `D:\output\*_space-MNI152_spm.nii`，并使用对应 SPM 工作目录中的：

- `y_*.nii`：T1w 到 MNI 的形变场；
- `c2*.nii`、`c3*.nii`：原生空间 WM/CSF 概率图；
- `wc2*.nii`、`wc3*.nii`：MNI 空间 WM/CSF 概率图。

脚本还会尝试从 `D:\openneuro data` 查找原始 BOLD JSON 中的 `RepetitionTime`；找不到时使用默认 TR `3.0 s`。频段默认为 `0.01–0.08 Hz`。去混杂设计包括 Friston 24 头动参数、WM/CSF 平均信号、截距和线性趋势；脚本计算低频 ALFF，并除以脑掩模内的全局平均 ALFF 得到 mALFF。输出目录为 `data preprocessing/malff_output`，状态文件为 `batch_malff_status.csv`。

### 4.3 AAL3 ROI 均值提取

`平均激活值提取.py` 默认需要以下文件位于 `data preprocessing`：

- `malff_output/`：`*_mALFF_nuisanceRegressed.nii*`；
- `mni152.nii`：MNI 模板；
- `AAL3v1_1mm.nii`：AAL3 标签图谱。

脚本将影像和图谱重采样到模板网格（连续插值用于 mALFF，最近邻插值用于标签），为每个有效 AAL3 ROI 计算均值，结果写入 `data preprocessing/nii_stats/`：

- 每个影像一个 `*_AAL3_mean.csv`；
- `all_mALFF_nuisanceRegressed_AAL3_mean_long.csv`：合并长表，列为 `Subject`、`Image`、`ROI`、`MeanValue`；
- `batch_AAL3_mean_status.csv`：处理状态。

脚本只负责重采样和 ROI 汇总，不会替代 MNI 配准；输入 mALFF 应已完成 MNI 标准化。

## 5. 运行预处理

1. 打开 MATLAB，将当前目录切换到 `D:\predata\代码\data preprocessing`。
2. 修改两个 `.m` 文件开头的 `SPM_PATH`、输入目录和输出目录。
3. 先运行：

```matlab
run('spm_bold_to_mni.m')
```

4. 再运行：

```matlab
run('compute_malff.m')
```

5. 在 PowerShell 中运行 ROI 汇总：

```powershell
Set-Location 'D:\predata\代码\data preprocessing'
python '.\平均激活值提取.py'
```

批处理脚本默认 `OVERWRITE_OUTPUT = false`，重新计算时请确认是否需要改为 `true`，以免覆盖已有结果。

## 6. ROI 组间筛选

两个比较脚本读取 NPZ 文件。NPZ 至少应包含以下字段：

- `data` 或 `meanvalue_matrix`：形状为 `ROI × subject` 的数值矩阵；
- `roi_ids`：每一行对应的 ROI 编号；
- `subject_ids`：每一列对应的受试者 ID。

脚本先按 ROI 行执行 Grubbs 异常值替换，再计算组均值差异；以所有 ROI 差异的标准差计算 z 值，并筛选 `abs(z_value) > 2.0` 的 ROI。输出还包括筛选 ROI 的逐样本校正值。

患者与健康对照：

```powershell
Set-Location 'D:\predata\代码\z ROI selection'
python '.\compare_patient_healthy_malff.py' `
  --healthy 'D:\predata\npz_data\healthy_mALFF_AAL3_mean.npz' `
  --patient 'D:\predata\npz_data\patient_mALFF_AAL3_mean.npz' `
  --output 'D:\predata\patient_vs_healthy_large_diff_rois.csv' `
  --values-output 'D:\predata\patient_vs_healthy_selected_rois_malff_grubbs.csv'
```

患者男性与女性：

```powershell
python '.\compare_patient_sex_malff.py' `
  --female 'D:\predata\npz_data\patient_female_mALFF_AAL3_mean.npz' `
  --male 'D:\predata\npz_data\patient_male_mALFF_AAL3_mean.npz'
```

可用参数：`--alpha`（默认 `0.05`）、`--std-times`（默认 `2.0`）。性别比较脚本另外输出男性/女性筛选 ROI 值和所有 ROI 的校正值。

运行 `compare_patient_sex_malff.py` 时，脚本通过同目录导入 `compare_patient_healthy_malff.py`，因此不要单独移动该文件。

## 7. 构建 OLS/GLM 模型输入

`ols model/build_model_inputs.py` 默认读取 `data/`（相对于运行命令时的当前目录）。所需文件结构如下；mALFF 表提供长表或宽表其中一种即可，脚本优先使用长表：

```text
data/
├─ 参与者信息总表.xlsx
├─ nii_stats/
│  ├─ all_mALFF_nuisanceRegressed_AAL3_mean_long.csv  # 优先使用
│  └─ patient_male_vs_female_all_samples_malff_grubbs.csv  # 长表不存在时作为宽表回退
└─ matrices/
   └─ sub-*_fisher_z.csv
```

临床表至少需要 `BIDS_ID`、`sex`、`age`、`Pain_severity (average score)`、`Disease_duration (years)`、`BMI`、`Pain_side`；如果存在，脚本也会保留 `Pain_type` 和 `Sindou_grade`。连接矩阵必须是方阵、行列标签一致、对称且不含 NaN，文件名中的受试者 ID 应与 `BIDS_ID` 匹配。

运行：

```powershell
Set-Location 'D:\predata\代码\ols model'
python '.\build_model_inputs.py' `
  --data-dir 'D:\predata\data' `
  --output-dir '.\model_inputs'
```

输出包括：

- `participants_for_analysis.csv`：临床设计表；
- `mALFF_wide.csv`：每行一个受试者、每列一个 ROI；
- `connectivity_edges_fisher_z_wide.csv.gz`：Fisher-z 矩阵上三角边；
- `graph_metrics_wide.csv`：全局和节点级连接指标；
- `qc_summary.json`、`qc_summary.md`：样本匹配和数据质量检查；
- `model_formula_notes.md`：建议模型说明。

## 8. 运行 ROI 级 GLM

三个模型脚本都读取 `participants_for_analysis.csv` 和 `mALFF_wide.csv`，对每个 ROI 批量拟合 OLS/GLM，并计算系数、t 值、正态近似 p 值、t 分布 p 值、FDR q 值和拟合指标。检验项为 `sex`、`pain_severity` 和 `sex_x_pain`；FDR 按检验项分别在各 ROI 间校正。三个脚本使用的协变量不同：

### 普通模型

```powershell
python '.\all_malff_connectivity_prediction_LR.py' `
  --input-dir '.\model_inputs' `
  --output-dir '.\LRresults' `
  --plot-top-n 9
```

### 简化协变量模型

```powershell
python '.\all_malff_connectivity_prediction_LR year.py' `
  --input-dir '.\model_inputs' `
  --output-dir '.\year results' `
  --plot-top-n 9
```

### 加入平均头动 MeanFD 的模型

FD 模型额外要求 `model_inputs/motion_metrics.csv`，且必须包含 `SubjectID` 和 `MeanFD`，并与分析样本 ID 完全一致：

```powershell
python '.\all_malff_connectivity_prediction_FD.py' `
  --input-dir '.\model_inputs' `
  --output-dir '.\FD results' `
  --plot-top-n 9
```

具体协变量为：普通模型 `sex`、`pain_severity`、`age`、`disease_duration_years`、`BMI_numeric`；简化协变量模型省略 `disease_duration_years`；MeanFD 模型在普通模型基础上增加 `MeanFD`。连续协变量和疼痛严重度会标准化，`sex` 按输入数��使用；请确认数据编码符合研究定义。三个脚本都包含 sex 与疼痛严重度交互项。每个模型输出目录包含 `roi_malff_glm_results.csv` 和拟合/显著性图。图中按 `q_fdr_normal < 0.05` 标记；正式报告时请同时说明样本量、模型设定、协变量编码和多重比较范围。

## 9. 绘图脚本

### 脑激活 NIfTI 可视化

`plot/brain_3d.py` 将激活 NIfTI 叠加到 MNI152 模板，导出不同方向切片和投影图。默认只查找输入目录顶层的未压缩 `*.nii` 文件；默认资源为 `plot/AAL3v1_1mm.nii`、`plot/mni152.nii`，默认输入 `plot/nii`，输出 `plot/python results`。若未指定 `--keep-existing`，脚本会先删除输出目录顶层已有的 PNG/TIF/TIFF 文件，再生成新图；请确认输出目录中没有需要保留的图片。

```powershell
Set-Location 'D:\predata\代码\plot'
python '.\brain_3d.py' `
  --activation-folder '.\nii' `
  --output-folder '.\python results' `
  --num-slices 10 `
  --export-resolution 1200
```

常用参数：`--act-thresh`、`--color-threshold`、`--slice-tolerance`、`--keep-existing`。

### 指定 AAL3 ROI 切片

`plot/brain_3d_ROI.py` 默认绘制 ROI `17` 和 `43`，需要 AAL3 标签图谱、`aal3.csv` 标签表和 MNI 模板：

```powershell
python '.\brain_3d_ROI.py' `
  --roi-ids 17 43 `
  --output-folder '.\python ROI' `
  --dpi 1200
```

### Figure S1–S3

- `Figure_S1.py`：读取同目录 `参与者信息 .xlsx` 的 `总表` 工作表，要求 `性别`、`疼痛评分` 两列，输出 `性别_疼痛评分_散点图.tif`；性别编码为 1=男性、0=女性；
- `Figure_S2.py`：读取同目录 `motion_metrics分组.csv` 和 `motion_metrics男女.csv`，按诊断组及患者性别绘制 MeanFD、MaxFD 和高运动比例，输出 `Figure_S2_motion_QC.tif`；
- `Figure_S3.py`：读取同目录 `性别效应的稳健性.xlsx`，输出 `Figure_S3_forest.tif`。

运行示例：

```powershell
python '.\Figure_S1.py'
python '.\Figure_S2.py'
python '.\Figure_S3.py'
```

如果输入 Excel/CSV 不在脚本默认位置，请先修改文件顶部的 `DATA_FILE`、`GROUP_FILE`、`SEX_FILE` 或相关常量。

## 10. 输出文件和质量检查

建议每个阶段完成后检查：

1. `batch_status.csv` 或 `batch_malff_status.csv` 中是否存在 `FAILED`；
2. NIfTI 是否可被 SPM、nibabel 或 FSL 正常读取，且维度/仿射矩阵一致；
3. AAL3 长表中 `Subject`、`ROI`、`MeanValue` 是否完整；
4. NPZ 的 `roi_ids` 和 `subject_ids` 是否与矩阵维度一致；
5. `qc_summary.md/json` 中是否有缺失受试者或额外受试者；
6. GLM 结果中的样本量、残差自由度、p 值和 FDR q 值是否合理。

## 11. 常见问题

### 找不到 SPM 或 TPM

检查 `SPM_PATH` 是否指向 SPM12 根目录，并确认 `SPM_PATH/tpm/TPM.nii` 存在；MATLAB 中可先执行 `which spm` 验证。

### 找不到输入 BOLD 或 JSON

检查 BIDS 目录层级、文件名模式和 `RAW_INPUT_DIR`/`INPUT_DIR`。JSON 缺失时 mALFF 脚本会回退到 `TR_SECONDS = 3.0`，但正式分析建议使用每个 run 的真实 `RepetitionTime`。

### ROI 数量或网格不一致

确保 mALFF、MNI 模板和 AAL3 图谱处于同一空间；ROI 提取脚本会重采样，但不会修复错误的空间方向或错误配准。

### 模型提示 ID 或列缺失

统一 `BIDS_ID`、`SubjectID` 和 NPZ 的 `subject_ids` 格式，删除重复 ID，并确认临床表、mALFF 表、连接矩阵和 MeanFD 表覆盖同一批受试者。

### Windows 路径含空格

PowerShell 命令中的路径请使用单引号；Python 文件名如 `all_malff_connectivity_prediction_LR year.py` 也应使用引号包裹。

## 12. 可复现性建议

- 在每次分析前记录 MATLAB、SPM12、Python、numpy、pandas、scipy、nibabel 和 nilearn 版本。
- 保留各阶段的状态 CSV、QC 报告、模型输出和绘图参数。
- 不要直接覆盖原始 NIfTI、NPZ 或临床表；将修改后的路径写入脚本配置区。
- 报告组间筛选结果时同时给出 Grubbs `alpha`、ROI z 值阈值和 GLM 的 FDR 阈值。

本 README 依据当前目录中的脚本和默认配置编写；如果脚本常量、输入文件名或数据组织方式发生变化，请同步更新本文档。
