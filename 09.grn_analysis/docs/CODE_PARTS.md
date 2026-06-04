# Previous_code 代码结构总览

本文件只基于静态读取 `Previous_code` 目录中的 `ipynb` 和 `_3D_plot.py` 得出结论，没有运行任何代码。

## 总体判断

这批代码可以分成 **8 个分析模块 + 1 个通用工具文件**。整体流程基本是：

`原始/标注数据统计 -> 按区域拆分并合并 h5ad -> GRN 推断 -> 全脑汇总 -> 特定脑区专题分析 -> 补充分析`

## 模块划分

| 模块 | 文件 | 主要作用 |
| --- | --- | --- |
| 1. 原始数据统计与区域预处理 | `00_01_Bin100_Data_Summary.ipynb` `00_02_Cellbin_Data_Summary.ipynb` `00_03_Bin100_Data_ChP_Summary.ipynb` `01_01_Bin100_Data_Process.ipynb` `01_02_Bin100_3D_visualization.ipynb` | 统计原始 bin100 / cellbin 数据中每个文件的区域和细胞类型数量；从原始数据中拆出特定区域；将样本级 h5ad 按脑区合并成区域级 `combined_adata_<region>.h5ad`。这是后续 GRN 分析的基础输入层。 |
| 2. Bin100 GRN 主流程与方法实验 | `01_03_Bin100_GRN_Inference.ipynb` `01_03_Bin100_GRN_Inference_02.ipynb` `01_03_Bin100_GRN_Inference_03.ipynb` `01_04_LargeGRN_test.ipynb` `01_05_LargeGRN_function_define.ipynb` `01_06_LargeGRN_new_function_define.ipynb` | 在区域级 bin100 数据上运行 spaGRN / InferNetwork；比较 GPU、CPU、大规模推断和自定义函数实现；生成 GRN 结果目录、临时矩阵、regulon 结果和最终 `.h5ad`。 |
| 3. 全脑 GRN 汇总与可视化 | `02_01_GRN_summary.ipynb` `02_02_GRN_visualization.ipynb` | 汇总多个脑区的 regulon 结果，生成全脑层面的 GRN 统计表、热图、通路富集结果和每个区域的重点 TF/GRN 图。 |
| 4. Cellbin 中脑流程 | `03_01_Cellbin_Midbrain_Summary.ipynb` `03_02_Cellbin_GRNscore_Calculation.ipynb` | 从 cellbin 数据中拆出 midbrain，重建 `combined_adata_midbrain.h5ad`，并根据已有 regulon 结果计算 AUC/GRN score。 |
| 5. 小脑专题 | `04_01_Cellbin_cerebellum.ipynb` `04_02_cerebellum_visualization.ipynb` `04_03_cerebellum_heatmap.ipynb` | 对 cerebellum 做注释增强、AUC 计算、细胞类型特异 TF/GRN 汇总，以及 pathway / GRN 图和热图输出。 |
| 6. 海马专题 | `05_01_Cellbin_hip.ipynb` `05_02_hip_GRN_visualization.ipynb` `05_03_hip_GRN_3D_visualization.ipynb` `05_04_hip_GRN_module.ipynb` `05_05_hip_heatmap.ipynb` | 对 HIP 做增强、AUC 计算、TF/GRN 汇总、2D/3D 可视化和热图分析。`05_04` 更偏向模块探索，代码里没有显式写文件。 |
| 7. MHB 专题 | `06_01_Bin100_MHB_GRN_Inference.ipynb` `06_02_Cellbin_mhb.ipynb` `06_03_mhb_specific_TF_GRN.ipynb` `06_04_mhb_visualization.ipynb` `06_05_mhb_heatmap.ipynb` | 先把 Hindbrain / midbrain / PONS 合并为 MHB，再对 MHB 做 GRN 推断、AUC 计算、特异 TF/GRN 汇总、可视化和热图。 |
| 8. 补充分析 / Todo | `07_01_Todo_list12.ipynb` `07_02_Todo_list3.ipynb` `07_03_Todo_list4.ipynb` `07_04_Todo_list5.ipynb` `07_05_Todo_list6.ipynb` `07_06_SC_AON_GRN.ipynb` | 这一组是补充问题驱动的分析，包括脑区细胞比例图、伪时序 / CytoTRACE2、按 Table S2 跨区域批量绘图、thalamus 与 SC/AON 的专题 GRN 图，以及外部数据对比。`07_05` 主要是比较读取和分析，没有看到显式落盘语句。 |
| 9. 通用工具 | `_3D_plot.py` | 封装了 `plot_gene_3d_k3d`，用于把基因表达画成 3D K3D 图，并可导出 HTML 快照。它是 `05_03` 这类 notebook 的公共工具，而不是独立分析流程。 |

## 每个模块在整体流程中的位置

| 顺序 | 模块 | 在流程中的位置 |
| --- | --- | --- |
| 1 | 原始数据统计与区域预处理 | 把原始样本整理成后续可复用的区域级 h5ad。 |
| 2 | Bin100 GRN 主流程与方法实验 | 产出 regulon、AUC、GRN `.h5ad` 和中间文件，是大多数下游图表的来源。 |
| 3 | 全脑 GRN 汇总与可视化 | 对多个区域的 GRN 结果做横向比较。 |
| 4 | Cellbin 中脑流程 | 把 cellbin midbrain 接到 GRN score 计算。 |
| 5 | 小脑专题 | 对单一区域做更深入的细胞类型和 pathway 解释。 |
| 6 | 海马专题 | 同上，但对象换成 HIP，并增加 3D HTML 可视化。 |
| 7 | MHB 专题 | 同上，但对象换成 MHB。 |
| 8 | 补充分析 / Todo | 用于论文补图、补充统计或延伸问题。 |

## 额外说明

1. `01_03` 到 `06_04` 之间大量代码共享同一个模式：先读 `combined_adata_<region>.h5ad` 或 `auc_<region>.h5ad`，再读 `regulons.json`，最后输出 csv / pdf / h5ad。
2. `Output/GRN_GPU_output/*`、`Output/GRN_output/*`、`Output/MHB/MHB/*` 这几类目录是全套流程的核心中间层，后面很多 notebook 都直接依赖这些目录里的 `regulons.json` 或 GRN `.h5ad`。
3. 这套代码不是一个单入口工程，而是按 notebook 分阶段推进的分析仓库。
