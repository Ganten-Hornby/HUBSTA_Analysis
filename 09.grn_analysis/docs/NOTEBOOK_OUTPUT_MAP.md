# Previous_code 文件与产出对应表

本表把每个 notebook / 脚本对应到它的功能和主要产物，便于按文件追踪。

## 对照表

| 文件 | 所属部分 | 主要作用 | 主要产物 |
| --- | --- | --- | --- |
| `00_01_Bin100_Data_Summary.ipynb` | 原始数据统计 | 统计 bin100 原始 h5ad 的区域数和细胞类型数 | `Output/Data_Summary/h5ad_region_counts_bin100.csv` `Output/Data_Summary/h5ad_celltype_counts_bin100.csv` |
| `00_02_Cellbin_Data_Summary.ipynb` | 原始数据统计 | 统计 cellbin 原始 h5ad 的区域数和细胞类型数 | `Output/Data_Summary/h5ad_region_counts_cellbin.csv` `Output/Data_Summary/h5ad_celltype_counts_cellbin.csv` |
| `00_03_Bin100_Data_ChP_Summary.ipynb` | 原始数据统计 | 从注释表里筛出 ChP 相关区域并合并数据 | `Process_Data/bin100_3D_region/combined_adata_Di_ChP.h5ad` `Process_Data/bin100_3D_region/combined_adata_Hindbrain_ChP.h5ad` |
| `01_01_Bin100_Data_Process.ipynb` | 区域预处理 | 把 bin100 原始样本按区域拆成子 h5ad | `Process_Data/bin100_3D_region/<region>/<sample>.h5ad` |
| `01_02_Bin100_3D_visualization.ipynb` | 区域预处理 | 合并每个区域的样本级 h5ad，并输出统计图 | `Process_Data/bin100_3D_region/combined_adata_<region>.h5ad` `region_spot_counts_vertical_chart.png` |
| `01_03_Bin100_GRN_Inference.ipynb` | Bin100 GRN | 对选定区域运行 spaGRN / InferNetwork | `Output/GRN_GPU_output/<region>/` `Output/GRN_output/<region>/` `Output/GRN_output/test/` |
| `01_03_Bin100_GRN_Inference_02.ipynb` | Bin100 GRN | 对另一组区域运行 GPU GRN 推断 | `Output/GRN_GPU_output/<region>/` |
| `01_03_Bin100_GRN_Inference_03.ipynb` | Bin100 GRN | 对另一组区域运行 GPU GRN 推断 | `Output/GRN_GPU_output/<region>/` |
| `01_04_LargeGRN_test.ipynb` | GRN 方法实验 | 测试大规模 GRN 推断的 GPU / CPU 版本 | `Output/GRN_output/test_large_gpu/` `Output/GRN_output/test_large/` |
| `01_05_LargeGRN_function_define.ipynb` | GRN 方法实验 | 自定义大规模 GRN 函数，显式落盘各阶段结果 | `Output/GRN_output/test_large_function/split_infer_region_spagrn.h5ad` `tmp_files/geary_adj.csv` `tmp_files/motifs.csv` `tmp_files/auc_mtx.csv` `tmp_files/regulon_specificity_scores.txt` |
| `01_06_LargeGRN_new_function_define.ipynb` | GRN 方法实验 | 加速版自定义 GRN 函数，并比较快慢版本输出 | `Output/GRN_output/test_large_function_fast/split_infer_region_spagrn.h5ad` `tmp_files/geary_adj.csv` `tmp_files/more_stats.csv` `tmp_files/motifs.csv` `tmp_files/auc_mtx.csv` `tmp_files/regulon_specificity_scores.txt` |
| `02_01_GRN_summary.ipynb` | 全脑汇总 | 汇总多区域 GRN 结果并做热图 | `Output/GRN_GPU_output/GRN_summary.csv` `Figure/whole_brain_selected_GRN/region_GRN_heatmap.pdf` |
| `02_02_GRN_visualization.ipynb` | 全脑汇总 | 逐区域导出 selected GRN、pathway 结果和网络图 | `Output/whole_brain_selected_GRN/*.csv` `Figure/whole_brain_selected_GRN/Pathway/*.pdf` `Figure/whole_brain_selected_GRN/GRN/*.pdf` |
| `03_01_Cellbin_Midbrain_Summary.ipynb` | Midbrain | 从 cellbin 中拆 midbrain 并合并为区域级 h5ad | `Process_Data/cellbin_3D_region/midbrain/<sample>.h5ad` `Process_Data/cellbin_3D_region/combined_adata_midbrain.h5ad` |
| `03_02_Cellbin_GRNscore_Calculation.ipynb` | Midbrain | 根据 regulons 计算 midbrain 的 AUC / GRN score | `Process_Data/cellbin_3D_region/auc_adata/auc_anndata_group_<group>.h5ad` `Process_Data/cellbin_3D_region/auc_mtx.h5ad` |
| `04_01_Cellbin_cerebellum.ipynb` | Cerebellum | 做小脑增强、AUC、特异 TF / GRN 表 | `Process_Data/specific_region/enhanced_cerebellum.h5ad` `Process_Data/specific_region/auc_cerebellum.h5ad` `Output/Cere/celltype_specific_tf.csv` `Output/Cere/celltype_specific_grn.csv` |
| `04_02_cerebellum_visualization.ipynb` | Cerebellum | 输出多个小脑 cluster 的 pathway / GRN 图 | `Figure/Cere/Cere_sc2*` `Figure/Cere/Cere_sc17*` `Figure/Cere/Cere_sc18*` `Figure/Cere/Cere_sc19*` `Figure/Cere/Cere_sc20*` `Figure/Cere/Cere_sc21*` |
| `04_03_cerebellum_heatmap.ipynb` | Cerebellum | 输出小脑 TF / GRN 热图 | `Figure/heatmap/cerebellum_tf.pdf` `Figure/heatmap/cerebellum_grn.pdf` |
| `05_01_Cellbin_hip.ipynb` | HIP | 做 HIP 增强、AUC、特异 TF / GRN 表 | `Process_Data/specific_region/enhanced_hip.h5ad` `Process_Data/specific_region/auc_hip.h5ad` `Output/HIP/celltype_specific_tf.csv` `Output/HIP/celltype_specific_grn.csv` |
| `05_02_hip_GRN_visualization.ipynb` | HIP | 输出多个 HIP cluster 的 pathway / GRN 图 | `Figure/HIP/HIP_sc18*` `Figure/HIP/HIP_sc20*` `Figure/HIP/HIP_sc21*` `Figure/HIP/HIP_sc22*` `Figure/HIP/HIP_sc23*` |
| `05_03_hip_GRN_3D_visualization.ipynb` | HIP | 调用 `_3D_plot.py` 输出 3D HTML | `Output/HIP/3D_GRN_THRA.html` |
| `05_04_hip_GRN_module.ipynb` | HIP | 模块层面的探索分析 | 代码中没有显式写出新文件 |
| `05_05_hip_heatmap.ipynb` | HIP | 输出 HIP TF / 子群 TF / GRN 热图 | `Figure/heatmap/hip_tf.pdf` `Figure/heatmap/hip_tf_subcluster.pdf` `Figure/heatmap/hip_grn.pdf` |
| `06_01_Bin100_MHB_GRN_Inference.ipynb` | MHB | 生成 MHB 合并数据并启动 MHB 的 GRN 推断 | `Process_Data/bin100_3D_region/combined_adata_MHB.h5ad` `Output/MHB/MHB/` |
| `06_02_Cellbin_mhb.ipynb` | MHB | 做 MHB 增强、邻接计算和 AUC 输出 | `Process_Data/specific_region/enhanced_mhb.h5ad` `Process_Data/specific_region/mhb_tmp/*` `Process_Data/specific_region/auc_mhb.h5ad` |
| `06_03_mhb_specific_TF_GRN.ipynb` | MHB | 输出 MHB 特异 TF / GRN 表 | `Output/MHB/celltype_specific_tf.csv` `Output/MHB/celltype_specific_grn.csv` |
| `06_04_mhb_visualization.ipynb` | MHB | 输出 MHB cluster 的 pathway / GRN 图 | `Figure/MHB/MHB_sc17*` `Figure/MHB/MHB_sc21*` |
| `06_05_mhb_heatmap.ipynb` | MHB | 输出 MHB TF / GRN 热图 | `Figure/heatmap/mhb_tf.pdf` `Figure/heatmap/mhb_grn.pdf` |
| `07_01_Todo_list12.ipynb` | Todo | 细胞比例图、伪时序、CytoTRACE2 图和结果 | `Figure/To_do_list_1/brain_region_celltype_fraction.pdf` `Process_Data/to_do_list_2/cytotrace2_results/cytotrace2_results.txt` 以及多个 `To_do_list_2/*.pdf` |
| `07_02_Todo_list3.ipynb` | Todo | 生成 marker 图和跨区域表达比例矩阵 | `Process_Data/to_do_list_3/*.csv` `Figure/To_do_list_3/*.png` 以及 `To_do_list_3/*.pdf` |
| `07_03_Todo_list4.ipynb` | Todo | thalamus 的 pathway / GRN 图 | `Figure/To_do_list_4/thalamus_selected_GRN_pathway.pdf` `Figure/To_do_list_4/thalamus_GRN.pdf` |
| `07_04_Todo_list5.ipynb` | Todo | 按 Table S2 和区域映射批量输出 GRN 图 | `Figure/To_do_list_5/<region>/<region>__selected_GRN_pathway.pdf` `Figure/To_do_list_5/<region>/<region>__GRN.pdf` |
| `07_05_Todo_list6.ipynb` | Todo | 读取外部数据集并做比较分析 | 代码中没有显式写出新文件 |
| `07_06_SC_AON_GRN.ipynb` | Todo | 输出 SC/AON 的 GO 富集和 GRN 图 | `Figure/SC_AON_GRN/GO_term.csv` `Figure/SC_AON_GRN/SC_AON_selected_GRN_pathway.pdf` `Figure/SC_AON_GRN/SC_AON_GRN.pdf` |
| `_3D_plot.py` | 通用工具 | 3D K3D 绘图与 HTML 导出 | 任意 `save_path` 对应的 HTML 快照 |

## 额外备注

1. `01_03` 系列虽然没有把所有输出文件名逐个写死，但下游 notebook 反复读取 `tmp_files/regulons.json` 和 `grn_<region>_spagrn.h5ad`，说明这些是事实上的标准中间产物。
2. `02_02`、`04_02`、`05_02`、`06_04`、`07_03`、`07_04`、`07_06` 这几类 notebook 本质上都在做同一种事：从 `regulons.json` 里挑 TF，做 pathway 和 GRN 的图。
3. `05_04` 和 `07_05` 更像探索型 notebook，读入很多数据，但代码中没有看到明确的写文件语句。
