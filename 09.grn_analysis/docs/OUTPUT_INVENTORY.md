# Previous_code 产出文件清单

本文件只根据代码中的 `write_h5ad`、`to_csv`、`savefig`、`np.savetxt`、`save_path=`、`grn.infer(output_dir=...)` 等静态线索整理，不代表这些文件当前一定都已实际存在。

## 1. 数据统计与基础预处理产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `Output/Data_Summary/h5ad_region_counts_bin100.csv` | `00_01_Bin100_Data_Summary.ipynb` | CSV | 统计每个 bin100 h5ad 文件里的区域数量。 |
| `Output/Data_Summary/h5ad_celltype_counts_bin100.csv` | `00_01_Bin100_Data_Summary.ipynb` | CSV | 统计每个 bin100 h5ad 文件里的细胞类型数量。 |
| `Output/Data_Summary/h5ad_region_counts_cellbin.csv` | `00_02_Cellbin_Data_Summary.ipynb` | CSV | 统计每个 cellbin h5ad 文件里的区域数量。 |
| `Output/Data_Summary/h5ad_celltype_counts_cellbin.csv` | `00_02_Cellbin_Data_Summary.ipynb` | CSV | 统计每个 cellbin h5ad 文件里的细胞类型数量。 |
| `Process_Data/bin100_3D_region/combined_adata_Di_ChP.h5ad` | `00_03_Bin100_Data_ChP_Summary.ipynb` | H5AD | 合并得到 Di_ChP 的区域级 AnnData。 |
| `Process_Data/bin100_3D_region/combined_adata_Hindbrain_ChP.h5ad` | `00_03_Bin100_Data_ChP_Summary.ipynb` | H5AD | 合并得到 Hindbrain_ChP 的区域级 AnnData。 |
| `Process_Data/bin100_3D_region/<region>/<sample>.h5ad` | `01_01_Bin100_Data_Process.ipynb` | H5AD | 把原始 bin100 数据按脑区拆分后的样本级文件。 |
| `Process_Data/bin100_3D_region/combined_adata_<region>.h5ad` | `01_02_Bin100_3D_visualization.ipynb` | H5AD | 将同一区域下的多个样本重新合并为区域级 AnnData。 |
| `region_spot_counts_vertical_chart.png` | `01_02_Bin100_3D_visualization.ipynb` | PNG | 区域 spot 数量的柱状图。 |
| `Process_Data/bin100_3D_region/combined_adata_MHB.h5ad` | `06_01_Bin100_MHB_GRN_Inference.ipynb` | H5AD | 把 Hindbrain、midbrain、PONS 合并出的 MHB 数据。 |

## 2. GRN 推断主目录与中间文件

这一类输出最重要。代码里既有显式文件名，也有通过 `output_dir` 交给 spaGRN / InferNetwork 自动写出的目录。

| 输出路径模式 | 来源文件 | 文件类型 | 作用 | 说明 |
| --- | --- | --- | --- | --- |
| `Output/GRN_GPU_output/<region>/` | `01_03_Bin100_GRN_Inference.ipynb` `01_03_Bin100_GRN_Inference_02.ipynb` `01_03_Bin100_GRN_Inference_03.ipynb` | 目录 | 每个区域的 GPU GRN 推断结果目录。 | 代码里显式传入 `output_dir=output_dir_region`。 |
| `Output/GRN_output/<region>/` | `01_03_Bin100_GRN_Inference.ipynb` | 目录 | 每个区域的 CPU 或非 GPU GRN 推断结果目录。 | 同上。 |
| `Output/MHB/MHB/` | `06_01_Bin100_MHB_GRN_Inference.ipynb` | 目录 | MHB 区域的 GRN 推断结果目录。 | 下游代码显式读取 `Output/MHB/MHB/tmp_files/regulons.json`。 |
| `Output/GRN_GPU_output/<region>/grn_<region>_spagrn.h5ad` | `01_03` 系列，下游由 `03_01` 等文件读取 | H5AD | 某个区域的最终 GRN AnnData。 | 这是根据 `03_01` 读取 `midbrain/grn_midbrain_spagrn.h5ad` 反推得到的命名模式。 |
| `Output/GRN_output/test_large_gpu/` | `01_04_LargeGRN_test.ipynb` | 目录 | 大规模 GRN 测试的 GPU 输出目录。 | 目录内容未在 notebook 中逐个列出。 |
| `Output/GRN_output/test_large/` | `01_04_LargeGRN_test.ipynb` | 目录 | 大规模 GRN 测试的 CPU 输出目录。 | 同上。 |
| `Output/GRN_output/test_large_function/split_infer_region_spagrn.h5ad` | `01_05_LargeGRN_function_define.ipynb` | H5AD | 自定义大规模 GRN 函数的最终结果。 | 文件名在代码中显式给出。 |
| `Output/GRN_output/test_large_function_fast/split_infer_region_spagrn.h5ad` | `01_06_LargeGRN_new_function_define.ipynb` | H5AD | 加速版自定义大规模 GRN 函数的最终结果。 | 文件名在代码中显式给出。 |
| `Output/GRN_output/test_large_function/tmp_files/geary_adj.csv` | `01_05_LargeGRN_function_define.ipynb` | CSV | GRN 推断阶段的邻接/局部相关矩阵。 | 由 `fn=os.path.join(tmp_dir, f'{mode}_adj.csv')` 推出。 |
| `Output/GRN_output/test_large_function/tmp_files/motifs.csv` | `01_05_LargeGRN_function_define.ipynb` | CSV | motif / cisTarget 剪枝结果。 | 显式传给 `prune_modules(... fn=...)`。 |
| `Output/GRN_output/test_large_function/tmp_files/auc_mtx.csv` | `01_05_LargeGRN_function_define.ipynb` | CSV | AUCell 结果矩阵。 | 显式传给 `cal_auc(... fn=...)`。 |
| `Output/GRN_output/test_large_function/tmp_files/regulon_specificity_scores.txt` | `01_05_LargeGRN_function_define.ipynb` | TXT | regulon specificity score。 | 显式传给 `cal_regulon_score(... fn=...)`。 |
| `Output/GRN_output/test_large_function_fast/tmp_files/geary_adj.csv` | `01_06_LargeGRN_new_function_define.ipynb` | CSV | 加速版 GRN 邻接/局部相关矩阵。 | 与上一条同模式。 |
| `Output/GRN_output/test_large_function_fast/tmp_files/more_stats.csv` | `01_06_LargeGRN_new_function_define.ipynb` | CSV | 空间统计补充结果。 | 在同 notebook 中被再次读取。 |
| `Output/GRN_output/test_large_function_fast/tmp_files/motifs.csv` | `01_06_LargeGRN_new_function_define.ipynb` | CSV | 加速版 motif 剪枝结果。 | 显式路径。 |
| `Output/GRN_output/test_large_function_fast/tmp_files/auc_mtx.csv` | `01_06_LargeGRN_new_function_define.ipynb` | CSV | 加速版 AUCell 结果。 | 显式路径。 |
| `Output/GRN_output/test_large_function_fast/tmp_files/regulon_specificity_scores.txt` | `01_06_LargeGRN_new_function_define.ipynb` | TXT | 加速版 RSS。 | 显式路径。 |
| `Output/GRN_GPU_output/<region>/tmp_files/regulons.json` | `02_02` `03_02` `04_01` `05_01` `07_03` `07_04` `07_06` 等下游 notebook | JSON | 下游几乎所有可视化和 TF/GRN 提取的核心输入。 | 代码反复读取该文件，因此可以确定 GRN 推断阶段会产出它。 |

## 3. 全脑 GRN 汇总产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `Output/GRN_GPU_output/GRN_summary.csv` | `02_01_GRN_summary.ipynb` | CSV | 汇总各区域 regulon/GRN 统计结果。 |
| `Figure/whole_brain_selected_GRN/region_GRN_heatmap.pdf` | `02_01_GRN_summary.ipynb` | PDF | 区域层面的 GRN 热图。 |
| `Output/whole_brain_selected_GRN/<region>_selected_GRN.csv` | `02_02_GRN_visualization.ipynb` | CSV | 每个区域筛出的重点 TF / regulon 列表。 |
| `Output/whole_brain_selected_GRN/0_<region>_selected_GRN_pathway.csv` | `02_02_GRN_visualization.ipynb` | CSV | 每个区域的 pathway 富集结果。 |
| `Figure/whole_brain_selected_GRN/Pathway/<region>_pathway.pdf` | `02_02_GRN_visualization.ipynb` | PDF | 每个区域的 pathway 网络图。 |
| `Figure/whole_brain_selected_GRN/GRN/<region>_GRN.pdf` | `02_02_GRN_visualization.ipynb` | PDF | 每个区域的 GRN 网络图。 |

`02_02` 代码里明确处理了这些区域：

- `midbrain`
- `HIP`
- `VZ`
- `SVZ`
- `subthalamic_nucleus`
- `CP_SP`
- `thalamus`
- `Di_ChP`
- `Hindbrain_ChP`

## 4. Cellbin midbrain 产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `Process_Data/cellbin_3D_region/midbrain/<sample>.h5ad` | `03_01_Cellbin_Midbrain_Summary.ipynb` | H5AD | 从 cellbin 中拆出的 midbrain 样本级数据。 |
| `Process_Data/cellbin_3D_region/combined_adata_midbrain.h5ad` | `03_01_Cellbin_Midbrain_Summary.ipynb` | H5AD | 合并后的 midbrain 区域级数据。 |
| `Process_Data/cellbin_3D_region/auc_adata/auc_anndata_group_<group>.h5ad` | `03_02_Cellbin_GRNscore_Calculation.ipynb` | H5AD | 每个 group 的 AUC 子结果。 |
| `Process_Data/cellbin_3D_region/auc_mtx.h5ad` | `03_02_Cellbin_GRNscore_Calculation.ipynb` | H5AD | midbrain 的 AUC 矩阵。 |

## 5. Cerebellum 专题产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `Process_Data/specific_region/enhanced_cerebellum.h5ad` | `04_01_Cellbin_cerebellum.ipynb` | H5AD | 注释增强后的小脑数据。 |
| `Process_Data/specific_region/cerebellum_tmp/enhanced_group_<suffix>.h5ad` | `04_01_Cellbin_cerebellum.ipynb` | H5AD | 按 group 拆开的增强小脑数据。 |
| `Process_Data/specific_region/auc_cerebellum_adata/auc_anndata_group_<group>.h5ad` | `04_01_Cellbin_cerebellum.ipynb` | H5AD | 小脑每个 group 的 AUC 结果。 |
| `Process_Data/specific_region/auc_cerebellum.h5ad` | `04_01_Cellbin_cerebellum.ipynb` | H5AD | 小脑整合 AUC 矩阵。 |
| `Output/Cere/celltype_specific_tf.csv` | `04_01_Cellbin_cerebellum.ipynb` | CSV | 小脑细胞类型特异 TF 表。 |
| `Output/Cere/celltype_specific_grn.csv` | `04_01_Cellbin_cerebellum.ipynb` | CSV | 小脑细胞类型特异 GRN 表。 |
| `Figure/Cere/Cere_sc2_selected_GRN_pathway.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc2 的 pathway 图。 |
| `Figure/Cere/Cere_sc2_GRN.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc2 的 GRN 图。 |
| `Figure/Cere/Cere_sc17_selected_GRN_pathway.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc17 的 pathway 图。 |
| `Figure/Cere/Cere_sc17_GRN.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc17 的 GRN 图。 |
| `Figure/Cere/Cere_sc18_selected_GRN_pathway.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc18 的 pathway 图。 |
| `Figure/Cere/Cere_sc18_GRN.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc18 的 GRN 图。 |
| `Figure/Cere/Cere_sc19_selected_GRN_pathway.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc19 的 pathway 图。 |
| `Figure/Cere/Cere_sc19_GRN.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc19 的 GRN 图。 |
| `Figure/Cere/Cere_sc20_selected_GRN_pathway.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc20 的 pathway 图。 |
| `Figure/Cere/Cere_sc20_GRN.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc20 的 GRN 图。 |
| `Figure/Cere/Cere_sc21_selected_GRN_pathway.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc21 的 pathway 图。 |
| `Figure/Cere/Cere_sc21_GRN.pdf` | `04_02_cerebellum_visualization.ipynb` | PDF | 小脑 sc21 的 GRN 图。 |
| `Figure/heatmap/cerebellum_tf.pdf` | `04_03_cerebellum_heatmap.ipynb` | PDF | 小脑 TF 热图。 |
| `Figure/heatmap/cerebellum_grn.pdf` | `04_03_cerebellum_heatmap.ipynb` | PDF | 小脑 GRN 热图。 |

## 6. HIP 专题产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `Process_Data/specific_region/enhanced_hip.h5ad` | `05_01_Cellbin_hip.ipynb` | H5AD | 注释增强后的 HIP 数据。 |
| `Process_Data/specific_region/auc_hip_adata/auc_anndata_group_<group>.h5ad` | `05_01_Cellbin_hip.ipynb` | H5AD | HIP 每个 group 的 AUC 结果。 |
| `Process_Data/specific_region/auc_hip.h5ad` | `05_01_Cellbin_hip.ipynb` | H5AD | HIP 整合 AUC 矩阵。 |
| `Output/HIP/celltype_specific_tf.csv` | `05_01_Cellbin_hip.ipynb` | CSV | HIP 细胞类型特异 TF 表。 |
| `Output/HIP/celltype_specific_grn.csv` | `05_01_Cellbin_hip.ipynb` | CSV | HIP 细胞类型特异 GRN 表。 |
| `Figure/HIP/HIP_sc18_selected_GRN_pathway.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc18 的 pathway 图。 |
| `Figure/HIP/HIP_sc18_GRN.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc18 的 GRN 图。 |
| `Figure/HIP/HIP_sc20_selected_GRN_pathway.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc20 的 pathway 图。 |
| `Figure/HIP/HIP_sc20_GRN.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc20 的 GRN 图。 |
| `Figure/HIP/HIP_sc21_selected_GRN_pathway.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc21 的 pathway 图。 |
| `Figure/HIP/HIP_sc21_GRN.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc21 的 GRN 图。 |
| `Figure/HIP/HIP_sc22_selected_GRN_pathway.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc22 的 pathway 图。 |
| `Figure/HIP/HIP_sc22_GRN.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc22 的 GRN 图。 |
| `Figure/HIP/HIP_sc23_selected_GRN_pathway.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc23 的 pathway 图。 |
| `Figure/HIP/HIP_sc23_GRN.pdf` | `05_02_hip_GRN_visualization.ipynb` | PDF | HIP sc23 的 GRN 图。 |
| `Output/HIP/3D_GRN_THRA.html` | `05_03_hip_GRN_3D_visualization.ipynb` | HTML | HIP 中 THRA 的 3D K3D 可视化快照。 |
| `Figure/heatmap/hip_tf.pdf` | `05_05_hip_heatmap.ipynb` | PDF | HIP TF 热图。 |
| `Figure/heatmap/hip_tf_subcluster.pdf` | `05_05_hip_heatmap.ipynb` | PDF | HIP 子群 TF 热图。 |
| `Figure/heatmap/hip_grn.pdf` | `05_05_hip_heatmap.ipynb` | PDF | HIP GRN 热图。 |

说明：`05_02` 中 `HIP_sc18` 的 `save_prefix` 被使用了两次，因此第二次调用有覆盖第一次同名输出的可能。

## 7. MHB 专题产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `Process_Data/specific_region/enhanced_mhb.h5ad` | `06_02_Cellbin_mhb.ipynb` | H5AD | 注释增强后的 MHB 数据。 |
| `Process_Data/specific_region/mhb_tmp/raw_mh_sc_02.h5ad` | `06_02_Cellbin_mhb.ipynb` | H5AD | MHB 中间原始子集。 |
| `Process_Data/specific_region/mhb_tmp/all_neighbor_indices.csv` | `06_02_Cellbin_mhb.ipynb` | CSV | MHB 邻接索引中间结果。 |
| `Process_Data/specific_region/mhb_tmp/enhanced_group_mh_sc_02.h5ad` | `06_02_Cellbin_mhb.ipynb` | H5AD | 增强后的 MHB group 文件。 |
| `Process_Data/specific_region/auc_mhb_adata/<processed>.h5ad` | `06_02_Cellbin_mhb.ipynb` | H5AD | MHB 按 group 输出的 AUC 文件。 |
| `Process_Data/specific_region/auc_mhb.h5ad` | `06_02_Cellbin_mhb.ipynb` | H5AD | MHB 整合 AUC 矩阵。 |
| `Output/MHB/celltype_specific_tf.csv` | `06_03_mhb_specific_TF_GRN.ipynb` | CSV | MHB 细胞类型特异 TF 表。 |
| `Output/MHB/celltype_specific_grn.csv` | `06_03_mhb_specific_TF_GRN.ipynb` | CSV | MHB 细胞类型特异 GRN 表。 |
| `Figure/MHB/MHB_sc17_selected_GRN_pathway.pdf` | `06_04_mhb_visualization.ipynb` | PDF | MHB sc17 的 pathway 图。 |
| `Figure/MHB/MHB_sc17_GRN.pdf` | `06_04_mhb_visualization.ipynb` | PDF | MHB sc17 的 GRN 图。 |
| `Figure/MHB/MHB_sc21_selected_GRN_pathway.pdf` | `06_04_mhb_visualization.ipynb` | PDF | MHB sc21 的 pathway 图。 |
| `Figure/MHB/MHB_sc21_GRN.pdf` | `06_04_mhb_visualization.ipynb` | PDF | MHB sc21 的 GRN 图。 |
| `Figure/heatmap/mhb_tf.pdf` | `06_05_mhb_heatmap.ipynb` | PDF | MHB TF 热图。 |
| `Figure/heatmap/mhb_grn.pdf` | `06_05_mhb_heatmap.ipynb` | PDF | MHB GRN 热图。 |

## 8. Todo / 补充分析产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `Figure/To_do_list_1/brain_region_celltype_fraction.pdf` | `07_01_Todo_list12.ipynb` | PDF | 脑区细胞类型比例图。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_2/celltype_umap.pdf` | `07_01_Todo_list12.ipynb` | PDF | UMAP 图。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_2/dpt_pseudotime.pdf` | `07_01_Todo_list12.ipynb` | PDF | DPT 伪时序图。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_2/dpt_pseudotime_violin.pdf` | `07_01_Todo_list12.ipynb` | PDF | DPT 伪时序小提琴图。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_2/dotplot_marker.pdf` | `07_01_Todo_list12.ipynb` | PDF | marker dotplot。 |
| `Process_Data/to_do_list_2/cytotrace2_results/cytotrace2_results.txt` | `07_01_Todo_list12.ipynb` | TXT | CytoTRACE2 输出分数表。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_2/CytoTRACE2_score.pdf` | `07_01_Todo_list12.ipynb` | PDF | CytoTRACE2 分数图。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_2/CytoTRACE2_Score_violin.pdf` | `07_01_Todo_list12.ipynb` | PDF | CytoTRACE2 分数小提琴图。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_3/umap_markers.pdf` | `07_02_Todo_list3.ipynb` | PDF | marker UMAP 图。 |
| `/home/dataset-assist-0/yaosen/lihan/hulei/Analysis/fetal_brain/Figure/To_do_list_3/umap_markers_20251219.pdf` | `07_02_Todo_list3.ipynb` | PDF | 另一版 marker UMAP 图。 |
| `Process_Data/to_do_list_3/Matrix_Total_Spots_per_Region.csv` | `07_02_Todo_list3.ipynb` | CSV | 各区域总 spots 数矩阵。 |
| `Process_Data/to_do_list_3/Matrix_Gene_Counts_per_Region.csv` | `07_02_Todo_list3.ipynb` | CSV | 各区域基因表达计数矩阵。 |
| `Process_Data/to_do_list_3/Matrix_Gene_Proportions_per_Region.csv` | `07_02_Todo_list3.ipynb` | CSV | 各区域基因表达比例矩阵。 |
| `Figure/To_do_list_3/<gene>_proportion_comparison.png` | `07_02_Todo_list3.ipynb` | PNG | 单个基因在不同区域的比例比较图。 |
| `Figure/To_do_list_4/thalamus_selected_GRN_pathway.pdf` | `07_03_Todo_list4.ipynb` | PDF | thalamus pathway 图。 |
| `Figure/To_do_list_4/thalamus_GRN.pdf` | `07_03_Todo_list4.ipynb` | PDF | thalamus GRN 图。 |
| `Figure/To_do_list_5/<region>/<region>__selected_GRN_pathway.pdf` | `07_04_Todo_list5.ipynb` | PDF | 按 Table S2 批量生成的区域 pathway 图。 |
| `Figure/To_do_list_5/<region>/<region>__GRN.pdf` | `07_04_Todo_list5.ipynb` | PDF | 按 Table S2 批量生成的区域 GRN 图。 |
| `Figure/SC_AON_GRN/GO_term.csv` | `07_06_SC_AON_GRN.ipynb` | CSV | SC/AON 的 GO 富集表。 |
| `Figure/SC_AON_GRN/SC_AON_selected_GRN_pathway.pdf` | `07_06_SC_AON_GRN.ipynb` | PDF | SC/AON pathway 图。 |
| `Figure/SC_AON_GRN/SC_AON_GRN.pdf` | `07_06_SC_AON_GRN.ipynb` | PDF | SC/AON GRN 图。 |

说明：

1. `07_04` 的 `save_prefix` 本身以 `_` 结尾，而函数内部又拼接了 `_selected_GRN_pathway.pdf` 和 `_GRN.pdf`，所以按代码字面推断文件名中会出现双下划线。
2. `07_05_Todo_list6.ipynb` 只看到了读文件和分析过程，没有明确写出新文件。

## 9. 通用工具 `_3D_plot.py` 的产物

| 输出路径 | 来源文件 | 文件类型 | 作用 |
| --- | --- | --- | --- |
| `<save_path>.html` | `_3D_plot.py` | HTML | `plot_gene_3d_k3d` 可以把 3D K3D 图快照保存为 HTML。 |
| `/data/work/gfap_plot_v_function.html` | `_3D_plot.py` 中的注释示例 | HTML | 示例输出路径，不是默认自动产物。 |

## 10. 最值得关注的核心产物

如果只看最核心的分析结果，优先关注这些目录：

- `Process_Data/bin100_3D_region/combined_adata_<region>.h5ad`
- `Output/GRN_GPU_output/<region>/`
- `Output/whole_brain_selected_GRN/`
- `Process_Data/specific_region/auc_<region>.h5ad`
- `Output/Cere/` `Output/HIP/` `Output/MHB/`
- `Figure/whole_brain_selected_GRN/` `Figure/Cere/` `Figure/HIP/` `Figure/MHB/`
