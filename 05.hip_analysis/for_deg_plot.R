library(presto)
library(sceasy)
library(reticulate)
library(Seurat)
library(tidyverse)

sceasy::convertFormat('/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sp.h5ad', 
                      from="anndata", 
                      to="seurat",
                      outFile='/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sp.rds')
rds = readRDS('/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sp.rds')
rds = NormalizeData(rds)
csv = wilcoxauc(rds, 'dmt_leiden_merge')
write.table(csv, file = "/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sp.csv", sep = ",", row.names = FALSE)



sceasy::convertFormat('/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sc.h5ad', 
                      from="anndata", 
                      to="seurat",
                      outFile='/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sc.rds')
rds = readRDS('/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sc.rds')
rds = NormalizeData(rds)
csv = wilcoxauc(rds, 'dmt_leiden_anno')
write.table(csv, file = "/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sc.csv", sep = ",", row.names = FALSE)