library(presto)
library(sceasy)
library(reticulate)
library(Seurat)
library(tidyverse)

sceasy::convertFormat('/data/work/05.cluster/FuseMap/0117/tha_deg_plot/tha_sp.h5ad', 
                      from="anndata", 
                      to="seurat",
                      outFile='/data/work/05.cluster/FuseMap/0117/tha_deg_plot/tha_sp.rds')
rds = readRDS('/data/work/05.cluster/FuseMap/0117/tha_deg_plot/tha_sp.rds')
rds = NormalizeData(rds)
csv = wilcoxauc(rds, 'dmt_leiden_annotation_0115')
write.table(csv, file = "/data/work/05.cluster/FuseMap/0117/tha_deg_plot/tha_sp.csv", sep = ",", row.names = FALSE)