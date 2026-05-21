library(presto)
library(sceasy)
library(reticulate)
library(Seurat)
library(tidyverse)

sceasy::convertFormat('/data/work/05.cluster/FuseMap/0313/mh_deg_plot/sc/hm_sc.h5ad', 
                      from="anndata", 
                      to="seurat",
                      outFile='/data/work/05.cluster/FuseMap/0313/mh_deg_plot/sc/hm_sc.rds')
rds = readRDS('/data/work/05.cluster/FuseMap/0313/mh_deg_plot/sc/hm_sc.rds')
rds = NormalizeData(rds)
csv = wilcoxauc(rds, 'dmt_leiden_merge')
write.table(csv, file = "/data/work/05.cluster/FuseMap/0313/mh_deg_plot/sc/hm_sc.csv", sep = ",", row.names = FALSE)

