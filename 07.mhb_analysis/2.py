import scanpy as sc
import anndata as ad
# import rapids_singlecell as rsc
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, rgb2hex
from collections import defaultdict

import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib.pyplot as plt

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/mid_hind_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()

adata = adata[:, ~adata.var_names.str.startswith('MT')]
adata = adata[:,[False if '.' in i else True for i in adata.var.index.tolist()]]

adatas = []
for i in set(adata.obs['slice_code']):
    temp = adata[adata.obs['slice_code'] == i].copy()
    sc.pp.normalize_total(temp)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    adatas.append(temp)
adata = ad.concat(adatas)



var = [
    'NKX2-4', 'NKX2-8', 'NKX6-2','CRH',
'HBG2', 'HBA2', 'HBG1', 'HBA1', 'OTX2',
 'PMCH', 'ALDH1A1', 'PITX2', 'NTS', 'TH',
'CABP2',
 'TTR', 'PCDH9',
 'TFAP2B', 'CCK',
 'TCF7L2', 'PTH2', 'GPR151', 'KITLG', 'LHX9',
 'CRABP1', 'NFIB', 'LMO4', 'MAP1B',
 'HOXB5', 'HOXA5', 'HOXB8', 'HOXB6', 'MAFA',
 'CST3', 'C1orf61',
 'NPY', 'LHX9', 'TFAP2D',
 'LAMP5', 'PEG10', 'MEIS2',
 'SPP1', 'FTL', 'MAGEA1', 'LINC00293', 'APOE',
 'OTP', 'LHX5-AS1', 'GATA3',
 'NEFL', 'NEFM', 'CARTPT', 'PRPH',
'PAX5', 'TPH2', 'SLC6A4', 'FEV', 'DBH',
 'IGFBP7', 'CLDN5', 'FN1', 'ITM2A',
 'SST', 'PHOX2B', 'CBLN1', 'LINC00682']


sc.pl.matrixplot(adata, 
                 var, 
                 'dmt_leiden_merge', 
                 standard_scale = 'var',
                 # dendrogram = True,
                 show = False
                )
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_spatial/matrixplot.pdf', bbox_inches = 'tight', dpi = 600)
plt.close()

sc.pl.dotplot(adata, 
                 var, 
                 'dmt_leiden_merge', 
                 standard_scale = 'var',
              show = False
                )
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_spatial/dotplot.pdf', bbox_inches = 'tight', dpi = 600)
plt.close()

colormap = {'mh_12': '#244320',
 'mh_8': '#a374bd',
 'mh_16': '#a4f2d4',
 'mh_2': '#db8c2c',
 'mh_14': '#d359aa',
 'mh_9': '#5f31de',
 'mh_1': '#ae83b8',
 'mh_4': '#26f77e',
 'mh_18': '#f5ee2f',
 'mh_11': '#dbb1c9',
 'mh_17': '#0cd53a',
 'mh_10': '#1607d3',
 'mh_3': '#b58c76',
 'mh_15': '#2c8fb2',
 'mh_6': '#727d30',
 'mh_7': '#3bbc7d',
 'mh_5': '#68754f',
 'mh_0': '#c9cd19'}

plot = sc.pl.embedding(adata, basis = 'X_dmt',color = 'dmt_leiden_merge', palette = colormap, show = False, 
                       title = 'Midbrain&Hindbrain SpatialRegion Annotation'); 
plot.set_aspect('equal')
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_spatial/dmt_plot.png', bbox_inches = 'tight', dpi = 600)
plt.close()


names = [
    '20_B03606F3G5_WT202405020032.h5ad',
 '22_B03606C4E6_WT202403310050.h5ad',
 '23_B03609A4D6_WT202404150263.h5ad',
 '27_B03610C1E3_WT202403310051.h5ad',
 '31_B03619A1D3_WT202403310052.h5ad',
 '35_B03619E4G6_WT202403310053.h5ad',
 '39_A03589A1D4_WT202403310046.h5ad',
 '43_A03590E1G4_WT202403310064.h5ad',
 '47_A03593C1F3_WT202403310068.h5ad',
 '51_B03605C2E5_WT202406020126.h5ad',
 '55_B03613E3G6_WT202403310069.h5ad',
 '59_B03612E4G6_WT202403310059.h5ad',
 '63_B03606C1E3_WT202403310061.h5ad',
 '67_A03595A1D3_WT202403310062.h5ad',
 '71_A03595A4D6_WT202403310063.h5ad',
    '75_D03468D1E3_WT202403310066.h5ad',
    '80_D03473D4E6_WT202403310070.h5ad',
    '84_B03423D1E3_WT202403310065.h5ad',
'A03587A5C6_WT2024071215080.h5ad',
'A03988A1C2_WT202407161208.h5ad',
'Y00547PC_WT202407282759.h5ad',
# 'A03591D4E5_WT2024071215074.h5ad',
'A03590A3D6_WT202407192652.h5ad',
'B03618D3F6_WT202407152793.h5ad',
'B03607C4E6_WT2024071214941.h5ad',
'A03994F1G2_WT2024071215067.h5ad',
'A03588A1C2_WT202407161185.h5ad',
]









import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

fig = plt.figure(figsize=(64, 24))
gs = GridSpec(4, 7, figure=fig)

x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
for name in names:
    adata_temp = adata[adata.obs['slice_code'] == name].copy()
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
    
    x_min = min(x_min, adata_temp.obsm['align_spatial_2d'][:, 0].min())
    x_max = max(x_max, adata_temp.obsm['align_spatial_2d'][:, 0].max())
    y_min = min(y_min, adata_temp.obsm['align_spatial_2d'][:, 1].min())
    y_max = max(y_max, adata_temp.obsm['align_spatial_2d'][:, 1].max())

count = 0
for name in names:
    adata_temp = adata[adata.obs['slice_code'] == name].copy()
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
    row = (count // 7) + 1
    col = count % 7  
    ax = fig.add_subplot(gs[row-1, col])
    
    sc.pl.embedding(
        adata_temp, basis="align_spatial_2d", color='dmt_leiden_merge',
        show=False, s=0.5, title='', legend_loc=None, ax=ax, palette=colormap
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.axis('off')
    ax.set_aspect('equal')
    if count == 0:
        scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
        ax.add_artist(scalebar)
    count += 1
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_spatial/spatial_plot_all.png', bbox_inches = 'tight', dpi = 600)
plt.close()



for i in set(adata.obs['dmt_leiden_merge']):
    fig = plt.figure(figsize=(64, 24))
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
    for name in names:
        adata_temp = adata[adata.obs['slice_code'] == name].copy()
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]

        x_min = min(x_min, adata_temp.obsm['align_spatial_2d'][:, 0].min())
        x_max = max(x_max, adata_temp.obsm['align_spatial_2d'][:, 0].max())
        y_min = min(y_min, adata_temp.obsm['align_spatial_2d'][:, 1].min())
        y_max = max(y_max, adata_temp.obsm['align_spatial_2d'][:, 1].max())
    
    gs = GridSpec(4, 7, figure=fig)
    count = 0
    for name in names:
        adata_temp = adata[adata.obs['slice_code'] == name].copy()
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]

        row = (count // 7) + 1
        col = count % 7  
        ax = fig.add_subplot(gs[row-1, col])
        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='dmt_leiden_merge',
            show=False, s=0.5, title='', legend_loc=None, ax=ax, palette=colormap, groups = i
        )
        
        # 设置统一的坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.axis('off')
        ax.set_aspect('equal')
        
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count+=1

    plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_spatial/spatial_plot_{i}.png', bbox_inches = 'tight', dpi = 600)
    plt.close()
