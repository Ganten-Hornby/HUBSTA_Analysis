import scanpy as sc
import anndata as ad
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, rgb2hex
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 # ADOBE AI 字帖

from matplotlib.font_manager import fontManager, FontProperties
import os
fontManager.addfont('/data/work/Arial.ttf')

font = FontProperties(fname='/data/work/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/Hippocampus_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')

names = [
    'B03607C4E6_WT2024071214941.h5ad',
    
    '12_B03605F3G5_WT202403310048.h5ad',
    '13_B03612A1C3_WT202403310056.h5ad',
    '14_A03591A1C3_WT202403310045.h5ad',
    '16_A03592A4C6_WT202403310044.h5ad',
    '18_B03602C4D6_WT202405020031.h5ad',
    '20_B03606F3G5_WT202405020032.h5ad',
    '22_B03606C4E6_WT202403310050.h5ad',
    '23_B03609A4D6_WT202404150263.h5ad',
    '27_B03610C1E3_WT202403310051.h5ad',
    '31_B03619A1D3_WT202403310052.h5ad',
    '35_B03619E4G6_WT202403310053.h5ad',
    '39_A03589A1D4_WT202403310046.h5ad',
    '43_A03590E1G4_WT202403310064.h5ad',
    '47_A03593C1F3_WT202403310068.h5ad',
]

dic_dmt_leiden = {
   
    
    '2': 'ctx_sc_01',
    '13': 'ctx_sc_02',
    '19': 'ctx_sc_03',
    '20': 'ctx_sc_04',
    
    '4': 'ctx_sc_05',
    '21': 'ctx_sc_05',
    
    '8': 'ctx_sc_06',
    '10': 'ctx_sc_07',
    '22': 'ctx_sc_08',
    
    '9': 'ctx_sc_09',
    '11': 'ctx_sc_09',
    '16': 'ctx_sc_10',
    '18': 'ctx_sc_10',
    '26': 'ctx_sc_10',
    '31': 'ctx_sc_10',
    
    '12': 'ctx_sc_11',
    '15': 'ctx_sc_12',
    '25': 'ctx_sc_12',
    '29': 'ctx_sc_12',
    '17': 'ctx_sc_13',
    
    '24': 'z_delete',
    
    
    '5': 'ctx_sc_14',
    '23': 'ctx_sc_15',
    '27': 'ctx_sc_16',
    
    '30': 'ctx_sc_17',
    
     '0': 'hip_sc_18',
    '7': 'hip_sc_19',
    
    '1': 'hip_sc_20',
    '3': 'hip_sc_21',
    '6': 'hip_sc_21',
    '14': 'hip_sc_22',
    '28': 'hip_sc_23',
}
adata.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata.obs['dmt_leiden']]

adata = adata[adata.obs['dmt_leiden_anno'] != 'z_delete']

colormap = {
 
  'ctx_sc_01' : '#374898',
  'ctx_sc_02' : '#6d85c7',
  'ctx_sc_03' : '#35c498',
  'ctx_sc_04' : '#9e2dc6',
  'ctx_sc_05' : '#2d7476',
  'ctx_sc_06' : '#cb0d6c',
  'ctx_sc_07' : '#20ea38',
  'ctx_sc_08' : '#0fabb6',
  'ctx_sc_09' : '#a59099',
  'ctx_sc_10' : '#2bea3a',
  'ctx_sc_11' : '#17b064',
  'ctx_sc_12' : '#52b8d5',
  'ctx_sc_13' : '#da2ef2',
  'ctx_sc_14' : '#6240f7',
  'ctx_sc_15' : '#c47233',
  'ctx_sc_16':'#a83b23',
  'ctx_sc_17':'#9994da',
  'hip_sc_18' : '#9b38e9',
  'hip_sc_19' : '#a89630',
  'hip_sc_20': '#5b798b',
  'hip_sc_21' : '#cb2505',
  'hip_sc_22' : '#62e7dd',
  'hip_sc_23' : '#245200',
}

adata.obs_names_make_unique()


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# 创建画布和网格布局
fig = plt.figure(figsize=(64, 6))
gs = GridSpec(1, 15, figure=fig)

# 计算全局坐标范围
x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
for name in names:
    adata_temp = adata[adata.obs['slice_code'] == name].copy()
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
    
    x_min = min(x_min, adata_temp.obsm['align_spatial_2d'][:, 0].min())
    x_max = max(x_max, adata_temp.obsm['align_spatial_2d'][:, 0].max())
    y_min = min(y_min, adata_temp.obsm['align_spatial_2d'][:, 1].min())
    y_max = max(y_max, adata_temp.obsm['align_spatial_2d'][:, 1].max())

# 绘制子图
count = 0
for name in names:
    adata_temp = adata[adata.obs['slice_code'] == name].copy()
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
    col = count % 15
    ax = fig.add_subplot(gs[0, col])
    
    # 绘制嵌入图
    sc.pl.embedding(
        adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno',
        show=False, s=1, title='', legend_loc=None, ax=ax, palette=colormap,
        groups = ['hip_sc_18', 'hip_sc_19', 'hip_sc_20', 'hip_sc_21', 'hip_sc_22', 'hip_sc_23']
    )
    
    # 设置统一的坐标轴范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 关闭坐标轴并设置等比例缩放
    ax.axis('off')
    ax.set_aspect('equal')
    # scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
    # ax.add_artist(scalebar)
    if count == 0:
        scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
        ax.add_artist(scalebar)
    count += 1
plt.savefig(f'/data/output/hip_02/spatial_plot.png', bbox_inches = 'tight', dpi = 600)
plt.close()
# plt.show()