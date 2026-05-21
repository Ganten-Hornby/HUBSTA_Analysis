import scanpy as sc
import anndata as ad
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, rgb2hex
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 # ADOBE AI 字帖
import os
from matplotlib.font_manager import fontManager, FontProperties

fontManager.addfont('/data/work/Arial.ttf')

font = FontProperties(fname='/data/work/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

colors = [
    "navy",                 
    "white",                
    (205 / 255, 41 / 255, 41 / 255)  
]

cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=100)

adata = sc.read_h5ad('/data/input/Files/ResultData/Notebook/NB2026022610032617570349/cerebellum_scAnno_20260226.h5ad')
adata.obs_names_make_unique()

adata = adata[:,[False if '.' in i else True for i in adata.var.index.tolist()]].copy()

adata = adata[adata.obs['dmt_leiden_anno'] != 'z_delete']




colormap = {'Cere_sc_22': '#b720f0',
 'Cere_sc_1': '#9ee6fe',
 'Cere_sc_2': '#84e0ee',
 'Cere_sc_14': '#405312',
 'Cere_sc_20': '#1af63c',
 'Cere_sc_7': '#51346b',
 'Cere_sc_19': '#97a50d',
 'Cere_sc_23': '#cdf318',
 'Cere_sc_15': '#728cfd',
 'Cere_sc_5': '#65e03c',
 'Cere_sc_4': '#cb6780',
 'Cere_sc_9': '#b572a1',
 'Cere_sc_13': '#51e78f',
 'Cere_sc_8': '#89c066',
 'Cere_sc_10': '#8163f4',
 'Cere_sc_24': '#c517d2',
 'Cere_sc_17_endo': '#f30494',
 'Cere_sc_17_others': '#f30449', 
 'z_delete': '#557eae',
 'Cere_sc_12': '#c93c35',
 'Cere_sc_21': '#10650a',
 'Cere_sc_3': '#aae898',
 'Cere_sc_16': '#505238',
 'Cere_sc_18': '#1f609e',
 'Cere_sc_6': '#f216bd',
 'Cere_sc_11': '#8945d4'}


names = ['15_C03627F5_WT202403180043.h5ad',
         '17_C03627F6_WT202403270557.h5ad',
'19_D03657F1_WT202403110530.h5ad',
'21_D03657F2_WT202403110531.h5ad',
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
'76_D03656A5_WT202403280404.h5ad',
'81_D03657C6_WT202403110520.h5ad',
'85_B03611D2_WT202403110546.h5ad',
'90_A03592D3_WT202403110532.h5ad',
'95_B03602D1_WT202403110535.h5ad',
'100_B03609G1_WT202403280406.h5ad',
         'A03590A3D6_WT202407192652.h5ad', # gw13
         'A03588A1C2_WT202407161185.h5ad', # gw13
         'A03988A1C2_WT202407161208.h5ad', # gw13
         'A03994F1G2_WT2024071215067.h5ad',# gw13
         # 'A03591D4E5_WT2024071215074.h5ad',
         'A03587A5C6_WT2024071215080.h5ad', # gw10
         'B03607C4E6_WT2024071214941.h5ad', # gw12
         'B03618D3F6_WT202407152793.h5ad', # gw16
         'B04122A3F6_WT202407282762.h5ad', # gw18
]
x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
for name in names:
    adata_temp = adata[adata.obs['slice_code'] == name].copy()
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
    
    x_min = min(x_min, adata_temp.obsm['align_spatial_2d'][:, 0].min())
    x_max = max(x_max, adata_temp.obsm['align_spatial_2d'][:, 0].max())
    y_min = min(y_min, adata_temp.obsm['align_spatial_2d'][:, 1].min())
    y_max = max(y_max, adata_temp.obsm['align_spatial_2d'][:, 1].max())


if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/8_cerebellum_single_re/5_sub_plot/single_17_18.png'):
    fig = plt.figure(figsize=(64, 24))
    gs = GridSpec(4, 8, figure=fig)
    count = 0
    for name in names:
        adata_temp = adata[adata.obs['slice_code'] == name].copy()
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
        row = (count // 8) + 1
        col = count % 8
        ax = fig.add_subplot(gs[row-1, col])

        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno', groups = ['Cere_sc_17_endo', 'Cere_sc_17_others','Cere_sc_18'],
            show=False, s=2, title='', legend_loc=None, ax=ax, palette=colormap
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axis('off')
        ax.set_aspect('equal')
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count += 1
    plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/8_cerebellum_single_re/5_sub_plot/single_17_18.png', bbox_inches = 'tight', dpi = 600)
    plt.close()

if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/8_cerebellum_single_re/5_sub_plot/single_2_20_21_17_18.png'):
    fig = plt.figure(figsize=(64, 24))
    gs = GridSpec(4, 8, figure=fig)
    count = 0
    for name in names:
        adata_temp = adata[adata.obs['slice_code'] == name].copy()
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
        row = (count // 8) + 1
        col = count % 8
        ax = fig.add_subplot(gs[row-1, col])

        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno', groups = ['Cere_sc_2', 'Cere_sc_20', 'Cere_sc_21', 'Cere_sc_17_endo', 'Cere_sc_17_others','Cere_sc_18'],
            show=False, s=2, title='', legend_loc=None, ax=ax, palette=colormap
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axis('off')
        ax.set_aspect('equal')
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count += 1
    plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/8_cerebellum_single_re/5_sub_plot/single_2_20_21_17_18.png', bbox_inches = 'tight', dpi = 600)
    plt.close()
    
if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/8_cerebellum_single_re/5_sub_plot/single_2_17_18_19_20_21.png'):
    fig = plt.figure(figsize=(64, 24))
    gs = GridSpec(4, 8, figure=fig)
    count = 0
    for name in names:
        adata_temp = adata[adata.obs['slice_code'] == name].copy()
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
        row = (count // 8) + 1
        col = count % 8
        ax = fig.add_subplot(gs[row-1, col])

        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno', groups = ['Cere_sc_2', 'Cere_sc_20', 'Cere_sc_21','Cere_sc_17_endo', 'Cere_sc_17_others','Cere_sc_18'],
            show=False, s=2, title='', legend_loc=None, ax=ax, palette=colormap
        )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axis('off')
        ax.set_aspect('equal')
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count += 1
    plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/8_cerebellum_single_re/5_sub_plot/single_2_17_18_19_20_21.png', bbox_inches = 'tight', dpi = 600)
    plt.close()