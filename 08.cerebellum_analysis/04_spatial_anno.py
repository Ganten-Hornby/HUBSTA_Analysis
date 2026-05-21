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


colormap = {'Cebe_3': '#d590d7',
 'Cebe_4': '#e31fe9',
 'Cebe_7': '#ec2426',
 'Cebe_0': '#cc6c42',
 'Cebe_1': '#708365',
 'Cebe_11': '#8980c0',
 'Cebe_5': '#6e1f94',
 'Cebe_10': '#6fbf5e',
 'Cebe_6': '#3da672',
 'Cebe_2': '#902b32',
 'Cebe_9': '#cde114',
 'Cebe_8': '#bcf19a'}

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

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/cerebellum_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
adata = adata[:, ~adata.var_names.str.startswith('MT')]
adata = adata[:, ~adata.var_names.str.startswith('-')]
adata = adata[:, adata.var_names != '']
adata.obs_names_make_unique()
dmt_leiden_merge = {
    '0': 'Cebe_0',
    '8': 'Cebe_0',
    '44': 'Cebe_0',
    '32': 'Cebe_0',
    '15': 'Cebe_0',
    '20': 'Cebe_0',
    '13': 'Cebe_0',
    '3': 'Cebe_0',
    '24': 'Cebe_1',
    '27': 'Cebe_1',
    '7': 'Cebe_1',
    
    '16': 'Cebe_2',
    '10': 'Cebe_2',
    
    '30': 'Cebe_3',
    '35': 'Cebe_3',
    '1': 'Cebe_3',
    
    '17': 'Cebe_4',
    '36': 'Cebe_4',
    '28': 'Cebe_4',
    
    '5': 'Cebe_5',
    
    '31': 'Cebe_6',
    '33': 'Cebe_6',
    '11': 'Cebe_6',
    '22': 'Cebe_6',
    '41': 'Cebe_6',
    '21': 'Cebe_6',
    
    
    '37': 'Cebe_7',
    '25': 'Cebe_7',
    
    '45': 'Cebe_8',
    '34': 'Cebe_8',
    '18': 'Cebe_8',
    '38': 'Cebe_8',
    '9': 'Cebe_8',
    '4': 'Cebe_8',
    '19': 'Cebe_8',
    '26': 'Cebe_8',
    
    '43': 'z_delete',
    
    '6': 'Cebe_9',
    '12': 'Cebe_9',
    '14': 'Cebe_9',
    '2': 'Cebe_9',
    '29': 'Cebe_9',
    
    '23': 'Cebe_10',
    '39': 'Cebe_10',
    '42': 'Cebe_10',
    
    '40': 'Cebe_11',

}
adata.obs['dmt_leiden_anno'] = [dmt_leiden_merge[i] for i in adata.obs['dmt_leiden'] ]
adata = adata[adata.obs['dmt_leiden_anno']!='z_delete']
adatas = []

for i in set(adata.obs['slice_code']):
    temp = adata[adata.obs['slice_code'] == i].copy()
    sc.pp.normalize_total(temp)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    del temp.obsm['cell_border']
    # temp = temp[temp.obs.sample(frac = 0.3).index]
    adatas.append(temp)
adata = ad.concat(adatas)
var = [
'STMN2','TUBA1A','TUBB2A','NEFL',
                           'HBG2','HBG1',
                           'SLC1A6','PCDH9','CALB1',
                           'RPL37A',
                           'INSM1','ERBB4',#'HNRNPA1','RPS6','PAX6',
                           'FN1','ITM2A',#'CLDN5','COL4A1',
                           'CA8','ID2','FOXP2','CPLX2','EBF1',
                           'CRABP1','TMSB4X','HNRNPA1','HES6',
                           'RORA','PCP4','FOXP2','DAB1',
                           'TTR','IGFBP7','FOLR1','SERPINF1',
                          'IGFBP5','TFAP2B','NEUROD2','ZIC1',
                          'PTN','VIM','NTRK2','SMOC1',
    
]

if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/matrixplot.pdf'):
    sc.pl.matrixplot(adata, 
                     var, 
                     'dmt_leiden_anno', 
                     standard_scale = 'var',
                     # dendrogram = True,
                     show = False,
                     cmap = cmap
                    )
    plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/matrixplot.pdf', bbox_inches = 'tight', dpi = 600)
    plt.close()
if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/dotplot.pdf'):
    sc.pl.dotplot(adata, 
                     var, 
                     'dmt_leiden_anno', 
                     standard_scale = 'var',
                  show = False,
                  cmap = cmap
                    )
    plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/dotplot.pdf', bbox_inches = 'tight', dpi = 600)
    plt.close()


if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/dmt_plot.png'):
    plot = sc.pl.embedding(adata, basis = 'X_dmt',color = 'dmt_leiden_anno', palette = colormap, show = False, 
                           title = 'Cerebellum SpatialRegion Annotation'); 
    plot.set_aspect('equal')
    plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/dmt_plot.png', bbox_inches = 'tight', dpi = 600)
    plt.close()

x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
for name in names:
    adata_temp = adata[adata.obs['slice_code'] == name].copy()
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
    
    x_min = min(x_min, adata_temp.obsm['align_spatial_2d'][:, 0].min())
    x_max = max(x_max, adata_temp.obsm['align_spatial_2d'][:, 0].max())
    y_min = min(y_min, adata_temp.obsm['align_spatial_2d'][:, 1].min())
    y_max = max(y_max, adata_temp.obsm['align_spatial_2d'][:, 1].max())


if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/spatial_plot_all.png'):
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
            adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno',
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
    plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/spatial_plot_all.png', bbox_inches = 'tight', dpi = 600)
    plt.close()


for i in set(adata.obs['dmt_leiden_anno']):
    if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/region_spatial_plot/{i}.pdf'):
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
                adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno',
                show=False, s=2, title='', legend_loc=None, ax=ax, palette=colormap, groups = i
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

        plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/region_spatial_plot/{i}.png', bbox_inches = 'tight', dpi = 600)
        plt.close()
        
for gene in var:
    try:
        if not os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/gene_plot/{gene}.pdf'):
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
                    adata_temp, basis="align_spatial_2d", color=gene,
                    show=False, s=0.5, title='', legend_loc=None, ax=ax, cmap = 'Reds',
                )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                ax.axis('off')
                ax.set_aspect('equal')
                if count == 0:
                    scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
                    ax.add_artist(scalebar)
                count += 1
            plt.savefig(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/gene_plot/{gene}.png', bbox_inches = 'tight', dpi = 600)
            plt.close()
    except:
        continue