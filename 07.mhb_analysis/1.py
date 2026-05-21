#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

from matplotlib.font_manager import fontManager, FontProperties

fontManager.addfont('/data/work/Arial.ttf')

font = FontProperties(fname='/data/work/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name


# In[3]:


adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/mid_hind_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()


# In[11]:


dic = {
    '0': 'mh_sc_00',
    '9': 'mh_sc_30',
    '1': 'mh_sc_01',
    '4': 'mh_sc_01',
    '12': 'mh_sc_01',
    '2': 'mh_sc_02',
    '26': 'mh_sc_02',
    '3': 'mh_sc_03',
    '5': 'mh_sc_04',
    '6': 'mh_sc_05',
    '19': 'mh_sc_05',
    '7': 'mh_sc_06',
    '8': 'mh_sc_07',
    '10': 'mh_sc_08',
    '11': 'mh_sc_09',
    '31': 'mh_sc_09',
    '13': 'mh_sc_10',
    '14': 'mh_sc_11',
    '15': 'mh_sc_12',
    '16': 'mh_sc_13',
    '17': 'mh_sc_14',
    '18': 'mh_sc_15',
    '35': 'mh_sc_15',
    '20': 'mh_sc_16',
    '21': 'mh_sc_17',
    '22': 'mh_sc_18',
    '23': 'mh_sc_19',
    '24': 'mh_sc_20',
    '25': 'mh_sc_21',
    '27': 'mh_sc_22',
    '32': 'mh_sc_22',
    '28': 'mh_sc_23',
    '29': 'mh_sc_24',
    '30': 'mh_sc_25',
    '33': 'mh_sc_26',
    '34': 'mh_sc_27',
    '36': 'mh_sc_28',
    '38': 'mh_sc_28',
    '37': 'mh_sc_29',
}
adata.obs['dmt_leiden_anno'] = [dic[i] for i in adata.obs['dmt_leiden']]

# adata = adata[:, ~adata.var_names.str.startswith('MT')]
# adata = adata[:,[False if '.' in i else True for i in adata.var.index.tolist()]]
# del adata.uns
# del adata.obsm
# del adata.obsp
# adata.write('mid_hind.h5ad')
# sorted_dict = {
#     # '0': ['HES3', 'CAPN3', 'AC132803.1', 'AC004817.3', 'BX284656.2'],
#     # '0': ['PTTG2', 'AL360181.2', 'AC018442.2', 'FASLG', 'LINC00442'],
#     '0': ['DEFB126',],
#  '1': ['LMO3',],
#  '2': ['CRABP1', 'OR5M3', ],
#  '3': ['NTS', 'PITX2', 'NTNG2'],
#  '4': ['TTR',],
#  '5': ['HBG2', 'HBG1', 'HBA1', 'HBA2'],
#  '6': ['CARTPT', 'ISLR2', 'SYT4'],
#  '7': ['TAC1', ],
#  '8': ['SST',],
#  '9': ['GFAP', 'AC125421.2', ],
#  '10': ['TCF7L2', 'LHX9',],
#  '11': ['HOXA5', 'HOXB5',],
#  '12': ['OTX2-AS1', 'OTX2', ],
#  '13': ['PMCH', 'AC092611.1', ],
#  '14': ['NPY', 'LHX1', ],
#  '15': ['CCK', 'TFAP2B', ],
#  '16': ['GATA3', 'SIX3', ],
#  '17': ['PAX5', 'EN2', 'CNPY1',],
#  '18': ['SPARCL1', 'CST3',],
#  '19': ['AC012462.2'],
#  '20': ['SPP1', 'APOE', 'FTL', 'CTSB'],
#  '21': ['ALDH1A1', 'TH', 'PBX1', 'NR4A2'],
#  '22': ['NR2F2', 'NR2F1',],
#  '23': ['NEFM', 'NEFL', 'PRPH', 'LGALS1'],
#  '24': ['CBLN1', ],
#  # '25': ['KAAG1', 'LINC02058', 'AC096759.2', 'TAB3-AS2', 'LINC02251'],
#     '25': ['VIM'],
#  '26': ['IGFBP7', 'CLDN5', 'FN1', 'ITM2A', 'SLC2A1'],
#  '27': ['ETV1', 'RNASE13'],
#  '28': ['ZIC1', 'KITLG', 'ZIC2', 'ZIC4'],
#  '29': ['RTP3', ]}
# In[ ]:





# In[4]:


adatas = []
for i in set(adata.obs['slice_code']):
    temp = adata[adata.obs['slice_code'] == i].copy()
    sc.pp.normalize_total(temp)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    adatas.append(temp)
adata = ad.concat(adatas)




var = [
    # '0': ['HES3', 'CAPN3', 'AC132803.1', 'AC004817.3', 'BX284656.2'],
    # '0': ['PTTG2', 'AL360181.2', 'AC018442.2', 'FASLG', 'LINC00442'],
    'DEFB126',
    'LMO3',
    'CRABP1', 'NFIB', 
    'NTS', 'PITX2', 'NTNG2',
    'TTR',
    'HBG2', 'HBG1', 'HBA1', 'HBA2',
    'CARTPT', 'ISLR2', 'SYT4',
    'TAC1', 
    'SST',
    'GFAP', 'AC125421.2',
    'TCF7L2', 'LHX9',
    'HOXA5', 'HOXB5',
    'OTX2-AS1', 'OTX2', 
    'PMCH', 'AC092611.1', 
    'NPY', 'LHX1', 
    'CCK', 'TFAP2B', 
    'GATA3', 'SIX3', 
    'PAX5', 'EN2', 'CNPY1',
    'SPARCL1', 'CST3',
    'AC012462.2', 'MALAT1',
    'SPP1', 'APOE', 'FTL', 'CTSB',
    'ALDH1A1', 'TH', 'PBX1', 'NR4A2',
    'NR2F2', 'NR2F1',
    'NEFM', 'NEFL', 'PRPH', 'LGALS1',
    'CBLN1', 
        'VIM',
 'IGFBP7', 'CLDN5', 'FN1', 'ITM2A', 'SLC2A1',
 'ETV1', 'RNASE13',
 'ZIC1', 'KITLG', 'ZIC2', 'ZIC4',
 'RTP3'
       ]




adata = adata[adata.obs['dmt_leiden_anno'] != 'mh_sc_30']








sc.pl.matrixplot(adata, 
                 var, 
                 'dmt_leiden_anno', 
                 standard_scale = 'var',
                 # dendrogram = True,
                 show = False
                )
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_single/matrixplot.pdf', bbox_inches = 'tight', dpi = 600)
plt.close()

sc.pl.dotplot(adata, 
                 var, 
                 'dmt_leiden_anno', 
                 standard_scale = 'var',
              show = False
                )
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_single/dotplot.pdf', bbox_inches = 'tight', dpi = 600)
plt.close()




colormap = {'mh_sc_29': '#a3dc1f',
 'mh_sc_00': '#5ef377',
 'mh_sc_02': '#654c0d',
 'mh_sc_27': '#aea266',
 'mh_sc_03': '#e4f36b',
 'mh_sc_23': '#2a74d1',
 'mh_sc_06': '#1426db',
 'mh_sc_25': '#e93f06',
 'mh_sc_16': '#8caec2',
 'mh_sc_28': '#db3529',
 'mh_sc_18': '#215789',
 'mh_sc_22': '#490ea7',
 'mh_sc_21': '#7a7bc3',
 'mh_sc_14': '#0d204c',
 'mh_sc_19': '#4ae86f',
 'mh_sc_20': '#efb6a4',
 'mh_sc_04': '#f55fb7',
 'mh_sc_13': '#ccb347',
 'mh_sc_26': '#b36428',
 'mh_sc_09': '#efea60',
 'mh_sc_17': '#30fac0',
 'mh_sc_12': '#cee3d7',
 'mh_sc_08': '#8a1ecb',
 'mh_sc_07': '#607f79',
 'mh_sc_15': '#3a6152',
 'mh_sc_24': '#91a6de',
 'mh_sc_10': '#779d25',
 'mh_sc_11': '#524ae3',
 'mh_sc_05': '#1164b8',
 'mh_sc_01': '#1f8071'}




plot = sc.pl.embedding(adata, basis = 'X_dmt',color = 'dmt_leiden_anno', palette = colormap, show = False, 
                       title = 'Midbrain&Hindbrain SpatialSingelCell Annotation'); 
plot.set_aspect('equal')
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_single/dmt_plot.png', bbox_inches = 'tight', dpi = 600)
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
        adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno',
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
plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_single/spatial_plot_all.png', bbox_inches = 'tight', dpi = 600)
plt.close()




for i in set(adata.obs['dmt_leiden_anno']):
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
            adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno',
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

    plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_single/spatial_plot_{i}.png', bbox_inches = 'tight', dpi = 600)
    plt.close()







gene_plot = {
'genes':['EN1', 'EN2', 'OTX2', 'GBX2', 'FGF8', 'WNT1'],
'FGFpathway': ['FGF2','FGF4','FGF8','FGF9','FGF10','FGF17','FGF19','FGF20','FGFR1','FGFR2','FGFR3','FGFR4'],
'WNTpathway':['WNT7A','WNT3A','WNT2','WNT8A','WNT5A','WNT5B','WNT6','FZD5','FZD9','FZD10','CER1','DKK1'],
'BMPpathway':['BMP8A','BMP3','BMP4','BMP5','BMP8B','GDF6','GDF7','BMPR1A','BMPR2','ACTR2','NOG','CHRD'],
'NODALpathway': ['NODAL','TDGF1','CFC1','ACTR1B','LEFTY2'],
'SHHpathway': ['SHH','CREBBP','SMO','PTCH1','SUFU'],
}

for k,v in gene_plot.items():

    for gene in v:
        try:
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
                    adata_temp, basis="align_spatial_2d", color=gene,
                    show=False, s=0.5, title='', legend_loc=None, ax=ax, cmap = 'Reds',
                )

                # 设置统一的坐标轴范围
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                ax.axis('off')
                ax.set_aspect('equal')
                if count == 0:
                    scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
                    ax.add_artist(scalebar)
                count += 1
            plt.savefig(f'/data/work/05.cluster/FuseMap/0313/mid_hind_single/gene_plot/{k}/{gene}.png', bbox_inches = 'tight', dpi = 600)
            plt.close()
        except:
            continue





