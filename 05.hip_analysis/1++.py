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


adatas = []
for i in set(adata.obs['slice_code']):
    temp = adata[adata.obs['slice_code'] == i].copy()
    sc.pp.normalize_total(temp)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    adatas.append(temp)
adata = ad.concat(adatas)


adata.obs_names_make_unique()


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm



x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
for name in names:
    adata_temp = adata[adata.obs['slice_code'] == name].copy()
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
    
    x_min = min(x_min, adata_temp.obsm['align_spatial_2d'][:, 0].min())
    x_max = max(x_max, adata_temp.obsm['align_spatial_2d'][:, 0].max())
    y_min = min(y_min, adata_temp.obsm['align_spatial_2d'][:, 1].min())
    y_max = max(y_max, adata_temp.obsm['align_spatial_2d'][:, 1].max())

    
# gene_list = ['ADNP', 'ANXA1', 'ARX', 'ASCL1', 'BMP4', 'BMP7', 'BTBD11', 'CALB1', 'CCND1', 'CDK4', 'CDKN1A', 'CDKN2B', 'NR2F1', 'CREB1', 'CTIP2', 'CCSER1', 'DAB1', 'DACT1', 'DCX', 'DLL1', 'DPF1', 'DVL2', 'EPHA4', 'EPHRINA5', 'EMX1', 'EMX2', 'EZH2', 'FGF8', 'FGF17', 'FGFR1', 'FOXG1', 'FAM107A', 'FCAR', 'GABRA1', 'GFAP', 'GLI3', 'GAP43', 'GRIA1', 'GRIK4', 'GRIN2B', 'HES1', 'HES5', 'HOPX', 'ID2', 'ID4', 'LILRA5', 'LRP6', 'LHX2', 'LHX9', 'LIN54', 'MAP1B', 'MAP2', 'MECP2', 'MKI67', 'MYT1L', 'NEFL', 'NEUROD1', 'NEUROG2', 'NES', 'NKX3-1', 'NLGN1', 'NOTCH1', 'NR2F1', 'NRXN1', 'PCNA', 'PCP4', 'PAX6', 'PLK2', 'PLK3', 'PLXNA2', 'PROX1', 'PRKCD', 'PTN', 'REELIN', 'RIC8B', 'SATB2', 'SCN1A', 'SCN2A', 'SEMA3A', 'SHANK3', 'SHH', 'SLFN11', 'SOX2', 'SOX11', 'STMN2', 'S100A2', 'S100A12', 'TBR1', 'TBR2', 'TNIP3', 'TUBA1A', 'UBE3A', 'VIM', 'WFS1', 'WNT3A', 'WNT5A', 'ZBTB20', 'BDNF', 'CCNE1', 'DNMT1', 'DLG4', 'FGF2', 'FMR4', 'HDAC1', 'HDAC2', 'JAG1', 'LIS1', 'MIR132', 'NEUROD2', 'NGF', 'NOTCH3', 'NTRK2', 'RELN', 'SYN1', 'SYN2', 'TRKB']
# gene_list = ['SHH', 'PTCH1', 'SMO', 'NKX2-1', 'NKX2-2', 'NKX6-1', 'OLIG2', 'FOXA2']
# gene_list = ['APP', 'PSEN1', 'PSEN2', 'APOE', 'TREM2', 'SORL1', 'BIN1', 'PICALM', 'CLU', 'ABCA7', 'CR1', 'CD33', 'MS4A6A', 
#     'INPP5D', 'FERMT2', 'PTK2B', 'SPI1', 'PLCG2', 'MEF2C', 'TOMM40', 'CHD8',]
gene_list = ['LMX1A',
'WNT3A',
'WNT2B',
'RSPO2',
'BMP4',
'BMP7',
'WNT7A',
'WNT7B',
'EMX2',
'PAX6',
'GLI3',
'LHX2',
'ZIC1',
'ZIC2',
'LEF1',
'PROX1',
'NEUROD1',
'ZBTB20',
'GRIK4',
'WIP1',
'RELN',
'CXCR4',
'CXCL12',
'NR2F2',]
for gene in gene_list:
    if os.path.exists(f'/data/output/hip_03/{gene}.png'):
        continue
    try:
        fig = plt.figure(figsize=(64, 6))
        gs = GridSpec(1, 15, figure=fig)
        count = 0
        for name in names:
            adata_temp = adata[adata.obs['slice_code'] == name].copy()
            adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
            adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
            col = count % 15
            ax = fig.add_subplot(gs[0, col])

            # 绘制嵌入图
            sc.pl.embedding(
                adata_temp, basis="align_spatial_2d", color=gene,
                show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds'
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
        plt.savefig(f'/data/output/hip_03/{gene}.png', bbox_inches = 'tight', dpi = 600)
        plt.close()
    except:
        continue
        

        
# gene_list = ['BMP4', 'BMP7', 'GDF7', 'BMPR1A', 'BMPR1B', 'SMAD1', 'SMAD5', 'MSX1', 'MSX2', 'PAX3', 'PAX7']
# for gene in gene_list:
#     if os.path.exists(f'/data/output/BMPs/{gene}.png'):
#         continue
#     try:
#         fig = plt.figure(figsize=(64, 6))
#         gs = GridSpec(1, 15, figure=fig)
#         count = 0
#         for name in names:
#             adata_temp = adata[adata.obs['slice_code'] == name].copy()
#             adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
#             adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
#             col = count % 15
#             ax = fig.add_subplot(gs[0, col])

#             # 绘制嵌入图
#             sc.pl.embedding(
#                 adata_temp, basis="align_spatial_2d", color=gene,
#                 show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds'
#             )

#             # 设置统一的坐标轴范围
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             # 关闭坐标轴并设置等比例缩放
#             ax.axis('off')
#             ax.set_aspect('equal')
#             # scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#             # ax.add_artist(scalebar)
#             if count == 0:
#                 scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#                 ax.add_artist(scalebar)
#             count += 1
#         plt.savefig(f'/data/output/BMPs/{gene}.png', bbox_inches = 'tight', dpi = 600)
#         plt.close()
#     except:
#         continue
        
        
        
# gene_list = ['WNT1', 'WNT3A', 'WNT8B', 'FZD1-10', 'LRP5', 'LRP6', 'CTNNB1', 'TCF7', 'LEF1', 'EMX1', 'EMX2']
# for gene in gene_list:
#     if os.path.exists(f'/data/output/WNTs/{gene}.png'):
#         continue
#     try:
#         fig = plt.figure(figsize=(64, 6))
#         gs = GridSpec(1, 15, figure=fig)
#         count = 0
#         for name in names:
#             adata_temp = adata[adata.obs['slice_code'] == name].copy()
#             adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
#             adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
#             col = count % 15
#             ax = fig.add_subplot(gs[0, col])

#             # 绘制嵌入图
#             sc.pl.embedding(
#                 adata_temp, basis="align_spatial_2d", color=gene,
#                 show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds'
#             )

#             # 设置统一的坐标轴范围
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             # 关闭坐标轴并设置等比例缩放
#             ax.axis('off')
#             ax.set_aspect('equal')
#             # scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#             # ax.add_artist(scalebar)
#             if count == 0:
#                 scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#                 ax.add_artist(scalebar)
#             count += 1
#         plt.savefig(f'/data/output/WNTs/{gene}.png', bbox_inches = 'tight', dpi = 600)
#         plt.close()
#     except:
#         continue
        
# gene_list = ['FGF8', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'EN1', 'EN2', 'OTX2', 'GBX2', 'SP8']
# for gene in gene_list:
#     if os.path.exists(f'/data/output/FGF8/{gene}.png'):
#         continue
#     try:
#         fig = plt.figure(figsize=(64, 6))
#         gs = GridSpec(1, 15, figure=fig)
#         count = 0
#         for name in names:
#             adata_temp = adata[adata.obs['slice_code'] == name].copy()
#             adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
#             adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
#             col = count % 15
#             ax = fig.add_subplot(gs[0, col])

#             # 绘制嵌入图
#             sc.pl.embedding(
#                 adata_temp, basis="align_spatial_2d", color=gene,
#                 show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds'
#             )

#             # 设置统一的坐标轴范围
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             # 关闭坐标轴并设置等比例缩放
#             ax.axis('off')
#             ax.set_aspect('equal')
#             # scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#             # ax.add_artist(scalebar)
#             if count == 0:
#                 scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#                 ax.add_artist(scalebar)
#             count += 1
#         plt.savefig(f'/data/output/FGF8/{gene}.png', bbox_inches = 'tight', dpi = 600)
#         plt.close()
#     except:
#         continue
        
# gene_list = ['ALDH1A2', 'ALDH1A3', 'CYP26A1', 'CYP26B1', 'RARA', 'RARB', 'RARG', 'RXRA', 'RXRB', 'RXRG', 'HOXA1', 'HOXB1', 'HOXC4', 'HOXD4']
# for gene in gene_list:
#     if os.path.exists(f'/data/output/RA/{gene}.png'):
#         continue
#     try:
#         fig = plt.figure(figsize=(64, 6))
#         gs = GridSpec(1, 15, figure=fig)
#         count = 0
#         for name in names:
#             adata_temp = adata[adata.obs['slice_code'] == name].copy()
#             adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
#             adata_temp.obsm['align_spatial_2d'][:, 1] = -adata_temp.obsm['align_spatial_2d'][:, 1]
#             col = count % 15
#             ax = fig.add_subplot(gs[0, col])

#             # 绘制嵌入图
#             sc.pl.embedding(
#                 adata_temp, basis="align_spatial_2d", color=gene,
#                 show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds'
#             )

#             # 设置统一的坐标轴范围
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)

#             # 关闭坐标轴并设置等比例缩放
#             ax.axis('off')
#             ax.set_aspect('equal')
#             # scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#             # ax.add_artist(scalebar)
#             if count == 0:
#                 scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#                 ax.add_artist(scalebar)
#             count += 1
#         plt.savefig(f'/data/output/RA/{gene}.png', bbox_inches = 'tight', dpi = 600)
#         plt.close()
#     except:
#         continue
# plt.show()

# adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0116/xgb/adatas/GW22.h5ad')

# sc.pp.normalize_total(adata)
# sc.pp.log1p(adata)
# sc.pp.scale(adata, zero_center=False, max_value=10)


# for gene in gene_list:
#     if os.path.exists(f'/data/output/gipGW22/{gene}.png'):
#         continue
#     try:

#         ax = sc.pl.embedding(adata, basis = 'spatial', color = gene, show = False, cmap = 'Reds', title = '')
#         ax.axis('off')
#         ax.set_aspect('equal')
#         scalebar = ScaleBar(1/2000, "mm", fixed_value=1, location = 'lower left', frameon = False,)
#         ax.add_artist(scalebar)
#         # plt.show()
#         plt.savefig(f'/data/output/gipGW22/{gene}.png', dpi = 600, bbox_inches = 'tight')
#         plt.close()
#     except:
#         continue
