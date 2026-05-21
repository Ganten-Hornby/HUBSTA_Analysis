# genes = ['DLX2','CALB1','NR2F2','DRD2','OPRM1','ZIC4','IRX3','FOXP2','GBX2','NTNG1','NF2F1','RGS16','TRH','TCF7L2','GAD1','GAD2','SLC17A7','SLC17A6','ETV1','MEF2C','NR4A2','POU4F1','RORA','ZIC1','ZNF804A','PTCHD1','ERBB4']
# genes = ['SST','LHX1','DLX6','SIX3','DLX1','DLX5','SOX14','NTS','OTX2','SSTR1','LHX5','GATA2','GBX2','DLX2']
# genes = ['SP8', 'CCK', 'PVALB']
genes = ['Tac2',
'Gal',
'Agt',
'Uts2b',
'Nucb2',
'Calca',
'Adcyap1',
'Calcb',
'Avp',
'Pdyn',
'Apln',
'Pomc',
'Npvf',
'Grp',
'Gcg',
'Pnoc',
'Cart',
'Ucn',
'Crh',
'Penk',
'Tac1',
'Vip',
'Cbln4',
'Nxphl',
'Cbln2',
'Scg2',
'Dbi',
'Scg5',
'Igf2',
'Chga',
'Chgab',
'Vgf',
'Nxph3',
'Scg3',
'Pthlh',
'Gnrh1',
'Trh',
'Prl2c2']
import scanpy as sc
import anndata as ad
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, rgb2hex
import numpy as np
import os
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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/thalamus_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')


adatas = []
for i in set(adata.obs['slice_code']):
    temp = adata[adata.obs['slice_code'] == i].copy()
    sc.pp.normalize_total(temp)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    adatas.append(temp)
adata = ad.concat(adatas)

names = [
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
 '51_B03605C2E5_WT202406020126.h5ad',
 '55_B03613E3G6_WT202403310069.h5ad',
 '59_B03612E4G6_WT202403310059.h5ad',
 '63_B03606C1E3_WT202403310061.h5ad',
 '67_A03595A1D3_WT202403310062.h5ad',
 '71_A03595A4D6_WT202403310063.h5ad',
 '75_D03468D1E3_WT202403310066.h5ad',
 '80_D03473D4E6_WT202403310070.h5ad',
 '84_B03423D1E3_WT202403310065.h5ad',
 '89_D03466A1C3_WT202403310058.h5ad',
 '99_D03470D1E3_WT202404290071.h5ad',
 '104_B03615F3G5_WT202405020035.h5ad',
 '105_D03468A4C6_WT202403310067.h5ad',
 'A03988A1C2_WT202407161208.h5ad',
 # 'A03591D4E5_WT2024071215074.h5ad',
 'A03590A3D6_WT202407192652.h5ad',
 'A03994F1G2_WT2024071215067.h5ad',
    'A03587A5C6_WT2024071215080.h5ad', # GW10
 'B03607C4E6_WT2024071214941.h5ad', # GW12
    'B03618D3F6_WT202407152793.h5ad', # GW16
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

for gene in genes:
    gene = gene.upper()
    try:
        if os.path.exists(f'/data/output/thalamus/{gene}.png'):
            continue
        fig = plt.figure(figsize=(32, 12))
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
                show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds'
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.axis('off')
            ax.set_aspect('equal')
            if count == 0:
                scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
                ax.add_artist(scalebar)
            count += 1
        plt.savefig(f'/data/output/thalamus/{gene}.png', bbox_inches = 'tight', dpi = 600)
        plt.close()
    except:
        continue