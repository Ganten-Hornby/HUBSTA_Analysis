genes = [
    # 'SLC17A6','SLC17A8','GAD2','SLC6A1','SLC32A1','DRD5','DRD2','HTR2C','CHAT','TAC1','OPRM1','GPR139','GPR151',
    #     'TRH','RGS16','PROX1','KITLG','PDE5A','LHX9','NTNG1',
        #'AIFM1','ALOX5','AMFR','AQP11','ATF4','ATF6','ATG10','ATP2A1','ATP2A2','ATP2A3','AUP1','BAG6','BAK1','BBC3','BCAP31','BCL2','BCL2L1','BFAR','BOK','BRSK2','CALR3','CANX','CASP4','CFTR','CREB3L1','CREB3L2','CREB3L4','CREBZF','CXCL8','DAB2IP','DDRGK1','DDX3X','DERL1','DERL2','DERL3','DNAJB12','DNAJB2','DNAJB9','DNAJC10','DNAJC18','DNAJC3','ECPAS','EDEM2','EIF2AK2','EIF2AK4','EIF2B5','EIF4G1','ELAVL4','ERLEC1','ERLIN1','ERN1','ERP29','ERP44','FAF2','FAM8A1','FBXO17','FBXO27','FBXO44','FBXO6','FICD','FLOT1','FOXRED2','GSK3B','HERPUD2','HM13','HSP90B1','HSPA5','ITPR1','JKAMP','LPCAT3','MANF','MAP3K5','MARCKS','MBTPS1','NGLY1','NR1H3','NRBF2','OPA1','P4HB','PARP16','PDIA3','PDIA6','PIK3R1','PIK3R2','PLA2G6','PML','PPP1R15B','PPP2CB','PTPN2','QRICH1','RASGRF1','RHBDD2','RNF103','RNF139','RNF183','RNF185','SEL1L2','SERP1','SESN2','SGTA','SGTB','SIRT1','SRPX','STT3B','STUB1','SYVN1','TMBIM6','TMED2','TMEM117','TMEM259','TMEM33','TMUB1','TOR1A','TRAF2','TRIB3','UBAC2','UBE2G2','UBE2J1','UBE2J2','UBE4A','UBE4B','UBQLN1','UBQLN2','UBXN4','UBXN6','UBXN8','UFM1','UMOD','USP14','USP19','VAPB','YOD1',

        #'GRSF1', 'HNRNPA1', 'HNRNPA1L2', 'HNRNPA2B1', 'HNRNPA3', 'HNRNPC', 'HNRNPF', 'HNRNPH2', 'HNRNPH3', 'HNRNPK', 'HNRNPL', 'HNRNPLL', 'HNRNPM', 'HNRNPR', 'HSPA8', 'HTATSF1', 'IK', 'ILDR1', 'ILDR2', 'ISY1', 'IWS1', 'JMJD6', 'KHDRBS1', 'LARP7', 'LGALS3', 'LSM1', 'LSM10', 'LSM2', 'LSM3', 'LSM4', 'LSM5', 'LSM6', 'LSM7', 'LSM8', 'MAGOH', 'MAGOHB', 'MBNL3', 'METTL14', 'METTL3', 'METTL4', 'MFAP1', 'MPHOSPH10', 'MTREX', 'MYOD1', 'NCBP2', 'NCBP2L', 'NCL', 'NOL3', 'NONO', 'NPM1', 'NRDE2', 'NSRP1', 'PABPC1', 'PDCD7', 'PHF5A', 'PLRG1', 'PNN', 'PPIE', 'PPIG', 'PPIH', 'PPIL1', 'PPIL3', 'PPP2CA', 'PPP2R1A', 'PQBP1', 'PRDX6', 'PRMT1', 'PRMT5', 'PRPF18', 'PRPF31', 'PRPF38A', 'PRPF38B', 'PRPF4', 'PRPF40A', 'PRPF8', 'PSIP1', 'PUF60', 'RALY', 'RBM12', 'RBM15B', 'RBM17', 'RBM20', 'RBM22', 'RBM23', 'RBM28', 'RBM3', 'RBM38', 'RBM39', 'RBM4', 'RBM42', 'RBM4B', 'RBM7', 'RBM8A', 'RBMX', 'RBMX2', 'RNF113A', 'RNPS1', 'RP9', 'RPS13', 'RPUSD4', 'RRAGC', 'RSRC1', 'RTCB', 'RTRAF', 'SAP18', 'SART3', 'SCNM1', 'SF3A1', 'SF3A3', 'SF3B1', 'SF3B2', 'SF3B4', 'SF3B5', 'SF3B6', 'SFPQ', 'SLU7', 'SMN1', 'SMNDC1', 'SNRNP200', 'SNRNP25', 'SNRNP27', 'SNRNP35', 'SNRNP40', 'SNRNP70', 'SNRPA', 'SNRPA1', 'SNRPB', 'SNRPB2', 'SNRPC', 'SNRPD1', 'SNRPD2', 'SNRPD3', 'SNRPE', 'SNRPF', 'SNRPG', 'SNRPN', 'SNU13', 'SNW1', 'SON', 'SREK1IP1', 'SRPK1', 'SRPK3', 'SRRM1', 'SRRM2', 'SRSF1', 'SRSF10', 'SRSF11', 'SRSF2', 'SRSF3', 'SRSF5', 'SRSF7', 'SRSF8', 'SRSF9','STRAP', 'SUGP1', 'SUPT20H', 'SUPT7L', 'SYF2', 'SYNCRIP', 'TADA1', 'TADA2B', 'TADA3', 'TAF10', 'TAF12', 'TAF9', 'TARDBP', 'TFIP11', 'THOC1', 'THOC3', 'THOC7', 'TMBIM6', 'TRA2B', 'TRPT1','TSEN15', 'TSEN34', 'TXNL4A', 'U2AF1L4', 'UBL5', 'USP39', 'WBP11', 'WBP4', 'WDR77', 'WDR83', 'WTAP', 'YBX1', 'YTHDC1', 'ZCRB1', 'ZMAT5', 'ZNF830', 'ZPR1'        
    'GBX2',
'WNT3A',
'FGF8',
'PAX6',
'TBR2',
'NKX2.1',
'SOX1',
'SOX2',
'SOX3',
'POU4f1',
'EBF1',
'EBF2',
'LHX1',
'LHX5',
'BCL11B',
'SHH',
'NOTCH1',
'HES1',
]

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
    if os.path.exists(f'/data/output/tha/{gene}.png'):
        continue
    try:
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
        plt.savefig(f'/data/output/tha/{gene}.png', bbox_inches = 'tight', dpi = 600)
        plt.close()
    except:
        continue