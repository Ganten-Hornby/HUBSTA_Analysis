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

fontManager.addfont('/data/work/Arial.ttf')

font = FontProperties(fname='/data/work/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/Hippocampus_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
dic_dmt_leiden = {
    '0': 'hip_sc_01',
    '7': 'hip_sc_02',
    
    '1': 'hip_sc_03',
    '3': 'hip_sc_04',
    '6': 'hip_sc_04',
    '14': 'hip_sc_05',
    
    
    '2': 'hip_sc_06',
    '13': 'hip_sc_07',
    '19': 'hip_sc_08',
    '20': 'hip_sc_09',
    
    '4': 'hip_sc_10',
    '21': 'hip_sc_10',
    
    '8': 'hip_sc_11',
    '10': 'hip_sc_12',
    '22': 'hip_sc_13',
    
    '9': 'hip_sc_14',
    '11': 'hip_sc_14',
    '16': 'hip_sc_15',
    '18': 'hip_sc_15',
    '26': 'hip_sc_15',
    '31': 'hip_sc_15',
    
    '12': 'hip_sc_16',
    '15': 'hip_sc_17',
    '25': 'hip_sc_17',
    '29': 'hip_sc_17',
    '17': 'hip_sc_18',
    
    '24': 'z_delete',
    
    
    '5': 'hip_sc_19',
    '23': 'hip_sc_20',
    '27': 'hip_sc_21',
    '28': 'hip_sc_22',
    '30': 'hip_sc_23',
}
adata.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata.obs['dmt_leiden']]
adata = adata[adata.obs['dmt_leiden_anno'] != 'z_delete']
del adata.obsm
del adata.obsp
del adata.uns
adata.write('/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sc.h5ad')

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/Hippocampus_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
dic = {
    '0': 'Hip0_COLA4A1_vessel',
    '1': 'Hip1_ZBTB20_PCDH20_allo1',
    '2': 'Hip2_PPP1R17_ELMO3_SVZ',
    '3': 'Hip3_NTS_allo2',
    '4': 'Hip4_AC144831.1',
    '5': 'Hip5_POU3F1_IZ',
    '6': 'Hip2_PPP1R17_ELMO3_SVZ',
    '7': 'Hip7_WNT7B_GPR22_CP&SP',
    '8': 'Hip4_AC144831.1',
    '9': 'Hip7_WNT7B_GPR22_CP&SP',
    '10': 'Hip5_POU3F1_IZ',
    '11': 'Hip8_HOPX_FABP7_VZ',
    '12': 'Hip8_HOPX_FABP7_VZ',
    '13': 'Hip5_POU3F1_IZ',
    '14': 'Hip8_HOPX_FABP7_VZ',
    '15': 'Hip6_COL4A1_vessel',
    '16': 'Hip5_POU3F1_IZ',
    '17': 'Hip1_ZBTB20_PCDH20_allo1',
    '18': 'Hip8_HOPX_FABP7_VZ',
    '19': 'Hip7_WNT7B_GPR22_CP&SP',
    '20': 'Hip8_HOPX_FABP7_VZ',
    '21': 'Hip4_AC144831.1',
    '22': 'Hip1_ZBTB20_PCDH20_allo1',
    '23': 'Hip8_HOPX_FABP7_VZ',
    '24': 'Hip2_PPP1R17_ELMO3_SVZ',
    '25': 'Hip4_AC144831.1',
    '26': 'Hip9_PPP1R17_EOMES_SVZ',
    '27': 'Hip4_AC144831.1',
    '28': 'Hip1_ZBTB20_PCDH20_allo1',
    '29': 'Hip7_WNT7B_GPR22_CP&SP',
    '30': 'Hip8_HOPX_FABP7_VZ',
    '31': 'Hip8_HOPX_FABP7_VZ',
    '32': 'Hip1_ZBTB20_PCDH20_allo1',
    '33': 'Hip3_NTS_allo2',
    '34': 'Hip10_COL4A1_vessel',
    '35': 'NA',
    '36': 'NA',
    '37': 'Hip1_ZBTB20_PCDH20_allo1',
    '38': 'Hip7_WNT7B_GPR22_CP&SP',
    '39': 'Hip7_WNT7B_GPR22_CP&SP',
    '40': 'Hip7_WNT7B_GPR22_CP&SP',
}

adata.obs['dmt_leiden_merge'] = [dic[i] for i in adata.obs['dmt_leiden']]
adata = adata[adata.obs['dmt_leiden_merge'] != 'NA'].copy()
del adata.obsm
del adata.obsp
del adata.uns
adata.write('/data/work/05.cluster/FuseMap/0116/hip_deg_plot/hip_sp.h5ad')