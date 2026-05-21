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

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/thalamus_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
dic = {'0': 'T7_SST_LHX_SIX3',
 '1': 'T9_CRH',
 '10': 'T6_CLSTN2',
 '11': 'T4_COL4A1_vessel',
 '12': 'T3_BMP4_IGFBP5',
 '13': 'T7_SST_LHX_SIX3',
 '14': 'T0_PCP4_VGF',
 '15': 'T9_CRH',
 '16': 'T3_BMP4_IGFBP5',
 '17': 'T3_BMP4_IGFBP5',
 '18': 'T2_PRSS12',
 '19': 'T9_CRH',
 '2': 'T9_CRH',
 '20': 'T1_NTS_CALB1',
 '21': 'T1_NTS_CALB1',
 '22': 'T1_NTS_CALB1',
 '23': 'T10_HBZ',
 '24': 'T1_NTS_CALB1',
 '25': 'T1_NTS_CALB1',
 '26': 'T2_PRSS12',
 '27': 'T9_CRH',
 '28': 'T2_PRSS12',
 '29': 'T5_AQP4_gial_region',
 '3': 'T10_HBZ',
 '30': 'T8_CBLN4_CBLN1',
 '31': 'T7_SST_LHX_SIX3',
 '32': 'T9_CRH',
 '33': 'T1_NTS_CALB1',
 '34': 'T7_SST_LHX_SIX3',
 '35': 'T10_HBZ',
 '36': 'T8_CBLN4_CBLN1',
 '37': 'T9_CRH',
 '38': 'T3_BMP4_IGFBP5',
 '39': 'T1_NTS_CALB1',
 '4': 'T2_PRSS12',
 '40': 'T9_CRH',
 '41': 'T6_CLSTN2',
 '42': 'T10_HBZ',
 '5': 'T7_SST_LHX_SIX3',
 '6': 'T7_SST_LHX_SIX3',
 '7': 'T9_CRH',
 '8': 'T2_PRSS12',
 '9': 'T1_NTS_CALB1'}

adata.obs['dmt_leiden_annotation_0115'] = [dic[i] for i in adata.obs['dmt_leiden']]
del adata.obsm
del adata.obsp
del adata.uns
adata.write('/data/work/05.cluster/FuseMap/0117/tha_deg_plot/tha_sp.h5ad')
