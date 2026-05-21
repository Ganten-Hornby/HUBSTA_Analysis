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
'''
adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/mid_hind_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()

del adata.obsm
del adata.obsp
del adata.uns
adata = adata[adata.obs.sample(frac = 0.3).index].copy()
adata.write('/data/work/05.cluster/FuseMap/0313/mh_deg_plot/hm_sp.h5ad')
'''
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
adata = adata[adata.obs['dmt_leiden_anno'] != 'mh_sc_30']

del adata.obsm
del adata.obsp
del adata.uns
adata = adata[adata.obs.sample(frac = 0.3).index].copy()
adata.write('/data/work/05.cluster/FuseMap/0313/mh_deg_plot/sc/hm_sc.h5ad')