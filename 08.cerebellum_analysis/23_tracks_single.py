import spaTrack as spt
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity = 0

adata = sc.read_h5ad('/data/input/Files/ResultData/Notebook/NB2026022610032617570349/cerebellum_scAnno_20260226.h5ad')
adata.obs_names_make_unique()

adata = adata[:,[False if '.' in i else True for i in adata.var.index.tolist()]].copy()

adata = adata[adata.obs['dmt_leiden_anno'] != 'z_delete']


import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import seaborn as sns
import warnings
from IPython.display import display

def assess_start_cluster(adata, obs_key):
    cluster_name_list=list()
    entropy_list=list()
    cell_name_list=list()
    from scipy.sparse import issparse

    if issparse(adata.X):
        adata.X = adata.X.toarray()
    for i in range(0,len(adata.obs.index)):
        cell_id=adata.obs.index[i]
        cluster_name=adata.obs[obs_key][i]
        adata_cluster=adata[adata.obs.index.isin([cell_id])]
        matrix = np.array(adata_cluster.X)
        entropy_name=entropy(matrix[0])
        cluster_name_list.append(cluster_name)
        cell_name_list.append(cell_id)
        entropy_list.append(entropy_name)
    df_value=pd.DataFrame({'cluster_name': cluster_name_list,'cell_name':cell_name_list,'entropy':entropy_list})
    adata.obs['entropy']=df_value['entropy'].values
    df_obs=adata.obs
    cluster_order=list(pd.DataFrame(df_obs[[obs_key,'entropy']].groupby([obs_key]).mean()).sort_values(['entropy'],ascending=False).index)
    df_entropy=pd.DataFrame(df_obs[[obs_key,'entropy']].groupby([obs_key]).mean()).sort_values(['entropy'],ascending=False)
    df_obs[obs_key]= pd.Categorical(df_obs[obs_key], categories=cluster_order, ordered=True)
    print('Cluster order sorted by entropy value: ',list(df_entropy.index))
    df_entropy.index=list(df_entropy.index)
    adata.uns['entropy value']=df_obs
    adata.uns['entropy value order']=df_entropy
    return adata

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

import os
for slice_code in names:
    if os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/4_tracks/single_20260226_1/{slice_code}'):
        continue
    adata = adata_all[adata_all.obs['slice_code'] == slice_code].copy()
    print(slice_code)
    print(adata.shape)
    adata.X = adata.X.astype(int)
    adata=assess_start_cluster(adata, 'dmt_leiden_anno')
    adata.obs['cluster'] = adata.obs['dmt_leiden_anno'].copy()
    adata.obsm['align_spatial_2d'][:,1] = -adata.obsm['align_spatial_2d'][:,1]
    adata.obsm['X_spatial'] = adata.obsm['align_spatial_2d'].copy()
    adata.obsm['X_align_spatial_2d'] = adata.obsm['align_spatial_2d'].copy()
    adata.obsm['X_pca'] = adata.obsm['X_dmt_highdim'].copy()

    start_cells = np.where(adata.obs["cluster"].isin(['Cere_sc_21', 'Cere_sc_2', 'Cere_sc_20']))[0]
    start_cells = list(start_cells)
    adata.obsp["trans"] = spt.get_ot_matrix(adata, data_type="spatial",alpha1=0.3,alpha2=0.7)
    adata.obs["ptime"] = spt.get_ptime(adata, start_cells)
    adata.uns["E_grid"], adata.uns["V_grid"] = spt.get_velocity(adata, basis="align_spatial_2d", n_neigh_pos=100,n_neigh_gene=0)

    del adata.obsp
    adata.X = csr_matrix(adata.X)
    if os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/4_tracks/single_20260226_1/{slice_code}'):
        continue
    adata.write(f'/data/work/05.cluster/FuseMap/20251103/4_tracks/single_20260226_1/{slice_code}')
    

