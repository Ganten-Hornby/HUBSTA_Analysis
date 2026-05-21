import spaTrack as spt
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity = 0

adata_all = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/cerebellum_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
adata_all = adata_all[:, ~adata_all.var_names.str.startswith('MT')]
adata_all = adata_all[:, ~adata_all.var_names.str.startswith('-')]
adata_all = adata_all[:, adata_all.var_names != '']
adata_all.obs_names_make_unique()

single_dic = {
    '0': 'Cere_sc_1',
    '28': 'Cere_sc_1',
    '27': 'Cere_sc_2',
    
    '1': 'Cere_sc_3',
    '8': 'Cere_sc_3',
    '21': 'Cere_sc_4',
    '19': 'Cere_sc_5',
    
    '2': 'Cere_sc_6',
    '9': 'Cere_sc_7',
    '11': 'Cere_sc_7',
    '12': 'Cere_sc_8',
    '14': 'Cere_sc_9',
    
    '3': 'Cere_sc_10',
    '26': 'Cere_sc_10',
    '17': 'Cere_sc_11',
    '15': 'Cere_sc_12',
    '18': 'Cere_sc_13',
    
    '4': 'z_delete',
    
    '5': 'Cere_sc_14',
    '16': 'Cere_sc_14',
    '30': 'Cere_sc_15',
    '20': 'Cere_sc_16',
    
    '6': 'Cere_sc_17',
    '10': 'Cere_sc_18',
    
    '7': 'Cere_sc_19',
    '25': 'Cere_sc_20',
    '23': 'Cere_sc_21',
    
    '13': 'z_delete',

    '22': 'z_delete',
    
    '24': 'Cere_sc_22',
    '31': 'Cere_sc_22',
    '29': 'Cere_sc_22',
    '32': 'z_delete',
}
adata_all.obs['dmt_leiden_anno'] = [single_dic[i] if i in single_dic.keys() else i for i in adata_all.obs['dmt_leiden']]
# adata.obs['dmt_leiden_anno'] = adata.obs['dmt_leiden_anno'].astype('category')
adata_all = adata_all[adata_all.obs['dmt_leiden_anno'] != 'z_delete']


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
    #display(df_entropy)
    return adata

import os
for slice_code in set(adata_all.obs['slice_code']):
    if os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/4_tracks/single_20251104_1/{slice_code}'):
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
    adata.write(f'/data/work/05.cluster/FuseMap/20251103/4_tracks/single_20251104_1/{slice_code}')
    

