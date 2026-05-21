import spaTrack as spt
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity = 0

adata_all = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/cerebellum_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
adata_all = adata_all[:, ~adata_all.var_names.str.startswith('MT')]
adata_all = adata_all[:, ~adata_all.var_names.str.startswith('-')]
adata_all = adata_all[:, adata_all.var_names != '']
adata_all.obs_names_make_unique()
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
adata_all.obs['dmt_leiden_anno'] = [dmt_leiden_merge[i] for i in adata_all.obs['dmt_leiden'] ]
adata_all = adata_all[adata_all.obs['dmt_leiden_anno']!='z_delete']
adata_all.obs['dmt_leiden_anno'] = adata_all.obs['dmt_leiden_anno'].astype('category')


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
    if os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/4_tracks/spatial_20251104_1/{slice_code}'):
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
    start_cells = np.where(adata.obs["cluster"].isin(['Cebe_7', 'Cebe_11']))[0]
    start_cells = list(start_cells)
    adata.obsp["trans"] = spt.get_ot_matrix(adata, data_type="spatial",alpha1=0.3,alpha2=0.7)
    adata.obs["ptime"] = spt.get_ptime(adata, start_cells)
    adata.uns["E_grid"], adata.uns["V_grid"] = spt.get_velocity(adata, basis="align_spatial_2d", n_neigh_pos=100,n_neigh_gene=0)
    
    # adata.write(f'/data/work/05.cluster/FuseMap/0314/tracks/adata/{slice_code}')
    
    del adata.obsp
    adata.X = csr_matrix(adata.X)
    adata.write(f'/data/work/05.cluster/FuseMap/20251103/4_tracks/spatial_20251104_1/{slice_code}')
    

