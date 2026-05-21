import spaTrack as spt
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity = 0

adata_all = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/Hippocampus_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
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
adata_all.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata_all.obs['dmt_leiden']]
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
    if os.path.exists(f'/data/work/05.cluster/FuseMap/0116/tracks/data_20240402/{slice_code}'):
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
    start_cells = spt.set_start_cells(adata, select_way='cell_type', cell_type='hip_sc_21', )
    
    
    adata.obsp["trans"] = spt.get_ot_matrix(adata, data_type="spatial",alpha1=0.3,alpha2=0.7)
    adata.obs["ptime"] = spt.get_ptime(adata, start_cells)
    adata.uns["E_grid"], adata.uns["V_grid"] = spt.get_velocity(adata, basis="align_spatial_2d", n_neigh_pos=100,n_neigh_gene=0)

    # adata.write(f'/data/work/05.cluster/FuseMap/0314/tracks/adata/{slice_code}')
    
    del adata.obsp
    adata.X = csr_matrix(adata.X)
    adata.write(f'/data/work/05.cluster/FuseMap/0116/tracks/data_20240402/{slice_code}')

