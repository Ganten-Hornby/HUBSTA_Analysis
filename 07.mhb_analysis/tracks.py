import spaTrack as spt
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

sc.settings.verbosity = 0

adata_all = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/cerebellum_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
annotation = {
    '0': 'Cere_sc_01',
    '8': 'Cere_sc_01',
    
    '1': 'Cere_sc_02',
    
    '2': 'Cere_sc_03',
    '18': 'Cere_sc_03',
    
    '3': 'Cere_sc_04',
    
    '4': 'Cere_sc_05',
    '32': 'Cere_sc_05',
    '5': 'Cere_sc_06',

    '6': 'Cere_sc_07',
    
    '7': 'Cere_sc_07',
    '27': 'Cere_sc_07',
    
    '9': 'Cere_sc_08',
    '13': 'Cere_sc_08',
    '15': 'Cere_sc_08',
    '22': 'Cere_sc_08',
    '25': 'Cere_sc_08',
    '29': 'Cere_sc_08',
    
    
    '10': 'Cere_sc_09',

    '11': 'Cere_sc_10',
    
    '12': 'Cere_sc_11',
    

    '14': 'Cere_sc_12',
 

    '16': 'Cere_sc_13',

    '17': 'Cere_sc_14',
    '19': 'Cere_sc_14',

    '20': 'Cere_sc_15',
    '24': 'Cere_sc_15',
    '26': 'Cere_sc_15',
    
    '21': 'Cere_sc_16',

    

    '23': 'Cere_sc_17',

    '28': 'z_delete',

    '30': 'Cere_sc_18',

    '31': 'Cere_sc_19',
}
adata_all.obs['dmt_leiden_anno'] = [annotation[i] for i in adata_all.obs['dmt_leiden']]
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

for slice_code in adata.obs['slice_code']:
    adata = adata_all[adata_all.obs['slice_code'] == slice_code]
    adata.X = adata.X.astype(int)
    adata=assess_start_cluster(adata, 'dmt_leiden_anno')
    adata.obs['cluster'] = adata.obs['dmt_leiden_anno'].copy()
    adata.obsm['align_spatial_2d'][:,1] = -adata.obsm['align_spatial_2d'][:,1]
    
    adata.obsm['X_spatial'] = adata.obsm['align_spatial_2d'].copy()
    adata.obsm['X_align_spatial_2d'] = adata.obsm['align_spatial_2d'].copy()
    adata.obsm['X_pca'] = adata.obsm['X_dmt_highdim'].copy()
    start_cells = spt.set_start_cells(adata, select_way='cell_type', cell_type='Cere_sc_07', )
    adata.obsp["trans"] = spt.get_ot_matrix(adata, data_type="spatial",alpha1=0.3,alpha2=0.7)
    adata.obs["ptime"] = spt.get_ptime(adata, start_cells)
    adata.uns["E_grid"], adata.uns["V_grid"] = spt.get_velocity(adata, basis="align_spatial_2d", n_neigh_pos=100,n_neigh_gene=0)


    adata.write(f'/data/work/05.cluster/FuseMap/0314/tracks/adata/{slice_code}')
