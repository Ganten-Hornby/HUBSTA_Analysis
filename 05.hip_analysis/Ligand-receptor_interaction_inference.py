import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spateo as st
import scanpy as sc

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/Hippocampus_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()

dic_dmt_leiden = {
    '0': 'hip_sc_1',
    '7': 'hip_sc_2',
    
    '1': 'hip_sc_3',
    '3': 'hip_sc_4',
    '6': 'hip_sc_4',
    '14': 'hip_sc_5',
    
    
    '2': 'hip_sc_6',
    '13': 'hip_sc_7',
    '19': 'hip_sc_8',
    '20': 'hip_sc_9',
    
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

adata = adata[adata.obs['dmt_leiden_anno'] != 'z_delete'].copy()

adata.uns['__type'] = 'UMI'


colormap = {'hip_sc_1': '#9b38e9',
            'hip_sc_2': '#a89630',
 'hip_sc_3': '#5b798b',
 'hip_sc_4': '#cb2505',
 'hip_sc_5': '#62e7dd',
 'hip_sc_6': '#245200',
 'hip_sc_7': '#374898',
 'hip_sc_8': '#6d85c7',
 'hip_sc_9': '#35c498',
 'hip_sc_10': '#9e2dc6',
 'hip_sc_11': '#2d7476',
 'hip_sc_12': '#cb0d6c',
 'hip_sc_13': '#20ea38',
 'hip_sc_14': '#0fabb6',
 'hip_sc_15': '#a59099',
 'hip_sc_16': '#2bea3a',
 'hip_sc_17': '#17b064',
 'hip_sc_18': '#52b8d5',
            'hip_sc_19': '#da2ef2',
 'hip_sc_20': '#6240f7',
 'hip_sc_21': '#c47233',
 'hip_sc_22': '#a83b23',
            'hip_sc_23': '#9994da',}

for i in set(adata.obs['slice_code']):
    if i == 'B03607C4E6_WT2024071214941.h5ad' or i == '43_A03590E1G4_WT202403310064.h5ad':
        continue
    else:
        adata1 = adata[adata.obs['slice_code'] == i].copy()
        save = i.replace('.h5ad', '')


    _, adata1 = st.tl.neighbors(
        adata1,
        basis='spatial',
        spatial_key='spatial',
        n_neighbors=20
    )



    adata1.obs['dmt_leiden_anno'] =adata1.obs['dmt_leiden_anno'].astype('category')


    a = ['hip_sc_1', 'hip_sc_2', 'hip_sc_3', 'hip_sc_4', 'hip_sc_5', 'hip_sc_6', 'hip_sc_7', 'hip_sc_8', 'hip_sc_9', 'hip_sc_10', 
         'hip_sc_11', 'hip_sc_12', 'hip_sc_13', 'hip_sc_14', 'hip_sc_15', 'hip_sc_16', 'hip_sc_17', 'hip_sc_18', 'hip_sc_19', 'hip_sc_20', 
         'hip_sc_21', 'hip_sc_22', 'hip_sc_23']
    df = pd.DataFrame({
        "celltype_sender": np.repeat(a, len(a)),
        "celltype_receiver": list(a)*len(a),
    })
    df = df[df['celltype_sender'] != df['celltype_receiver']]
    df["celltype_pair"] = df["celltype_sender"].str.cat(
        df["celltype_receiver"], sep="-")
    df = df.reset_index(drop=True)

    db_dir = '/data/work/05.cluster/FuseMap/0116/db/'

    res = {}
    dropped = []
    for idx, i in enumerate(df['celltype_pair']):
        s, r = i.split(sep='-')
        result = st.tl.find_cci_two_group(adata1,
                                          path=db_dir,
                                          species='human',
                                          group='dmt_leiden_anno',
                                          sender_group=s,
                                          receiver_group=r,
                                          filter_lr='outer',
                                          min_pairs=0,
                                          min_pairs_ratio=0,
                                          top=20,)
        if result is not None:
            res[i] = result
        else:
            dropped.append(idx)
    result = pd.DataFrame(columns=res[df['celltype_pair'][1]]['lr_pair'].columns)
    for l in df.index:
        if l not in dropped:
            res[df['celltype_pair'][l]]['lr_pair'] = res[df['celltype_pair'][l]
                                                         ]['lr_pair'].sort_values('lr_co_exp_ratio', ascending=False)[0:3]
            result = pd.concat([result, res[df['celltype_pair'][l]]
                               ['lr_pair']], axis=0, join='outer')

    df_result = result.loc[result['lr_co_exp_num'] > 5]
    df_result.drop_duplicates(
        subset=['lr_pair', 'sr_pair', ], keep='first', inplace=True)
    df_result.to_csv(f'/data/work/05.cluster/FuseMap/0116/cci/{save}.csv')


