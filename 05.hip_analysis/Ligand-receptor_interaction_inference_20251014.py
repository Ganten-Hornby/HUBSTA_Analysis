import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spateo as st
import scanpy as sc

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/Hippocampus_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()

dic_dmt_leiden = {
   
    
    '2': 'ctx_sc_01',
    '13': 'ctx_sc_02',
    '19': 'ctx_sc_03',
    '20': 'ctx_sc_04',
    
    '4': 'ctx_sc_05',
    '21': 'ctx_sc_05',
    
    '8': 'ctx_sc_06',
    '10': 'ctx_sc_07',
    '22': 'ctx_sc_08',
    
    '9': 'ctx_sc_09',
    '11': 'ctx_sc_09',
    '16': 'ctx_sc_10',
    '18': 'ctx_sc_10',
    '26': 'ctx_sc_10',
    '31': 'ctx_sc_10',
    
    '12': 'ctx_sc_11',
    '15': 'ctx_sc_12',
    '25': 'ctx_sc_12',
    '29': 'ctx_sc_12',
    '17': 'ctx_sc_13',
    
    '24': 'z_delete',
    
    
    '5': 'ctx_sc_14',
    '23': 'ctx_sc_15',
    '27': 'ctx_sc_16',
    
    '30': 'ctx_sc_17',
    
     '0': 'hip_sc_18',
    '7': 'hip_sc_19',
    
    '1': 'hip_sc_20',
    '3': 'hip_sc_21',
    '6': 'hip_sc_21',
    '14': 'hip_sc_22',
    '28': 'hip_sc_23',
}

adata.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata.obs['dmt_leiden']]

adata = adata[adata.obs['dmt_leiden_anno'] != 'z_delete'].copy()
adata = adata[adata.obs['dmt_leiden_anno'] != 'ctx_sc_05'].copy()
adata.uns['__type'] = 'UMI'


colormap = {
 
  'ctx_sc_01' : '#374898',
  'ctx_sc_02' : '#6d85c7',
  'ctx_sc_03' : '#35c498',
  'ctx_sc_04' : '#9e2dc6',
  'ctx_sc_05' : '#2d7476',
  'ctx_sc_06' : '#cb0d6c',
  'ctx_sc_07' : '#20ea38',
  'ctx_sc_08' : '#0fabb6',
  'ctx_sc_09' : '#a59099',
  'ctx_sc_10' : '#2bea3a',
  'ctx_sc_11' : '#17b064',
  'ctx_sc_12' : '#52b8d5',
  'ctx_sc_13' : '#da2ef2',
  'ctx_sc_14' : '#6240f7',
  'ctx_sc_15' : '#c47233',
  'ctx_sc_16':'#a83b23',
  'ctx_sc_17':'#9994da',
  'hip_sc_18' : '#9b38e9',
  'hip_sc_19' : '#a89630',
  'hip_sc_20': '#5b798b',
  'hip_sc_21' : '#cb2505',
  'hip_sc_22' : '#62e7dd',
  'hip_sc_23' : '#245200',
}

for i in set(adata.obs['slice_code']):
    if i == 'B03607C4E6_WT2024071214941.h5ad':# or i == '43_A03590E1G4_WT202403310064.h5ad':
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


    a = ['ctx_sc_01', 'ctx_sc_02', 'ctx_sc_03', 'ctx_sc_04', 
         # 'ctx_sc_05', 
         'ctx_sc_06', 'ctx_sc_07', 'ctx_sc_08', 'ctx_sc_09', 'ctx_sc_10', 
         'ctx_sc_11', 'ctx_sc_12', 'ctx_sc_13', 'ctx_sc_14', 'ctx_sc_15', 'ctx_sc_16', 'ctx_sc_17', 'hip_sc_18', 'hip_sc_19', 'hip_sc_20', 
         'hip_sc_21', 'hip_sc_22', 'hip_sc_23']
    df = pd.DataFrame({
        "celltype_sender": np.repeat(a, len(a)),
        "celltype_receiver": list(a)*len(a),
    })
    df = df[df['celltype_sender'] != df['celltype_receiver']]
    df["celltype_pair"] = df["celltype_sender"].str.cat(
        df["celltype_receiver"], sep="-")
    df = df.reset_index(drop=True)

    db_dir = '/data/work/05.cluster/FuseMap/20250116/db/'

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
    df_result.to_csv(f'/data/work/05.cluster/FuseMap/20250116/cci_20251014/{save}.csv')


