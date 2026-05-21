import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spateo as st
import scanpy as sc

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/mid_hind_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()
adata = adata[:,[False if '.' in i else True for i in adata.var.index.tolist()]]

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


adata.uns['__type'] = 'UMI'


colormap = {'mh_sc_29': '#a3dc1f',
 'mh_sc_00': '#5ef377',
 'mh_sc_02': '#654c0d',
 'mh_sc_27': '#aea266',
 'mh_sc_03': '#e4f36b',
 'mh_sc_23': '#2a74d1',
 'mh_sc_06': '#1426db',
 'mh_sc_25': '#e93f06',
 'mh_sc_16': '#8caec2',
 'mh_sc_28': '#db3529',
 'mh_sc_18': '#215789',
 'mh_sc_22': '#490ea7',
 'mh_sc_21': '#7a7bc3',
 'mh_sc_14': '#0d204c',
 'mh_sc_19': '#4ae86f',
 'mh_sc_20': '#efb6a4',
 'mh_sc_04': '#f55fb7',
 'mh_sc_13': '#ccb347',
 'mh_sc_26': '#b36428',
 'mh_sc_09': '#efea60',
 'mh_sc_17': '#30fac0',
 'mh_sc_12': '#cee3d7',
 'mh_sc_08': '#8a1ecb',
 'mh_sc_07': '#607f79',
 'mh_sc_15': '#3a6152',
 'mh_sc_24': '#91a6de',
 'mh_sc_10': '#779d25',
 'mh_sc_11': '#524ae3',
 'mh_sc_05': '#1164b8',
 'mh_sc_01': '#1f8071'}


for i in set(adata.obs['slice_code']):

    
    adata1 = adata[adata.obs['slice_code'] == i].copy()
    save = i.replace('.h5ad', '')


    _, adata1 = st.tl.neighbors(
        adata1,
        basis='spatial',
        spatial_key='spatial',
        n_neighbors=20
    )


    adata1.obs['dmt_leiden_anno'] =adata1.obs['dmt_leiden_anno'].astype('category')


    a = ['mh_sc_00', 'mh_sc_01', 'mh_sc_02', 'mh_sc_03', 'mh_sc_04', 'mh_sc_05', 'mh_sc_06', 'mh_sc_07', 
    'mh_sc_08', 'mh_sc_09', 'mh_sc_10', 'mh_sc_11', 'mh_sc_12', 'mh_sc_13', 'mh_sc_14', 'mh_sc_15', 'mh_sc_16', 
    'mh_sc_17', 'mh_sc_18', 'mh_sc_19', 'mh_sc_20', 'mh_sc_21', 'mh_sc_22', 'mh_sc_23', 'mh_sc_24', 'mh_sc_25', 'mh_sc_26', 'mh_sc_27', 'mh_sc_28', 'mh_sc_29']
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
    df_result.to_csv(f'/data/work/05.cluster/FuseMap/0313/cci/{save}.csv')


