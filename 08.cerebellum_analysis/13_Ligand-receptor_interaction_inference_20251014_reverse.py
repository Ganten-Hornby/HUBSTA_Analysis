import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spateo as st
import scanpy as sc

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/cerebellum_latent_embeddings_all_single_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()

# adata = adata[:, ~adata.var_names.str.startswith('MT')].copy()
adata = adata[:,[False if '.' in i else True for i in adata.var.index.tolist()]].copy()
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
adata.obs['dmt_leiden_anno'] = [single_dic[i] if i in single_dic.keys() else i for i in adata.obs['dmt_leiden']]
# adata.obs['dmt_leiden_anno'] = adata.obs['dmt_leiden_anno'].astype('category')
adata = adata[adata.obs['dmt_leiden_anno'] != 'z_delete']
adata.uns['__type'] = 'UMI'


colormap = {'Cere_sc_22': '#b720f0',
 'Cere_sc_1': '#9ee6fe',
 'Cere_sc_2': '#84e0ee',
 'Cere_sc_14': '#405312',
 'Cere_sc_20': '#1af63c',
 'Cere_sc_7': '#51346b',
 'Cere_sc_19': '#97a50d',
 'Cere_sc_23': '#cdf318',
 'Cere_sc_15': '#728cfd',
 'Cere_sc_5': '#65e03c',
 'Cere_sc_4': '#cb6780',
 'Cere_sc_9': '#b572a1',
 'Cere_sc_13': '#51e78f',
 'Cere_sc_8': '#89c066',
 'Cere_sc_10': '#8163f4',
 'Cere_sc_24': '#c517d2',
 'Cere_sc_17': '#f30494',
 'z_delete': '#557eae',
 'Cere_sc_12': '#c93c35',
 'Cere_sc_21': '#10650a',
 'Cere_sc_3': '#aae898',
 'Cere_sc_16': '#505238',
 'Cere_sc_18': '#1f609e',
 'Cere_sc_6': '#f216bd',
 'Cere_sc_11': '#8945d4'}

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
names.reverse()
import os
for i in names:
    save = i.replace('.h5ad', '')
    if os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/2_cerebellum_single/cci_20251112/{save}.csv'):
        continue
    adata1 = adata[adata.obs['slice_code'] == i].copy()
    

    _, adata1 = st.tl.neighbors(
        adata1,
        basis='spatial',
        spatial_key='spatial',
        n_neighbors=30
    )


    adata1.obs['dmt_leiden_anno'] =adata1.obs['dmt_leiden_anno'].astype('category')


    a = list(colormap.keys())
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
        try:
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
        except:
            result = None
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
    df_result.to_csv(f'/data/work/05.cluster/FuseMap/20251103/2_cerebellum_single/cci_20251112/{save}.csv')


