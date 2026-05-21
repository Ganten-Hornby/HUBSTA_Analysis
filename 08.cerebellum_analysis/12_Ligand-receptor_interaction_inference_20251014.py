import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spateo as st
import scanpy as sc
import os
adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/20250106/cerebellum_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()

# adata = adata[:, ~adata.var_names.str.startswith('MT')].copy()
adata = adata[:,[False if '.' in i else True for i in adata.var.index.tolist()]].copy()
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
adata.obs['dmt_leiden_anno'] = [dmt_leiden_merge[i] for i in adata.obs['dmt_leiden'] ]
adata = adata[adata.obs['dmt_leiden_anno']!='z_delete']
adata.uns['__type'] = 'UMI'


colormap = {'Cebe_3': '#d590d7',
 'Cebe_4': '#e31fe9',
 'Cebe_7': '#ec2426',
 'Cebe_0': '#cc6c42',
 'Cebe_1': '#708365',
 'Cebe_11': '#8980c0',
 'Cebe_5': '#6e1f94',
 'Cebe_10': '#6fbf5e',
 'Cebe_6': '#3da672',
 'Cebe_2': '#902b32',
 'Cebe_9': '#cde114',
 'Cebe_8': '#bcf19a'}

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
for name in names:
    print(name)
    
    adata1 = adata[adata.obs['slice_code'] == name].copy()
    save = name.replace('.h5ad', '')
    if os.path.exists(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/cci_20251112/{save}.csv'):
        continue
    _, adata1 = st.tl.neighbors(
        adata1,
        basis='spatial',
        spatial_key='spatial',
        n_neighbors=30
    )


    adata1.obs['dmt_leiden_anno'] =adata1.obs['dmt_leiden_anno'].astype('category')


    # a = ['Cebe_3', 'Cebe_4', 'Cebe_7', 'Cebe_0', 'Cebe_1', 'Cebe_11', 'Cebe_5', 'Cebe_10', 
    #      'Cebe_6', 'Cebe_2', 'Cebe_9', 'Cebe_8'
    #      # 'Cebe_1'
    #     ]
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
        if s == 'Cebe_6' and r == 'Cebe_8':
            result = None
        elif s == 'Cebe_8' and r == 'Cebe_6':
            result = None
        elif s == 'Cebe_3' and r == 'Cebe_10':
            result = None
        elif s == 'Cebe_10' and r == 'Cebe_3':
            result = None
        elif s == 'Cebe_3' and r == 'Cebe_0':
            result = None
        elif s == 'Cebe_0' and r == 'Cebe_3':
            result = None
        elif s == 'Cebe_3' and r == 'Cebe_6':
            result = None
        elif s == 'Cebe_6' and r == 'Cebe_3':
            result = None
        # elif s == 'Cebe_3' and r == 'Cebe_10' and name == '31_B03619A1D3_WT202403310052.h5ad':
            # result = None
        # elif s == 'Cebe_10' and r == 'Cebe_3' and name == '31_B03619A1D3_WT202403310052.h5ad':
            # result = None
        else:
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
    df_result.to_csv(f'/data/work/05.cluster/FuseMap/20251103/3_cerebellum_spatial/cci_20251112/{save}.csv')


