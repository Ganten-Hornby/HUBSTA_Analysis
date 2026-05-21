import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spateo as st
import scanpy as sc

adata = sc.read_h5ad('/data/work/05.cluster/FuseMap/0106/thalamus_latent_embeddings_all_spatial_pretrain/dmt_leiden_20250108_1.h5ad')
adata.obs_names_make_unique()


dic = {'0': 'T7_SST_LHX_SIX3',
 '1': 'T9_CRH',
 '10': 'T6_CLSTN2',
 '11': 'T4_COL4A1_vessel',
 '12': 'T3_BMP4_IGFBP5',
 '13': 'T7_SST_LHX_SIX3',
 '14': 'T0_PCP4_VGF',
 '15': 'T9_CRH',
 '16': 'T3_BMP4_IGFBP5',
 '17': 'T3_BMP4_IGFBP5',
 '18': 'T2_PRSS12',
 '19': 'T9_CRH',
 '2': 'T9_CRH',
 '20': 'T1_NTS_CALB1',
 '21': 'T1_NTS_CALB1',
 '22': 'T1_NTS_CALB1',
 '23': 'T10_HBZ',
 '24': 'T1_NTS_CALB1',
 '25': 'T1_NTS_CALB1',
 '26': 'T2_PRSS12',
 '27': 'T9_CRH',
 '28': 'T2_PRSS12',
 '29': 'T5_AQP4_gial_region',
 '3': 'T10_HBZ',
 '30': 'T8_CBLN4_CBLN1',
 '31': 'T7_SST_LHX_SIX3',
 '32': 'T9_CRH',
 '33': 'T1_NTS_CALB1',
 '34': 'T7_SST_LHX_SIX3',
 '35': 'T10_HBZ',
 '36': 'T8_CBLN4_CBLN1',
 '37': 'T9_CRH',
 '38': 'T3_BMP4_IGFBP5',
 '39': 'T1_NTS_CALB1',
 '4': 'T2_PRSS12',
 '40': 'T9_CRH',
 '41': 'T6_CLSTN2',
 '42': 'T10_HBZ',
 '5': 'T7_SST_LHX_SIX3',
 '6': 'T7_SST_LHX_SIX3',
 '7': 'T9_CRH',
 '8': 'T2_PRSS12',
 '9': 'T1_NTS_CALB1'}

adata.obs['dmt_leiden_annotation_0115'] = [dic[i] for i in adata.obs['dmt_leiden']]
colormap = {'T2_PRSS12': '#31d6d3',
 'T6_CLSTN2': '#6b0c4d',
 'T3_BMP4_IGFBP5': '#e94a1d',
 'T4_COL4A1_vessel': '#cf58e5',
 'T10_HBZ': '#39d789',
 'T7_SST_LHX_SIX3': '#4e7c26',
 'T8_CBLN4_CBLN1': '#eaccd8',
 'T1_NTS_CALB1': '#9114fb',
 'T9_CRH': '#79f4ec',
 'T0_PCP4_VGF': '#b3fcdd',
 'T5_AQP4_gial_region': '#455edf'}


adata.uns['__type'] = 'UMI'


adatas_list = [
    '43_A03590E1G4_WT202403310064.h5ad', # GW 13
    'A03988A1C2_WT202407161208.h5ad', # GW13_rep_3
 'B03618D3F6_WT202407152793.h5ad', # GW16
 # 'A03591D4E5_WT2024071215074.h5ad',
 'A03590A3D6_WT202407192652.h5ad', # GW13_rep_1
 'A03587A5C6_WT2024071215080.h5ad', # GW 10
 'A03994F1G2_WT2024071215067.h5ad', # GW13_rep_2
 'B03607C4E6_WT2024071214941.h5ad', # GW12
]

DDIICC = {
    '43_A03590E1G4_WT202403310064.h5ad': 'GW13',
    'A03988A1C2_WT202407161208.h5ad': 'GW13_rep_3',
 'B03618D3F6_WT202407152793.h5ad': 'GW16',
 # 'A03591D4E5_WT2024071215074.h5ad',
 'A03590A3D6_WT202407192652.h5ad': 'GW13_rep_1',
 'A03587A5C6_WT2024071215080.h5ad': 'GW10',
 'A03994F1G2_WT2024071215067.h5ad': 'GW13_rep_2',
 'B03607C4E6_WT2024071214941.h5ad': 'GW12',

    
}


for li_i in adatas_list:
    li = DDIICC[li_i]
    adata1 = adata[adata.obs['slice_code'] == li_i].copy()
    


    _, adata1 = st.tl.neighbors(
        adata1,
        basis='spatial',
        spatial_key='spatial',
        n_neighbors=20
    )

    adata1.obs['dmt_leiden_annotation_0115'] = adata1.obs['dmt_leiden_annotation_0115'].astype('category')

    a = ['T0_PCP4_VGF', 'T10_HBZ', 'T1_NTS_CALB1', 'T2_PRSS12', 
         'T3_BMP4_IGFBP5', 'T4_COL4A1_vessel', 'T5_AQP4_gial_region', 
         'T6_CLSTN2', 'T7_SST_LHX_SIX3', 'T8_CBLN4_CBLN1', 'T9_CRH']
    
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
                                          group='dmt_leiden_annotation_0115',
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
    df_result.to_csv(f'/data/work/05.cluster/FuseMap/0117/cci/{li}.csv')

