import numba
numba.set_num_threads(15) # 如果适用的话，避免多线程问题
numba.config.DISABLE_JIT = True # 禁用 JIT 编译
import scanpy as sc
from train import *
import scanpy as sc
import os
import pandas as pd
os.chdir('/data/work/05.cluster/FuseMap/1130/for_fusemap/')
print(os.getcwd())
seed_all(42)

import os

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
'A03587A5C6_WT2024071215080.h5ad',
'A03588A1C2_WT202407161185.h5ad',
'A03590A3D6_WT202407192652.h5ad',
'A03591D4E5_WT2024071215074.h5ad',
'A03988A1C2_WT202407161208.h5ad',
'A03994F1G2_WT2024071215067.h5ad',
'B03607C4E6_WT2024071214941.h5ad',
'B03618D3F6_WT202407152793.h5ad',]

X_input=[]
raw_path = '/data/work/05.cluster/FuseMap/1130/datas/cerebellums_preprocess/'
for name in names:
    adata = sc.read_h5ad(raw_path+name)
    adata.obs['name']=f'{name}'
    X_input.append(adata)
kneighbor = ['delaunay']*len(X_input)
input_identity = ['ST']*len(X_input)
save_dir = f"/data/work/05.cluster/FuseMap/1130/for_fusemap/cerebellums"
network = main(X_input, save_dir, kneighbor, input_identity,preprocess_save = True)