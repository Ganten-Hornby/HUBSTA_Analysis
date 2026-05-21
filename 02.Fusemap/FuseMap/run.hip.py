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

names = [
'12_B03605F3G5_WT202403310048.h5ad',
'13_B03612A1C3_WT202403310056.h5ad',
'14_A03591A1C3_WT202403310045.h5ad',
'16_A03592A4C6_WT202403310044.h5ad',
'18_B03602C4D6_WT202405020031.h5ad',
'20_B03606F3G5_WT202405020032.h5ad',
'22_B03606C4E6_WT202403310050.h5ad',
'23_B03609A4D6_WT202404150263.h5ad',
'27_B03610C1E3_WT202403310051.h5ad',
'31_B03619A1D3_WT202403310052.h5ad',
'35_B03619E4G6_WT202403310053.h5ad',
'39_A03589A1D4_WT202403310046.h5ad',
'43_A03590E1G4_WT202403310064.h5ad',
'47_A03593C1F3_WT202403310068.h5ad',
'B03607C4E6_WT2024071214941.h5ad',
]

X_input=[]
raw_path = '/data/work/05.cluster/FuseMap/1130/datas/Hippocampus_preprocess/'
for name in names:
    adata = sc.read_h5ad(raw_path+name)
    adata.obs['name']=f'{name}'
    X_input.append(adata)

kneighbor = ['delaunay']*len(X_input)
input_identity = ['ST']*len(X_input)
save_dir = f"/data/work/05.cluster/FuseMap/1130/for_fusemap/Hippocampus"
network = main(X_input, save_dir, kneighbor, input_identity, preprocess_save = True)