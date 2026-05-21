## packages loading

import tangram as tg
import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import re
parser = argparse.ArgumentParser()

parser.add_argument('--sc_data_path', type=str, help='Path to the sc_data file path')
parser.add_argument('--sc_label', type=str, help='Path to the sc_data laebl')
parser.add_argument('--st_data_path', type=str, help='Path to the st_data file path')
parser.add_argument('--output_path', type=str, help='Path to the output path')

## data loading
args = parser.parse_args()
ad_sc_path = args.sc_data_path
sc_label = args.sc_label
ad_sp_path = args.st_data_path
save_sp_path = args.output_path


'''
ad_sc_path = 'data/ref.h5ad'
ad_sp_path = 'data/qry.h5ad'
save_sp_path = 'result_data/merfish_tangram.h5ad'
'''

ad_sc = sc.read_h5ad(ad_sc_path) ## reference single cell data
ad_sp = sc.read_h5ad(ad_sp_path) ## query spatial data


use_col = sc_label



# sc.tl.rank_genes_groups(ad_sc, groupby=use_col, use_raw=False)
# markers_df = pd.DataFrame(ad_sc.uns["rank_genes_groups"]["names"]).iloc[0:20, :]
# markers = list(np.unique(markers_df.melt().value.values))
import torch
tg.pp_adatas(ad_sc, ad_sp, genes=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ad_map_cl = tg.map_cells_to_space(ad_sc,
ad_sp,
mode='clusters',
cluster_label=use_col,
learning_rate = 0.001,
device = device,
num_epochs = 5000)
ad_map_cl.write(save_sp_path)
