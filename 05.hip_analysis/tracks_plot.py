import scanpy as sc
import anndata as ad
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, rgb2hex
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import jaccard_score
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 # ADOBE AI 字帖
import os
from matplotlib.font_manager import fontManager, FontProperties
import gc
from scipy.sparse import csr_matrix
fontManager.addfont('/data/work/Arial.ttf')

font = FontProperties(fname='/data/work/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name



names = [
    'B03607C4E6_WT2024071214941.h5ad',
    
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
]


# colormap = {
#   'hip_sc_01' : '#9b38e9',
#   'hip_sc_02' : '#a89630',
#   'hip_sc_03': '#5b798b',
#   'hip_sc_04' : '#cb2505',
#   'hip_sc_05' : '#62e7dd',
#   'hip_sc_06' : '#245200',
#   'hip_sc_07' : '#374898',
#   'hip_sc_08' : '#6d85c7',
#   'hip_sc_09' : '#35c498',
#   'hip_sc_10' : '#9e2dc6',
#   'hip_sc_11' : '#2d7476',
#   'hip_sc_12' : '#cb0d6c',
#   'hip_sc_13' : '#20ea38',
#   'hip_sc_14' : '#0fabb6',
#   'hip_sc_15' : '#a59099',
#   'hip_sc_16' : '#2bea3a',
#   'hip_sc_17' : '#17b064',
#   'hip_sc_18' : '#52b8d5',
#   'hip_sc_19' : '#da2ef2',
#   'hip_sc_20' : '#6240f7',
#   'hip_sc_21' : '#c47233',
#   'hip_sc_22':'#a83b23',
#   'hip_sc_23':'#9994da'
# }

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





import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

save_path = '/data/work/05.cluster/FuseMap/0116/tracks/figs_20240612/'
h5ad_path = '/data/work/05.cluster/FuseMap/0116/tracks/data_20240402/'

x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
for name in names:
    adata_temp = sc.read_h5ad(f'{h5ad_path}{name}')
    
    adata_temp.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata_temp.obs['dmt_leiden']]
    adata_temp = adata_temp[adata_temp.obs['dmt_leiden_anno'] != 'z_delete'].copy()
    
    adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
    x_min = min(x_min, adata_temp.obsm['align_spatial_2d'][:, 0].min())
    x_max = max(x_max, adata_temp.obsm['align_spatial_2d'][:, 0].max())
    y_min = min(y_min, adata_temp.obsm['align_spatial_2d'][:, 1].min())
    y_max = max(y_max, adata_temp.obsm['align_spatial_2d'][:, 1].max())



if not os.path.exists(f'{save_path}ptime_spatial_plot_all.png'):
    fig = plt.figure(figsize=(64, 6))
    gs = GridSpec(1, 15, figure=fig)
    count = 0
    for name in names:
        adata_temp = sc.read_h5ad(f'{h5ad_path}{name}')
        
        adata_temp.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata_temp.obs['dmt_leiden']]
        adata_temp = adata_temp[adata_temp.obs['dmt_leiden_anno'] != 'z_delete'].copy()
        
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        row = (count // 15) + 1
        col = count % 15
        ax = fig.add_subplot(gs[row-1, col])

        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='entropy',
            show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'viridis',
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axis('off')
        ax.set_aspect('equal')
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count += 1
    plt.savefig(f'{save_path}entropy_spatial_plot_all.png', bbox_inches = 'tight', dpi = 600)
    plt.close()



if not os.path.exists(f'{save_path}ptime_spatial_plot_all.png'):
    fig = plt.figure(figsize=(64, 6))
    gs = GridSpec(1, 15, figure=fig)
    count = 0
    adatas = []
    for name in names:
        adata_temp = sc.read_h5ad(f'{h5ad_path}{name}')
        
        adata_temp.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata_temp.obs['dmt_leiden']]
        adata_temp = adata_temp[adata_temp.obs['dmt_leiden_anno'] != 'z_delete'].copy()
        
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        row = (count // 15) + 1
        col = count % 15
        ax = fig.add_subplot(gs[row-1, col])

        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='ptime',
            show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds',
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axis('off')
        ax.set_aspect('equal')
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count += 1
        adatas.append(adata_temp)
    plt.savefig(f'{save_path}ptime_spatial_plot_all.png', bbox_inches = 'tight', dpi = 600)
    plt.close()

    import anndata as ad
    adata = ad.concat(adatas)
    adata.obs_names_make_unique()


    if not os.path.exists(f'{save_path}ptime_dmt_plot.png'):
        plot = sc.pl.embedding(adata, basis = 'X_dmt',color = 'ptime', cmap = 'Reds', show = False, 
                               title = 'Cerebellum ptime'); 
        plot.set_aspect('equal')
        plt.savefig(f'{save_path}ptime_dmt_plot.png', bbox_inches = 'tight', dpi = 600)
        plt.close()

    if not os.path.exists(f'{save_path}entropy_dmt_plot.png'):
        plot = sc.pl.embedding(adata, basis = 'X_dmt',color = 'entropy', cmap = 'viridis', show = False, 
                               title = 'Cerebellum ptime'); 
        plot.set_aspect('equal')
        plt.savefig(f'{save_path}entropy_dmt_plot.png', bbox_inches = 'tight', dpi = 600)
        plt.close()






if not os.path.exists(f'{save_path}ptime_stream_plot_all.png'):
    count = 0
    fig = plt.figure(figsize=(64, 6))
    gs = GridSpec(1, 15, figure=fig)
    for name in names:
        print(name)
        adata_temp = sc.read_h5ad(f'{h5ad_path}{name}')
        
        adata_temp.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata_temp.obs['dmt_leiden']]
        adata_temp = adata_temp[adata_temp.obs['dmt_leiden_anno'] != 'z_delete'].copy()
        
        means = adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - means
        row = (count // 15) + 1
        col = count % 15
        ax = fig.add_subplot(gs[row-1, col])

        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='ptime',
            show=False, s=1, title='', legend_loc=None, ax=ax, cmap = 'Reds',
        )

        X_grid, V_grid = adata_temp.uns['E_grid'],adata_temp.uns['V_grid']
        X_grid = X_grid.T - means
        X_grid = X_grid.T
        linewidth = 1
        density = 2
        arrow_color = 'k'
        arrow_size = 1
        arrow_style = '-|>'
        max_length = 4
        integration_direction = "both"
        lengths = np.sqrt((V_grid**2).sum(0))
        linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
        stream_kwargs = {
                "linewidth": linewidth,
                "density": density or 2,
                "zorder": 3,
                'arrow_color': 'k',
                "arrowsize": arrow_size or 1,
                "arrowstyle": arrow_style or "-|>",
                "maxlength": max_length or 4,
                "integration_direction": integration_direction or "both",
            }
        stream_kwargs["color"] = stream_kwargs.pop("arrow_color", "k")
        ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **stream_kwargs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axis('off')
        ax.set_aspect('equal')
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count += 1
    plt.savefig(f'{save_path}ptime_stream_plot_all.png', bbox_inches = 'tight', dpi = 600)
    plt.close()

if not os.path.exists(f'{save_path}celltype_stream_plot_all.png'):
    count = 0
    fig = plt.figure(figsize=(64, 6))
    gs = GridSpec(1, 15, figure=fig)
    for name in names:
        adata_temp = sc.read_h5ad(f'{h5ad_path}{name}')
        
        adata_temp.obs['dmt_leiden_anno'] = [dic_dmt_leiden[i] for i in adata_temp.obs['dmt_leiden']]
        adata_temp = adata_temp[adata_temp.obs['dmt_leiden_anno'] != 'z_delete'].copy()
        
        means = adata_temp.obsm['align_spatial_2d'].max(axis = 0)
        adata_temp.obsm['align_spatial_2d'] = adata_temp.obsm['align_spatial_2d'] - means
        row = (count // 15) + 1
        col = count % 15
        ax = fig.add_subplot(gs[row-1, col])

        sc.pl.embedding(
            adata_temp, basis="align_spatial_2d", color='dmt_leiden_anno',
            show=False, s=1, title='', legend_loc=None, ax=ax, palette=colormap
        )
        X_grid, V_grid = adata_temp.uns['E_grid'],adata_temp.uns['V_grid']
        X_grid = X_grid.T - means
        X_grid = X_grid.T
        linewidth = 1 
        density = 2
        arrow_color = 'k'
        arrow_size = 1
        arrow_style = '-|>'
        max_length = 4
        integration_direction = "both"
        lengths = np.sqrt((V_grid**2).sum(0))
        linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
        stream_kwargs = {
                "linewidth": linewidth,
                "density": density or 2,
                "zorder": 3,
                'arrow_color': 'k',
                "arrowsize": arrow_size or 1,
                "arrowstyle": arrow_style or "-|>",
                "maxlength": max_length or 4,
                "integration_direction": integration_direction or "both",
            }
        stream_kwargs["color"] = stream_kwargs.pop("arrow_color", "k")
        ax.streamplot(X_grid[0], X_grid[1], V_grid[0], V_grid[1], **stream_kwargs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.axis('off')
        ax.set_aspect('equal')
        if count == 0:
            scalebar = ScaleBar(0.0097, "mm", fixed_value=1, location = 'lower left', frameon = False,)
            ax.add_artist(scalebar)
        count += 1
    plt.savefig(f'{save_path}celltype_stream_plot_all.png', bbox_inches = 'tight', dpi = 600)
    plt.close()