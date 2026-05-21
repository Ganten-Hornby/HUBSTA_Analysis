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

from matplotlib.font_manager import fontManager, FontProperties

fontManager.addfont('/data/work/Arial.ttf')

font = FontProperties(fname='/data/work/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name
import os

names = [
 '1_C03627A3_WT202403110495.h5ad',
 '2_C03627A6_WT202403110494.h5ad',
 '3_C03627F1_WT202403110496.h5ad',
 '4_C03627G1_WT202403110489.h5ad',
 '5_D03657E5_WT202403110487.h5ad',
 '6_C02941A1B2_WT202403310083.h5ad',
 '7_D03473F1G2_WT202403310080.h5ad',
 '8_B03422F5G6_WT202403310079.h5ad',
 '9_D03466F1G2_WT202403310042.h5ad',
 '10_D03470F1G2_WT202403310077.h5ad',
 '11_B03612A4C6_WT202403310047.h5ad',
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
 '51_B03605C2E5_WT202406020126.h5ad',
 '55_B03613E3G6_WT202403310069.h5ad',
 '59_B03612E4G6_WT202403310059.h5ad',
 '63_B03606C1E3_WT202403310061.h5ad',
 '67_A03595A1D3_WT202403310062.h5ad',
 '71_A03595A4D6_WT202403310063.h5ad',
 '75_D03468D1E3_WT202403310066.h5ad',
 '80_D03473D4E6_WT202403310070.h5ad',
 '84_B03423D1E3_WT202403310065.h5ad',
 '89_D03466A1C3_WT202403310058.h5ad',
 '99_D03470D1E3_WT202404290071.h5ad',
 '104_B03615F3G5_WT202405020035.h5ad',
 '105_D03468A4C6_WT202403310067.h5ad',
 '106_B03610F5G6_WT202403310041.h5ad',
 '108_B03423F5G6_WT202403310082.h5ad',
 '110_A03587D5E6_WT202403310038.h5ad',
 '113_A03589A5C6_WT202403310040.h5ad',
 '114_A03590A1C2_WT202403310039.h5ad',
 '115_B03609F5G6_WT202403310081.h5ad',
 '116_B03610D5E6_WT202403310078.h5ad',
 '117_B03614F1_WT202403110556.h5ad',
 '118_B03614G3_WT202403280407.h5ad',
 '119_B03616F1_WT202403280411.h5ad',
]

adatas = []
for name in names:
    # temp = adata[adata.obs['slice_code'] == i].copy()
    temp = sc.read_h5ad(f'/data/work/05.cluster/FuseMap/20241130/datas/whole_brain/{name}')
    if name == '47_A03593C1F3_WT202403310068.h5ad':
        temp.obsm['align_spatial_3d'] = np.array([temp.obsm['align_spatial_2d'][:,0],
                                                   temp.obsm['align_spatial_2d'][:,1],
                                                   [1187.5 for i in range(len(temp))]
                                                   ]).T
    temp = temp[temp.obs['region_h2'].isin(['CP_SP', 'VZ', 'SVZ', 'IZ'])]
    sc.pp.normalize_total(temp)
    sc.pp.log1p(temp)
    sc.pp.scale(temp, zero_center=False, max_value=10)
    temp = temp[:, ['CDR1','DOK6','OCIAD2','PPP1R17','PRDM8','SEPT5','SHB','ST6GAL2']]
    adatas.append(temp)
adata = ad.concat(adatas)


import numpy as np
import pandas as pd
import k3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scanpy as sc
import k3d
import numpy as np
import pandas as pd
import trimesh
import anndata as ad


spatial_key = 'align_spatial_3d'
point_size  = 1
coords = adata.obsm[spatial_key].copy()
coords[:, [1, 2]] = coords[:, [2, 1]]   # y 与 z 互转
coords[:, 2] = -coords[:, 2]


mesh = trimesh.load('/data/work/12.pl_ply/mesh/brain.obj')
mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
mesh.vertices[:, 2] = -mesh.vertices[:, 2]


for gene_name in ['CDR1','DOK6','OCIAD2','PPP1R17','PRDM8','SEPT5','SHB','ST6GAL2']:
    try:
        if os.path.exists(f'/data/work/12.pl_ply/20251119_result/cortex_plot_01/{gene_name}.html'):
            continue
        if adata.raw is not None:
            expr = adata.raw[:, gene_name].X
        else:
            expr = adata[:, gene_name].X


        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        vmin, vmax = np.percentile(expr, [0, 100])        
        norm   = Normalize(vmin=vmin, vmax=vmax)
        cmap   = plt.cm.Reds
        sm     = ScalarMappable(norm=norm, cmap=cmap)


        colors_rgb = sm.to_rgba(expr, bytes=False)[:, :3] 
        colors_uint = (colors_rgb * 255).astype(np.uint8)
        colors_hex = (colors_uint[:, 0].astype(np.uint32) << 16) | \
                     (colors_uint[:, 1].astype(np.uint32) << 8)  | \
                      colors_uint[:, 2]
        plot = k3d.plot(background_color=0xffffff)
        expr = expr/10
        alpha_min, alpha_max = 0, 1          # 最低/最高透明度
        opacity = expr * (alpha_max - alpha_min)  # 反向


        plt_points = k3d.points(positions=coords,
                                colors=colors_hex,
                                opacities=opacity,   # ← 关键
                                point_size=point_size,
                                shader='3dSpecular',
                                name=gene_name)
        plot += plt_points

        stl_mesh = k3d.mesh(
            vertices=mesh.vertices.astype(np.float32),
            indices  =mesh.faces.astype(np.uint32),
            color    =0xB0B0B0,
            opacity  =0.30,
            wireframe=False,
            name='STL'
        )
        plot += stl_mesh
        plot.camera = [7431.183780371341,
                         -2718.585696189546,
                         -890.0932313888161,
                         2798.4393920898438,
                         1246.624137878418,
                         -2679.1115112304688,
                         -0.16464848235296514,
                         0.15813353942384692,
                         0.9735936837141818]
        plot.grid_visible = False
        with open(f'/data/work/12.pl_ply/20251119_result/cortex_plot_01/{gene_name}.html', 'w') as fp:
            fp.write(plot.get_snapshot())
    # break
        del plot
    except Exception as e:
        print(e)
        print(gene_name)
        continue