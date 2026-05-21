import os
import warnings
from tqdm import *
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from sklearn.metrics import jaccard_score
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, rgb2hex
from matplotlib.font_manager import fontManager, FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
warnings.filterwarnings('ignore')
plt.rcParams['pdf.fonttype'] = 42
fontManager.addfont('/data/work/Arial.ttf')
font = FontProperties(fname='/data/work/Arial.ttf')
font_name = font.get_name()
plt.rcParams['font.family'] = font_name

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
target_genes = ['RORB','NR2F1','BRINP2','WNT7B','HOPX','SOX2','HES1','YPEL2',
                'TRIM71', 'PTEN', 'PIK3CA', 'SMARCC1', 'FMN2', 'MTOR', 
                'FOXJ1', 'PTCH1', 'FXYD2']

csv = pd.read_csv('/data/work/0521.csv', index_col = 'Unnamed: 0')
adatas = []
for name in tqdm(names):
    slice_code = name.replace('.h5ad', '')
    idx, chip, task = slice_code.split('_')
    z = csv[csv['chip'] == chip]['z'].values[0]
    if os.path.exists(f'/data/input/Files/zhoutao3/store_20250103/05.cluster/FuseMap/20241130/datas/whole_brain_20251031/{name}'):
        adata_path = f'/data/input/Files/zhoutao3/store_20250103/05.cluster/FuseMap/20241130/datas/whole_brain_20251031/{name}'
        adata_temp = sc.read_h5ad(adata_path)
        
        if chip == 'B03610C1E3' and task == 'WT202403310051':
            adata_temp.obsm['align_spatial_2d'] = np.array([adata_temp.obsm['align_spatial_2d'][:,0],
                                                       adata_temp.obsm['align_spatial_2d'][:,1]*1.1]).T.copy()
        elif chip == 'B03609A4D6' and task == 'WT202404150263':
            adata_temp.obsm['align_spatial_2d'] = np.array([adata_temp.obsm['align_spatial_2d'][:,0],
                                                       adata_temp.obsm['align_spatial_2d'][:,1]*1.1]).T.copy()
        elif chip == 'B03606C4E6' and task == 'WT202403310050':
            adata_temp.obsm['align_spatial_2d'] = np.array([adata_temp.obsm['align_spatial_2d'][:,0],
                                                       adata_temp.obsm['align_spatial_2d'][:,1]*1.1]).T.copy()
        elif chip == 'B03619A1D3' and task == 'WT202403310052':
            adata_temp.obsm['align_spatial_2d'] = np.array([adata_temp.obsm['align_spatial_2d'][:,0],
                                                       adata_temp.obsm['align_spatial_2d'][:,1]*1.1]).T.copy()
        elif chip == 'B03612E4G6' and task == 'WT202403310059':
            adata_temp.obsm['align_spatial_2d'] = np.array([adata_temp.obsm['align_spatial_2d'][:,0],
                                                       adata_temp.obsm['align_spatial_2d'][:,1]+100]).T.copy()
        
        adata_temp.obsm['align_spatial_3d'] = np.array([adata_temp.obsm['align_spatial_2d'][:, 0],
                                                        adata_temp.obsm['align_spatial_2d'][:, 1],
                                                        [z for i in range(len(adata_temp))]]).T
    else:
        adata_path = f'/data/input/Files/zhoutao3/store_20250103/05.cluster/FuseMap/20241130/datas/whole_brain/{name}'
        adata_temp = sc.read_h5ad(adata_path)
        
    adata_temp.var_names_make_unique()
    adata_temp.obs_names_make_unique()
    sc.pp.normalize_total(adata_temp)
    sc.pp.log1p(adata_temp)
    sc.pp.scale(adata_temp, zero_center=False, max_value=10)
    available_genes = [g for g in target_genes if g in adata_temp.var_names]

    adata_temp = adata_temp[:, available_genes].copy()
    
    adatas.append(adata_temp)
adata = ad.concat(adatas, join = 'outer')


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


for gene_name in target_genes:
    try:
        # 检查文件是否已存在
        if os.path.exists(f'/data/work/12.pl_ply/20260305_result/plot_02/{gene_name}.html'):
            continue
            
        # 获取表达量数据
        if adata.raw is not None:
            expr = adata.raw[:, gene_name].X
        else:
            expr = adata[:, gene_name].X

        # 转换为numpy数组
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        else:
            expr = np.asarray(expr).flatten()

        # ========== 关键修改：过滤表达量为0的点 ==========
        # 创建掩码，只保留表达量 > 0 的细胞
        mask = expr > 0
        
        # 如果没有表达该基因的细胞，跳过
        if not np.any(mask):
            print(f"{gene_name}: 无表达细胞，跳过")
            continue
            
        # 应用掩码过滤表达量和坐标
        expr_filtered = expr[mask]
        coords_filtered = coords[mask]  # 确保coords是numpy数组或支持布尔索引
        
        print(f"{gene_name}: {np.sum(mask)}/{len(expr)} 细胞保留 ({np.sum(mask)/len(expr)*100:.1f}%)")
        
        # ========== 颜色映射（基于过滤后的数据） ==========
        # 使用过滤后的数据计算vmin/vmax，避免0值影响
        vmin, vmax = np.percentile(expr_filtered, [0, 100])
        
        # 如果所有值相同，避免除零错误
        if vmin == vmax:
            vmax = vmin + 1e-6
            
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.jet
        sm = ScalarMappable(norm=norm, cmap=cmap)

        # 生成颜色（过滤后的数据）
        colors_rgb = sm.to_rgba(expr_filtered, bytes=False)[:, :3]
        colors_uint = (colors_rgb * 255).astype(np.uint8)
        colors_hex = (colors_uint[:, 0].astype(np.uint32) << 16) | \
                     (colors_uint[:, 1].astype(np.uint32) << 8) | \
                      colors_uint[:, 2]

        # ========== 透明度计算（基于过滤后的数据） ==========
        # 归一化表达量用于透明度（可选：使用log转换增强可视化）
        expr_norm = expr_filtered / 10  # 你的原始缩放
        
        # 限制透明度范围
        alpha_min, alpha_max = 0.1, 1.0  # 最小0.1避免完全透明
        opacity = alpha_min + (expr_norm - expr_norm.min()) / (expr_norm.max() - expr_norm.min() + 1e-6) * (alpha_max - alpha_min)
        
        # 确保透明度在有效范围内
        opacity = np.clip(opacity, alpha_min, alpha_max)

        # ========== 创建K3D可视化 ==========
        plot = k3d.plot(background_color=0xffffff)
        
        plt_points = k3d.points(
            positions=coords_filtered,      # 使用过滤后的坐标
            colors=colors_hex,              # 使用过滤后的颜色
            opacities=opacity,              # 使用过滤后的透明度
            point_size=point_size,
            shader='3dSpecular',
            name=gene_name
        )
        plot += plt_points

        # 添加STL网格（保持不变）
        stl_mesh = k3d.mesh(
            vertices=mesh.vertices.astype(np.float32),
            indices=mesh.faces.astype(np.uint32),
            color=0xB0B0B0,
            opacity=0.30,
            wireframe=False,
            name='STL'
        )
        plot += stl_mesh
        
        # 设置相机和网格
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
        
        # 保存HTML
        output_path = f'/data/work/12.pl_ply/20260305_result/plot_02/{gene_name}.html'
        with open(output_path, 'w') as fp:
            fp.write(plot.get_snapshot())
            
        # 清理内存
        del plot, expr, expr_filtered, coords_filtered, colors_hex, opacity
        
    except Exception as e:
        print(f"Error processing {gene_name}: {e}")
        import traceback
        traceback.print_exc()
        continue