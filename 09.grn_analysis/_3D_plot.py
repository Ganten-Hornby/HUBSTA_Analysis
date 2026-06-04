import numpy as np
import pandas as pd
import scanpy as sc
import k3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, Colormap
from matplotlib.cm import ScalarMappable
from typing import Optional, Tuple, Union

def plot_gene_3d_k3d(
    adata: sc.AnnData,
    gene_name: str,
    spatial_key: str = 'ccf',
    point_size: float = 1.0,
    cmap: Union[str, Colormap] = 'Reds',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
    coord_scale: float = 10.0,
    n_bin: int = 50,
    base_alpha: float = 0.05,
    max_alpha_scale: float = 0.8,
    background_color: int = 0x1e1e1e,
    shader: str = '3dSpecular',
    save_path: Optional[str] = None
):
    """
    使用 K3D 绘制单个基因在 3D 空间中的表达情况。

    采用分箱（binning）策略来优化透明度渲染性能。

    参数:
    ----------
    adata : sc.AnnData
        包含空间数据和基因表达的 AnnData 对象。
    gene_name : str
        要可视化的基因名称（应存在于 adata.var_names 或 adata.raw.var_names 中）。
    spatial_key : str, optional
        在 adata.obsm 中存储 3D 坐标的键名。默认为 'ccf'。
    point_size : float, optional
        K3D 中点的大小。默认为 1.0。
    cmap : Union[str, Colormap], optional
        用于基因表达的 Matplotlib 颜色映射表。默认为 'Reds'。
    vmin : Optional[float], optional
        颜色映射的最小值。如果为 None，则使用 'percentile_clip' 计算。默认为 None。
    vmax : Optional[float], optional
        颜色映射的最大值。如果为 None，则使用 'percentile_clip' 计算。默认为 None。
    percentile_clip : Tuple[float, float], optional
        如果 vmin 或 vmax 为 None，用于计算它们的分位数范围。默认为 (2.0, 98.0)。
    coord_scale : float, optional
        坐标的缩放因子（原始代码中为 *10）。默认为 10.0。
    n_bin : int, optional
        用于透明度分箱的数量。越高性能越好，但越不平滑。默认为 50。
    base_alpha : float, optional
        表达量为 0 的点的基础透明度。默认为 0.05。
    max_alpha_scale : float, optional
        添加到基础透明度上的最大透明度缩放。
        (总最大透明度 = base_alpha + max_alpha_scale)。默认为 0.8。
    background_color : int, optional
        K3D 图形的背景颜色（十六进制）。默认为 0x1e1e1e (深灰色)。
    shader : str, optional
        K3D 点的着色器。默认为 '3dSpecular'。
    save_path : Optional[str], optional
        如果提供，则将 K3D 图形的快照保存到此 HTML 文件路径。默认为 None。

    返回:
    -------
    k3d.plot.Plot
        K3D 绘图对象。可以在 Jupyter Notebook 中直接显示。
    """
    
    print(f"--- Starting 3D visualization for gene: '{gene_name}' ---")

    # --- 1. Extract coordinates ---
    if spatial_key not in adata.obsm:
        raise KeyError(f"在 adata.obsm 中未找到空间坐标键: '{spatial_key}'")
    coords = adata.obsm[spatial_key] * coord_scale
    print(f"Coordinates extracted and scaled (factor: {coord_scale}).")

    # --- 2. Extract gene expression ---
    expr = None
    if adata.raw is not None and gene_name in adata.raw.var_names:
        expr = adata.raw[:, gene_name].X
        print(f"Extracting expression from adata.raw.")
    elif gene_name in adata.var_names:
        expr = adata[:, gene_name].X
        print(f"Extracting expression from adata.X.")
    else:
        raise KeyError(f"在 adata.var_names 或 adata.raw.var_names 中均未找到基因: '{gene_name}'")

    # Convert (sparse) matrix to a flat numpy array
    if hasattr(expr, 'toarray'):
        expr = expr.toarray().flatten()
    else:
        expr = np.asarray(expr).flatten()
    print(f"Expression data flattened. Total points: {len(expr)}")

    # --- 3. Set up color and opacity mapping ---
    if vmin is None:
        vmin = np.percentile(expr, percentile_clip[0])
    if vmax is None:
        vmax = np.percentile(expr, percentile_clip[1])
    
    # Prevent division by zero if vmin and vmax are equal
    if vmin == vmax:
        vmax = vmin + 1e-6

    print(f"Color range (vmin, vmax): ({vmin:.2f}, {vmax:.2f})")
    
    # Set up Matplotlib colormapper
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    
    # Calculate normalized opacity values (0 to 1)
    alpha_norm = np.clip((expr - vmin) / (vmax - vmin), 0, 1)

    # --- 4. K3D Plotting (Binned Rendering) ---
    alpha_edges = np.linspace(0, 1, n_bin + 1)
    plot = k3d.plot(background_color=background_color, name=f"{gene_name} Expression")
    
    print(f"Adding points to K3D plot ({n_bin} bins)...")
    points_added = 0
    
    for i in range(n_bin):
        # Find points belonging to the current opacity bin
        mask = (alpha_norm >= alpha_edges[i]) & (alpha_norm < alpha_edges[i + 1])
        
        # If no points in this bin, skip
        if mask.sum() == 0:
            continue

        # --- Calculate properties for this bin ---
        # 1. Calculate the uniform opacity for this bin
        seg_alpha_norm = (alpha_edges[i] + alpha_edges[i + 1]) / 2  # Use the midpoint of the bin
        seg_alpha_final = base_alpha + max_alpha_scale * seg_alpha_norm
        
        # 2. Get coordinates for this bin
        seg_pos = coords[mask].astype(np.float32)
        
        # 3. Get colors for this bin (convert from RGB to K3D's uint32 format)
        seg_colors_rgb = sm.to_rgba(expr[mask], bytes=False)[:, :3]
        seg_colors_hex = (seg_colors_rgb * 255).astype(np.uint8)
        seg_colors_uint = (seg_colors_hex[:, 0].astype(np.uint32) << 16) | \
                          (seg_colors_hex[:, 1].astype(np.uint32) << 8)  | \
                           seg_colors_hex[:, 2]

        # --- Add to K3D plot ---
        plt_points = k3d.points(
            positions=seg_pos,
            colors=seg_colors_uint,
            point_size=point_size,
            shader=shader,
            opacity=float(seg_alpha_final),  # K3D requires a float for opacity
            name=f'expr_bin_{i}'
        )
        plot += plt_points
        points_added += len(seg_pos)

    print(f"K3D plot created. Total points added: {points_added}.")

    # --- 5. Save (if required) ---
    if save_path:
        print(f"Saving snapshot to: {save_path}")
        try:
            with open(save_path, 'w') as fp:
                fp.write(plot.get_snapshot())
            print(f"Snapshot saved successfully.")
        except Exception as e:
            print(f"[Error] Failed to save snapshot: {e}")

    print(f"--- Visualization function finished ---")
    return plot

# -----------------------------------------------------------------
# ---                          Usage Example                        ---
# -----------------------------------------------------------------

if __name__ == "__main__":
    
    # This is an example of how to use the function.
    # You need to uncomment and run it in your own environment.

    # # --- 1. Import libraries (already at top of file) ---
    # import scanpy as sc
    # import numpy as np

    # # --- 2. Load data ---
    # # (You need to ensure you have access to this file)
    # print("Loading AnnData file...")
    # try:
    #     adata = sc.read_h5ad('/data/work/stereoseq.h5ad')
    #     print("AnnData file loaded successfully.")
    # except Exception as e:
    #     print(f"Failed to load file: {e}")
    #     print("Please ensure the file path '/data/work/stereoseq.h5ad' is correct and accessible.")
    #     # # If loading fails, create dummy data for demonstration
    #     # print("Creating dummy data for demonstration...")
    #     # n_obs = 10000
    #     # adata = sc.AnnData(X=np.random.rand(n_obs, 5))
    #     # adata.raw = adata.copy()
    #     # adata.var_names = ['Gfap', 'GeneB', 'GeneC', 'GeneD', 'GeneE']
    #     # adata.obsm['ccf'] = np.random.rand(n_obs, 3) * 50
    #     # adata[:, 'Gfap'].X = np.random.choice([0, 0.5, 1, 2, 5], n_obs)


    # # --- 3. Call the function ---
    # if 'adata' in locals():
    #     print("\nCalling plot_gene_3d_k3d function...")
    #     gfap_plot = plot_gene_3d_k3d(
    #         adata=adata,
    #         gene_name='Gfap',
    #         spatial_key='ccf',
    #         point_size=1.0,
    #         cmap='Reds',
    #         percentile_clip=(2, 98),
    #         save_path='/data/work/gfap_plot_v_function.html'
    #     )
        
    #     # --- 4. Display in Jupyter ---
    #     # In a Jupyter Notebook or Jupyter Lab,
    #     # you just need to have the `gfap_plot` variable as the last line,
    #     # and it will display automatically.
        
    #     # print("\nFunction call finished.")
    #     # print("If you are in a Jupyter environment, type 'gfap_plot' in a new cell to display the image.")
        
    #     # Alternatively, if not in Jupyter, you can only rely on the saved HTML file.
    #     print("Please check the saved HTML file: /data/work/gfap_plot_v_function.html")

    pass

