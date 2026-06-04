# HUBSTA_Analysis

> **H**uman **U**nified **B**rain **S**patial **T**ranscriptomic **A**nalysis — Human Fetal Brain Development Atlas Pipeline

A comprehensive spatial transcriptomics analysis pipeline for human fetal brain development, built on **Stereo-seq** (Spatial Enhanced Resolution Omics-sequencing) data. This repository covers the full workflow from cell segmentation to downstream biological interpretation, including deep learning-based embedding, GPU-accelerated clustering, region-specific DEG analysis, ligand-receptor interaction inference, spatial GRN inference, pseudotime analysis, and 3D visualization.

---

## Overview of Brain Regions Analyzed

| Region | Directory | Key Analyses |
|--------|-----------|--------------|
| **Hippocampus** | `05.hip_analysis/` | DEG, Ligand-Receptor, Pseudotime, Tracks plots, Spatial gene plots |
| **Thalamus** | `06.thalamus_analysis/` | DEG, Ligand-Receptor, Spatial gene plots, Tangram mapping |
| **Mid-Hindbrain (MHB)** | `07.mhb_analysis/` | DEG, Ligand-Receptor, Tracks plots |
| **Cerebellum** | `08.cerebellum_analysis/` | DEG, Single-cell annotation, Spatial annotation, Pseudotime, Ligand-Receptor, Circos plots |
| **Spatial GRN** | `09.grn_analysis/` | GPU SpaGRN submodule, whole-brain and region-specific GRN notebooks, TF/regulon visualization |
| **Cortex** | `12.3D_ply_plot/` | 3D PLY mesh gene visualization |

---

## Pipeline Workflow

```
┌─────────────────────────┐
│  01. Cell Segmentation  │  ← Stereo-seq raw data (.gem, .h5ad)
│     (Stereopy + ONNX)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  02. FuseMap Integration│  ← Spatial data integration & embedding
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  03. DMT-HI Latent      │  ← Deep Manifold Transformation
│     Representation      │     with Hyperbolic embedding
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  04. RAPIDS GPU         │  ← GPU-accelerated Leiden clustering
│     Clustering          │     on DMT embeddings
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  05-08. Region Analysis │  ← Per-region downstream analyses
│                         │     · DEG / Gene expression plots
│                         │     · Ligand-Receptor (CCI) inference
│                         │     · Pseudotime trajectory
│                         │     · Spatial tracks visualization
│                         │     · Cell type annotation
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  09. Spatial GRN        │  ← GPU SpaGRN inference and
│     Analysis            │     TF/regulon visualization
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  12. 3D PLY Plot        │  ← 3D mesh visualization of
│     3D Visualization    │     gene expression on brain models
└─────────────────────────┘
```

---

## Repository Structure

```
HUBSTA_Analysis/
├── 01.cell_segmentation/      # Cell segmentation (Stereopy + ONNX model)
│   ├── 01.cell_segmentation.py
│   ├── 02.cell_correct.py
│   ├── 03.make_cell_mask.py
│   └── cell_segmetation_v3.0.onnx
│
├── 02.Fusemap/                # FuseMap: spatial data integration framework
│   ├── FuseMap/
│   │   ├── config.py          # Configuration
│   │   ├── dataset.py         # Data loading
│   │   ├── model.py           # Core model architecture
│   │   ├── train.py           # Training pipeline
│   │   ├── loss.py            # Loss functions
│   │   ├── preprocess.py      # Preprocessing
│   │   ├── run.*.py           # Region-specific run scripts
│   │   └── paper_code/        # Paper reproduction code
│   │       ├── benchmark/     # Benchmarks (cell integration, gene imputation)
│   │       ├── cell_cell_interation/
│   │       ├── reference_mapping/
│   │       ├── universal_cell_type/
│   │       ├── universal_gene_embedding/
│   │       └── universal_tissue_region/
│
├── 03.DMT-HI/                 # Deep Manifold Transformation (Hyperbolic)
│   ├── main.py                # Main entry point
│   ├── model/                 # Model definitions
│   ├── conf_new/              # YAML config files
│   ├── dataloader/            # Data loaders
│   ├── manifolds/             # Hyperbolic manifold utilities
│   ├── sweep/                 # Hyperparameter sweep configs
│   └── run*.sh                # Run scripts
│
├── 04.rapids_cluster/         # GPU-accelerated clustering (RAPIDS)
│   ├── *_single_1.py          # Single-cell embedding clustering
│   ├── *_spatial_1.py         # Spatial embedding clustering
│   └── run.*.sh               # Region-specific run scripts
│
├── 05.hip_analysis/           # Hippocampus analysis
│   ├── for_deg_plot.py/R/sh   # DEG analysis
│   ├── Ligand-receptor_interaction_inference.py
│   ├── tracks_single.py / tracks_plot.py
│   └── *.ipynb                # Analysis notebooks
│
├── 06.thalamus_analysis/      # Thalamus analysis
│   ├── for_deg_plot.py/R/sh   # DEG analysis
│   ├── Ligand-receptor_interaction_inference.py
│   ├── tangram_yes.py         # Tangram mapping
│   └── *.ipynb                # Analysis notebooks
│
├── 07.mhb_analysis/           # Mid-Hindbrain analysis
│   ├── for_deg_plot.py/R/sh   # DEG analysis
│   ├── Ligand-receptor_interaction_inference.py
│   └── *.ipynb                # Analysis notebooks
│
├── 08.cerebellum_analysis/    # Cerebellum analysis
│   ├── *read_process_deg.ipynb
│   ├── *single_anno.py        # Single-cell annotation
│   ├── *spatial_anno.py       # Spatial annotation
│   ├── *tracks*.py            # Tracks visualization
│   ├── *Ligand-receptor*.py   # CCI inference
│   ├── *pseudotime_plot.ipynb # Pseudotime trajectory
│   ├── *circosplot*.ipynb     # Circos plots
│   └── *.ipynb
│
├── 09.grn_analysis/           # Spatial GRN analysis and GPU SpaGRN submodule
│   ├── SpaGRN/                 # Git submodule: https://github.com/DBinary/SpaGRN
│   ├── notebooks/              # Fetal brain GRN analysis notebooks
│   ├── docs/                   # Notebook and output inventories
│   ├── _3D_plot.py             # K3D 3D expression helper
│   └── requirements.txt
│
├── 12.3D_ply_plot/            # 3D PLY mesh visualization
│   ├── 2_mhb_gene_plot_*.py   # MHB 3D gene plot
│   ├── 3_cortex_gene_plot_*.py # Cortex 3D gene plot
│   ├── 04_brain_gene_plot_*.py # Whole brain 3D gene plot
│   ├── 4_makemesh.ipynb       # Mesh generation
│   └── 5_thalamus_gene_plot_*.py # Thalamus 3D gene plot
│
└── README.md
```

---

## Key Features

### 1. Cell Segmentation (`01.cell_segmentation`)
- Stereo-seq cell segmentation using the **Stereopy** framework
- ONNX-accelerated deep learning model (v3.0) for cell boundary detection
- Cell mask generation and correction

### 2. FuseMap — Spatial Data Integration (`02.Fusemap`)
- Universal gene embedding for spatial transcriptomics
- Tissue region identification and harmonization
- Reference mapping (MERFISH, STARmap PLUS, Slide-seq V2, Stereo-seq)
- Cell-cell interaction analysis
- Gene imputation and targeted gene panel selection
- Benchmark suite for cell integration and gene imputation

### 3. DMT-HI — Deep Manifold Transformation (`03.DMT-HI`)
- Hyperbolic embedding for spatial transcriptomics data
- Configurable transformer and MLP backbones
- YAML-based experiment configuration
- Weights & Biases (wandb) integration for experiment tracking
- Hyperparameter sweep support

### 4. GPU Clustering (`04.rapids_cluster`)
- **RAPIDS cuML** accelerated Leiden clustering
- Processes both single-cell and spatial embeddings
- Covers: Cerebellum, Cortex (P10), Hippocampus, Mid-Hindbrain, Thalamus, Whole Brain

### 5. Downstream Analysis (per brain region)
| Analysis | Description |
|----------|-------------|
| **DEG Analysis** | Differential expression with Python + R (`for_deg_plot.py/R/sh`) |
| **Ligand-Receptor (CCI)** | Cell-cell interaction inference between spatial regions |
| **Pseudotime** | Trajectory analysis of cell lineages |
| **Tracks Plot** | Spatial gene expression tracks across tissue slices |
| **Cell Annotation** | Single-cell and spatial annotation of cell types |
| **Tangram Mapping** | Spatial mapping of cell types |
| **Circos Plots** | Circos-style visualization of cell-cell interactions |

### 6. Spatial GRN Analysis (`09.grn_analysis`)
- Fetal brain spatial gene regulatory network analysis notebooks
- GPU-rewritten SpaGRN inference code included as a git submodule
- Whole-brain and region-specific TF/regulon summaries
- GRN pathway, heatmap, and 3D visualization helpers

### 7. 3D Visualization (`12.3D_ply_plot`)
- 3D mesh (PLY format) rendering of brain regions
- Gene expression mapped onto 3D brain surfaces
- Supports Cortex, Thalamus, Mid-Hindbrain, and whole brain

---

## Dependencies

The pipeline relies on several key Python packages:

- **Spatial Transcriptomics**: `stereopy`, `spateo`
- **Single-cell / Spatial**: `scanpy`, `anndata`, `squidpy`
- **GPU Clustering**: `rapids-singlecell` (requires NVIDIA GPU + RAPIDS)
- **Deep Learning**: `pytorch`, `pytorch-lightning`
- **FuseMap**: See `02.Fusemap/FuseMap/fusemap_environment.yaml`
- **DMT-HI**: See `03.DMT-HI/requirements.txt` and `install_env.sh`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **GRN inference**: `spagrn`, `pyscenic`, `arboreto`, `omicverse`, `gseapy`, `k3d`
- **3D**: `open3d`, `trimesh`

For detailed environment setup, refer to:
- `01.cell_segmentation/readme.md` — Stereopy installation
- `02.Fusemap/FuseMap/fusemap_environment.yaml` — FuseMap conda environment
- `03.DMT-HI/readme.md` — DMT-HI environment
- `09.grn_analysis/requirements.txt` — spatial GRN analysis dependencies

---

## Quick Start

### Step 1: Cell Segmentation
```bash
cd 01.cell_segmentation
python 01.cell_segmentation.py
python 02.cell_correct.py
python 03.make_cell_mask.py
```

### Step 2: Run FuseMap Integration
```bash
cd 02.Fusemap/FuseMap
python run.thalamus.py   # or run.cerebellum.py / run.hip.py / run.mhb.py / run.ctx.py
```

### Step 3: Train DMT-HI Embedding
```bash
cd 03.DMT-HI
python main.py fit -c=conf_new/transf_cond_mnist.yaml
```

### Step 4: GPU Clustering
```bash
cd 04.rapids_cluster
bash run.thalamus.sh     # or run.cerebellum.sh / run.Hippocampus.sh etc.
```

### Step 5: Region-specific Analysis
```bash
# Navigate to the desired brain region directory (e.g., Hippocampus)
cd 05.hip_analysis

# Run DEG analysis
bash for_deg_plot.sh

# Run ligand-receptor inference
python Ligand-receptor_interaction_inference.py

# Open notebooks for downstream visualization
jupyter lab 1.ipynb
```

### Step 6: Spatial GRN Analysis
```bash
git submodule update --init --recursive
cd 09.grn_analysis
jupyter lab notebooks/01_bin100_preprocessing_grn/01_03_Bin100_GRN_Inference.ipynb
```

### Step 7: 3D Visualization
```bash
cd 12.3D_ply_plot
python 04_brain_gene_plot_20260306.py  # Whole brain 3D gene expression
```

---

## Data

The pipeline expects **Stereo-seq** data in `.h5ad` (AnnData) format. Each brain region has multiple tissue slices (identified by `slice_code`), each containing:
- Spatial coordinates (`obsm['align_spatial_2d']`)
- Gene expression matrix
- DMT latent embeddings (`obsm['X_dmt']`, `obsm['X_dmt_highdim']`)
- Leiden cluster assignments

The spatial GRN notebooks in `09.grn_analysis/` also expect GRN resources and intermediate AnnData outputs such as `GRN_resource/`, `Process_Data/`, `Output/`, and `Figure/`. Large data and generated outputs are intentionally excluded from the repository.

---

## Citation

If you use this pipeline or any of its components in your research, please cite the relevant tools:

- **Stereopy**: [https://stereopy.readthedocs.io/](https://stereopy.readthedocs.io/)
- **FuseMap**: See `02.Fusemap/FuseMap/README.md` for citation
- **RAPIDS**: [https://rapids.ai/](https://rapids.ai/)

---

## License

This project is for research purposes. See individual component directories for specific licenses (e.g., `02.Fusemap/FuseMap/LICENSE`).

---

## Author & Contact

For questions or collaboration inquiries, please open an issue or contact the repository maintainer.