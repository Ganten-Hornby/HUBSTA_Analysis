# Spatial GRN Analysis

Fetal brain spatial gene regulatory network analysis notebooks and helper code.

## Repository Layout

```text
.
├── _3D_plot.py
├── SpaGRN/
├── docs/
│   ├── CODE_PARTS.md
│   ├── NOTEBOOK_OUTPUT_MAP.md
│   └── OUTPUT_INVENTORY.md
├── notebooks/
│   ├── 00_data_summary/
│   ├── 01_bin100_preprocessing_grn/
│   ├── 02_whole_brain_grn/
│   ├── 03_cellbin_midbrain/
│   ├── 04_cerebellum/
│   ├── 05_hip/
│   ├── 06_mhb/
│   └── 07_supplementary/
└── requirements.txt
```

`SpaGRN/` is a git submodule pointing to
`https://github.com/DBinary/SpaGRN`. It contains the GPU-rewritten spatial GRN
inference implementation used by the analysis notebooks.

## Analysis Flow

The notebooks are ordered by their filename prefixes:

1. `00_*`: summarize bin100 and cellbin input data.
2. `01_*`: preprocess bin100 data and run GRN inference experiments.
3. `02_*`: summarize and visualize whole-brain GRN results.
4. `03_*`: process cellbin midbrain data and calculate GRN scores.
5. `04_*`: cerebellum-specific GRN analysis.
6. `05_*`: hippocampus-specific GRN analysis, including 3D visualization.
7. `06_*`: MHB-specific GRN inference and downstream analysis.
8. `07_*`: supplementary analyses and manuscript-driven follow-up tasks.

See `docs/CODE_PARTS.md` for a module-level overview,
`docs/NOTEBOOK_OUTPUT_MAP.md` for notebook-to-output mapping, and
`docs/OUTPUT_INVENTORY.md` for the inferred generated output inventory.

## External Data And Resources

Large input data and generated outputs are intentionally not committed. The
notebooks expect these paths to exist relative to the repository working
directory unless edited locally:

- `fetal_brain_hongtaoyu/`
- `fetal_brain_data_ChP/`
- `GRN_resource/`
- `Process_Data/`
- `Output/`
- `Figure/`

Common GRN resource files referenced by the notebooks include:

- `GRN_resource/lr_network_human.csv`
- `GRN_resource/lr_network_mouse.csv`
- motif databases, motif annotations, and transcription factor lists used by
  spaGRN / pySCENIC workflows

Initialize the SpaGRN submodule after cloning this repository:

```bash
git submodule update --init --recursive
```

Some notebooks also reference historical absolute paths under
`/home/dataset-assist-0/...`. Those paths are preserved from the original
analysis notebooks and should be adjusted locally before rerunning.

## Python Environment

Install the Python packages listed in `requirements.txt` in an analysis
environment that also has access to the external datasets and GRN resources.
The original notebooks were exploratory and may require additional local helper
modules, including `enhance_h5ad_fast` and `plot_grn`, if rerunning the
corresponding notebook sections.

## Notes

- This repository is a code archive and workflow reference, not a single-entry
  executable pipeline.
- Large `.h5ad`, `.rds`, generated figures, and GRN output directories are
  excluded by `.gitignore`.
