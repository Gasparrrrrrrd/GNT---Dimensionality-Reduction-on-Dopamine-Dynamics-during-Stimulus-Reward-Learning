# GNT — Dimensionality Reduction on Dopamine/GABA Dynamics during Stimulus-Reward Learning

## Project Overview
This repository contains a complete data analysis framework for exploring population neural activity dynamics (Dopamine + GABA) in stimulus-reward learning tasks in mice.

The goal is to:
- Perform dimensionality reduction (PCA) on neural timecourses across datasets (`CRFB`, `ToneFB`, `SpontFB`).
- Quantify how much of the neural variance is captured by low-dimensional trajectories.
- Visualize class-level and epoch-level representational geometry.
- Compare cross-experiment embeddings in neural state space.

## What we are trying to do
- Neural circuits often encode behaviorally-relevant signals in high-dimensional activity patterns. PCA reveals latent structure, separability, and temporal organization.
- Dopamine and GABA dynamics are said to be key modulators of reward-driven learning and decision-making. This project enables a parallel view of how these neuromodulatory networks can be understood to modulate something else, namely movement.
- Combining spontaneous and task-locked conditions can decouple intrinsic dynamics from learned stimulus responses, and thus disentangle what is movement and what is reward.

## Repository Structure
- `data*.mat` — raw preprocessed recordings (repository examples)
- `plot_pca.py` — PCA analytics, basis projection, scatter/trajectory generation
- `plot_pca_framework.py` — pipeline orchestration and cross-dataset analysis
- `outputs/` — generated figures, HTML 3D trajectories, overlays, diagnostics
- `PCA_*.ipynb` — interactive notebooks with exploratory analysis and figure recipes
- `HIGHLIGHTS.md` — strong findings summary created in this branch

## Key outputs to inspect
- `outputs/.*/*_trajectory.png` — PC1/PC2/PC3 trace plots for each condition
- `outputs/overlays/*` — dataset-to-dataset comparisons
- `outputs/diagnostics/*` — separation statistics and clustering quality

## Analysis flow
1. Load FR trajectories.
2. Select relevant neuron groups: D/DB/DF/DFB and G/GB/GF/GFB.
3. Compute PCA on 2s sliding windows per condition.
4. Build visualizations:
   - 2D scattered points (`PC1 vs PC2`)
   - 3D trajectories (`PC1/2/3` components over time)
   - Overlay plots across groups and datasets
5. Quantify explained variance, cross-epoch stability, and cluster separability.

## Run locally
```bash
python plot_pca_framework.py --dataset CRFB --mode combined
python plot_pca.py --dataset ToneFB --component 3
```


---
