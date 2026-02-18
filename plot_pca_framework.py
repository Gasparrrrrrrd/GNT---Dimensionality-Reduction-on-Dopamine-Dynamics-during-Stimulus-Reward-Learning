"""
plot_pca_framework.py
=====================
Extended PCA analysis framework for VTA Dopamine/GABA neural dynamics.

Builds on the existing plot_pca.py module, adding:
- Separate PCA fit / project functions for cross-dataset/cross-epoch analysis
- Group-averaged neuron extraction for cross-dataset projection
- Epoch slicing for cross-epoch PCA
- Quantitative trajectory metrics (speed, curvature, separation, etc.)
- Subspace overlap, Procrustes distance, reconstruction R-squared
- 1D PC timecourse plots, overlay figures, scree plots, speed profiles
- Cross-epoch R-squared heatmaps
"""

import os
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.spatial import procrustes as scipy_procrustes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import reusable pieces from existing module
from plot_pca import (
    matlab_struct_to_dict,
    arrays_to_dfs,
    load_dataset,
    EVENT_RULES,
    get_event_markers,
    cmap_to_rgb_strings,
    build_plotly_colorscale,
    endpoint_trace,
    colorbar_trace,
    SCENE_AXES,
    smooth_trajectories,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Data extraction (NaN-robust)
# ===========================================================================

def extract_neuron_data(data, neuron_groups):
    """Extract z-scored firing rates for requested neuron groups.

    Like plot_pca.extract_neuron_data but logs per-group NaN statistics
    so the DFB NaN issue (rows 19/65 in CRFB/ToneFB) is visible.

    Returns:
        X:  pd.DataFrame of shape (total_neurons, 2 * timesteps)
        timesteps: number of timesteps per half (e.g. 1201)
        stats: dict with per-group {group: {total, kept, dropped}} info
    """
    data_dfs = arrays_to_dfs(data)
    available = list(data['firing_rate'].keys())
    missing = [g for g in neuron_groups if g not in available]
    if missing:
        raise ValueError(
            f"Neuron group(s) {missing} not found. Available: {available}"
        )

    parts = []
    timesteps = None
    stats = {}
    for group in neuron_groups:
        fr = data_dfs['firing_rate'][f'firing_rate_{group}']
        zfwd = fr[f'firing_rate_{group}_zforward']
        zbwd = fr[f'firing_rate_{group}_zbackward']
        if timesteps is None:
            timesteps = zfwd.shape[1]
        combined = pd.concat([zfwd, zbwd], axis=1)
        n_total = combined.shape[0]
        combined = combined.dropna()
        n_kept = combined.shape[0]
        n_dropped = n_total - n_kept
        stats[group] = {'total': n_total, 'kept': n_kept, 'dropped': n_dropped}
        if n_dropped > 0:
            logger.warning(
                f"Group {group}: dropped {n_dropped}/{n_total} neurons with NaN"
            )
        if n_kept > 0:
            parts.append(combined)
        else:
            logger.warning(f"Group {group}: ALL neurons dropped (NaN). Skipping.")

    if not parts:
        raise ValueError("No neurons remaining after NaN removal.")

    X = pd.concat(parts, axis=0)
    return X, timesteps, stats


def extract_group_averaged_data(data, neuron_groups):
    """Average z-scored firing rates within each neuron group.

    Returns X of shape (n_groups, 2*timesteps) where each row is the
    group-mean firing rate. Enables cross-dataset projection when neuron
    identities don't match (e.g. SpontFB <-> CRFB/ToneFB).

    Returns:
        X: pd.DataFrame of shape (n_groups, 2*timesteps)
        timesteps: int
        group_labels: list of str (group names in order)
    """
    data_dfs = arrays_to_dfs(data)
    available = list(data['firing_rate'].keys())

    rows = []
    group_labels = []
    timesteps = None
    for group in neuron_groups:
        if group not in available:
            logger.warning(f"Group {group} not in dataset, skipping.")
            continue
        fr = data_dfs['firing_rate'][f'firing_rate_{group}']
        zfwd = fr[f'firing_rate_{group}_zforward']
        zbwd = fr[f'firing_rate_{group}_zbackward']
        if timesteps is None:
            timesteps = zfwd.shape[1]
        combined = pd.concat([zfwd, zbwd], axis=1).dropna()
        if combined.shape[0] == 0:
            logger.warning(f"Group {group}: no valid neurons for averaging.")
            continue
        group_mean = combined.mean(axis=0)
        rows.append(group_mean.values)
        group_labels.append(group)

    if not rows:
        raise ValueError("No groups with valid data for averaging.")

    X = pd.DataFrame(np.array(rows))
    return X, timesteps, group_labels


# ===========================================================================
# PCA: fit / project / align
# ===========================================================================

def fit_pca(X, n_components=3):
    """Fit PCA on X.T (timepoints as samples, neurons as features).

    Returns the fitted sklearn PCA object.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X.T if hasattr(X, 'T') else X.T)
    return pca


def project_onto_pca(pca, X):
    """Project data X onto a previously fitted PCA basis.

    X must have the same number of rows (neurons/features) as the
    data used to fit pca.

    Returns:
        projections: np.ndarray of shape (n_components, n_columns)
    """
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    return pca.components_ @ X_arr


def align_pca_signs(pca_target, pca_reference):
    """Flip component signs of pca_target to maximize correlation with
    pca_reference components. Modifies pca_target in-place.

    This addresses PCA's inherent sign ambiguity when comparing across
    datasets.
    """
    for i in range(pca_target.n_components):
        if np.dot(pca_target.components_[i], pca_reference.components_[i]) < 0:
            pca_target.components_[i] *= -1
    return pca_target


def run_pca(X, n_components=3):
    """Legacy-compatible wrapper: fit + project in one step.

    Returns:
        projections: np.ndarray (n_components, 2*timesteps)
        pca: fitted PCA object
        explained_variance_ratio: list of floats
    """
    pca = fit_pca(X, n_components)
    projections = project_onto_pca(pca, X)
    return projections, pca, list(pca.explained_variance_ratio_)


# ===========================================================================
# Epoch slicing (before PCA fitting)
# ===========================================================================

def slice_epoch(X, timesteps, start_idx, end_idx):
    """Slice both fwd and bwd halves of X to a specific time epoch.

    X has shape (n_neurons, 2*timesteps) where columns [0:timesteps] are
    forward and [timesteps:2*timesteps] are backward.

    Returns:
        X_sliced: same type as X, shape (n_neurons, 2*(end_idx - start_idx))
        epoch_timesteps: end_idx - start_idx
    """
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    fwd_slice = X_arr[:, start_idx:end_idx]
    bwd_slice = X_arr[:, timesteps + start_idx:timesteps + end_idx]
    X_sliced = np.hstack([fwd_slice, bwd_slice])
    epoch_timesteps = end_idx - start_idx
    if hasattr(X, 'values'):
        X_sliced = pd.DataFrame(X_sliced, index=X.index)
    return X_sliced, epoch_timesteps


def slice_window(projections, timesteps, event_idx=600, window=120, dt=0.01):
    """Slice PCA projections into forward/backward windows around event.

    Returns dict with fwd, bwd, n_plot, plot_time.
    """
    fwd_start = event_idx - window
    fwd_end = event_idx + window
    bwd_start = timesteps + event_idx - window
    bwd_end = timesteps + event_idx + window

    n_plot = 2 * window + 1
    plot_time = (np.arange(n_plot) - window) * dt

    fwd = projections[:, fwd_start:fwd_end + 1]
    bwd = projections[:, bwd_start:bwd_end + 1]

    return {'fwd': fwd, 'bwd': bwd, 'n_plot': n_plot, 'plot_time': plot_time}


# ===========================================================================
# Predefined epochs
# ===========================================================================

EPOCHS = {
    'pre_tone':       {'dataset': 'ToneFB', 'start': 480, 'end': 600,
                       'desc': 'Baseline before CS'},
    'tone_to_reward': {'dataset': 'ToneFB', 'start': 600, 'end': 700,
                       'desc': 'CS processing period'},
    'post_reward':    {'dataset': 'ToneFB', 'start': 700, 'end': 820,
                       'desc': 'Reward response'},
    'pre_CR':         {'dataset': 'CRFB',   'start': 480, 'end': 600,
                       'desc': 'Baseline before movement'},
    'post_CR':        {'dataset': 'CRFB',   'start': 600, 'end': 720,
                       'desc': 'Movement execution'},
    'pre_spont':      {'dataset': 'SpontFB','start': 480, 'end': 600,
                       'desc': 'Baseline before spontaneous movement'},
    'post_spont':     {'dataset': 'SpontFB','start': 600, 'end': 720,
                       'desc': 'Spontaneous movement'},
    'full_window':    {'dataset': 'Any',    'start': 480, 'end': 720,
                       'desc': 'Full ±120 window'},
}


# ===========================================================================
# Quantitative trajectory metrics
# ===========================================================================

def compute_trajectory_metrics(smooth_data, window_data, dt=0.01):
    """Compute speed, curvature, arc length, and fwd-bwd separation.

    Args:
        smooth_data: dict with 'fwd_smooth', 'bwd_smooth' (n_components, n_plot)
        window_data: dict with 'fwd', 'bwd', 'n_plot', 'plot_time'
        dt: timestep in seconds

    Returns dict with:
        fwd_speed: (n_plot-1,) instantaneous speed
        bwd_speed: (n_plot-1,)
        fwd_curvature: (n_plot-2,) unsigned curvature
        bwd_curvature: (n_plot-2,)
        fwd_arc_length: float
        bwd_arc_length: float
        separation: (n_plot,) fwd-bwd Euclidean distance
        mean_separation: float
        plot_time: time axis from window_data
    """
    fwd = smooth_data['fwd_smooth']  # (n_comp, n_plot)
    bwd = smooth_data['bwd_smooth']

    def _speed(traj):
        diff = np.diff(traj, axis=1) / dt
        return np.sqrt(np.sum(diff**2, axis=0))

    def _curvature(traj):
        # curvature = |v x a| / |v|^3  (in 3D)
        v = np.diff(traj, axis=1) / dt          # velocity (n_comp, n-1)
        a = np.diff(v, axis=1) / dt             # acceleration (n_comp, n-2)
        v_mid = v[:, :-1]                        # match dims
        if traj.shape[0] == 3:
            cross = np.cross(v_mid.T, a.T).T     # (3, n-2)
            cross_mag = np.sqrt(np.sum(cross**2, axis=0))
        else:
            # 2D: |v1*a2 - v2*a1|
            cross_mag = np.abs(v_mid[0]*a[1] - v_mid[1]*a[0])
        v_mag = np.sqrt(np.sum(v_mid**2, axis=0))
        v_mag = np.maximum(v_mag, 1e-12)
        return cross_mag / (v_mag**3)

    def _arc_length(traj):
        diff = np.diff(traj, axis=1)
        return np.sum(np.sqrt(np.sum(diff**2, axis=0)))

    fwd_speed = _speed(fwd)
    bwd_speed = _speed(bwd)

    fwd_curv = _curvature(fwd)
    bwd_curv = _curvature(bwd)

    separation = np.sqrt(np.sum((fwd - bwd)**2, axis=0))

    return {
        'fwd_speed': fwd_speed,
        'bwd_speed': bwd_speed,
        'fwd_curvature': fwd_curv,
        'bwd_curvature': bwd_curv,
        'fwd_arc_length': _arc_length(fwd),
        'bwd_arc_length': _arc_length(bwd),
        'separation': separation,
        'mean_separation': float(np.mean(separation)),
        'plot_time': window_data['plot_time'],
    }


def compute_reconstruction_r2(pca, X):
    """R-squared: fraction of variance in X captured by the PCA basis.

    Projects X into the PCA subspace and back, then computes
    1 - SS_residual / SS_total.
    """
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    X_projected = pca.components_ @ X_arr           # (n_comp, n_time)
    X_reconstructed = pca.components_.T @ X_projected  # (n_neurons, n_time)
    ss_res = np.sum((X_arr - X_reconstructed) ** 2)
    ss_tot = np.sum((X_arr - X_arr.mean()) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_subspace_overlap(components_a, components_b):
    """Principal angles between two subspaces.

    Args:
        components_a: (n_components, n_features) from PCA A
        components_b: (n_components, n_features) from PCA B

    Returns:
        cosines: array of cosines of principal angles (values near 1 = aligned)
    """
    M = components_a @ components_b.T
    _, s, _ = np.linalg.svd(M)
    return np.clip(s, 0, 1)


def compute_procrustes_distance(traj_a, traj_b):
    """Procrustes distance between two trajectories.

    Each trajectory: (n_components, n_timepoints). Transposes to (n_time, n_comp)
    for scipy.spatial.procrustes.

    Returns:
        disparity: float (Procrustes distance, lower = more similar)
        traj_a_aligned: aligned version of traj_a
        traj_b_aligned: aligned version of traj_b
    """
    a = traj_a.T.copy()
    b = traj_b.T.copy()
    mtx1, mtx2, disparity = scipy_procrustes(a, b)
    return float(disparity), mtx1.T, mtx2.T


def compute_cross_correlation(timecourse_a, timecourse_b, dt=0.01):
    """Cross-correlation of two 1D timecourses.

    Returns:
        lags: array of lag values in seconds
        correlation: normalized cross-correlation values
    """
    a = timecourse_a - np.mean(timecourse_a)
    b = timecourse_b - np.mean(timecourse_b)
    corr = np.correlate(a, b, mode='full')
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm > 0:
        corr = corr / norm
    n = len(a)
    lags = np.arange(-(n - 1), n) * dt
    return lags, corr


# ===========================================================================
# Cross-epoch R-squared matrix
# ===========================================================================

def compute_cross_epoch_r2_matrix(X, timesteps, epoch_configs, neuron_groups_label,
                                  n_components=3):
    """Compute R-squared for every pair of (fit_epoch, project_epoch).

    Args:
        X: neuron data (n_neurons, 2*timesteps)
        timesteps: int
        epoch_configs: dict of {name: {'start': int, 'end': int}}
        n_components: int

    Returns:
        r2_matrix: pd.DataFrame (fit_epoch x project_epoch)
        pca_dict: dict of {epoch_name: fitted PCA object}
    """
    epoch_names = list(epoch_configs.keys())
    n = len(epoch_names)
    r2_matrix = np.zeros((n, n))
    pca_dict = {}

    # Fit PCA on each epoch
    for name, cfg in epoch_configs.items():
        X_epoch, _ = slice_epoch(X, timesteps, cfg['start'], cfg['end'])
        pca_dict[name] = fit_pca(X_epoch, n_components)

    # Compute R-squared for each pair
    for i, fit_name in enumerate(epoch_names):
        for j, proj_name in enumerate(epoch_names):
            cfg = epoch_configs[proj_name]
            X_proj, _ = slice_epoch(X, timesteps, cfg['start'], cfg['end'])
            r2_matrix[i, j] = compute_reconstruction_r2(pca_dict[fit_name], X_proj)

    return pd.DataFrame(r2_matrix, index=epoch_names, columns=epoch_names), pca_dict


# ===========================================================================
# Visualization: 1D PC timecourses
# ===========================================================================

def plot_1d_pc_timecourses(window_data, smooth_data, event_markers, title,
                           dataset_name=None, n_components=3):
    """Plot each PC as a 1D function of time with fwd/bwd overlaid.

    Returns a matplotlib figure with n_components subplots.
    """
    plot_time = window_data['plot_time']
    fwd = smooth_data['fwd_smooth']
    bwd = smooth_data['bwd_smooth']

    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3 * n_components),
                             sharex=True)
    if n_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(plot_time, fwd[i], color='orangered', linewidth=2,
                label='Forward')
        ax.plot(plot_time, bwd[i], color='royalblue', linewidth=2,
                label='Backward')

        # Event markers as vertical lines
        for marker in event_markers:
            t_event = window_data['plot_time'][marker['idx']] if marker['idx'] < len(plot_time) else None
            if t_event is not None:
                ax.axvline(t_event, color=marker['color'], linestyle='--',
                           linewidth=1.5, alpha=0.7, label=marker['label'])

        ax.set_ylabel(f'PC{i+1}', fontsize=12)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ===========================================================================
# Visualization: overlay figure (multiple datasets in shared PC space)
# ===========================================================================

def build_overlay_figure(trajectory_sets, title, width=1200, height=900):
    """Build 3D Plotly figure overlaying trajectories from multiple datasets.

    Args:
        trajectory_sets: list of dicts, each with:
            'fwd_smooth': (3, n_plot)
            'bwd_smooth': (3, n_plot)
            'label': str (e.g. 'SpontFB GABA')
            'fwd_color': str (CSS color)
            'bwd_color': str (CSS color)
            'dash': str ('solid' or 'dash')
            'event_markers': list of marker dicts (optional)
            'plot_time': array (optional, for hover)
        title: str

    Returns: go.Figure
    """
    fig = go.Figure()

    for tset in trajectory_sets:
        fwd = tset['fwd_smooth']
        bwd = tset['bwd_smooth']
        label = tset['label']
        dash = tset.get('dash', 'solid')

        # Forward trajectory
        fig.add_trace(go.Scatter3d(
            x=fwd[0], y=fwd[1], z=fwd[2],
            mode='lines', name=f'{label} Fwd',
            line=dict(color=tset['fwd_color'], width=5, dash=dash),
        ))

        # Backward trajectory
        fig.add_trace(go.Scatter3d(
            x=bwd[0], y=bwd[1], z=bwd[2],
            mode='lines', name=f'{label} Bwd',
            line=dict(color=tset['bwd_color'], width=5, dash=dash),
        ))

        # Event markers
        for marker in tset.get('event_markers', []):
            idx = marker['idx']
            if idx < fwd.shape[1]:
                fig.add_trace(endpoint_trace(
                    fwd[0][idx], fwd[1][idx], fwd[2][idx],
                    color=marker['color'], name=f"{label} {marker['label']}",
                    symbol=marker.get('symbol', 'diamond'),
                    size=marker.get('size', 10),
                ))

    fig.update_layout(
        title=dict(text=title, y=0.95, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title="PC1", **SCENE_AXES),
            yaxis=dict(title="PC2", **SCENE_AXES),
            zaxis=dict(title="PC3", **SCENE_AXES),
        ),
        font=dict(family="Times New Roman, serif", size=12, color="black"),
        margin=dict(l=80, r=220, t=80, b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="v", x=1.05, y=0.5),
        width=width, height=height,
    )
    return fig


# ===========================================================================
# Visualization: scree plot comparison
# ===========================================================================

def plot_scree_comparison(variance_ratios_dict, title="Scree Plot Comparison",
                          n_show=10):
    """Side-by-side eigenvalue spectra.

    Args:
        variance_ratios_dict: {label: array_of_explained_variance_ratios}
        n_show: max number of components to show
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, evr in variance_ratios_dict.items():
        evr = np.asarray(evr)[:n_show]
        ax.plot(range(1, len(evr) + 1), evr * 100, 'o-', label=label,
                linewidth=2, markersize=6)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ===========================================================================
# Visualization: speed profiles
# ===========================================================================

def plot_speed_profiles(metrics_dict, title="Speed Profiles"):
    """Plot speed profiles for multiple analyses with event markers.

    Args:
        metrics_dict: {label: trajectory_metrics_dict}
            Each value is the output of compute_trajectory_metrics().
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for label, m in metrics_dict.items():
        t = m['plot_time']
        # Speed has one fewer point than trajectory
        t_speed = (t[:-1] + t[1:]) / 2
        axes[0].plot(t_speed, m['fwd_speed'], linewidth=2, label=label)
        axes[1].plot(t_speed, m['bwd_speed'], linewidth=2, label=label)

    axes[0].set_title('Forward Speed', fontsize=12)
    axes[1].set_title('Backward Speed', fontsize=12)
    for ax in axes:
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Speed (PC units/s)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ===========================================================================
# Visualization: cross-epoch R-squared heatmap
# ===========================================================================

def plot_cross_epoch_r2_matrix(r2_df, title="Cross-Epoch R² Matrix"):
    """Heatmap of R-squared values for every fit-epoch x project-epoch pair.

    Args:
        r2_df: pd.DataFrame from compute_cross_epoch_r2_matrix()
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(r2_df.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(r2_df.columns)))
    ax.set_xticklabels(r2_df.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(r2_df.index)))
    ax.set_yticklabels(r2_df.index, fontsize=9)
    ax.set_xlabel('Project Epoch', fontsize=12)
    ax.set_ylabel('Fit Epoch', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Annotate cells
    for i in range(len(r2_df.index)):
        for j in range(len(r2_df.columns)):
            val = r2_df.values[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label='R²', shrink=0.8)
    fig.tight_layout()
    return fig


# ===========================================================================
# Visualization: metric comparison table
# ===========================================================================

def plot_metric_comparison_table(metrics_dict, title="Trajectory Metrics Comparison"):
    """Summary bar charts of key metrics across analyses.

    Args:
        metrics_dict: {label: trajectory_metrics_dict}
    """
    labels = list(metrics_dict.keys())
    fwd_arcs = [metrics_dict[l]['fwd_arc_length'] for l in labels]
    bwd_arcs = [metrics_dict[l]['bwd_arc_length'] for l in labels]
    mean_seps = [metrics_dict[l]['mean_separation'] for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(labels))
    w = 0.35

    axes[0].bar(x - w/2, fwd_arcs, w, label='Forward', color='orangered', alpha=0.8)
    axes[0].bar(x + w/2, bwd_arcs, w, label='Backward', color='royalblue', alpha=0.8)
    axes[0].set_ylabel('Arc Length')
    axes[0].set_title('Trajectory Arc Length')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[0].legend()

    axes[1].bar(x, mean_seps, color='purple', alpha=0.8)
    axes[1].set_ylabel('Mean Separation')
    axes[1].set_title('Fwd-Bwd Mean Separation')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

    # Peak speeds
    fwd_peak = [np.max(metrics_dict[l]['fwd_speed']) for l in labels]
    bwd_peak = [np.max(metrics_dict[l]['bwd_speed']) for l in labels]
    axes[2].bar(x - w/2, fwd_peak, w, label='Forward', color='orangered', alpha=0.8)
    axes[2].bar(x + w/2, bwd_peak, w, label='Backward', color='royalblue', alpha=0.8)
    axes[2].set_ylabel('Peak Speed')
    axes[2].set_title('Peak Trajectory Speed')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    axes[2].legend()

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ===========================================================================
# Full pipeline: single dataset analysis (extended)
# ===========================================================================

def analyze_dataset(mat_file, var_name, dataset_name, neuron_groups,
                    combo_label, n_components=3, event_idx=600, window=150,
                    dt=0.01, sg_window=11, sg_order=3):
    """Run the full analysis pipeline for one dataset + neuron combo.

    Like plot_pca.plot_pca but returns intermediate objects for framework use.
    Uses window=150 by default for better post-reward coverage.

    Returns dict with all intermediate results.
    """
    data = load_dataset(mat_file, var_name)
    X, timesteps, stats = extract_neuron_data(data, neuron_groups)
    pca_obj = fit_pca(X, n_components)
    projections = project_onto_pca(pca_obj, X)
    evr = list(pca_obj.explained_variance_ratio_)

    win_data = slice_window(projections, timesteps, event_idx, window, dt)
    smooth_data = smooth_trajectories(win_data, sg_window, sg_order)
    markers, marker_warnings = get_event_markers(dataset_name, window)
    metrics = compute_trajectory_metrics(smooth_data, win_data, dt)

    return {
        'data': data,
        'X': X,
        'timesteps': timesteps,
        'stats': stats,
        'pca': pca_obj,
        'projections': projections,
        'explained_variance_ratio': evr,
        'n_neurons': X.shape[0],
        'window_data': win_data,
        'smooth_data': smooth_data,
        'event_markers': markers,
        'marker_warnings': marker_warnings,
        'metrics': metrics,
        'config': {
            'mat_file': mat_file, 'var_name': var_name,
            'dataset_name': dataset_name, 'neuron_groups': neuron_groups,
            'combo_label': combo_label, 'n_components': n_components,
            'event_idx': event_idx, 'window': window, 'dt': dt,
            'sg_window': sg_window, 'sg_order': sg_order,
        },
    }


# ===========================================================================
# Cross-dataset projection pipeline
# ===========================================================================

def _extract_raw_neuron_matrix(data, neuron_groups):
    """Extract the full (un-dropped) neuron matrix for a set of groups.

    Returns a pd.DataFrame of shape (total_neurons, 2*timesteps) with a
    MultiIndex of (group_name, local_row_index) to uniquely identify each
    neuron across groups. No NaN rows are dropped yet.
    """
    data_dfs = arrays_to_dfs(data)
    parts = []
    indices = []
    timesteps = None
    for group in neuron_groups:
        fr = data_dfs['firing_rate'][f'firing_rate_{group}']
        zfwd = fr[f'firing_rate_{group}_zforward']
        zbwd = fr[f'firing_rate_{group}_zbackward']
        if timesteps is None:
            timesteps = zfwd.shape[1]
        combined = pd.concat([zfwd, zbwd], axis=1)
        parts.append(combined)
        # Build MultiIndex tuples: (group_name, local_row_0), (group_name, local_row_1), ...
        local_indices = combined.index.tolist()
        indices.extend([(group, i) for i in local_indices])

    X_raw = pd.concat(parts, axis=0, ignore_index=True)
    # Assign the MultiIndex
    X_raw.index = pd.MultiIndex.from_tuples(indices, names=['group', 'local_row'])
    return X_raw, timesteps


def _align_neuron_data(data_fit, data_proj, neuron_groups):
    """Return two neuron matrices with the same rows (neurons).

    Extracts the raw (un-dropped) matrices for both datasets, finds every
    row that contains a NaN in *either* dataset, and drops those rows from
    *both*.  This guarantees that row i in the returned fit matrix and row i
    in the returned project matrix always refer to the same neuron.

    Uses a MultiIndex (group, local_row) to uniquely identify neurons across
    groups, since simple integer indices would be duplicated when groups are
    concatenated.

    Returns:
        X_fit_clean:  pd.DataFrame (n_shared_neurons, 2*ts_fit)
        X_proj_clean: pd.DataFrame (n_shared_neurons, 2*ts_proj)
        n_dropped:    int  (number of rows removed)
    """
    X_fit_raw, ts_fit = _extract_raw_neuron_matrix(data_fit, neuron_groups)
    X_proj_raw, ts_proj = _extract_raw_neuron_matrix(data_proj, neuron_groups)

    # Rows that have any NaN in the fit dataset (MultiIndex tuples)
    nan_rows_fit = set(X_fit_raw.index[X_fit_raw.isnull().any(axis=1)].tolist())
    # Rows that have any NaN in the project dataset (MultiIndex tuples)
    nan_rows_proj = set(X_proj_raw.index[X_proj_raw.isnull().any(axis=1)].tolist())
    # Union: drop a row if it is bad in *either* dataset
    bad_rows = nan_rows_fit | nan_rows_proj

    keep_rows = [idx for idx in X_fit_raw.index if idx not in bad_rows]
    n_dropped = len(bad_rows)

    if n_dropped > 0:
        # Format for readable logging
        nan_fit_str = {f"{g}[{r}]" for g, r in nan_rows_fit}
        nan_proj_str = {f"{g}[{r}]" for g, r in nan_rows_proj}
        logger.warning(
            f"_align_neuron_data: dropping {n_dropped} neuron row(s) that have "
            f"NaN in at least one dataset (fit NaN: {sorted(nan_fit_str)}, "
            f"project NaN: {sorted(nan_proj_str)})."
        )

    X_fit_clean = X_fit_raw.loc[keep_rows]
    X_proj_clean = X_proj_raw.loc[keep_rows]
    return X_fit_clean, X_proj_clean, n_dropped


def cross_project(result_fit, result_project, use_group_avg=False,
                  neuron_groups=None, window=150, event_idx=600, dt=0.01,
                  sg_window=11, sg_order=3):
    """Project one dataset onto another's PCA basis.

    If use_group_avg=True, uses group-averaged data for both (required when
    neuron identities don't match, e.g. SpontFB <-> CRFB/ToneFB).

    If use_group_avg=False, assumes the two datasets share the same neurons
    (e.g. CRFB <-> ToneFB).  NaN-dropping may remove *different* rows in
    each dataset (e.g. DFB row 19 in ToneFB, row 65 in CRFB).  To keep the
    neuron dimension consistent, both raw matrices are re-extracted and any
    row that is NaN in *either* dataset is dropped from *both* before fitting
    or projecting.

    Args:
        result_fit: output of analyze_dataset() for the fitting dataset
        result_project: output of analyze_dataset() for the dataset to project
        use_group_avg: bool
        neuron_groups: list of group names (required if use_group_avg=True)

    Returns dict with projected results.
    """
    if use_group_avg:
        if neuron_groups is None:
            raise ValueError("neuron_groups required when use_group_avg=True")
        # Re-extract as group averages
        X_fit_avg, ts_fit, groups_fit = extract_group_averaged_data(
            result_fit['data'], neuron_groups)
        X_proj_avg, ts_proj, groups_proj = extract_group_averaged_data(
            result_project['data'], neuron_groups)

        n_comp = result_fit['pca'].n_components
        pca_fit = fit_pca(X_fit_avg, min(n_comp, X_fit_avg.shape[0]))
        projections = project_onto_pca(pca_fit, X_proj_avg)
        timesteps = ts_proj
        X_for_r2 = X_proj_avg
    else:
        # Re-extract both datasets and drop NaN rows from both simultaneously
        # so that the neuron dimension is identical for fitting and projection.
        cfg_fit = result_fit['config']
        cfg_proj = result_project['config']
        groups = cfg_fit['neuron_groups']

        X_fit_clean, X_proj_clean, n_dropped = _align_neuron_data(
            result_fit['data'], result_project['data'], groups
        )

        n_comp = result_fit['pca'].n_components
        pca_fit = fit_pca(X_fit_clean, n_comp)
        projections = project_onto_pca(pca_fit, X_proj_clean)
        timesteps = result_project['timesteps']
        X_for_r2 = X_proj_clean

    win_data = slice_window(projections, timesteps, event_idx, window, dt)
    smooth_data = smooth_trajectories(win_data, sg_window, sg_order)

    ds_name = result_project['config']['dataset_name']
    markers, warnings = get_event_markers(ds_name, window)
    metrics = compute_trajectory_metrics(smooth_data, win_data, dt)

    r2 = compute_reconstruction_r2(pca_fit, X_for_r2)

    return {
        'projections': projections,
        'window_data': win_data,
        'smooth_data': smooth_data,
        'event_markers': markers,
        'metrics': metrics,
        'r2': r2,
        'pca_fit': pca_fit,
        'use_group_avg': use_group_avg,
        'fit_dataset': result_fit['config']['dataset_name'],
        'project_dataset': ds_name,
    }
