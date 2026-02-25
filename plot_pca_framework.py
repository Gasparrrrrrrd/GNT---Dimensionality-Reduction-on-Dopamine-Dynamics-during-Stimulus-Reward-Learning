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
    build_figure,
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
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    pca.fit(X_arr.T)
    return pca


def project_onto_pca(pca, X):
    """Project data X onto a previously fitted PCA basis.

    X must have the same number of rows (neurons/features) as the
    data used to fit pca.  pca.mean_ (per-neuron temporal mean of the
    training data) is subtracted before projecting.  For z-scored data
    pca.mean_ ≈ 0, so results are numerically unchanged within-dataset,
    but the subtraction is essential for correctness when pca was fitted
    on a *different* dataset (cross-dataset projection).

    Returns:
        projections: np.ndarray of shape (n_components, n_columns)
    """
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    return pca.components_ @ (X_arr - pca.mean_[:, np.newaxis])


def align_pca_signs(pca_target, pca_reference):
    """Optimally permute and sign-flip pca_target's components to align with
    pca_reference.  Modifies pca_target in-place.

    Why permutation matters: PCA orders components by decreasing variance,
    which differs across datasets.  Index-based matching (PC1↔PC1) fails when
    the leading variance axis is different (e.g. reward dynamics dominate
    ToneFB-PC1 while direction dominates SpontFB-PC1).

    Algorithm:
      1. Build cross-Gram matrix M = target.components_ @ reference.components_.T
         (shape n_comp × n_comp); M[i,j] = dot between target PC-i and ref PC-j.
      2. Use the Hungarian algorithm on −|M| to find the optimal bijection
         (permute target rows to maximise |alignment| with reference rows).
      3. Invert the permutation so that new[k] aligns with reference[k].
      4. Sign-flip each matched pair so their dot product is positive.

    Returns the modified pca_target and the permutation array (inv_perm),
    where inv_perm[k] is the original target PC index now at position k.
    """
    from scipy.optimize import linear_sum_assignment

    M = pca_target.components_ @ pca_reference.components_.T  # (n_comp, n_comp)
    row_ind, col_ind = linear_sum_assignment(-np.abs(M))

    # col_ind[i] = the reference PC that target PC i is assigned to.
    # We need the INVERSE permutation: inv_perm[k] = which target PC should
    # be placed at position k (to align with reference PC k).
    n = len(col_ind)
    inv_perm = np.empty(n, dtype=col_ind.dtype)
    inv_perm[col_ind] = np.arange(n)

    new_components = pca_target.components_[inv_perm].copy()

    # Sign-flip so each matched dot product is positive.
    # new[k] = original target PC inv_perm[k], should align with ref PC k.
    for k in range(n):
        if M[inv_perm[k], k] < 0:
            new_components[k] *= -1

    pca_target.components_ = new_components
    return pca_target, inv_perm


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
    'pre_tone':       {'dataset': 'ToneFB', 'start': 450, 'end': 600,
                       'desc': 'Baseline before CS'},
    # NOTE: 'cs_period' ends exactly at reward delivery (idx 700 = tone + 100).
    # Previously named 'tone_to_reward' with end=750, which overlapped 'post_reward'
    # by 50 timesteps (700-749) — that caused artificially inflated cross-epoch R²
    # between these two epochs.  Now they are strictly non-overlapping.
    'cs_period':      {'dataset': 'ToneFB', 'start': 600, 'end': 700,
                       'desc': 'CS processing window (tone onset → reward delivery)'},
    'post_reward':    {'dataset': 'ToneFB', 'start': 700, 'end': 850,
                       'desc': 'Reward response (reward delivery onward)'},
    # CRFB epochs: non-overlapping to avoid inflated cross-epoch R².
    # Previous definition had 'during_CR' [525,675) overlapping both
    # 'pre_CR' [450,600) and 'post_CR' [600,750) by 75 timesteps each.
    'pre_CR':         {'dataset': 'CRFB',   'start': 450, 'end': 525,
                       'desc': 'Well before movement'},
    'peri_CR':        {'dataset': 'CRFB',   'start': 525, 'end': 600,
                       'desc': 'Immediately before movement (ramp-up)'},
    'post_CR':        {'dataset': 'CRFB',   'start': 600, 'end': 675,
                       'desc': 'Movement execution (early)'},
    'late_CR':        {'dataset': 'CRFB',   'start': 675, 'end': 750,
                       'desc': 'Movement execution (late)'},
    'pre_spont':      {'dataset': 'SpontFB','start': 450, 'end': 600,
                       'desc': 'Baseline before spontaneous movement'},
    'post_spont':     {'dataset': 'SpontFB','start': 600, 'end': 750,
                       'desc': 'Spontaneous movement'},
    'full_window':    {'dataset': 'Any',    'start': 450, 'end': 750,
                       'desc': 'Full ±150 window'},
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
        mean_separation: float — mean over ALL timepoints (includes pre-event)
        peak_separation: float — maximum separation across all timepoints
        post_event_mean_separation: float — mean separation strictly after
            t=0 (event onset), the scientifically relevant window for testing
            whether the two conditions diverge post-stimulus.
        plot_time: time axis from window_data
    """
    fwd = smooth_data['fwd_smooth']  # (n_comp, n_plot)
    bwd = smooth_data['bwd_smooth']
    plot_time = window_data['plot_time']

    def _speed(traj):
        diff = np.diff(traj, axis=1) / dt
        return np.sqrt(np.sum(diff**2, axis=0))

    def _curvature(traj):
        # curvature = |v x a| / |v|^3
        ndim = traj.shape[0]
        if ndim < 2:
            # Curvature is undefined in 1D
            return np.zeros(max(traj.shape[1] - 2, 0))
        v = np.diff(traj, axis=1) / dt          # velocity (n_comp, n-1)
        a = np.diff(v, axis=1) / dt             # acceleration (n_comp, n-2)
        v_mid = v[:, :-1]                        # match dims
        if ndim == 2:
            cross_mag = np.abs(v_mid[0]*a[1] - v_mid[1]*a[0])
        elif ndim == 3:
            cross = np.cross(v_mid.T, a.T).T     # (3, n-2)
            cross_mag = np.sqrt(np.sum(cross**2, axis=0))
        else:
            # General N-D: kappa = sqrt(|v|^2|a|^2 - (v·a)^2) / |v|^3
            v_dot_a = np.sum(v_mid * a, axis=0)
            v_sq = np.sum(v_mid**2, axis=0)
            a_sq = np.sum(a**2, axis=0)
            cross_mag = np.sqrt(np.maximum(v_sq * a_sq - v_dot_a**2, 0.0))
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

    # Post-event separation: only timepoints where t >= 0
    post_event_mask = plot_time >= 0
    post_sep = separation[post_event_mask]
    post_event_mean_sep = float(np.mean(post_sep)) if post_sep.size > 0 else float(np.mean(separation))

    return {
        'fwd_speed': fwd_speed,
        'bwd_speed': bwd_speed,
        'fwd_curvature': fwd_curv,
        'bwd_curvature': bwd_curv,
        'fwd_arc_length': _arc_length(fwd),
        'bwd_arc_length': _arc_length(bwd),
        'separation': separation,
        'mean_separation': float(np.mean(separation)),
        'peak_separation': float(np.max(separation)),
        'post_event_mean_separation': post_event_mean_sep,
        'plot_time': plot_time,
    }


def compute_reconstruction_r2(pca, X):
    """R-squared: fraction of temporal variance in X captured by the PCA basis.

    Projects X into the PCA subspace and back, then computes
    1 - SS_residual / SS_total.

    SS_total uses the *per-neuron* temporal mean (axis=1), which matches what
    PCA actually optimises and is invariant to the absolute firing-rate baseline
    of the projected dataset.  The previous implementation used the global
    scalar mean, which inflated SS_total with inter-neuron variance and gave
    misleading values for cross-dataset projections.
    """
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    # Mean-correct using pca.mean_ (training-data per-neuron mean) before
    # projecting, consistent with project_onto_pca().
    X_centered = X_arr - pca.mean_[:, np.newaxis]
    X_projected = pca.components_ @ X_centered      # (n_comp, n_time)
    X_reconstructed = pca.components_.T @ X_projected  # (n_neurons, n_time)
    ss_res = np.sum((X_centered - X_reconstructed) ** 2)
    # SS_total: total variance after removing each neuron's own temporal mean.
    # For z-scored data this equals n_neurons × n_timepoints.
    ss_tot = np.sum(X_centered ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def compute_pc_behavioral_correlation(projections, timesteps, window, event_idx,
                                       force_fwd=None, force_bwd=None,
                                       lick_fwd=None, lick_bwd=None, dt=0.01):
    """Correlate each PC time-course with behavioral traces (force / lick rate).

    This function quantitatively links PC axes to behavioral variables,
    providing statistical grounding for the interpretation that a given PC
    encodes kinematics rather than reward.

    *** DATA AVAILABILITY NOTE ***
    Behavioral traces (force, lick rate) are recorded per-trial and have not
    yet been matched to the per-neuron trial-averaged firing rates used for PCA.
    Once per-trial behavioral data is aligned to the same trials and time-base
    as the z-scored firing rates, pass the trial-averaged force/lick traces
    (shape: n_timesteps) as ``force_fwd`` / ``force_bwd`` / ``lick_fwd`` /
    ``lick_bwd``.

    Args:
        projections: np.ndarray (n_components, 2*timesteps) — output of
            project_onto_pca().
        timesteps: int — number of timepoints per half.
        window: int — half-window size in samples (same as slice_window).
        event_idx: int — event position within each half (default 600).
        force_fwd: 1-D array of length timesteps (trial-averaged force, forward
            condition). Pass None until data are available.
        force_bwd: same for backward condition.
        lick_fwd: 1-D array of length timesteps (trial-averaged lick rate, fwd).
        lick_bwd: same for backward condition.
        dt: float — seconds per timestep.

    Returns:
        pd.DataFrame with columns ['PC', 'condition', 'variable', 'r', 'p']
        when behavioral data are provided, or None with a NotImplementedError
        warning when all behavioral inputs are None.
    """
    from scipy.stats import pearsonr

    behavioral_vars = {
        'force_fwd':  force_fwd,
        'force_bwd':  force_bwd,
        'lick_fwd':   lick_fwd,
        'lick_bwd':   lick_bwd,
    }
    if all(v is None for v in behavioral_vars.values()):
        logger.warning(
            "compute_pc_behavioral_correlation: no behavioral traces provided. "
            "Per-trial force/lick data must be aligned to the z-scored firing "
            "rates before this analysis can run.  Returning None."
        )
        return None

    # Slice projections to the same window used for trajectory visualisation
    fwd_start = event_idx - window
    fwd_end   = event_idx + window + 1
    bwd_start = timesteps + event_idx - window
    bwd_end   = timesteps + event_idx + window + 1

    pc_fwd = projections[:, fwd_start:fwd_end]   # (n_comp, 2*window+1)
    pc_bwd = projections[:, bwd_start:bwd_end]

    n_comp = projections.shape[0]
    records = []

    for cond_label, pc_half, beh_label, beh_trace in [
        ('fwd', pc_fwd, 'force', force_fwd),
        ('bwd', pc_bwd, 'force', force_bwd),
        ('fwd', pc_fwd, 'lick',  lick_fwd),
        ('bwd', pc_bwd, 'lick',  lick_bwd),
    ]:
        if beh_trace is None:
            continue
        beh_arr = np.asarray(beh_trace)
        # Slice behavioral trace to the same window
        beh_win = beh_arr[event_idx - window: event_idx + window + 1]
        if len(beh_win) != pc_half.shape[1]:
            logger.warning(
                f"Behavioral trace length {len(beh_win)} does not match "
                f"PC window length {pc_half.shape[1]} — skipping {cond_label}/{beh_label}."
            )
            continue
        for k in range(n_comp):
            r, p = pearsonr(pc_half[k], beh_win)
            records.append({
                'PC': f'PC{k+1}',
                'condition': cond_label,
                'variable': beh_label,
                'r': round(float(r), 4),
                'p': float(p),
            })

    return pd.DataFrame(records) if records else None


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
# RSA: Representational Similarity Analysis
# ===========================================================================

def compute_rdm(X, metric='correlation'):
    """Compute Representational Dissimilarity Matrix from population vectors.

    Each column of X is a population vector at one timepoint.
    The RDM is a symmetric (n_timepoints x n_timepoints) matrix where
    entry (i, j) measures dissimilarity between the population vectors
    at timepoints i and j.

    Args:
        X: np.ndarray of shape (n_features, n_timepoints).
           For raw neural data, n_features = n_neurons.
           For PCA projections, n_features = n_components.
        metric: str, one of 'correlation', 'euclidean', 'cosine'.
            'correlation' uses 1 - Pearson r (standard RSA choice, scale-
            invariant — appropriate for z-scored data).
            'euclidean' uses pairwise Euclidean distance.
            'cosine' uses 1 - cosine similarity.

    Returns:
        rdm: np.ndarray of shape (n_timepoints, n_timepoints), symmetric,
             zero diagonal.
    """
    from scipy.spatial.distance import pdist, squareform

    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    # pdist expects (n_observations, n_features): each row = one timepoint
    X_T = X_arr.T  # (n_timepoints, n_features)

    if metric not in ('correlation', 'euclidean', 'cosine'):
        raise ValueError(
            f"Unknown metric '{metric}'. Use 'correlation', 'euclidean', or 'cosine'."
        )

    rdm = squareform(pdist(X_T, metric=metric))

    # Guard: if any timepoint has zero variance across features,
    # correlation is undefined → NaN.  Replace with 1.0 (max dissimilarity).
    if metric == 'correlation' and np.any(np.isnan(rdm)):
        logger.warning(
            "compute_rdm: NaN values in correlation RDM (likely zero-variance "
            "timepoints). Replacing NaN with 1.0 (max dissimilarity)."
        )
        rdm = np.nan_to_num(rdm, nan=1.0)

    np.fill_diagonal(rdm, 0.0)
    return rdm


def compute_rsa(rdm_a, rdm_b, method='spearman', n_permutations=10000):
    """Compare two RDMs using Representational Similarity Analysis.

    Extracts the upper triangle of each RDM (excluding diagonal) and
    computes the correlation between them.  Significance is assessed via
    a Mantel permutation test (row/column shuffle of one RDM).

    **Caveat — temporal autocorrelation**: neural timeseries are strongly
    autocorrelated, so adjacent rows/columns of the RDM are not independent.
    The Mantel test can therefore be anti-conservative.  Always report the
    effect size (r) alongside p, and consider subsampling every k-th
    timepoint as a robustness check.

    Args:
        rdm_a: np.ndarray (T x T) — first RDM.
        rdm_b: np.ndarray (T x T) — second RDM (must be same shape).
        method: 'spearman' (default, rank correlation) or 'pearson'.
        n_permutations: int — number of permutations for significance test.
            Set to 0 to skip the permutation test (returns p=NaN).

    Returns:
        r_observed: float — observed correlation between RDM upper triangles.
        p_value: float — Mantel test p-value (proportion of permuted r >=
            observed r).
    """
    from scipy.stats import spearmanr, pearsonr

    if rdm_a.shape != rdm_b.shape:
        raise ValueError(
            f"RDM shapes must match: {rdm_a.shape} vs {rdm_b.shape}"
        )

    triu_idx = np.triu_indices(rdm_a.shape[0], k=1)
    vec_a = rdm_a[triu_idx]
    vec_b = rdm_b[triu_idx]

    if method == 'spearman':
        r_observed, _ = spearmanr(vec_a, vec_b)
    elif method == 'pearson':
        r_observed, _ = pearsonr(vec_a, vec_b)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'spearman' or 'pearson'.")

    if n_permutations <= 0:
        return float(r_observed), float('nan')

    # Mantel permutation test: shuffle rows AND columns simultaneously
    count_ge = 0
    n = rdm_a.shape[0]
    corr_fn = spearmanr if method == 'spearman' else pearsonr
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        rdm_b_perm = rdm_b[np.ix_(perm, perm)]
        vec_b_perm = rdm_b_perm[triu_idx]
        r_perm, _ = corr_fn(vec_a, vec_b_perm)
        if r_perm >= r_observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)
    return float(r_observed), float(p_value)


def compute_procrustes_comparison(smooth_data_a, smooth_data_b,
                                   directions=('fwd', 'bwd', 'both')):
    """Batch Procrustes comparison between two sets of smoothed trajectories.

    Compares forward↔forward, backward↔backward, and/or both concatenated.
    Uses the existing ``compute_procrustes_distance`` function.

    Note: ``scipy.spatial.procrustes`` normalises both matrices to unit
    Frobenius norm, so Procrustes disparity measures *shape* similarity
    only, not amplitude.  Complements ``mean_separation`` which captures
    scale.

    Args:
        smooth_data_a: dict with 'fwd_smooth', 'bwd_smooth' (n_comp × n_plot)
        smooth_data_b: dict with 'fwd_smooth', 'bwd_smooth' (n_comp × n_plot)
        directions: tuple of which comparisons to make.
            'fwd': forward trajectories only.
            'bwd': backward trajectories only.
            'both': concatenate fwd+bwd and compare as single trajectory.

    Returns:
        dict of {direction: {'disparity': float,
                             'aligned_a': array, 'aligned_b': array}}
    """
    out = {}

    if 'fwd' in directions:
        d, a, b = compute_procrustes_distance(
            smooth_data_a['fwd_smooth'], smooth_data_b['fwd_smooth']
        )
        out['fwd'] = {'disparity': d, 'aligned_a': a, 'aligned_b': b}

    if 'bwd' in directions:
        d, a, b = compute_procrustes_distance(
            smooth_data_a['bwd_smooth'], smooth_data_b['bwd_smooth']
        )
        out['bwd'] = {'disparity': d, 'aligned_a': a, 'aligned_b': b}

    if 'both' in directions:
        traj_a = np.hstack([smooth_data_a['fwd_smooth'],
                            smooth_data_a['bwd_smooth']])
        traj_b = np.hstack([smooth_data_b['fwd_smooth'],
                            smooth_data_b['bwd_smooth']])
        d, a, b = compute_procrustes_distance(traj_a, traj_b)
        out['both'] = {'disparity': d, 'aligned_a': a, 'aligned_b': b}

    return out


# ===========================================================================
# Null models and permutation tests (Issues B & F)
# ===========================================================================

def null_cross_projection_r2(data_fit, data_project, neuron_groups,
                              n_components=3, n_permutations=500,
                              seed=42):
    """Null distribution for group-averaged cross-projection R².

    With only ~4 group-average features and 3 PCs, cross-projection R²
    is structurally biased upward.  This function builds a null
    distribution by independently phase-randomising each group's
    time-course (preserving autocorrelation and power spectrum) in the
    *project* dataset before projecting onto the fit PCA basis.

    Args:
        data_fit:  raw data dict for the fitting dataset.
        data_project: raw data dict for the dataset to project.
        neuron_groups: list of group names (e.g. ['DF','DB','D','DFB']).
        n_components: int.
        n_permutations: number of null samples.
        seed: int — RNG seed for reproducibility.

    Returns:
        observed_r2: float — real cross-projection R².
        null_r2s: np.ndarray (n_permutations,) — null R² values.
        p_value: float — fraction of null R² >= observed.
    """
    rng = np.random.default_rng(seed)

    # --- observed ---
    X_fit, ts_fit, _ = extract_group_averaged_data(data_fit, neuron_groups)
    X_proj, ts_proj, _ = extract_group_averaged_data(data_project, neuron_groups)
    n_comp = min(n_components, X_fit.shape[0])
    pca_fit = fit_pca(X_fit, n_comp)
    observed_r2 = compute_reconstruction_r2(pca_fit, X_proj)

    # --- null distribution via phase randomisation ---
    null_r2s = np.empty(n_permutations)
    X_proj_arr = X_proj.values if hasattr(X_proj, 'values') else np.asarray(X_proj)

    for i in range(n_permutations):
        X_null = _phase_randomise(X_proj_arr, rng)
        X_null_df = pd.DataFrame(X_null)
        null_r2s[i] = compute_reconstruction_r2(pca_fit, X_null_df)

    p_value = float((np.sum(null_r2s >= observed_r2) + 1) / (n_permutations + 1))
    return observed_r2, null_r2s, p_value


def null_separation(smooth_data, window_data, dt=0.01,
                    n_permutations=500, seed=42):
    """Null distribution for fwd-bwd trajectory separation.

    Phase-randomises each PC's projected time-course independently for
    both forward and backward, then recomputes post-event mean
    separation.  This preserves the autocorrelation and power spectrum
    of each PC while destroying the cross-condition structure.

    Args:
        smooth_data: dict with 'fwd_smooth', 'bwd_smooth'.
        window_data: dict with 'plot_time'.
        dt: timestep in seconds.
        n_permutations: number of null samples.
        seed: int.

    Returns:
        observed_sep: float — real post-event mean separation.
        null_seps: np.ndarray (n_permutations,).
        p_value: float.
    """
    rng = np.random.default_rng(seed)

    fwd = smooth_data['fwd_smooth']
    bwd = smooth_data['bwd_smooth']
    plot_time = window_data['plot_time']
    post_mask = plot_time >= 0

    # Observed
    sep = np.sqrt(np.sum((fwd - bwd)**2, axis=0))
    observed_sep = float(np.mean(sep[post_mask])) if np.any(post_mask) else float(np.mean(sep))

    # Null: phase-randomise fwd and bwd independently
    null_seps = np.empty(n_permutations)
    for i in range(n_permutations):
        fwd_null = _phase_randomise(fwd, rng)
        bwd_null = _phase_randomise(bwd, rng)
        sep_null = np.sqrt(np.sum((fwd_null - bwd_null)**2, axis=0))
        null_seps[i] = float(np.mean(sep_null[post_mask])) if np.any(post_mask) else float(np.mean(sep_null))

    p_value = float((np.sum(null_seps >= observed_sep) + 1) / (n_permutations + 1))
    return observed_sep, null_seps, p_value


def null_cross_class_r2(result_a, result_b, n_permutations=200,
                         n_folds=5, seed=42):
    """Null distribution for cross-class projection R².

    Shuffles neuron assignments between class A and class B (keeping
    the total pool fixed) before fitting the lstsq mapping, then
    computes CV R².  This tests whether the observed cross-class
    mapping captures genuine shared structure vs. fitting noise.

    Args:
        result_a: analyze_dataset output for class A (reference).
        result_b: analyze_dataset output for class B (projected).
        n_permutations: number of shuffles (lower default — CV is slow).
        n_folds: int.
        seed: int.

    Returns:
        observed_r2: float — real CV R².
        null_r2s: np.ndarray (n_permutations,).
        p_value: float.
    """
    rng = np.random.default_rng(seed)

    X_a = result_a['X']
    X_b = result_b['X']
    pca_a = result_a['pca']

    X_a_arr = X_a.values if hasattr(X_a, 'values') else np.asarray(X_a)
    X_b_arr = X_b.values if hasattr(X_b, 'values') else np.asarray(X_b)

    Z_a = pca_a.components_ @ (X_a_arr - pca_a.mean_[:, np.newaxis])

    # Observed CV R²
    observed_r2 = _cv_r2_lstsq(X_b_arr, Z_a, n_folds)

    # Pool all neurons and reshuffle
    X_pool = np.vstack([X_a_arr, X_b_arr])  # (n_a + n_b, 2T)
    n_a = X_a_arr.shape[0]

    null_r2s = np.empty(n_permutations)
    for i in range(n_permutations):
        idx = rng.permutation(X_pool.shape[0])
        X_a_shuf = X_pool[idx[:n_a], :]
        X_b_shuf = X_pool[idx[n_a:], :]

        # Refit PCA on shuffled A
        pca_shuf = fit_pca(pd.DataFrame(X_a_shuf), pca_a.n_components)
        Z_a_shuf = pca_shuf.components_ @ (X_a_shuf - pca_shuf.mean_[:, np.newaxis])

        null_r2s[i] = _cv_r2_lstsq(X_b_shuf, Z_a_shuf, n_folds)

    p_value = float((np.sum(null_r2s >= observed_r2) + 1) / (n_permutations + 1))
    return observed_r2, null_r2s, p_value


def _cv_r2_lstsq(X_b, Z_a, n_folds):
    """Internal: k-fold CV R² for lstsq mapping X_b -> Z_a (with intercept)."""
    n_time = X_b.shape[1]
    fold_size = n_time // n_folds
    fold_r2s = []

    for fold in range(n_folds):
        ts = fold * fold_size
        te = ts + fold_size if fold < n_folds - 1 else n_time
        test_mask = np.zeros(n_time, dtype=bool)
        test_mask[ts:te] = True
        train_mask = ~test_mask

        X_train = np.column_stack([X_b[:, train_mask].T, np.ones(train_mask.sum())])
        Z_train = Z_a[:, train_mask]
        X_test = np.column_stack([X_b[:, test_mask].T, np.ones(test_mask.sum())])
        Z_test = Z_a[:, test_mask]

        W_aug, _, _, _ = np.linalg.lstsq(X_train, Z_train.T, rcond=None)
        Z_pred = (X_test @ W_aug).T

        r2s = []
        for k in range(Z_a.shape[0]):
            ss_tot = float(np.sum((Z_test[k] - Z_test[k].mean()) ** 2))
            ss_res = float(np.sum((Z_test[k] - Z_pred[k]) ** 2))
            r2s.append(1.0 - ss_res / (ss_tot + 1e-12))
        fold_r2s.append(np.mean(r2s))

    return float(np.mean(fold_r2s))


def _phase_randomise(X, rng):
    """Phase-randomise each row of X independently (preserves power spectrum).

    For each row: FFT → randomise phases → IFFT → take real part.
    The autocorrelation structure and marginal variance are preserved,
    but cross-row temporal alignment is destroyed.
    """
    n_rows, n_cols = X.shape
    X_out = np.empty_like(X)
    for r in range(n_rows):
        freq = np.fft.rfft(X[r])
        phases = rng.uniform(0, 2 * np.pi, size=freq.shape)
        # Preserve DC and Nyquist (real-valued)
        phases[0] = 0.0
        if n_cols % 2 == 0:
            phases[-1] = 0.0
        freq_rand = np.abs(freq) * np.exp(1j * phases)
        X_out[r] = np.fft.irfft(freq_rand, n=n_cols)
    return X_out


# ===========================================================================
# Visualization: RDM heatmap
# ===========================================================================

def plot_rdm(rdm, title="RDM", time_axis=None, cmap='viridis',
             vmin=None, vmax=None):
    """Plot a Representational Dissimilarity Matrix as a heatmap.

    Args:
        rdm: np.ndarray (T x T).
        title: str.
        time_axis: np.ndarray (T,) — optional time values for axis labels.
        cmap: str — matplotlib colormap.
        vmin, vmax: float — colorbar range (auto-scales if None).

    Returns:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    if time_axis is not None:
        extent = [time_axis[0], time_axis[-1], time_axis[-1], time_axis[0]]
        im = ax.imshow(rdm, cmap=cmap, aspect='auto', extent=extent,
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
    else:
        im = ax.imshow(rdm, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel('Timepoint', fontsize=12)

    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, label='Dissimilarity', shrink=0.8)
    fig.tight_layout()
    return fig


def plot_rsa_comparison(rsa_results, title="RSA Comparison"):
    """Bar chart of RSA similarity values with significance markers.

    Args:
        rsa_results: dict of {label: (r_value, p_value)}
        title: str

    Returns:
        fig: matplotlib Figure
    """
    labels = list(rsa_results.keys())
    # Accept both tuple (r, p) and dict {'r': ..., 'p': ...} formats
    def _get_rp(v):
        if isinstance(v, dict):
            return v['r'], v['p']
        return v[0], v[1]
    r_values = [_get_rp(rsa_results[l])[0] for l in labels]
    p_values = [_get_rp(rsa_results[l])[1] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#e74c3c' if 'DA' in l or 'Dopamine' in l else
              '#3498db' if 'GABA' in l else '#2ecc71'
              for l in labels]
    ax.bar(range(len(labels)), r_values, color=colors, alpha=0.8)

    for i, (r, p) in enumerate(zip(r_values, p_values)):
        if np.isnan(p):
            marker = ''
        elif p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = 'n.s.'
        y_pos = r + 0.02 if r >= 0 else r - 0.05
        ax.text(i, y_pos, marker, ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('RSA Correlation (r)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


def plot_procrustes_comparison(procrustes_dict, title="Procrustes Distance Comparison"):
    """Bar chart of Procrustes disparity values across comparisons.

    Args:
        procrustes_dict: {label: disparity_float} or
                         {label: {'fwd': {...}, 'bwd': {...}, 'both': {...}}}
        title: str

    Returns:
        fig: matplotlib Figure
    """
    flat = {}
    for label, val in procrustes_dict.items():
        if isinstance(val, dict) and 'disparity' not in val:
            for direction, v in val.items():
                disp = v['disparity'] if isinstance(v, dict) else v
                flat[f"{label} ({direction})"] = disp
        elif isinstance(val, dict):
            flat[label] = val['disparity']
        else:
            flat[label] = val

    labels = list(flat.keys())
    values = list(flat.values())

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#e74c3c' if 'DA' in l or 'Dopamine' in l else
              '#3498db' if 'GABA' in l else '#2ecc71'
              for l in labels]
    ax.bar(range(len(labels)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Procrustes Disparity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


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

        # Event markers as vertical lines.
        # Labels are only added in the first subplot (i == 0) to avoid
        # duplicate legend entries across subplots.
        for marker in event_markers:
            t_event = window_data['plot_time'][marker['idx']] if marker['idx'] < len(plot_time) else None
            if t_event is not None:
                ax.axvline(t_event, color=marker['color'], linestyle='--',
                           linewidth=1.5, alpha=0.7,
                           label=marker['label'] if i == 0 else '_nolegend_')

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

    # Only keep indices present in both DataFrames and not in bad_rows
    shared_indices = set(X_fit_raw.index) & set(X_proj_raw.index)
    keep_rows = [idx for idx in shared_indices if idx not in bad_rows]
    n_dropped = len(set(X_fit_raw.index) | set(X_proj_raw.index)) - len(keep_rows)

    if n_dropped > 0:
        nan_fit_str = {f"{g}[{r}]" for g, r in nan_rows_fit}
        nan_proj_str = {f"{g}[{r}]" for g, r in nan_rows_proj}
        logger.warning(
            f"_align_neuron_data: dropping {n_dropped} neuron row(s) that have "
            f"NaN in at least one dataset or are not shared (fit NaN: {sorted(nan_fit_str)}, "
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


def cross_class_project(result_a, result_b, window=150, event_idx=600, dt=0.01,
                        sg_window=11, sg_order=3, n_folds=5):
    """Project class B neurons into class A's PC space via least-squares regression.

    Class A and class B are *different* neuron populations (different feature
    dimensions), so direct PCA projection is impossible.  Instead, we learn a
    linear map W that maps class B activations to class A PC scores::

        X_b.T @ W  ≈  Z_a.T      (least squares; W shape: n_b × n_components)

    and represent class B activity in A's PC coordinate frame as::

        Z_b_in_a = (X_b.T @ W).T      shape: (n_components × 2T)

    Both results **must** come from the same dataset so that the time axes are
    aligned (same trials, same timesteps).

    R² is reported both as *training* R² (same data used to fit W, upward-
    biased) **and** as *cross-validated* R² using k-fold splits on the time
    axis.  The CV R² is the unbiased estimate.

    Scientific use-case: fit on SpontFB_DF, project SpontFB_DB.
    If the DB-backward trajectory in DF PC space resembles the DF-forward
    trajectory, both populations encode the same direction variable.

    Args:
        result_a: analyze_dataset output for the *reference* class (PC axes).
        result_b: analyze_dataset output for the *projected* class.
        window, event_idx, dt, sg_window, sg_order: same as analyze_dataset.
        n_folds: int — number of folds for cross-validated R² (default 5).
            Each fold holds out a contiguous block of timepoints so that the
            temporal autocorrelation structure of neural data is preserved.

    Returns dict with: smooth_data, window_data, event_markers, metrics,
        r2_train (mean R² across PCs, biased), r2_cv (mean cross-validated R²),
        r2_cv_per_pc (list[float]), r2_per_pc (list[float], training),
        W (mapping matrix n_b × n_comp), fit_class, project_class, dataset.
    """
    X_a = result_a['X']
    X_b = result_b['X']
    pca_a = result_a['pca']

    X_a_arr = X_a.values if hasattr(X_a, 'values') else np.asarray(X_a)
    X_b_arr = X_b.values if hasattr(X_b, 'values') else np.asarray(X_b)

    # Class A PC time-courses: (n_comp × 2T) — use mean-corrected projection
    Z_a = pca_a.components_ @ (X_a_arr - pca_a.mean_[:, np.newaxis])

    # ---- Full-data (training) fit ----
    # Least-squares with intercept: [X_b.T | 1] @ [W; b] ≈ Z_a.T
    # The intercept absorbs any mean offset between populations.
    X_b_aug = np.column_stack([X_b_arr.T, np.ones(X_b_arr.shape[1])])
    W_aug, _, _, _ = np.linalg.lstsq(X_b_aug, Z_a.T, rcond=None)  # (n_b+1, n_comp)
    W = W_aug[:-1, :]          # (n_b, n_comp) mapping weights
    b = W_aug[-1, :]           # (n_comp,)     intercept

    # Class B projected into class A's PC space: (n_comp × 2T)
    Z_b_in_a = (X_b_arr.T @ W + b).T

    # Training R² per PC
    r2_per_pc = []
    for k in range(Z_a.shape[0]):
        ss_tot = float(np.sum((Z_a[k] - Z_a[k].mean()) ** 2))
        ss_res = float(np.sum((Z_a[k] - Z_b_in_a[k]) ** 2))
        r2_per_pc.append(1.0 - ss_res / (ss_tot + 1e-12))
    r2_train = float(np.mean(r2_per_pc))

    # ---- Cross-validated R² (k-fold, contiguous time blocks) ----
    # Splits are on the time axis so autocorrelation within a block is
    # preserved.  Each fold trains on the rest and evaluates on the held-out
    # block.
    n_time = X_b_arr.shape[1]   # 2T
    fold_size = n_time // n_folds
    cv_r2_folds = []  # shape: (n_folds, n_comp)

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end   = test_start + fold_size if fold < n_folds - 1 else n_time

        # Boolean masks
        test_mask  = np.zeros(n_time, dtype=bool)
        test_mask[test_start:test_end] = True
        train_mask = ~test_mask

        X_b_train = X_b_arr[:, train_mask]   # (n_b, n_train)
        Z_a_train  = Z_a[:, train_mask]       # (n_comp, n_train)
        X_b_test   = X_b_arr[:, test_mask]
        Z_a_test   = Z_a[:, test_mask]

        X_b_train_aug = np.column_stack([X_b_train.T, np.ones(X_b_train.shape[1])])
        W_fold_aug, _, _, _ = np.linalg.lstsq(X_b_train_aug, Z_a_train.T, rcond=None)
        W_fold = W_fold_aug[:-1, :]
        b_fold = W_fold_aug[-1, :]
        Z_pred_test = (X_b_test.T @ W_fold + b_fold).T  # (n_comp, n_test)

        fold_r2 = []
        for k in range(Z_a.shape[0]):
            ss_tot = float(np.sum((Z_a_test[k] - Z_a_test[k].mean()) ** 2))
            ss_res = float(np.sum((Z_a_test[k] - Z_pred_test[k]) ** 2))
            fold_r2.append(1.0 - ss_res / (ss_tot + 1e-12))
        cv_r2_folds.append(fold_r2)

    cv_r2_arr = np.array(cv_r2_folds)          # (n_folds, n_comp)
    r2_cv_per_pc = cv_r2_arr.mean(axis=0).tolist()
    r2_cv_per_pc_std = cv_r2_arr.std(axis=0).tolist()
    r2_cv = float(np.mean(r2_cv_per_pc))

    if r2_train - r2_cv > 0.1:
        logger.warning(
            f"cross_class_project: large gap between training R² ({r2_train:.3f}) "
            f"and CV R² ({r2_cv:.3f}) — mapping may be overfit.  "
            f"Consider reducing n_b or interpreting the CV value."
        )

    timesteps = result_a['timesteps']
    win_data = slice_window(Z_b_in_a, timesteps, event_idx, window, dt)
    smooth_data = smooth_trajectories(win_data, sg_window, sg_order)

    ds_name = result_a['config']['dataset_name']
    markers, _ = get_event_markers(ds_name, window)
    metrics = compute_trajectory_metrics(smooth_data, win_data, dt)

    return {
        'projections': Z_b_in_a,
        'window_data': win_data,
        'smooth_data': smooth_data,
        'event_markers': markers,
        'metrics': metrics,
        'r2_train': r2_train,
        'r2_per_pc': r2_per_pc,
        'r2_cv': r2_cv,
        'r2_cv_per_pc': r2_cv_per_pc,
        'r2_cv_per_pc_std': r2_cv_per_pc_std,
        'W': W,
        'fit_class': result_a['config']['combo_label'],
        'project_class': result_b['config']['combo_label'],
        'dataset': ds_name,
    }


# ===========================================================================
# Epoch-specific analysis and saving
# ===========================================================================

# Absolute event positions used by get_epoch_event_markers
_ABSOLUTE_EVENTS = {
    'SpontFB': [
        {'label': 'Spont', 'abs_idx': 600, 'color': 'purple',
         'symbol': 'diamond', 'size': 14},
    ],
    'CRFB': [
        {'label': 'CR', 'abs_idx': 600, 'color': 'red',
         'symbol': 'diamond', 'size': 14},
    ],
    'ToneFB': [
        {'label': 'Tone', 'abs_idx': 600, 'color': 'gold',
         'symbol': 'diamond', 'size': 14},
        {'label': 'Reward', 'abs_idx': 700, 'color': 'dodgerblue',
         'symbol': 'square', 'size': 14},
    ],
}


def get_epoch_event_markers(dataset_name, epoch_start, epoch_end):
    """Event markers with indices relative to epoch start.

    Translates absolute positions (event=600, reward=700) into epoch-
    relative indices.  Events outside the epoch are filtered out.

    Returns:
        markers: list of dicts
        warnings: list of str
    """
    epoch_len = epoch_end - epoch_start
    markers = [
        {'label': 'Start', 'idx': 0,
         'color': 'black', 'symbol': 'circle', 'size': 12},
        {'label': 'End', 'idx': epoch_len - 1,
         'color': 'green', 'symbol': 'circle', 'size': 12},
    ]
    for rule in _ABSOLUTE_EVENTS.get(dataset_name, []):
        rel_idx = rule['abs_idx'] - epoch_start
        if 0 <= rel_idx < epoch_len:
            markers.append({
                'label': rule['label'], 'idx': rel_idx,
                'color': rule['color'], 'symbol': rule['symbol'],
                'size': rule['size'],
            })
    return markers, []


def analyze_epoch(data, neuron_groups, dataset_name, combo_label,
                  epoch_name, epoch_start, epoch_end,
                  n_components=3, dt=0.01, sg_window=11, sg_order=3):
    """Run PCA on a specific time epoch within a dataset.

    Slices the neuron data to [epoch_start, epoch_end) for both fwd and
    bwd halves, fits PCA on that slice, and produces trajectory data.

    Returns dict with epoch-specific results.
    """
    X, ts, stats = extract_neuron_data(data, neuron_groups)
    X_epoch, epoch_ts = slice_epoch(X, ts, epoch_start, epoch_end)
    epoch_len = epoch_end - epoch_start

    n_comp = min(n_components, X_epoch.shape[0] - 1, epoch_len - 1)
    if n_comp < 1:
        raise ValueError(
            f"Not enough neurons or timepoints for PCA "
            f"(n_neurons={X_epoch.shape[0]}, epoch_len={epoch_len})"
        )

    pca_obj = fit_pca(X_epoch, n_comp)
    projections = project_onto_pca(pca_obj, X_epoch)
    evr = list(pca_obj.explained_variance_ratio_)

    # Build window_data: fwd is first half, bwd is second half
    fwd = projections[:, :epoch_len]
    bwd = projections[:, epoch_len:]
    plot_time = (np.arange(epoch_len) + epoch_start - 600) * dt  # relative to event=600

    win_data = {
        'fwd': fwd, 'bwd': bwd,
        'n_plot': epoch_len, 'plot_time': plot_time,
    }

    # Smooth (only if epoch_len > sg_window)
    eff_sg = min(sg_window, epoch_len - 1)
    if eff_sg % 2 == 0:
        eff_sg -= 1
    eff_sg = max(eff_sg, 3)
    eff_order = min(sg_order, eff_sg - 1)
    smooth_data = smooth_trajectories(win_data, eff_sg, eff_order)

    markers, _ = get_epoch_event_markers(dataset_name, epoch_start, epoch_end)
    metrics = compute_trajectory_metrics(smooth_data, win_data, dt)

    return {
        'X_epoch': X_epoch,
        'pca': pca_obj,
        'projections': projections,
        'explained_variance_ratio': evr,
        'n_neurons': X_epoch.shape[0],
        'stats': stats,
        'window_data': win_data,
        'smooth_data': smooth_data,
        'event_markers': markers,
        'metrics': metrics,
        'epoch_name': epoch_name,
        'epoch_start': epoch_start,
        'epoch_end': epoch_end,
        'config': {
            'dataset_name': dataset_name, 'combo_label': combo_label,
            'neuron_groups': neuron_groups, 'n_components': n_comp,
        },
    }


def save_epoch_trajectories(datasets_config, neuron_combos, epochs_config,
                            output_base='outputs', n_components=3,
                            dt=0.01, sg_window=11, sg_order=3,
                            fig_width=1100, fig_height=800):
    """Generate and save epoch-specific trajectory PNGs for all combos.

    Args:
        datasets_config: dict like {'SpontFB': {'mat_file': ..., 'var_name': ...}}
        neuron_combos: dict like {'Dopamine': ['DF','DB','D','DFB'], ...}
        epochs_config: dict like EPOCHS
        output_base: root output directory

    Returns list of saved file paths.
    """
    saved_files = []

    for ds_name, ds_cfg in datasets_config.items():
        data = load_dataset(ds_cfg['mat_file'], ds_cfg['var_name'])

        for combo_name, groups in neuron_combos.items():
            for ep_name, ep_cfg in epochs_config.items():
                # Only run epochs relevant to this dataset
                if ep_cfg['dataset'] != ds_name and ep_cfg['dataset'] != 'Any':
                    continue

                try:
                    ep_result = analyze_epoch(
                        data, groups, ds_name, combo_name,
                        ep_name, ep_cfg['start'], ep_cfg['end'],
                        n_components=n_components, dt=dt,
                        sg_window=sg_window, sg_order=sg_order,
                    )
                except Exception as e:
                    logger.warning(
                        f"Epoch {ep_name} for {ds_name}/{combo_name} failed: {e}"
                    )
                    continue

                evr = ep_result['explained_variance_ratio']
                evr_str = '+'.join(f'{v:.1%}' for v in evr)
                title = (
                    f"{ds_name} – {combo_name} – {ep_name}\n"
                    f"({ep_cfg['desc']}, "
                    f"n={ep_result['n_neurons']}, EVR={evr_str})"
                )

                fig = build_figure(
                    ep_result['window_data'],
                    ep_result['smooth_data'],
                    ep_result['event_markers'],
                    title,
                    plot_type='trajectory',
                    width=fig_width, height=fig_height,
                )

                out_dir = os.path.join(
                    output_base, ds_name, combo_name, 'epochs'
                )
                os.makedirs(out_dir, exist_ok=True)
                fpath = os.path.join(out_dir, f'{ep_name}_trajectory.png')
                try:
                    fig.write_image(fpath, scale=2)
                    saved_files.append(fpath)
                    print(f"  Saved: {fpath}")
                except Exception as e:
                    logger.warning(f"PNG export failed for {fpath}: {e}")

    return saved_files


# ===========================================================================
# Sub-population selectivity comparison (Issue 12)
# ===========================================================================

def compare_selectivity_subpopulations(mat_file, var_name, dataset_name,
                                        all_groups, selective_groups,
                                        n_components=3, event_idx=600,
                                        window=150, dt=0.01,
                                        sg_window=11, sg_order=3):
    """Compare PCA structure between direction-selective-only and full populations.

    Tests whether including direction-non-selective neurons (D, DFB) dilutes
    the directional structure in PC space.  Runs PCA independently on:

        1. ``selective_groups`` — e.g. ['DF', 'DB'] (pure direction-selective)
        2. ``selective_groups + mixed`` — e.g. ['DF', 'DB', 'DFB']
        3. ``all_groups``          — e.g. ['DF', 'DB', 'D', 'DFB'] (full combo)

    and returns a summary DataFrame with EVR and post-event separation for
    each sub-population, making it easy to see whether unclassified or mixed-
    selectivity neurons add or remove directional structure.

    Args:
        mat_file: str — path to .mat file.
        var_name: str — variable name within the .mat file.
        dataset_name: str — e.g. 'SpontFB'.
        all_groups: list[str] — full group list, e.g. ['DF','DB','D','DFB'].
        selective_groups: list[str] — direction-selective subset, e.g. ['DF','DB'].
        n_components, event_idx, window, dt, sg_window, sg_order: as elsewhere.

    Returns:
        summary_df: pd.DataFrame with columns:
            subpopulation, groups, n_neurons, evr_pc1, evr_pc2, evr_pc3,
            evr_total, peak_separation, post_event_mean_separation
        results: dict keyed by subpopulation label -> analyze_dataset output
    """
    data = load_dataset(mat_file, var_name)

    # Build sub-population definitions
    non_selective = [g for g in all_groups if g not in selective_groups]
    mixed_groups   = [g for g in non_selective if g.endswith('FB') or g.endswith('fb')]
    unclassified   = [g for g in non_selective if g not in mixed_groups]

    subpops = {'selective_only': selective_groups}
    if mixed_groups:
        subpops['selective_and_mixed'] = selective_groups + mixed_groups
    subpops['all'] = all_groups

    records = []
    results = {}

    for label, groups in subpops.items():
        # Filter to groups that actually exist in data
        available = list(data['firing_rate'].keys())
        groups_present = [g for g in groups if g in available]
        if not groups_present:
            logger.warning(f"compare_selectivity_subpopulations: no groups available for '{label}', skipping.")
            continue
        try:
            res = analyze_dataset(
                mat_file, var_name, dataset_name, groups_present,
                combo_label=label, n_components=n_components,
                event_idx=event_idx, window=window, dt=dt,
                sg_window=sg_window, sg_order=sg_order,
            )
        except Exception as e:
            logger.warning(f"compare_selectivity_subpopulations: '{label}' failed: {e}")
            continue

        evr = res['explained_variance_ratio']
        m   = res['metrics']
        records.append({
            'subpopulation':            label,
            'groups':                   '+'.join(groups_present),
            'n_neurons':                res['n_neurons'],
            'evr_pc1':                  evr[0] if len(evr) > 0 else None,
            'evr_pc2':                  evr[1] if len(evr) > 1 else None,
            'evr_pc3':                  evr[2] if len(evr) > 2 else None,
            'evr_total':                sum(evr),
            'peak_separation':          m['peak_separation'],
            'post_event_mean_separation': m['post_event_mean_separation'],
        })
        results[label] = res

    summary_df = pd.DataFrame(records)
    return summary_df, results


# ===========================================================================
# Participation ratio (effective dimensionality)
# ===========================================================================

def compute_participation_ratio(X, max_components=None):
    """Compute the participation ratio of the covariance spectrum.

    PR = (sum lambda_i)^2 / sum(lambda_i^2)

    This gives the effective dimensionality of the data.
    A value of 1 means all variance is in one dimension; a value of N
    means variance is uniformly distributed across N dimensions.

    Args:
        X: data matrix (n_neurons, n_timepoints) or (n_neurons, 2*timesteps)
        max_components: if set, use only this many components

    Returns:
        pr: float (participation ratio)
        eigenvalues: array of explained variances
    """
    X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
    n_comp = max_components or min(X_arr.shape) - 1
    n_comp = min(n_comp, min(X_arr.shape) - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(X_arr.T)
    lambdas = pca.explained_variance_
    pr = float(np.sum(lambdas) ** 2 / np.sum(lambdas ** 2))
    return pr, lambdas


# ===========================================================================
# Trajectory divergence onset
# ===========================================================================

def compute_divergence_onset(smooth_data, window_data, dt=0.01,
                             baseline_window=20, threshold_factor=2.0):
    """Find when fwd and bwd trajectories diverge significantly.

    Computes fwd-bwd Euclidean distance over time, estimates baseline
    separation from the first `baseline_window` timepoints, and finds
    the first timepoint where separation exceeds `threshold_factor` ×
    baseline.

    Args:
        smooth_data: dict with 'fwd_smooth', 'bwd_smooth'
        window_data: dict with 'plot_time'
        dt: timestep in seconds
        baseline_window: number of initial timepoints for baseline estimate
        threshold_factor: multiplier on baseline to define divergence

    Returns:
        onset_time: float (seconds relative to event), or None if never
        onset_idx: int index, or None
        separation: array of fwd-bwd distance over time
        threshold: float (the separation threshold used)
    """
    fwd = smooth_data['fwd_smooth']
    bwd = smooth_data['bwd_smooth']
    separation = np.sqrt(np.sum((fwd - bwd) ** 2, axis=0))
    plot_time = window_data['plot_time']

    bw = min(baseline_window, len(separation) // 4)
    baseline = np.mean(separation[:bw])
    threshold = baseline * threshold_factor

    exceeding = np.where(separation > threshold)[0]
    if len(exceeding) == 0:
        return None, None, separation, threshold

    onset_idx = int(exceeding[0])
    onset_time = float(plot_time[onset_idx])
    return onset_time, onset_idx, separation, threshold


# ===========================================================================
# PC loading analysis
# ===========================================================================

def compute_pc_loadings_by_group(pca_obj, neuron_groups, stats):
    """Compute mean absolute loading per neuron group for each PC.

    Args:
        pca_obj: fitted PCA object (components_ shape: n_comp x n_neurons)
        neuron_groups: list of group names in order they were concatenated
        stats: dict from extract_neuron_data with per-group {kept} counts

    Returns:
        loadings_df: DataFrame (n_components x n_groups) with mean |loading|
    """
    components = pca_obj.components_  # (n_comp, n_neurons)
    n_comp = components.shape[0]

    group_loadings = {}
    offset = 0
    for group in neuron_groups:
        n_kept = stats[group]['kept']
        if n_kept == 0:
            group_loadings[group] = [0.0] * n_comp
            continue
        group_slice = components[:, offset:offset + n_kept]
        group_loadings[group] = np.mean(np.abs(group_slice), axis=1).tolist()
        offset += n_kept

    return pd.DataFrame(group_loadings, index=[f'PC{i+1}' for i in range(n_comp)])


def plot_pc_loadings(loadings_df, title="PC Loadings by Neuron Group"):
    """Bar chart of mean absolute loading per group for each PC.

    Args:
        loadings_df: DataFrame from compute_pc_loadings_by_group()
    """
    n_comp = len(loadings_df)
    groups = loadings_df.columns.tolist()
    x = np.arange(len(groups))
    width = 0.8 / n_comp

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(n_comp):
        ax.bar(x + i * width, loadings_df.iloc[i], width,
               label=f'PC{i+1}', alpha=0.8)

    ax.set_xticks(x + width * (n_comp - 1) / 2)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel('Mean |Loading|', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


# ===========================================================================
# Visualization: participation ratio comparison
# ===========================================================================

def plot_participation_ratio_comparison(pr_dict, title="Participation Ratio"):
    """Bar chart comparing effective dimensionality across analyses.

    Args:
        pr_dict: {label: participation_ratio_value}
    """
    labels = list(pr_dict.keys())
    values = [pr_dict[l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#e74c3c' if 'Dopamine' in l else
              '#3498db' if 'GABA' in l else '#2ecc71'
              for l in labels]
    ax.bar(range(len(labels)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Participation Ratio', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    return fig


# ===========================================================================
# Visualization: divergence onset comparison
# ===========================================================================

def plot_divergence_comparison(div_dict, title="Fwd-Bwd Trajectory Divergence"):
    """Plot separation curves with divergence onset markers.

    Args:
        div_dict: {label: (onset_time, onset_idx, separation, threshold, plot_time)}
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    for label, (onset_t, onset_i, sep, thresh, t) in div_dict.items():
        line, = ax.plot(t, sep, linewidth=1.5, label=label)
        if onset_t is not None:
            ax.axvline(onset_t, color=line.get_color(), linestyle='--',
                       alpha=0.5)
            ax.plot(onset_t, sep[onset_i], 'o', color=line.get_color(),
                    markersize=8)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Fwd-Bwd Separation', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig