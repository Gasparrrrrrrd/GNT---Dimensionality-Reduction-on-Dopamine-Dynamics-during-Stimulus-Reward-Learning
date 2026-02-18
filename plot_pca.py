import os
import json
import datetime
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Data loading helpers (unchanged from original)
# ---------------------------------------------------------------------------

def matlab_struct_to_dict(matlab_struct):
    if isinstance(matlab_struct, np.ndarray) and matlab_struct.dtype.names:
        matlab_struct = matlab_struct[0, 0] if matlab_struct.size > 0 else matlab_struct
    if not hasattr(matlab_struct, 'dtype') or not matlab_struct.dtype.names:
        return matlab_struct
    result = {}
    for field in matlab_struct.dtype.names:
        val = matlab_struct[field]
        if isinstance(val, np.ndarray) and val.dtype.names:
            result[field] = matlab_struct_to_dict(val)
        else:
            result[field] = val.squeeze() if isinstance(val, np.ndarray) else val
    return result


def arrays_to_dfs(d, parent_key=''):
    result = {}
    for key, val in d.items():
        full_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(val, dict):
            result[full_key] = arrays_to_dfs(val, full_key)
        elif isinstance(val, np.ndarray):
            if val.dtype == float:
                result[full_key] = pd.DataFrame(val)
        else:
            result[full_key] = val
    return result


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def load_dataset(mat_file, var_name):
    """Load a .mat file and return the data as a nested Python dict."""
    mat_data = sio.loadmat(mat_file)
    if var_name not in mat_data:
        raise KeyError(
            f"Variable '{var_name}' not found in '{mat_file}'. "
            f"Available keys: {[k for k in mat_data if not k.startswith('__')]}"
        )
    data = matlab_struct_to_dict(mat_data[var_name])
    return data


def extract_neuron_data(data, neuron_groups):
    """Extract z-scored firing rates for requested neuron groups and concatenate.

    For each group, horizontally concatenates [zforward | zbackward].
    Then stacks all groups vertically.

    Returns:
        X: pd.DataFrame of shape (total_neurons, 2 * timesteps)
        timesteps: number of timesteps per half (e.g. 1201)
    """
    data_dfs = arrays_to_dfs(data)
    available = list(data['firing_rate'].keys())
    missing = [g for g in neuron_groups if g not in available]
    if missing:
        raise ValueError(
            f"Neuron group(s) {missing} not found in dataset. "
            f"Available groups: {available}"
        )

    parts = []
    timesteps = None
    for group in neuron_groups:
        fr = data_dfs['firing_rate'][f'firing_rate_{group}']
        zfwd = fr[f'firing_rate_{group}_zforward']
        zbwd = fr[f'firing_rate_{group}_zbackward']
        if timesteps is None:
            timesteps = zfwd.shape[1]
        combined = pd.concat([zfwd, zbwd], axis=1)
        parts.append(combined)

    X = pd.concat(parts, axis=0)
    return X, timesteps


# ---------------------------------------------------------------------------
# PCA computation
# ---------------------------------------------------------------------------

def run_pca(X, n_components=3):
    """Fit PCA on X.T and project all data onto the components.

    Returns:
        projections: np.ndarray of shape (n_components, 2*timesteps)
        components: np.ndarray of shape (n_components, total_neurons)
        explained_variance_ratio: list of floats
    """
    pca = PCA(n_components=n_components)
    pca.fit(X.T)
    components = pca.components_
    projections = np.array(components @ X)
    return projections, components, list(pca.explained_variance_ratio_)


def slice_window(projections, timesteps, event_idx=600, window=100, dt=0.01):
    """Slice PCA projections into forward/backward windows around the event.

    Returns dict with keys:
        fwd: np.ndarray (n_components, 2*window+1) - forward window
        bwd: np.ndarray (n_components, 2*window+1) - backward window
        n_plot: int (2*window + 1)
        plot_time: np.ndarray - time axis centred on event (seconds)
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


def smooth_trajectories(window_data, sg_window=11, sg_order=3):
    """Apply Savitzky-Golay smoothing to forward and backward projections.

    Returns dict with keys:
        fwd_smooth: np.ndarray (n_components, n_plot)
        bwd_smooth: np.ndarray (n_components, n_plot)
    """
    n_comp = window_data['fwd'].shape[0]
    fwd_smooth = np.array([
        savgol_filter(window_data['fwd'][i], sg_window, sg_order)
        for i in range(n_comp)
    ])
    bwd_smooth = np.array([
        savgol_filter(window_data['bwd'][i], sg_window, sg_order)
        for i in range(n_comp)
    ])
    return {'fwd_smooth': fwd_smooth, 'bwd_smooth': bwd_smooth}


# ---------------------------------------------------------------------------
# Event markers
# ---------------------------------------------------------------------------

EVENT_RULES = {
    'SpontFB': [
        {'label': 'Spont', 'offset': 0, 'color': 'purple', 'symbol': 'diamond', 'size': 14},
    ],
    'CRFB': [
        {'label': 'CR', 'offset': 0, 'color': 'red', 'symbol': 'diamond', 'size': 14},
    ],
    'ToneFB': [
        {'label': 'Tone', 'offset': 0, 'color': 'gold', 'symbol': 'diamond', 'size': 14},
        {'label': 'Reward', 'offset': 100, 'color': 'dodgerblue', 'symbol': 'square', 'size': 14},
    ],
}


def get_event_markers(dataset_name, window=100):
    """Return a list of event marker dicts for the given dataset.

    Each dict: {label, idx, color, symbol, size}.
    Always includes Start (idx=0) and End (idx=2*window).
    Dataset-specific events are added based on EVENT_RULES.
    Events outside the window are skipped with a warning.

    Returns:
        markers: list of dicts
        warnings: list of warning strings
    """
    warnings = []
    max_idx = 2 * window

    markers = [
        {'label': 'Start', 'idx': 0, 'color': 'black', 'symbol': 'circle', 'size': 12},
        {'label': 'End', 'idx': max_idx, 'color': 'green', 'symbol': 'circle', 'size': 12},
    ]

    for rule in EVENT_RULES.get(dataset_name, []):
        idx = window + rule['offset']
        if idx < 0 or idx > max_idx:
            warnings.append(
                f"Event '{rule['label']}' at offset {rule['offset']} "
                f"falls outside window [0, {max_idx}], skipping"
            )
            continue
        markers.append({
            'label': rule['label'],
            'idx': idx,
            'color': rule['color'],
            'symbol': rule['symbol'],
            'size': rule['size'],
        })

    return markers, warnings


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def cmap_to_rgb_strings(cmap, t):
    colors = []
    for v in t:
        r, g, b, _ = cmap(v)
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    return colors


def build_plotly_colorscale(cmap, t_range=(0.15, 1.0), n=10):
    stops = np.linspace(t_range[0], t_range[1], n)
    scale = []
    for i, v in enumerate(stops):
        r, g, b, _ = cmap(v)
        scale.append([i / (n - 1), f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'])
    return scale


def endpoint_trace(x, y, z, color, name, symbol='circle', size=12, showlegend=True):
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers',
        name=name,
        showlegend=showlegend,
        legendgroup=name,
        marker=dict(size=size, color=color, symbol=symbol,
                    line=dict(color='black', width=1.5))
    )


def colorbar_trace(plotly_cs, title, x_pos, tickvals, ticktext):
    return go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        showlegend=False,
        marker=dict(
            size=0,
            color=[0, 1],
            colorscale=plotly_cs,
            showscale=True,
            colorbar=dict(
                title=dict(text=title, side='right'),
                x=x_pos, len=0.5, y=0.75, thickness=15,
                tickvals=tickvals,
                ticktext=ticktext
            )
        )
    )


SCENE_AXES = dict(
    showgrid=True, gridcolor='lightgrey',
    zeroline=True, zerolinewidth=2, zerolinecolor='lightgrey',
    showline=True, linewidth=2, linecolor='black', mirror=True
)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(window_data, smooth_data, event_markers, title,
                 plot_type='scatter', fwd_cmap_name='YlOrRd',
                 bwd_cmap_name='Blues', t_norm_range=(0.15, 1.0),
                 width=1100, height=800):
    """Build a Plotly 3D figure with scatter, trajectory, and event markers.

    plot_type:
        'scatter'    - scatter gradient (opacity 0.85) + smooth trajectory
        'trajectory' - scatter dots (opacity 0.6) + smooth trajectory (showlegend)
    """
    n_plot = window_data['n_plot']
    plot_time = window_data['plot_time']

    fwd = window_data['fwd']
    bwd = window_data['bwd']
    fwd_s = smooth_data['fwd_smooth']
    bwd_s = smooth_data['bwd_smooth']

    # Color mapping
    fwd_cmap = plt.get_cmap(fwd_cmap_name)
    bwd_cmap = plt.get_cmap(bwd_cmap_name)
    t_norm = np.linspace(t_norm_range[0], t_norm_range[1], n_plot)

    fwd_colors = cmap_to_rgb_strings(fwd_cmap, t_norm)
    bwd_colors = cmap_to_rgb_strings(bwd_cmap, t_norm)
    fwd_end_color = fwd_colors[-1]
    bwd_end_color = bwd_colors[-1]

    fwd_plotly_cs = build_plotly_colorscale(fwd_cmap, t_norm_range)
    bwd_plotly_cs = build_plotly_colorscale(bwd_cmap, t_norm_range)

    cb_tickvals = [0, 0.5, 1.0]
    cb_ticktext = [f'{plot_time[0]:.1f}s', '0s', f'{plot_time[-1]:.1f}s']

    is_scatter = (plot_type == 'scatter')
    scatter_opacity = 0.85 if is_scatter else 0.6
    traj_showlegend = not is_scatter

    fig = go.Figure()

    # Colorbars
    fig.add_trace(colorbar_trace(fwd_plotly_cs, 'Forward<br>time (s)', 1.02,
                                 cb_tickvals, cb_ticktext))
    fig.add_trace(colorbar_trace(bwd_plotly_cs, 'Backward<br>time (s)', 1.14,
                                 cb_tickvals, cb_ticktext))

    # Forward scatter
    fig.add_trace(go.Scatter3d(
        x=fwd[0], y=fwd[1], z=fwd[2],
        mode='markers', name='Forward', legendgroup='Forward',
        marker=dict(size=5, color=fwd_colors, opacity=scatter_opacity),
        hovertemplate='Forward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
    ))

    # Backward scatter
    fig.add_trace(go.Scatter3d(
        x=bwd[0], y=bwd[1], z=bwd[2],
        mode='markers', name='Backward', legendgroup='Backward',
        marker=dict(size=5, color=bwd_colors, opacity=scatter_opacity),
        hovertemplate='Backward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
    ))

    # Forward trajectory
    fig.add_trace(go.Scatter3d(
        x=fwd_s[0], y=fwd_s[1], z=fwd_s[2],
        mode='lines', name='Forward traj.',
        legendgroup='Forward', showlegend=traj_showlegend,
        line=dict(color=fwd_end_color, width=5 if is_scatter else 6)
    ))

    # Backward trajectory
    fig.add_trace(go.Scatter3d(
        x=bwd_s[0], y=bwd_s[1], z=bwd_s[2],
        mode='lines', name='Backward traj.',
        legendgroup='Backward', showlegend=traj_showlegend,
        line=dict(color=bwd_end_color, width=5 if is_scatter else 6)
    ))

    # Event markers
    seen_labels = set()
    for marker in event_markers:
        idx = marker['idx']
        label = marker['label']
        show = label not in seen_labels
        seen_labels.add(label)

        # Forward event marker
        fig.add_trace(endpoint_trace(
            fwd[0][idx], fwd[1][idx], fwd[2][idx],
            color=marker['color'], name=label,
            symbol=marker['symbol'], size=marker['size'],
            showlegend=show,
        ))
        # Backward event marker
        fig.add_trace(endpoint_trace(
            bwd[0][idx], bwd[1][idx], bwd[2][idx],
            color=marker['color'], name=label,
            symbol=marker['symbol'], size=marker['size'],
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, y=0.95, x=0.5, xanchor='center', yanchor='top'),
        scene=dict(
            xaxis=dict(title="PC1", **SCENE_AXES),
            yaxis=dict(title="PC2", **SCENE_AXES),
            zaxis=dict(title="PC3", **SCENE_AXES),
        ),
        font=dict(family="Times New Roman, serif", size=12, color="black"),
        margin=dict(l=80, r=220, t=80, b=80),
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(title="", orientation="v", x=1.28, y=0.4,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='lightgrey', borderwidth=1),
        width=width, height=height,
    )

    return fig


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def plot_pca(mat_file, var_name, dataset_name, neuron_groups, neuron_combo_label,
             n_components=3, event_idx=600, window=100, dt=0.01,
             sg_window=11, sg_order=3, fwd_cmap_name='YlOrRd', bwd_cmap_name='Blues',
             output_dir=None, show=True, fig_width=1100, fig_height=800):
    """Full PCA pipeline: load -> extract -> PCA -> slice -> smooth -> plot -> save.

    Returns dict with keys:
        figures: {'scatter': go.Figure, 'trajectory': go.Figure}
        explained_variance_ratio: list of floats
        n_neurons: int
        neuron_groups_used: list of str
        saved_files: list of str (paths to PNGs if output_dir was set)
        warnings: list of str
    """
    warnings_list = []

    # Load and extract
    data = load_dataset(mat_file, var_name)
    X, timesteps = extract_neuron_data(data, neuron_groups)
    n_neurons = X.shape[0]

    # PCA
    projections, components, evr = run_pca(X, n_components)

    # Window and smooth
    win_data = slice_window(projections, timesteps, event_idx, window, dt)
    smooth_data = smooth_trajectories(win_data, sg_window, sg_order)

    # Event markers
    markers, marker_warnings = get_event_markers(dataset_name, window)
    warnings_list.extend(marker_warnings)

    # Build figures
    figures = {}
    for plot_type in ['scatter', 'trajectory']:
        type_label = 'scatter' if plot_type == 'scatter' else 'smooth trajectory'
        title = (f"{dataset_name} – {neuron_combo_label} – "
                 f"PCA projection – {type_label} "
                 f"(±{window} timesteps around event)")
        fig = build_figure(
            win_data, smooth_data, markers, title,
            plot_type=plot_type,
            fwd_cmap_name=fwd_cmap_name, bwd_cmap_name=bwd_cmap_name,
            width=fig_width, height=fig_height,
        )
        figures[plot_type] = fig
        if show:
            fig.show()

    # Save PNGs
    saved_files = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for plot_type, fig in figures.items():
            fname = f"{dataset_name}_{neuron_combo_label}_{plot_type}.png"
            fpath = os.path.join(output_dir, fname)
            try:
                fig.write_image(fpath, scale=2)
                saved_files.append(fpath)
            except Exception as e:
                warnings_list.append(f"PNG export failed for {fname}: {e}")

    return {
        'figures': figures,
        'explained_variance_ratio': evr,
        'n_neurons': n_neurons,
        'neuron_groups_used': list(neuron_groups),
        'saved_files': saved_files,
        'warnings': warnings_list,
    }


# ---------------------------------------------------------------------------
# Batch analysis driver
# ---------------------------------------------------------------------------

def run_analysis(config):
    """Iterate all datasets x neuron combos, call plot_pca(), save outputs.

    Returns dict with keys:
        results: list of dicts (one per dataset x combo, suitable for DataFrame)
        figures: dict keyed by '{dataset}_{combo}' -> {'scatter': fig, 'trajectory': fig}
        summary_csv_path: str or None
    """
    results = []
    all_figures = {}

    datasets = config['datasets']
    combos = config['neuron_combos']
    pca_cfg = config.get('pca', {})
    win_cfg = config.get('window', {})
    smooth_cfg = config.get('smoothing', {})
    vis_cfg = config.get('visualization', {})
    out_cfg = config.get('output', {})

    base_dir = out_cfg.get('base_dir', 'outputs')
    save_png = out_cfg.get('save_png', True)
    save_manifest = out_cfg.get('save_manifest', True)
    show_figures = out_cfg.get('show_figures', False)
    plot_types = vis_cfg.get('plot_types', ['scatter', 'trajectory'])

    for ds_name, ds_cfg in datasets.items():
        for combo_name, neuron_groups in combos.items():
            output_dir = os.path.join(base_dir, ds_name, combo_name)
            row = {
                'dataset': ds_name,
                'neuron_combo': combo_name,
                'neuron_groups': '+'.join(neuron_groups),
            }

            try:
                result = plot_pca(
                    mat_file=ds_cfg['mat_file'],
                    var_name=ds_cfg['var_name'],
                    dataset_name=ds_name,
                    neuron_groups=neuron_groups,
                    neuron_combo_label=combo_name,
                    n_components=pca_cfg.get('n_components', 3),
                    event_idx=win_cfg.get('event_idx', 600),
                    window=win_cfg.get('window', 100),
                    dt=win_cfg.get('dt', 0.01),
                    sg_window=smooth_cfg.get('sg_window', 11),
                    sg_order=smooth_cfg.get('sg_order', 3),
                    fwd_cmap_name=vis_cfg.get('fwd_cmap', 'YlOrRd'),
                    bwd_cmap_name=vis_cfg.get('bwd_cmap', 'Blues'),
                    output_dir=output_dir if save_png else None,
                    show=show_figures,
                    fig_width=vis_cfg.get('fig_width', 1100),
                    fig_height=vis_cfg.get('fig_height', 800),
                )

                evr = result['explained_variance_ratio']
                row.update({
                    'n_neurons': result['n_neurons'],
                    'explained_var_pc1': evr[0] if len(evr) > 0 else None,
                    'explained_var_pc2': evr[1] if len(evr) > 1 else None,
                    'explained_var_pc3': evr[2] if len(evr) > 2 else None,
                    'explained_var_total': sum(evr),
                    'scatter_png': None,
                    'trajectory_png': None,
                    'manifest_json': None,
                    'status': 'success',
                    'warnings': '; '.join(result['warnings']) if result['warnings'] else '',
                    'error': None,
                })

                # Record saved file paths
                for fpath in result['saved_files']:
                    if '_scatter.png' in fpath:
                        row['scatter_png'] = fpath
                    elif '_trajectory.png' in fpath:
                        row['trajectory_png'] = fpath

                # Save manifest JSON
                if save_manifest:
                    os.makedirs(output_dir, exist_ok=True)
                    manifest = {
                        'dataset': ds_name,
                        'neuron_combo': combo_name,
                        'neuron_groups': list(neuron_groups),
                        'n_neurons': result['n_neurons'],
                        'n_components': pca_cfg.get('n_components', 3),
                        'explained_variance_ratio': evr,
                        'explained_variance_total': sum(evr),
                        'parameters': {
                            'event_idx': win_cfg.get('event_idx', 600),
                            'window': win_cfg.get('window', 100),
                            'dt': win_cfg.get('dt', 0.01),
                            'sg_window': smooth_cfg.get('sg_window', 11),
                            'sg_order': smooth_cfg.get('sg_order', 3),
                        },
                        'event_markers': [m['label'] for m in
                                          get_event_markers(ds_name, win_cfg.get('window', 100))[0]],
                        'files': {
                            'scatter_png': row.get('scatter_png'),
                            'trajectory_png': row.get('trajectory_png'),
                        },
                        'warnings': result['warnings'],
                        'timestamp': datetime.datetime.now().isoformat(),
                    }
                    manifest_path = os.path.join(
                        output_dir, f"{ds_name}_{combo_name}_manifest.json"
                    )
                    with open(manifest_path, 'w') as f:
                        json.dump(manifest, f, indent=2)
                    row['manifest_json'] = manifest_path

                key = f"{ds_name}_{combo_name}"
                all_figures[key] = result['figures']

            except Exception as e:
                row.update({
                    'n_neurons': None,
                    'explained_var_pc1': None,
                    'explained_var_pc2': None,
                    'explained_var_pc3': None,
                    'explained_var_total': None,
                    'scatter_png': None,
                    'trajectory_png': None,
                    'manifest_json': None,
                    'status': 'error',
                    'warnings': '',
                    'error': str(e),
                })

            results.append(row)

    # Save summary CSV
    summary_csv_path = None
    if out_cfg.get('save_summary_csv', True):
        os.makedirs(base_dir, exist_ok=True)
        summary_csv_path = os.path.join(base_dir, 'analysis_summary.csv')
        pd.DataFrame(results).to_csv(summary_csv_path, index=False)

    return {
        'results': results,
        'figures': all_figures,
        'summary_csv_path': summary_csv_path,
    }
