import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load .mat file
mat_data = sio.loadmat('DataSpontFB.mat')
DataSpontFB = mat_data['dataSpontFB']

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

# Convert to clean dict
data = matlab_struct_to_dict(DataSpontFB)
data_dfs = arrays_to_dfs(data)

# Get the z-scored firing rates for forward and backward
firing_rate_zscore_forward_dfs = {}
firing_rate_zscore_backward_dfs = {}

for neuron_name in data['firing_rate'].keys():
    firing_rate_zscore_forward_dfs[neuron_name] = data_dfs['firing_rate'][f'firing_rate_{neuron_name}'][f'firing_rate_{neuron_name}_zforward']
    firing_rate_zscore_backward_dfs[neuron_name] = data_dfs['firing_rate'][f'firing_rate_{neuron_name}'][f'firing_rate_{neuron_name}_zbackward']

# Concatenate data: X has shape (neurons, 2*time)
X = pd.concat([firing_rate_zscore_forward_dfs['DF'], firing_rate_zscore_backward_dfs['DF']], axis=1)

# Get time information (number of timesteps per event)
neurons, time = firing_rate_zscore_forward_dfs['DF'].shape

# Fit PCA on ALL timesteps (full X, both forward and backward)
pca = PCA(n_components=3)
pca.fit(X.T)

PC1 = pca.components_[0]
PC2 = pca.components_[1]
PC3 = pca.components_[2]

dt = 0.01

# --- Define window around events for plotting ---
event_idx = 600   # event position within each half (index)
window    = 70    # ±window timesteps around each event

fwd_start = event_idx - window
fwd_end   = event_idx + window
bwd_start = time + event_idx - window
bwd_end   = time + event_idx + window

n_plot = 2 * window + 1

# Time axis centred on the event (in seconds)
plot_time = (np.arange(n_plot) - window) * dt

# PCA projections on ALL timesteps (numpy to avoid pandas index issues)
PC1_proj = np.array(PC1 @ X)
PC2_proj = np.array(PC2 @ X)
PC3_proj = np.array(PC3 @ X)

# --- Slice projections to ±window timesteps around each event ---
PC1_fwd = PC1_proj[fwd_start:fwd_end + 1]
PC2_fwd = PC2_proj[fwd_start:fwd_end + 1]
PC3_fwd = PC3_proj[fwd_start:fwd_end + 1]

PC1_bwd = PC1_proj[bwd_start:bwd_end + 1]
PC2_bwd = PC2_proj[bwd_start:bwd_end + 1]
PC3_bwd = PC3_proj[bwd_start:bwd_end + 1]

# --- Smooth trajectory via Savitzky-Golay ---
sg_window = 11
sg_order  = 3

fwd_sx = savgol_filter(PC1_fwd, sg_window, sg_order)
fwd_sy = savgol_filter(PC2_fwd, sg_window, sg_order)
fwd_sz = savgol_filter(PC3_fwd, sg_window, sg_order)

bwd_sx = savgol_filter(PC1_bwd, sg_window, sg_order)
bwd_sy = savgol_filter(PC2_bwd, sg_window, sg_order)
bwd_sz = savgol_filter(PC3_bwd, sg_window, sg_order)

# --- Per-point explicit RGB colors (the only reliable way to get a gradient
#     in Plotly 3D scatter) ---
# Forward: yellow → orange → deep red  (YlOrRd colormap)
# Backward: white → blue → deep navy   (Blues colormap)
fwd_cmap = plt.get_cmap('YlOrRd')
bwd_cmap = plt.get_cmap('Blues')

t_norm = np.linspace(0.15, 1.0, n_plot)   # avoid the very pale end

def cmap_to_rgb_strings(cmap, t):
    """Return list of 'rgb(r,g,b)' strings for each value in t."""
    colors = []
    for v in t:
        r, g, b, _ = cmap(v)
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    return colors

fwd_colors = cmap_to_rgb_strings(fwd_cmap, t_norm)
bwd_colors = cmap_to_rgb_strings(bwd_cmap, t_norm)

# Representative colors for legend swatches (midpoint and endpoint)
fwd_mid_color = fwd_colors[n_plot // 2]
bwd_mid_color = bwd_colors[n_plot // 2]
fwd_end_color = fwd_colors[-1]
bwd_end_color = bwd_colors[-1]

# --- Helper: single-point endpoint marker ---
def endpoint_trace(x, y, z, color, name, showlegend=True):
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers',
        name=name,
        showlegend=showlegend,
        legendgroup=name,
        marker=dict(size=12, color=color, symbol='circle',
                    line=dict(color='black', width=1.5))
    )

scene_axes = dict(
    showgrid=True, gridcolor='lightgrey',
    zeroline=True, zerolinewidth=2, zerolinecolor='lightgrey',
    showline=True, linewidth=2, linecolor='black', mirror=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Scatter with per-point colour gradient + smooth trajectory line
# ─────────────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Forward scatter — explicit per-point RGB strings give a true gradient
fig.add_trace(go.Scatter3d(
    x=PC1_fwd, y=PC2_fwd, z=PC3_fwd,
    mode='markers',
    name='Forward',
    legendgroup='Forward',
    marker=dict(size=5, color=fwd_colors, opacity=0.85),
    hovertemplate='Forward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
))

# Backward scatter
fig.add_trace(go.Scatter3d(
    x=PC1_bwd, y=PC2_bwd, z=PC3_bwd,
    mode='markers',
    name='Backward',
    legendgroup='Backward',
    marker=dict(size=5, color=bwd_colors, opacity=0.85),
    hovertemplate='Backward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
))

# Smooth trajectory lines
fig.add_trace(go.Scatter3d(
    x=fwd_sx, y=fwd_sy, z=fwd_sz,
    mode='lines', name='Forward traj.',
    legendgroup='Forward', showlegend=False,
    line=dict(color=fwd_end_color, width=5)
))
fig.add_trace(go.Scatter3d(
    x=bwd_sx, y=bwd_sy, z=bwd_sz,
    mode='lines', name='Backward traj.',
    legendgroup='Backward', showlegend=False,
    line=dict(color=bwd_end_color, width=5)
))

# Start (black) and End (green) markers
fig.add_trace(endpoint_trace(PC1_fwd[0],  PC2_fwd[0],  PC3_fwd[0],  'black', 'Start'))
fig.add_trace(endpoint_trace(PC1_bwd[0],  PC2_bwd[0],  PC3_bwd[0],  'black', 'Start', showlegend=False))
fig.add_trace(endpoint_trace(PC1_fwd[-1], PC2_fwd[-1], PC3_fwd[-1], 'green', 'End'))
fig.add_trace(endpoint_trace(PC1_bwd[-1], PC2_bwd[-1], PC3_bwd[-1], 'green', 'End',   showlegend=False))

fig.update_layout(
    title=dict(text=f"PCA projection – scatter (±{window} timesteps around event)",
               y=0.95, x=0.5, xanchor='center', yanchor='top'),
    scene=dict(
        xaxis=dict(title="PC1", **scene_axes),
        yaxis=dict(title="PC2", **scene_axes),
        zaxis=dict(title="PC3", **scene_axes),
    ),
    font=dict(family="Times New Roman, serif", size=12, color="black"),
    margin=dict(l=80, r=160, t=80, b=80),
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(title="", orientation="v", x=1.02, y=0.9,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgrey', borderwidth=1),
    width=950, height=800
)
fig.show()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Scatter dots + smooth trajectory line
# ─────────────────────────────────────────────────────────────────────────────
fig2 = go.Figure()

# Forward scatter dots
fig2.add_trace(go.Scatter3d(
    x=PC1_fwd, y=PC2_fwd, z=PC3_fwd,
    mode='markers',
    name='Forward',
    legendgroup='Forward',
    marker=dict(size=5, color=fwd_colors, opacity=0.6),
    hovertemplate='Forward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
))

# Backward scatter dots
fig2.add_trace(go.Scatter3d(
    x=PC1_bwd, y=PC2_bwd, z=PC3_bwd,
    mode='markers',
    name='Backward',
    legendgroup='Backward',
    marker=dict(size=5, color=bwd_colors, opacity=0.6),
    hovertemplate='Backward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
))

# Smooth trajectory lines
fig2.add_trace(go.Scatter3d(
    x=fwd_sx, y=fwd_sy, z=fwd_sz,
    mode='lines', name='Forward traj.',
    legendgroup='Forward', showlegend=True,
    line=dict(color=fwd_end_color, width=6)
))
fig2.add_trace(go.Scatter3d(
    x=bwd_sx, y=bwd_sy, z=bwd_sz,
    mode='lines', name='Backward traj.',
    legendgroup='Backward', showlegend=True,
    line=dict(color=bwd_end_color, width=6)
))

# Start (black) and End (green) markers
fig2.add_trace(endpoint_trace(PC1_fwd[0],  PC2_fwd[0],  PC3_fwd[0],  'black', 'Start'))
fig2.add_trace(endpoint_trace(PC1_bwd[0],  PC2_bwd[0],  PC3_bwd[0],  'black', 'Start', showlegend=False))
fig2.add_trace(endpoint_trace(PC1_fwd[-1], PC2_fwd[-1], PC3_fwd[-1], 'green', 'End'))
fig2.add_trace(endpoint_trace(PC1_bwd[-1], PC2_bwd[-1], PC3_bwd[-1], 'green', 'End',   showlegend=False))

fig2.update_layout(
    title=dict(text=f"PCA projection – smooth trajectory (±{window} timesteps around event)",
               y=0.95, x=0.5, xanchor='center', yanchor='top'),
    scene=dict(
        xaxis=dict(title="PC1", **scene_axes),
        yaxis=dict(title="PC2", **scene_axes),
        zaxis=dict(title="PC3", **scene_axes),
    ),
    font=dict(family="Times New Roman, serif", size=12, color="black"),
    margin=dict(l=80, r=160, t=80, b=80),
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(title="", orientation="v", x=1.02, y=0.9,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgrey', borderwidth=1),
    width=950, height=800
)
fig2.show()
