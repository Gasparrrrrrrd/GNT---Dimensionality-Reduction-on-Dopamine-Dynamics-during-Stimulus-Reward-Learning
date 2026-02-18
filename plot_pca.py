import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go
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
window    = 80    # ±window timesteps around each event

fwd_start = event_idx - window
fwd_end   = event_idx + window
bwd_start = time + event_idx - window
bwd_end   = time + event_idx + window

n_plot = 2 * window + 1

# Time axis centred on the event (in seconds)
plot_time = (np.arange(n_plot) - window) * dt   # e.g. -0.80 … +0.80 s

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

# --- Build a very smooth trajectory approximation via B-spline ---
# Use a low number of knots (k=3 cubic spline, few interior knots) so the
# curve is a smooth summary rather than an interpolation through every point.
def smooth_spline_3d(x, y, z, n_out=400, k=3, n_knots=8):
    """Fit a parametric B-spline through (x,y,z) with few knots for a smooth curve."""
    t = np.linspace(0, 1, len(x))
    t_out = np.linspace(0, 1, n_out)
    # Place interior knots uniformly (excluding endpoints)
    knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    spl_x = make_interp_spline(t, x, k=k, t=knots)
    spl_y = make_interp_spline(t, y, k=k, t=knots)
    spl_z = make_interp_spline(t, z, k=k, t=knots)
    return spl_x(t_out), spl_y(t_out), spl_z(t_out)

fwd_sx, fwd_sy, fwd_sz = smooth_spline_3d(PC1_fwd, PC2_fwd, PC3_fwd)
bwd_sx, bwd_sy, bwd_sz = smooth_spline_3d(PC1_bwd, PC2_bwd, PC3_bwd)

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

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Scatter with per-trace colour gradient + smooth trajectory line
# ─────────────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Forward scatter (orange gradient, colour = time relative to event)
fig.add_trace(go.Scatter3d(
    x=PC1_fwd, y=PC2_fwd, z=PC3_fwd,
    mode='markers',
    name='Forward',
    legendgroup='Forward',
    marker=dict(
        size=4,
        color=plot_time,
        cmin=plot_time[0], cmax=plot_time[-1],
        colorscale=[[0, '#ffcec6'], [1, '#a71900']],
        colorbar=dict(
            title='Time (s)',
            ticks='outside', ticklen=4,
            orientation='v', len=0.4,
            x=1.02, y=0.7,
            thickness=12
        ),
        showscale=True,
        opacity=0.7
    ),
    hovertemplate='Forward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>t: %{marker.color:.2f}s'
))

# Backward scatter (blue gradient)
fig.add_trace(go.Scatter3d(
    x=PC1_bwd, y=PC2_bwd, z=PC3_bwd,
    mode='markers',
    name='Backward',
    legendgroup='Backward',
    marker=dict(
        size=4,
        color=plot_time,
        cmin=plot_time[0], cmax=plot_time[-1],
        colorscale=[[0, '#ACC4E3'], [1, '#020078']],
        colorbar=dict(
            title='Time (s)',
            ticks='outside', ticklen=4,
            orientation='v', len=0.4,
            x=1.12, y=0.7,
            thickness=12
        ),
        showscale=True,
        opacity=0.7
    ),
    hovertemplate='Backward<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>t: %{marker.color:.2f}s'
))

# Smooth trajectory lines
fig.add_trace(go.Scatter3d(
    x=fwd_sx, y=fwd_sy, z=fwd_sz,
    mode='lines', name='Forward (smooth)',
    legendgroup='Forward', showlegend=False,
    line=dict(color='#a71900', width=5)
))
fig.add_trace(go.Scatter3d(
    x=bwd_sx, y=bwd_sy, z=bwd_sz,
    mode='lines', name='Backward (smooth)',
    legendgroup='Backward', showlegend=False,
    line=dict(color='#020078', width=5)
))

# Start (black) and End (green) markers
fig.add_trace(endpoint_trace(PC1_fwd[0],  PC2_fwd[0],  PC3_fwd[0],  'black', 'Start'))
fig.add_trace(endpoint_trace(PC1_bwd[0],  PC2_bwd[0],  PC3_bwd[0],  'black', 'Start', showlegend=False))
fig.add_trace(endpoint_trace(PC1_fwd[-1], PC2_fwd[-1], PC3_fwd[-1], 'green', 'End'))
fig.add_trace(endpoint_trace(PC1_bwd[-1], PC2_bwd[-1], PC3_bwd[-1], 'green', 'End',   showlegend=False))

scene_axes = dict(
    showgrid=True, gridcolor='lightgrey',
    zeroline=True, zerolinewidth=2, zerolinecolor='lightgrey',
    showline=True, linewidth=2, linecolor='black', mirror=True
)

fig.update_layout(
    title=dict(text="PCA projection – scatter (±{}ts around event)".format(window),
               y=0.95, x=0.5, xanchor='center', yanchor='top'),
    scene=dict(
        xaxis=dict(title="PC1", **scene_axes),
        yaxis=dict(title="PC2", **scene_axes),
        zaxis=dict(title="PC3", **scene_axes),
    ),
    font=dict(family="Times New Roman, serif", size=12, color="black"),
    margin=dict(l=80, r=120, t=80, b=80),
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(title="", orientation="v", x=1.25, y=0.5),
    width=900, height=800
)
fig.show()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Scatter dots + smooth trajectory line (no dot-to-dot connection)
# ─────────────────────────────────────────────────────────────────────────────
fig2 = go.Figure()

# Forward scatter dots (orange gradient)
fig2.add_trace(go.Scatter3d(
    x=PC1_fwd, y=PC2_fwd, z=PC3_fwd,
    mode='markers',
    name='Forward',
    legendgroup='Forward',
    marker=dict(
        size=4,
        color=plot_time,
        cmin=plot_time[0], cmax=plot_time[-1],
        colorscale=[[0, '#ffcec6'], [1, '#a71900']],
        colorbar=dict(
            title='Time (s)',
            ticks='outside', ticklen=4,
            orientation='v', len=0.4,
            x=1.02, y=0.7,
            thickness=12
        ),
        showscale=True,
        opacity=0.5
    ),
    hovertemplate='Forward<br>t: %{marker.color:.2f}s'
))

# Backward scatter dots (blue gradient)
fig2.add_trace(go.Scatter3d(
    x=PC1_bwd, y=PC2_bwd, z=PC3_bwd,
    mode='markers',
    name='Backward',
    legendgroup='Backward',
    marker=dict(
        size=4,
        color=plot_time,
        cmin=plot_time[0], cmax=plot_time[-1],
        colorscale=[[0, '#ACC4E3'], [1, '#020078']],
        colorbar=dict(
            title='Time (s)',
            ticks='outside', ticklen=4,
            orientation='v', len=0.4,
            x=1.12, y=0.7,
            thickness=12
        ),
        showscale=True,
        opacity=0.5
    ),
    hovertemplate='Backward<br>t: %{marker.color:.2f}s'
))

# Smooth trajectory lines (the actual "trajectory approximation")
fig2.add_trace(go.Scatter3d(
    x=fwd_sx, y=fwd_sy, z=fwd_sz,
    mode='lines', name='Forward trajectory',
    legendgroup='Forward', showlegend=True,
    line=dict(color='#a71900', width=6)
))
fig2.add_trace(go.Scatter3d(
    x=bwd_sx, y=bwd_sy, z=bwd_sz,
    mode='lines', name='Backward trajectory',
    legendgroup='Backward', showlegend=True,
    line=dict(color='#020078', width=6)
))

# Start (black) and End (green) markers
fig2.add_trace(endpoint_trace(PC1_fwd[0],  PC2_fwd[0],  PC3_fwd[0],  'black', 'Start'))
fig2.add_trace(endpoint_trace(PC1_bwd[0],  PC2_bwd[0],  PC3_bwd[0],  'black', 'Start', showlegend=False))
fig2.add_trace(endpoint_trace(PC1_fwd[-1], PC2_fwd[-1], PC3_fwd[-1], 'green', 'End'))
fig2.add_trace(endpoint_trace(PC1_bwd[-1], PC2_bwd[-1], PC3_bwd[-1], 'green', 'End',   showlegend=False))

fig2.update_layout(
    title=dict(text="PCA projection – smooth trajectory (±{}ts around event)".format(window),
               y=0.95, x=0.5, xanchor='center', yanchor='top'),
    scene=dict(
        xaxis=dict(title="PC1", **scene_axes),
        yaxis=dict(title="PC2", **scene_axes),
        zaxis=dict(title="PC3", **scene_axes),
    ),
    font=dict(family="Times New Roman, serif", size=12, color="black"),
    margin=dict(l=80, r=120, t=80, b=80),
    paper_bgcolor="white", plot_bgcolor="white",
    legend=dict(title="", orientation="v", x=1.25, y=0.5),
    width=900, height=800
)
fig2.show()
    line=dict(color='#dd4d32', width=3),
    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Time: %{marker.color:.2f}s'
))

# Add backward trace (blue/teal) - markers and line
fig2.add_trace(go.Scatter3d(
    x=PC1_bwd_plot,
    y=PC2_bwd_plot,
    z=PC3_bwd_plot,
    mode='markers+lines',
    name='Backward',
    marker=dict(
        size=4,
        color=plot_time,
        colorscale=[[0, '#7ed9c9'], [1, '#32ae97']],
        colorbar=dict(
            title='Time rel. to event (s)',
            orientation='h',
            len=0.25,
            x=1.0,
            y=0.85
        ),
        showscale=True
    ),
    line=dict(color='#32ae97', width=3),
    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Time: %{marker.color:.2f}s'
))

# --- Start markers (black): first point of each trajectory ---
fig2.add_trace(endpoint_trace(
    PC1_fwd_plot[0], PC2_fwd_plot[0], PC3_fwd_plot[0],
    color='black', symbol='circle', name='Start'
))
fig2.add_trace(endpoint_trace(
    PC1_bwd_plot[0], PC2_bwd_plot[0], PC3_bwd_plot[0],
    color='black', symbol='circle', name='Start',
    showlegend=False   # same legend entry as above
))

# --- End markers (green): last point of each trajectory ---
fig2.add_trace(endpoint_trace(
    PC1_fwd_plot[-1], PC2_fwd_plot[-1], PC3_fwd_plot[-1],
    color='green', symbol='circle', name='End'
))
fig2.add_trace(endpoint_trace(
    PC1_bwd_plot[-1], PC2_bwd_plot[-1], PC3_bwd_plot[-1],
    color='green', symbol='circle', name='End',
    showlegend=False   # same legend entry as above
))

# Update layout for scientific journal-quality
fig2.update_layout(
    title={
        'text': "Projection of the data on the first three principal components (Continuous Trace)",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    scene=dict(
        xaxis=dict(
            title="PC1",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgrey',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            title="PC2",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgrey',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
        zaxis=dict(
            title="PC3",
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='lightgrey',
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True
        ),
    ),
    font=dict(
        family="Times New Roman, serif",
        size=12,
        color="black"
    ),
    margin=dict(l=80, r=40, t=80, b=80),
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(
        title="Movement",
        orientation="h",
        yanchor="top",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    width=800,
    height=800
)

# Display the continuous trace plot
fig2.show()
