import scipy.io as sio
import numpy as np

# Load both datasets
crfb = sio.loadmat('dataCRFB.mat')['dataCRFB']
tonefb = sio.loadmat('dataToneFB.mat')['dataToneFB']

def get_struct_info(struct):
    struct = struct[0, 0] if struct.size > 0 else struct
    info = {}
    for field in struct.dtype.names:
        val = struct[field]
        if field == 'firing_rate':
            fr_info = {}
            fr = val[0, 0] if val.size > 0 else val
            for g in fr.dtype.names:
                gdata = fr[g]
                # Handle nested structure
                if gdata.ndim > 0:
                    gdata = gdata[0, 0]
                zfwd = gdata['zforward']
                zbwd = gdata['zbackward']
                # Handle nested arrays
                if zfwd.ndim > 2:
                    zfwd = zfwd[0, 0]
                if zbwd.ndim > 2:
                    zbwd = zbwd[0, 0]
                # Check for NaN
                if zfwd.size > 0:
                    fwd_nan = np.isnan(zfwd).any(axis=1).sum() if zfwd.ndim == 2 else 0
                    n_neurons = zfwd.shape[0] if zfwd.ndim == 2 else 0
                    timesteps = zfwd.shape[1] if zfwd.ndim == 2 else 0
                else:
                    fwd_nan = 0
                    n_neurons = 0
                    timesteps = 0
                fr_info[g] = {
                    'n_neurons': n_neurons,
                    'timesteps': timesteps,
                    'fwd_nan_rows': fwd_nan,
                }
            info[field] = fr_info
        else:
            info[field] = val.shape
    return info

print('=== CRFB ===')
crfb_info = get_struct_info(crfb)
for g, ginfo in crfb_info['firing_rate'].items():
    print(f"  {g}: n={ginfo['n_neurons']}, ts={ginfo['timesteps']}, fwd_nan={ginfo['fwd_nan_rows']}")

print('\n=== ToneFB ===')
tonefb_info = get_struct_info(tonefb)
for g, ginfo in tonefb_info['firing_rate'].items():
    print(f"  {g}: n={ginfo['n_neurons']}, ts={ginfo['timesteps']}, fwd_nan={ginfo['fwd_nan_rows']}")

# Check DA groups specifically
DA_GROUPS = ['DF', 'DB', 'D', 'DFB']
print('\n=== DA Groups Comparison ===')
for g in DA_GROUPS:
    cr_n = crfb_info['firing_rate'][g]['n_neurons']
    to_n = tonefb_info['firing_rate'][g]['n_neurons']
    print(f"  {g}: CRFB={cr_n}, ToneFB={to_n}, match={cr_n == to_n}")