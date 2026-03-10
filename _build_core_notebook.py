"""Build PCA_Core_Tests.ipynb — 6 core tests for the movement hypothesis."""
import json

def md(text):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    """Create a code cell."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    return {"cell_type": "code", "metadata": {}, "source": lines,
            "outputs": [], "execution_count": None}

cells = []

# ============================================================
# [0] Title + Hypothesis
# ============================================================
cells.append(md("""\
# Core Tests: Does VTA Dopamine Encode Movement Direction, Not Reward?

## Central Hypothesis

**DA neurons encode kinematic/behavioral variables (movement direction, force generation) rather than reward prediction error (RPE).**

Forward vs Backward movements should produce distinct trajectories in PC space. The latent variables driving variance should reflect movement direction, not reward value.

## Experimental Setup

| Dataset | Aligned to | Contains | Reward? |
|---------|-----------|----------|---------|
| **SpontFB** | Spontaneous movement onset (t=600) | Pure movement, no task | No |
| **CRFB** | Conditioned Response onset (t=600) | CR movement, CS/reward shuffled | Yes (shuffled) |
| **ToneFB** | CS tone onset (t=600) | CS burst, then CR movement, then reward at t=700 | Yes (t=700) |

- **Forward/Backward:** Mouse moves toward a sugar tube placed in front or behind (session-determined, no choice).
- **Neuron classes:** DA (DF, DB, D, DFB) and GABA (GF, GB, G, GFB). F=forward-selective, B=backward-selective, FB=bidirectional.
- **Two phasic DA bursts in ToneFB:** (1) at CS onset = salience/sensory, (2) near reward delivery = our claim: CR movement signal.
- **CR timing:** Variable reaction time ~30-100 timesteps (0.3-1.0s) after CS.

## Six Core Tests

1. **SpontFB cross-projection:** Does the pure-movement subspace explain task data?
2. **Per-class separation:** Do DF/DB/GF/GB show significant direction selectivity?
3. **GF/GB reward insensitivity:** Are GABA direction neurons blind to reward?
4. **Reward-time deflection:** Is there a speed transient at reward delivery?
5. **CS vs CR direction sensitivity (NEW):** Is the CS burst direction-invariant while the CR burst is direction-selective?
6. **CR burst in movement subspace:** Does reward-time DA activity live in the SpontFB movement subspace?
"""))

# ============================================================
# [1] Imports + Configuration
# ============================================================
cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import importlib

import plot_pca_framework
importlib.reload(plot_pca_framework)

from plot_pca_framework import (
    load_dataset,
    extract_neuron_data,
    extract_group_averaged_data,
    fit_pca,
    project_onto_pca,
    analyze_dataset,
    cross_project,
    compute_reconstruction_r2,
    compute_trajectory_metrics,
    slice_window,
    smooth_trajectories,
    build_overlay_figure,
    plot_1d_pc_timecourses,
    plot_speed_profiles,
    compute_rdm,
    compare_rdms,
    EPOCHS,
    # Null models
    null_cross_projection_r2,
    null_separation,
    null_reward_deflection,
    # Private helpers
    _align_neuron_data,
    _phase_randomise,
)

logging.basicConfig(level=logging.WARNING)
%matplotlib inline
plt.rcParams["figure.dpi"] = 100

# ── Configuration ──
DATASETS = {
    'SpontFB': {'mat_file': 'dataSpontFB.mat', 'var_name': 'dataSpontFB'},
    'CRFB':    {'mat_file': 'dataCRFB.mat',    'var_name': 'dataCRFB'},
    'ToneFB':  {'mat_file': 'dataToneFB.mat',  'var_name': 'dataToneFB'},
}
DA_GROUPS   = ['DF', 'DB', 'D', 'DFB']
GABA_GROUPS = ['GF', 'GB', 'G', 'GFB']
ALL_GROUPS  = DA_GROUPS + GABA_GROUPS

N_COMPONENTS = 3
EVENT_IDX    = 600
WINDOW       = 150
DT           = 0.01
SG_WINDOW    = 11
SG_ORDER     = 3
MIN_NEURONS  = 10
"""))

# ============================================================
# [2] Load + Run PCA
# ============================================================
cells.append(code("""\
# Run all 9 dataset x population analyses
results = {}
combos = {'Dopamine': DA_GROUPS, 'GABA': GABA_GROUPS}

for ds_name, ds_cfg in DATASETS.items():
    for combo_label, groups in combos.items():
        key = f"{ds_name}_{combo_label}"
        try:
            r = analyze_dataset(
                mat_file=ds_cfg['mat_file'], var_name=ds_cfg['var_name'],
                dataset_name=ds_name, neuron_groups=groups, combo_label=combo_label,
                n_components=N_COMPONENTS, event_idx=EVENT_IDX, window=WINDOW,
                dt=DT, sg_window=SG_WINDOW, sg_order=SG_ORDER)
            results[key] = r
            evr = r['explained_variance_ratio']
            print(f"OK  {key:25s}  n={r['n_neurons']:4d}  "
                  f"EVR=[{'+'.join(f'{v:.3f}' for v in evr)}]  "
                  f"sep={r['metrics']['mean_separation']:.2f}")
        except Exception as e:
            print(f"FAIL {key:25s}  {e}")

# Per-class PCA
single_class_results = {}
for ds_name, ds_cfg in DATASETS.items():
    for group in ALL_GROUPS:
        key = f"{ds_name}_{group}"
        try:
            r = analyze_dataset(
                mat_file=ds_cfg['mat_file'], var_name=ds_cfg['var_name'],
                dataset_name=ds_name, neuron_groups=[group], combo_label=group,
                n_components=N_COMPONENTS, event_idx=EVENT_IDX, window=WINDOW,
                dt=DT, sg_window=SG_WINDOW, sg_order=SG_ORDER)
            if r['n_neurons'] < MIN_NEURONS:
                print(f"SKIP {key:25s}  n={r['n_neurons']} < {MIN_NEURONS}")
                continue
            single_class_results[key] = r
            sep = r['metrics']['mean_separation']
            print(f"OK  {key:25s}  n={r['n_neurons']:4d}  sep={sep:.2f}")
        except Exception as e:
            print(f"SKIP {key:25s}  {e}")

print(f"\\n9 population analyses: {len(results)}")
print(f"Per-class analyses (n >= {MIN_NEURONS}): {len(single_class_results)}")
"""))

# ============================================================
# [3] Data Summary
# ============================================================
cells.append(md("""\
### Data Summary

Examine the neuron counts and which classes were available above. Classes with fewer than 10 neurons are skipped (PCA unreliable). Key classes for hypothesis testing: **DF, DB, GF, GB** (direction-selective) and **GFB** (bidirectional GABA).

---
"""))

# ============================================================
# TEST 1: SpontFB → Task Cross-Projection
# ============================================================
cells.append(md("""\
---
## Test 1: Does the Spontaneous-Movement Subspace Explain Task Neural Variance?

SpontFB has **no CS, no reward** -- only spontaneous F/B movements. If DA and GABA encode movement direction, PCs fitted on SpontFB group-averaged data should capture the majority of variance in CRFB and ToneFB.

**Method:** Average neurons within selectivity groups (DF/DB/D/DFB or GF/GB/G/GFB) to create 4 pseudo-neurons. Fit PCA on SpontFB groups, project Task groups. Compute R-squared. **Null model:** phase-randomise each group's timecourse independently (destroys temporal alignment, preserves autocorrelation spectrum).

**Prediction (movement):** R-squared > 0.70 for both DA and GABA, significantly above null. SpontFB PCs capture task variance because the same F/B movement drives activity in all datasets.

**Prediction (RPE):** R-squared lower for DA than GABA in the Task datasets. DA has value-related variance (CS, reward signals) in ToneFB/CRFB that SpontFB cannot explain. If DA R-squared is notably lower than GABA R-squared, the extra DA variance is non-movement (candidate: value).
"""))

cells.append(code("""\
# Test 1: Group-averaged cross-projection + null model
cross_proj_results = {}

for combo_label, groups in [('GABA', GABA_GROUPS), ('Dopamine', DA_GROUPS)]:
    spont_key = f'SpontFB_{combo_label}'
    if spont_key not in results:
        continue

    for target_ds in ['CRFB', 'ToneFB']:
        target_key = f'{target_ds}_{combo_label}'
        if target_key not in results:
            continue

        # Forward: SpontFB → Task
        label = f'{combo_label}: Spont->{target_ds}'
        xp = cross_project(
            results[spont_key], results[target_key],
            use_group_avg=True, neuron_groups=groups,
            window=WINDOW, event_idx=EVENT_IDX, dt=DT,
            sg_window=SG_WINDOW, sg_order=SG_ORDER)
        r2_obs = xp['r2']

        # Null model
        null_res = null_cross_projection_r2(
            results[spont_key], results[target_key],
            neuron_groups=groups, n_permutations=1000,
            window=WINDOW, event_idx=EVENT_IDX, dt=DT,
            sg_window=SG_WINDOW, sg_order=SG_ORDER)

        null_mean = np.mean(null_res['null_r2'])
        null_std = np.std(null_res['null_r2'])
        z = (r2_obs - null_mean) / null_std if null_std > 0 else float('inf')
        p = float((np.sum(null_res['null_r2'] >= r2_obs) + 1) / (len(null_res['null_r2']) + 1))

        cross_proj_results[label] = {
            'r2': r2_obs, 'null_mean': null_mean, 'null_std': null_std,
            'z': z, 'p': p, 'null_values': null_res['null_r2']}
        print(f"  {label:35s}  R2={r2_obs:.4f}  null={null_mean:.4f}+/-{null_std:.4f}  "
              f"z={z:.2f}  p={p:.4f}")

# Plot
labels = list(cross_proj_results.keys())
r2_vals = [cross_proj_results[l]['r2'] for l in labels]
null_means = [cross_proj_results[l]['null_mean'] for l in labels]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))
ax.bar(x - 0.15, r2_vals, 0.3, label='Observed R-squared', color='steelblue')
ax.bar(x + 0.15, null_means, 0.3, label='Null mean', color='lightcoral', alpha=0.7)
# Error bars for null
ax.errorbar(x + 0.15, null_means,
            yerr=[cross_proj_results[l]['null_std'] for l in labels],
            fmt='none', color='darkred', capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('R-squared')
ax.set_title('Test 1: SpontFB → Task Cross-Projection')
ax.legend()
# Add significance stars
for i, l in enumerate(labels):
    if cross_proj_results[l]['p'] < 0.001:
        ax.text(i, r2_vals[i] + 0.02, '***', ha='center', fontsize=12)
    elif cross_proj_results[l]['p'] < 0.01:
        ax.text(i, r2_vals[i] + 0.02, '**', ha='center', fontsize=12)
    elif cross_proj_results[l]['p'] < 0.05:
        ax.text(i, r2_vals[i] + 0.02, '*', ha='center', fontsize=12)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
### Test 1 -- Interpretation

**Fill based on observed results:**
- Is R-squared > 0.70? → Movement subspace explains task data.
- Is DA R-squared similar to GABA R-squared? → Both encode movement (supports movement theory).
- Is DA R-squared notably lower than GABA? → DA may have extra non-movement variance (partial support for RPE).
- Are all p-values < 0.05? → Results significantly exceed autocorrelation-driven chance.

---
"""))

# ============================================================
# TEST 2: Per-Class Fwd-Bwd Separation + Null
# ============================================================
cells.append(md("""\
## Test 2: Do DF/DB Show Distinct Opposite Trajectories? Do GF/GB Show Direction Selectivity?

If DA encodes movement direction, forward-selective (DF) and backward-selective (DB) neurons should show strong, opposite F/B trajectories. Same for GF/GB in GABA.

**Method:** For each class with n >= 10 neurons, compute post-event mean fwd-bwd separation from the per-class PCA. **Null model:** phase-randomise each PC timecourse independently (`null_separation()`), testing whether observed separation exceeds what autocorrelation alone would produce.

**Prediction (movement):** DF and DB have high separation in SpontFB AND CRFB (both are movement contexts). GF and GB have high separation across all datasets. GFB may show moderate separation if bidirectional GABA still encodes direction.

**Prediction (RPE):** DA classes show lower separation than GABA classes (DA variance driven by value, not direction). DF/DB separation should be weaker in ToneFB (CS dominates DA variance, reducing directional structure).
"""))

cells.append(code("""\
# Test 2: Per-class separation + null model
sep_results = []

for key, r in sorted(single_class_results.items()):
    ds_name = key.split('_')[0]
    cls_name = '_'.join(key.split('_')[1:])

    # Run null separation
    ns = null_separation(
        r['smooth_data'], r['window_data'],
        n_permutations=1000, seed=42)

    obs_sep = ns['observed_separation']
    null_mean = float(np.mean(ns['null_separations']))
    null_std = float(np.std(ns['null_separations']))
    z = (obs_sep - null_mean) / null_std if null_std > 0 else float('inf')
    p = ns['p_value']

    sep_results.append({
        'key': key, 'dataset': ds_name, 'class': cls_name,
        'n_neurons': r['n_neurons'], 'separation': obs_sep,
        'null_mean': null_mean, 'z': z, 'p': p})
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {key:25s}  n={r['n_neurons']:4d}  sep={obs_sep:6.2f}  "
          f"z={z:5.2f}  p={p:.4f} {sig}")

# Plot: grouped by class, colored by dataset
sep_df = pd.DataFrame(sep_results)
classes_order = ['DF', 'DB', 'D', 'DFB', 'GF', 'GB', 'G', 'GFB']
ds_colors = {'SpontFB': 'steelblue', 'CRFB': 'coral', 'ToneFB': 'seagreen'}

fig, ax = plt.subplots(figsize=(14, 5))
bar_width = 0.25
for i, ds in enumerate(['SpontFB', 'CRFB', 'ToneFB']):
    subset = sep_df[sep_df['dataset'] == ds]
    positions = []
    heights = []
    stars = []
    for j, cls in enumerate(classes_order):
        row = subset[subset['class'] == cls]
        if len(row) == 0:
            continue
        positions.append(j + (i - 1) * bar_width)
        heights.append(row.iloc[0]['separation'])
        stars.append(row.iloc[0]['p'])
    ax.bar(positions, heights, bar_width, label=ds, color=ds_colors[ds], alpha=0.8)
    for pos, h, p in zip(positions, heights, stars):
        if p < 0.001:
            ax.text(pos, h + 0.3, '***', ha='center', fontsize=8)
        elif p < 0.01:
            ax.text(pos, h + 0.3, '**', ha='center', fontsize=8)
        elif p < 0.05:
            ax.text(pos, h + 0.3, '*', ha='center', fontsize=8)

ax.set_xticks(range(len(classes_order)))
ax.set_xticklabels(classes_order)
ax.set_ylabel('Post-event mean fwd-bwd separation')
ax.set_title('Test 2: Per-Class Direction Selectivity (*** p<0.001, ** p<0.01, * p<0.05)')
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
### Test 2 -- Interpretation

**Fill based on results:**
- Do DF and DB show significant separation across SpontFB and CRFB? → Direction encoding confirmed in DA.
- Do GF and GB show significant separation across all datasets? → Direction encoding confirmed in GABA.
- Is separation weaker in ToneFB for DA classes? → CS-evoked synchronisation may temporarily reduce directional structure.
- Does GFB show any separation? → Bidirectional GABA still has directional sensitivity.

---
"""))

# ============================================================
# TEST 3: GF/GB Reward Insensitivity
# ============================================================
cells.append(md("""\
## Test 3: Are GF/GB Neurons Blind to Reward Delivery?

If GABA direction neurons are pure movement encoders, their fwd-bwd separation should be **unchanged** at reward delivery time (t=700 in ToneFB). The separation timecourse should be flat through the reward window.

**Method:** Compute instantaneous fwd-bwd separation ||fwd(t) - bwd(t)|| at each timepoint for GF and GB in ToneFB. Compare mean separation in a pre-reward window [50, 90] (relative to event) to a post-reward window [110, 150]. **Null:** circularly shift the separation timecourse 1000 times and recompute the pre-vs-post difference.

**Control:** Same analysis on SpontFB at the equivalent post-event latency (no reward present).

**Prediction (movement):** No significant change in separation at reward time. The separation timecourse follows movement kinematics, not reward delivery.

**Prediction (RPE):** If GABA receives reward signals, separation should change at reward time (new deflection, acceleration, or collapse).
"""))

cells.append(code("""\
# Test 3: GF/GB reward insensitivity — separation timecourse around reward
reward_insensitivity_results = {}

# In the windowed data, event is at index WINDOW (=150).
# Reward is at event + 100 = index 250 in the windowed data (for ToneFB).
# Pre-reward: indices [200, 240] = event-relative [50, 90]
# Post-reward: indices [260, 300] = event-relative [110, 150]
PRE_REWARD_SLICE  = slice(200, 240)
POST_REWARD_SLICE = slice(260, 300)
# For SpontFB: same indices (matched post-event latency, no reward)
rng = np.random.default_rng(42)

for cls in ['GF', 'GB', 'GFB']:
    for ds in ['ToneFB', 'SpontFB']:
        key = f'{ds}_{cls}'
        if key not in single_class_results:
            continue
        r = single_class_results[key]
        sd = r['smooth_data']
        fwd = sd['fwd_smooth']  # shape (3, n_timepoints)
        bwd = sd['bwd_smooth']

        # Instantaneous separation timecourse
        sep_t = np.sqrt(np.sum((fwd - bwd)**2, axis=0))

        # Pre-vs-post reward difference
        pre_sep = np.mean(sep_t[PRE_REWARD_SLICE])
        post_sep = np.mean(sep_t[POST_REWARD_SLICE])
        obs_delta = post_sep - pre_sep

        # Null: circular shift
        n_t = len(sep_t)
        null_deltas = np.empty(1000)
        for i in range(1000):
            shift = rng.integers(1, n_t)
            sep_shifted = np.roll(sep_t, shift)
            null_deltas[i] = (np.mean(sep_shifted[POST_REWARD_SLICE])
                              - np.mean(sep_shifted[PRE_REWARD_SLICE]))

        p = float((np.sum(np.abs(null_deltas) >= np.abs(obs_delta)) + 1) / 1001)
        z = (obs_delta - np.mean(null_deltas)) / np.std(null_deltas) if np.std(null_deltas) > 0 else 0

        reward_insensitivity_results[key] = {
            'sep_timecourse': sep_t, 'pre': pre_sep, 'post': post_sep,
            'delta': obs_delta, 'p': p, 'z': z}
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {key:20s}  pre_sep={pre_sep:.2f}  post_sep={post_sep:.2f}  "
              f"delta={obs_delta:+.2f}  z={z:.2f}  p={p:.4f} {sig}")

# Plot separation timecourses
for cls in ['GF', 'GB', 'GFB']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, ds in zip(axes, ['ToneFB', 'SpontFB']):
        key = f'{ds}_{cls}'
        if key not in reward_insensitivity_results:
            ax.set_title(f'{key}: not available')
            continue
        ri = reward_insensitivity_results[key]
        t_axis = np.arange(len(ri['sep_timecourse'])) - WINDOW
        t_axis_s = t_axis * DT
        ax.plot(t_axis_s, ri['sep_timecourse'], color='black', linewidth=1)
        ax.axvline(0, color='gray', ls='--', alpha=0.5, label='Event')
        if ds == 'ToneFB':
            ax.axvline(1.0, color='gold', ls='--', linewidth=2, label='Reward')
        # Shade pre/post windows
        ax.axvspan(0.5, 0.9, alpha=0.15, color='blue', label='Pre-reward')
        ax.axvspan(1.1, 1.5, alpha=0.15, color='red', label='Post-reward')
        ax.set_xlabel('Time from event (s)')
        ax.set_ylabel('Fwd-bwd separation')
        p_val = ri['p']
        ax.set_title(f'{key}  delta={ri["delta"]:+.2f}  p={p_val:.3f}')
        ax.legend(fontsize=7)
    plt.suptitle(f'Test 3: {cls} Separation Timecourse Around Reward', fontsize=12)
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("""\
### Test 3 -- Interpretation

**Fill based on results:**
- Is the delta non-significant for GF/GB in ToneFB? → GABA direction neurons are blind to reward. **Strong support for movement theory.**
- Is the delta also non-significant in SpontFB? → Confirms baseline: no change expected at matched latency.
- If delta IS significant for GF/GB → GABA receives reward signals, complicating the pure-movement interpretation.

---
"""))

# ============================================================
# TEST 4: Reward-Time Deflection Null
# ============================================================
cells.append(md("""\
## Test 4: Is There a Reward-Specific Speed Transient?

If reward delivery triggers a distinct neural event, trajectory speed should spike at t=700 in ToneFB. SpontFB (no reward) is the control.

**Method:** `null_reward_deflection()` runs two tests:
1. **Within-ToneFB:** Is speed at reward time higher than at other post-event timepoints? (circular time-shift null)
2. **Between-datasets:** Is ToneFB speed at reward time different from SpontFB at matched latency? (bootstrap permutation)

**Prediction (movement):** Both p-values non-significant. Any speed at reward time reflects ongoing CR movement. ToneFB and SpontFB have similar speed at matched post-movement latency.

**Prediction (RPE):** Significant within-ToneFB p-value (speed spike at reward) AND significant between-dataset difference.

**Caveat:** A significant within-ToneFB result alone is ambiguous -- it could reflect the CR movement peaking near reward time. The between-dataset test is the more informative one.
"""))

cells.append(code("""\
# Test 4: Reward-time deflection null
for combo_label in ['GABA', 'Dopamine']:
    tone_key = f'ToneFB_{combo_label}'
    spont_key = f'SpontFB_{combo_label}'
    if tone_key not in results or spont_key not in results:
        continue

    print(f"\\n{'='*60}")
    print(f"  {combo_label}")
    print(f"{'='*60}")

    null_rd = null_reward_deflection(
        results[tone_key]['smooth_data'], results[tone_key]['window_data'],
        results[spont_key]['smooth_data'], results[spont_key]['window_data'],
        reward_offset=100, dt=DT, test_half_width=10,
        n_permutations=1000, seed=42)

    wt = null_rd['within_task']
    bt = null_rd['between_datasets']
    print(f"  Within-ToneFB:  observed_speed={wt['observed_speed']:.4f}  "
          f"p={wt['p_value']:.4f}")
    print(f"  Between-dataset: task={bt['task_speed']:.4f}  ctrl={bt['ctrl_speed']:.4f}  "
          f"diff={bt['observed_diff']:+.4f}  p={bt['p_value']:.4f}")

    # Plot null distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(wt['null_speeds'], bins=40, color='lightcoral', alpha=0.7, label='Null')
    ax.axvline(wt['observed_speed'], color='black', linewidth=2, label='Observed')
    ax.set_title(f'{combo_label}: Within-ToneFB (p={wt["p_value"]:.4f})')
    ax.set_xlabel('Speed at reward time')
    ax.legend()

    ax = axes[1]
    ax.hist(bt['null_diffs'], bins=40, color='lightblue', alpha=0.7, label='Null diffs')
    ax.axvline(bt['observed_diff'], color='black', linewidth=2, label='Observed diff')
    ax.set_title(f'{combo_label}: Between-dataset (p={bt["p_value"]:.4f})')
    ax.set_xlabel('Speed difference (ToneFB - SpontFB)')
    ax.legend()

    plt.suptitle(f'Test 4: Reward-Time Speed Deflection ({combo_label})')
    plt.tight_layout()
    plt.show()
"""))

cells.append(md("""\
### Test 4 -- Interpretation

**Fill based on results:**
- Both p-values non-significant → No reward-specific speed transient. Consistent with movement theory.
- Within significant, between non-significant → Speed peak at reward time exists but is similar to SpontFB at matched latency → likely movement-driven.
- Both significant → Evidence for a genuine reward-locked signal. Characterise its magnitude relative to the overall movement signal.

---
"""))

# ============================================================
# TEST 5: CS vs CR Direction Sensitivity (NEW)
# ============================================================
cells.append(md("""\
## Test 5: Is the CS Burst Direction-Invariant While the CR Burst Is Direction-Selective?

**This is the strongest test distinguishing salience from movement.**

In ToneFB, we compute fwd-bwd separation in two time windows:
- **CS window [0, 40 timesteps]** (= ToneFB indices [600, 640]): First 400ms after tone onset, before CR starts (reaction time ~30-100 timesteps).
- **Late window [60, 100 timesteps]** (= ToneFB indices [660, 700]): CR movement phase, approaching reward delivery.

**Prediction (movement):**
- DA CS window separation is **LOW** -- the CS burst is a salience response, same for fwd and bwd (direction-invariant).
- DA Late window separation is **HIGH** -- the CR movement is direction-selective.
- GABA: both windows show separation (GABA encodes direction throughout).

**Prediction (RPE):**
- Both bursts encode value → both should have **SIMILAR** direction sensitivity. RPE doesn't predict a CS-vs-Late dissociation.

**Why this is decisive:** A dissociation (CS = direction-invariant, Late = direction-selective) in DA cannot be explained by RPE, which predicts both bursts encode the same variable (value).
"""))

cells.append(code("""\
# Test 5: CS vs CR direction sensitivity
# In the windowed data (length 2*WINDOW+1 = 301), event is at index WINDOW=150.
# CS window: indices [WINDOW, WINDOW+40] = [150, 190]
# Late window: indices [WINDOW+60, WINDOW+100] = [210, 250]
CS_SLICE   = slice(WINDOW, WINDOW + 40)
LATE_SLICE = slice(WINDOW + 60, WINDOW + 100)

dir_sensitivity_results = {}
rng = np.random.default_rng(42)

# Run on combined populations and individual classes
test_keys = []
for ds in ['ToneFB']:
    for pop in ['Dopamine', 'GABA']:
        k = f'{ds}_{pop}'
        if k in results:
            test_keys.append((k, results[k]))
    for cls in ['DF', 'DB', 'GF', 'GB']:
        k = f'{ds}_{cls}'
        if k in single_class_results:
            test_keys.append((k, single_class_results[k]))

for key, r in test_keys:
    sd = r['smooth_data']
    fwd = sd['fwd_smooth']  # (3, n_t)
    bwd = sd['bwd_smooth']

    # Separation timecourse
    sep_t = np.sqrt(np.sum((fwd - bwd)**2, axis=0))

    cs_sep = float(np.mean(sep_t[CS_SLICE]))
    late_sep = float(np.mean(sep_t[LATE_SLICE]))
    obs_diff = late_sep - cs_sep

    # Permutation: circularly shift fwd and bwd independently
    n_t = fwd.shape[1]
    null_diffs = np.empty(1000)
    for i in range(1000):
        shift_f = rng.integers(1, n_t)
        shift_b = rng.integers(1, n_t)
        fwd_s = np.roll(fwd, shift_f, axis=1)
        bwd_s = np.roll(bwd, shift_b, axis=1)
        sep_s = np.sqrt(np.sum((fwd_s - bwd_s)**2, axis=0))
        null_diffs[i] = float(np.mean(sep_s[LATE_SLICE]) - np.mean(sep_s[CS_SLICE]))

    p = float((np.sum(np.abs(null_diffs) >= np.abs(obs_diff)) + 1) / 1001)
    z = (obs_diff - np.mean(null_diffs)) / np.std(null_diffs) if np.std(null_diffs) > 0 else 0

    dir_sensitivity_results[key] = {
        'cs_sep': cs_sep, 'late_sep': late_sep, 'diff': obs_diff,
        'p': p, 'z': z, 'sep_t': sep_t}
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"  {key:25s}  CS_sep={cs_sep:6.2f}  Late_sep={late_sep:6.2f}  "
          f"diff={obs_diff:+6.2f}  z={z:5.2f}  p={p:.4f} {sig}")

# Plot: separation timecourse for key populations
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey='row')
plot_keys = ['ToneFB_Dopamine', 'ToneFB_DF', 'ToneFB_DB',
             'ToneFB_GABA', 'ToneFB_GF', 'ToneFB_GB']
for ax, pk in zip(axes.flat, plot_keys):
    if pk not in dir_sensitivity_results:
        ax.set_title(f'{pk}: N/A')
        continue
    dr = dir_sensitivity_results[pk]
    t_axis = (np.arange(len(dr['sep_t'])) - WINDOW) * DT
    ax.plot(t_axis, dr['sep_t'], color='black', linewidth=1)
    ax.axvline(0, color='gray', ls='--', alpha=0.5)
    ax.axvline(1.0, color='gold', ls='--', linewidth=1.5, label='Reward')
    ax.axvspan(0, 0.4, alpha=0.2, color='cyan', label='CS window')
    ax.axvspan(0.6, 1.0, alpha=0.2, color='orange', label='Late window')
    ax.set_title(f'{pk}  CS={dr["cs_sep"]:.1f}  Late={dr["late_sep"]:.1f}  p={dr["p"]:.3f}')
    ax.set_xlabel('Time (s)')
    if ax in axes[:, 0]:
        ax.set_ylabel('Fwd-bwd separation')
    ax.legend(fontsize=6)

plt.suptitle('Test 5: CS vs Late Direction Sensitivity', fontsize=13)
plt.tight_layout()
plt.show()

# Bar plot: CS vs Late for each population
fig, ax = plt.subplots(figsize=(10, 5))
pop_keys = [k for k in dir_sensitivity_results.keys()]
x = np.arange(len(pop_keys))
cs_vals = [dir_sensitivity_results[k]['cs_sep'] for k in pop_keys]
late_vals = [dir_sensitivity_results[k]['late_sep'] for k in pop_keys]
ax.bar(x - 0.15, cs_vals, 0.3, label='CS window [0, 0.4s]', color='cyan', alpha=0.8)
ax.bar(x + 0.15, late_vals, 0.3, label='Late window [0.6, 1.0s]', color='orange', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([k.replace('ToneFB_', '') for k in pop_keys], rotation=30)
ax.set_ylabel('Mean fwd-bwd separation')
ax.set_title('Test 5: Direction Sensitivity — CS vs Late Window')
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
### Test 5 -- Interpretation

**Fill based on results:**
- DA: CS_sep << Late_sep (significant difference) → **CS burst is direction-invariant, CR burst is direction-selective.** This is the decisive dissociation. Cannot be explained by RPE (which predicts both encode the same variable).
- GABA: both CS_sep and Late_sep are high → GABA encodes direction throughout. Consistent with movement encoding.
- If DA CS_sep ≈ Late_sep → No dissociation. Both bursts have similar directional content. Would require alternative explanation.

---
"""))

# ============================================================
# TEST 6: CR Burst Matches Movement Subspace
# ============================================================
cells.append(md("""\
## Test 6: Does the Reward-Time DA Burst Live in the Movement Subspace?

If DA activity near reward delivery is actually a CR movement signal, it should be well-explained by SpontFB PCs (which capture pure movement). The CS burst may NOT be fully in the movement subspace (salience uses different dimensions).

**Method:** Fit PCA on SpontFB. Project two windows from ToneFB:
1. CS window [600, 640]: initial CS burst
2. Late window [660, 700]: CR/reward phase

Compute R-squared for each. **Null:** phase-randomise SpontFB neurons, refit PCA, re-project.

**Prediction (movement):** Late window R-squared is HIGH (the "reward burst" IS movement). CS window R-squared is LOWER (salience occupies partially different dimensions).

**Prediction (RPE):** Both windows have SIMILAR R-squared (both value signals, both equally different from movement).
"""))

cells.append(code("""\
# Test 6: Project ToneFB windows onto SpontFB PCA
subspace_results = {}
rng = np.random.default_rng(42)

for combo_label, groups in [('Dopamine', DA_GROUPS), ('GABA', GABA_GROUPS)]:
    spont_key = f'SpontFB_{combo_label}'
    tone_key = f'ToneFB_{combo_label}'
    if spont_key not in results or tone_key not in results:
        continue

    # Align neurons between SpontFB and ToneFB
    data_spont = results[spont_key]['data']
    data_tone = results[tone_key]['data']
    X_spont, X_tone, n_dropped = _align_neuron_data(data_spont, data_tone, groups)
    n_t = X_spont.shape[1] // 2
    n_neurons = X_spont.shape[0]

    # Fit PCA on SpontFB (full data)
    pca_spont = fit_pca(X_spont, n_components=min(N_COMPONENTS, n_neurons - 1))

    # Define windows (absolute indices in the data)
    cs_start, cs_end = 600, 640
    late_start, late_end = 660, 700

    for win_name, w_start, w_end in [('CS', cs_start, cs_end), ('Late', late_start, late_end)]:
        # Slice window from ToneFB (both fwd and bwd halves)
        X_win = np.hstack([X_tone[:, w_start:w_end], X_tone[:, n_t + w_start:n_t + w_end]])

        # Observed R-squared
        r2_obs = compute_reconstruction_r2(pca_spont, X_win)

        # Null: phase-randomise SpontFB, refit PCA, reproject
        null_r2 = np.empty(500)
        for i in range(500):
            X_spont_null = _phase_randomise(X_spont, rng)
            pca_null = fit_pca(X_spont_null, n_components=min(N_COMPONENTS, n_neurons - 1))
            null_r2[i] = compute_reconstruction_r2(pca_null, X_win)

        null_mean = float(np.mean(null_r2))
        null_std = float(np.std(null_r2))
        z = (r2_obs - null_mean) / null_std if null_std > 0 else float('inf')
        p = float((np.sum(null_r2 >= r2_obs) + 1) / 501)

        label = f'{combo_label}_{win_name}'
        subspace_results[label] = {
            'r2': r2_obs, 'null_mean': null_mean, 'null_std': null_std,
            'z': z, 'p': p}
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"  {label:25s}  R2={r2_obs:.4f}  null={null_mean:.4f}+/-{null_std:.4f}  "
              f"z={z:.2f}  p={p:.4f} {sig}")

# Plot: CS vs Late R-squared
fig, ax = plt.subplots(figsize=(8, 5))
for combo in ['Dopamine', 'GABA']:
    cs_key = f'{combo}_CS'
    late_key = f'{combo}_Late'
    if cs_key in subspace_results and late_key in subspace_results:
        cs_r2 = subspace_results[cs_key]['r2']
        late_r2 = subspace_results[late_key]['r2']
        cs_null = subspace_results[cs_key]['null_mean']
        late_null = subspace_results[late_key]['null_mean']
        x_pos = 0 if combo == 'Dopamine' else 1
        ax.bar(x_pos - 0.15, cs_r2, 0.3, color='cyan', alpha=0.8,
               label='CS window' if x_pos == 0 else '')
        ax.bar(x_pos + 0.15, late_r2, 0.3, color='orange', alpha=0.8,
               label='Late window' if x_pos == 0 else '')
        # Null baseline
        ax.hlines(cs_null, x_pos - 0.3, x_pos, colors='cyan', linestyles='--', alpha=0.5)
        ax.hlines(late_null, x_pos, x_pos + 0.3, colors='orange', linestyles='--', alpha=0.5)

ax.set_xticks([0, 1])
ax.set_xticklabels(['Dopamine', 'GABA'])
ax.set_ylabel('R-squared (projected onto SpontFB PCs)')
ax.set_title('Test 6: ToneFB Windows in SpontFB Movement Subspace')
ax.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
### Test 6 -- Interpretation

**Fill based on results:**
- DA Late R-squared > DA CS R-squared → The "reward burst" lives more in the movement subspace than the CS burst does. Supports: reward-time activity = CR movement.
- Both significantly above null → Both windows contain some movement-related variance (expected: even CS may trigger preparatory movement).
- GABA: both windows have similar R-squared → GABA encodes direction in both phases (consistent with Test 2/3).
- If DA CS R-squared ≈ DA Late R-squared → No subspace dissociation. Both live equally in the movement subspace. Would need to reconcile with Test 5.

---
"""))

# ============================================================
# SUMMARY
# ============================================================
cells.append(md("""\
---
## Summary of Results
"""))

cells.append(code("""\
# Summary table
print("=" * 90)
print(f"{'Test':50s} {'Result':>10s} {'p-value':>10s} {'Supports':>15s}")
print("=" * 90)

# Test 1: SpontFB cross-projection
for label, res in cross_proj_results.items():
    supports = 'Movement' if res['r2'] > 0.70 and res['p'] < 0.05 else 'Ambiguous'
    print(f"T1: {label:45s} R2={res['r2']:.3f}  p={res['p']:.4f}  {supports}")

# Test 2: Per-class separation (summary)
for row in sep_results:
    if row['class'] in ['DF', 'DB', 'GF', 'GB']:
        supports = 'Movement' if row['p'] < 0.05 else 'No signal'
        print(f"T2: {row['key']:45s} sep={row['separation']:.2f}  p={row['p']:.4f}  {supports}")

# Test 3: GF/GB reward insensitivity
for key, res in reward_insensitivity_results.items():
    if 'ToneFB' in key:
        supports = 'Movement' if res['p'] > 0.05 else 'RPE?'
        print(f"T3: {key:45s} delta={res['delta']:+.2f}  p={res['p']:.4f}  {supports}")

# Test 5: CS vs CR direction sensitivity
for key, res in dir_sensitivity_results.items():
    if 'ToneFB_Dopamine' in key or 'ToneFB_GABA' in key:
        supports = 'Movement' if res['diff'] > 0 and res['p'] < 0.05 else 'Ambiguous'
        print(f"T5: {key:45s} CS={res['cs_sep']:.1f} Late={res['late_sep']:.1f}  "
              f"p={res['p']:.4f}  {supports}")

# Test 6: CR burst in movement subspace
for label, res in subspace_results.items():
    print(f"T6: {label:45s} R2={res['r2']:.3f}  p={res['p']:.4f}")

print("=" * 90)
"""))

cells.append(md("""\
## Conclusion

**Fill based on combined evidence from all 6 tests.**

| Test | Question | Movement prediction | RPE prediction | Observed |
|------|----------|-------------------|---------------|----------|
| **T1** | SpontFB explains task data? | R2 > 0.70 | DA R2 < GABA R2 | |
| **T2** | DF/DB/GF/GB direction-selective? | All significant | DA weaker | |
| **T3** | GF/GB blind to reward? | p > 0.05 | p < 0.05 | |
| **T4** | Speed transient at reward? | p > 0.05 | p < 0.05 | |
| **T5** | CS direction-invariant, CR direction-selective? | CS low, Late high | Both similar | |
| **T6** | Reward-time burst in movement subspace? | Late R2 > CS R2 | Both similar | |

**Next steps:**
- Behavioural correlation: once per-trial force/lick data aligned, link PCs to kinematics directly.
- Trial-level PCA for cross-validation.
- Airpuff data: does aversive stimulus project onto same movement subspace?
"""))

# ============================================================
# Assemble notebook
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "cells": cells
}

with open('PCA_Core_Tests.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created PCA_Core_Tests.ipynb with {len(cells)} cells")
