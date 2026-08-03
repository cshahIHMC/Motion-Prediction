"""
locomotion_index.py

Derives a "Locomotion Index" (LI) from a trained FLD checkpoint's latent
phasor representation, on trial_10 (sit_stand_squat_transitions -- a good
test case since it should show real oscillatory/postural regime changes).

LI = phasor circular variance x phase velocity (compute_LI_phasor), computed
over a sliding window (WINDOW frames), then session-normalised by its own
95th percentile so 1.0 means "as fast/coherent as the brisker moments of
THIS trial" -- not an absolute, cross-trial-comparable bound the way the
energy-ratio formula this replaced was. See compute_LI_phasor's docstring.
Read-only: loads the latest FLD checkpoint, slides its encoder across the
trial exactly as training does (same normalisation, same windowing, same
model.forward() call), computes LI, and displays two figures via
plt.show() -- nothing is saved to disk, nothing is retrained.

Run:
    python3 locomotion_index.py

--------------------------------------------------------------------------
What's plotted, and why it's organised this way
--------------------------------------------------------------------------
Figure 1 (3 subplots, sharing a time axis):
  1. LI(t) with regime-shaded background (locomotion/transitional/postural)
     + the 0.15/0.40 regime-boundary lines. The first WINDOW frames are a
     warm-up artifact (LI=0 by construction, not a real postural reading).
  2. Per-channel phase coverage (CV x V), one line per of the 8 latent
     channels -- the per-channel quantity LI is the cross-channel mean of;
     there's no separate "per-channel LI" the way the old formula had.
  3. Insole ground-reaction force, L and R (raw sensor units) as a
     reference signal, so you can visually cross-check regime calls
     against actual stance/loading events.
Figure 2 (2x4 grid, one subplot per channel):
  Circular variance (CV) and phase velocity (V) over time per channel, on
  twin y-axes -- shows which channels are driving coverage (and therefore
  LI) up or down at each moment.

--------------------------------------------------------------------------
Notes
--------------------------------------------------------------------------
- Time axis is literally `window_index / sample_rate` (per the brief), not
  frame_indices/sample_rate -- the two differ by a fixed ~(H-1)/150s
  offset (window 0 in the latent series is anchored to raw trial frame
  H-1, not frame 0), but since every series here has the same length T and
  is plotted against the same axis, the only cost is that the x-axis
  doesn't read as "true" trial-elapsed time. knee_angle_r itself IS
  correctly aligned to the latent axis via frame_indices before plotting,
  it's just displayed against the simpler window-index time axis alongside
  everything else.
- task_labels (per the brief's optional parameter) isn't exercised here --
  a single trial's CSV has exactly one constant task label for its whole
  duration, so there's nothing to differentiate against a time axis within
  one trial. The plotting code below falls back to time-only, which the
  brief explicitly allows.
- frequency is converted from the model's raw "cycles-per-window" output
  to Hz before being used here (freq_hz = freq_raw * sample_rate / H) --
  same non-obvious unit fix documented in impedance_from_latents.py; f in
  the brief's formula is described as "Hz", so this conversion is applied.
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))

from plot_error_across_time import (
    _load_all_data, _discover_tasks, _find_checkpoint, _load_state_dict_flexible,
    INPUT_COLUMNS, OUTPUT_COLUMNS, INPUT_DIM, OUTPUT_DIM,
)
from Models.FLD import FLD

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLD_RUN  = ""              # blank = latest FLD run under runs/FLD_*
# Data/all_data.csv now prefixes trial_id with subject (S01_/S02_). S01_trial_10
# is sit_stand_squat_transitions -- an actual stand/squat trial (the old
# "trial_10" default here predates the subject-prefixed naming and its
# "walking_1_2_m_s" comment was already stale relative to the current data).
TRIAL_ID = "S01_trial_10"

# Known-segment spot-check (see main()'s "SQUAT/STANDING SEGMENT DIAGNOSTICS" block):
# indices are into the WINDOW/latent axis (A/f/c, length T), not raw CSV rows.
# Point SEGMENT_SLICE at an actual known hold within TRIAL_ID for this
# diagnostic to mean anything -- the default below is illustrative only.
SEGMENT_SLICE = slice(500, 700)

# compute_LI_phasor's sliding window, in frames (30 = 200ms at 150Hz). The
# first WINDOW entries of LI/CV/V are exactly 0 (not enough history yet to
# compute a circular stat) -- that's a warm-up artifact, not a real
# "postural" reading, and it does get labelled 'postural' by the regime
# thresholds below since 0 < 0.15. See the note printed in main().
WINDOW = 30

SAMPLE_RATE = 150.0
LATENT_STRIDE = 1         # frames between encoder windows -- 1 keeps dt exactly 1/SAMPLE_RATE
MAX_ENCODE_BATCH = 256
FORCE_COLUMNS = ["L_insole_force", "R_insole_force"]

REGIME_COLORS = {"locomotion": "#f7f7f7", "transitional": "#d0d0d0", "postural": "#a0a0a0"}
REGIME_LABELS = {"locomotion": "Locomotion", "transitional": "Transitional", "postural": "Postural"}


# ---------------------------------------------------------------------------
# Model / data loading -- same pattern as impedance_from_latents.py / plot_phase_vs_insole.py
# ---------------------------------------------------------------------------

def load_fld_model(device, pinned_run=""):
    run_dir, ckpt_path = _find_checkpoint("FLD", "fld_model", pinned_run)
    print(f"[LI] FLD checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FLD(
        observation_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        history_horizon=cfg["history_horizon"],
        latent_channel=cfg["latent_dim"],
        device=device,
        dt=1.0 / SAMPLE_RATE,
        encoder_shape=cfg["encoder_shape"],
        decoder_shape=cfg["decoder_shape"],
    )
    _load_state_dict_flexible(model, ckpt["fld_state_dict"])
    model.to(device)
    model.eval()
    mean_in = ckpt["in_mean"].cpu().numpy()
    std_in  = ckpt["in_std"].cpu().numpy()
    print(f"[LI]   epoch={ckpt['epoch']}  history_horizon={cfg['history_horizon']}  "
          f"latent_dim={cfg['latent_dim']}")
    return model, cfg, mean_in, std_in


def encode_latent_series(model, data_in, history_horizon, device, stride=1, batch_size=256):
    """Slide the encoder across the whole trial (encode-only, k=1, no
    forecasting) -- identical mechanism to analyze_fld_latents.py /
    impedance_from_latents.py."""
    N = data_in.shape[0]
    starts = list(range(0, N - history_horizon + 1, stride))
    if not starts:
        raise ValueError(f"Trial too short ({N} rows) for history_horizon={history_horizon}")

    phases, freqs, amps, offsets = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            wins = torch.stack([data_in[s:s + history_horizon].T for s in batch_starts]).to(device)
            _, _, _, params = model(wins, k=1)
            phase, freq, amp, offset = params
            phases.append(phase.cpu())
            freqs.append(freq.cpu())
            amps.append(amp.cpu())
            offsets.append(offset.cpu())

    phase  = torch.cat(phases,  dim=0).numpy()   # (T, N) cycles, wrapped (-0.5, 0.5]
    freq_w = torch.cat(freqs,   dim=0).numpy()   # (T, N) cycles PER WINDOW -- converted to Hz below
    amp    = torch.cat(amps,    dim=0).numpy()
    offset = torch.cat(offsets, dim=0).numpy()
    frame_indices = np.array(starts) + history_horizon - 1
    return phase, freq_w, amp, offset, frame_indices


def build_latent_32(phase, freq_w, amp, offset, history_horizon, sample_rate):
    """Assemble the (T, 32) phasor latent z, in the layout
    compute_locomotion_index expects: [A_cos, A_sin, f_Hz, c] per channel."""
    phi_rad = 2.0 * np.pi * phase
    freq_hz = freq_w * sample_rate / history_horizon
    A_cos = amp * np.cos(phi_rad)
    A_sin = amp * np.sin(phi_rad)
    T, N = phase.shape
    z = np.zeros((T, 4 * N), dtype=np.float64)
    z[:, 0::4] = A_cos
    z[:, 1::4] = A_sin
    z[:, 2::4] = freq_hz
    z[:, 3::4] = offset
    return z


# ---------------------------------------------------------------------------
# Locomotion Index -- exactly as specified
# ---------------------------------------------------------------------------

def compute_LI_phasor(z, N=8, window=30, eps=1e-8):
    """
    Locomotion Index from phasor circular variance x phase velocity.
    Fully within the FLD latent. No raw IMU needed. Subject-independent.

    Physical interpretation:
        CV     = how much of the phase circle is being visited
        v_mean = how fast the phase is moving
        CV x v_mean = phase coverage rate = locomotion signature

    window: sliding window in frames (30 frames = 200ms at 150Hz)
    """
    A_cos = z[:, 0::4]                              # (T, N)
    A_sin = z[:, 1::4]                              # (T, N)
    phi   = np.arctan2(A_sin, A_cos)                # (T, N)
    A     = np.sqrt(A_cos**2 + A_sin**2)            # (T, N)

    T   = len(z)
    CV  = np.zeros((T, N))
    V   = np.zeros((T, N))

    for t in range(window, T):
        phi_win = phi[t-window:t]                   # (window, N)

        # Circular variance
        R       = np.abs(np.mean(np.exp(1j * phi_win), axis=0))  # (N,)
        CV[t]   = 1 - R                             # (N,) in [0, 1]

        # Phase velocity -- unwrap to remove 2pi jumps
        phi_unwrap = np.unwrap(phi_win, axis=0)     # (window, N)
        dphi       = np.diff(phi_unwrap, axis=0)    # (window-1, N)
        V[t]       = np.abs(np.mean(dphi, axis=0))  # (N,) rad/frame

    # Combined signal: phase coverage rate
    coverage = CV * V                               # (T, N)

    # Global mean across channels
    LI_raw = np.mean(coverage, axis=1)             # (T,)

    # Session-normalise to [0, 1]
    # 95th percentile anchors to brisk walking within the session
    p95    = np.percentile(LI_raw[window:], 95)
    LI     = np.clip(LI_raw / (p95 + eps), 0, 1)

    # Regime classification
    # Thresholds set from your data:
    #   Walking LI_raw ~ 0.052, normalised ~ 1.0
    #   Squat   LI_raw ~ 0.008, normalised ~ 0.15
    #   Stand   LI_raw ~ 0.002, normalised ~ 0.04
    # Boundaries at 0.4 and 0.15 of normalised scale
    regime = np.where(LI > 0.40, 'locomotion',
             np.where(LI > 0.15, 'transitional',
                                 'postural'))

    return LI, CV, V, coverage, regime


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _shade_regimes(ax, t, regime):
    """Shade contiguous regime runs as background spans (grouping runs
    rather than one axvspan per sample -- there can be tens of thousands of
    windows). Returns legend handles for whichever regimes actually occur."""
    regime_arr = np.asarray(regime)
    change_idx = np.where(regime_arr[1:] != regime_arr[:-1])[0] + 1
    bounds = [0] + change_idx.tolist() + [len(regime_arr)]

    seen = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        r = regime_arr[s]
        t_end = t[e] if e < len(t) else t[-1]
        ax.axvspan(t[s], t_end, color=REGIME_COLORS[r], alpha=0.7, lw=0, zorder=0)
        if r not in seen:
            seen.append(r)

    order = ['locomotion', 'transitional', 'postural']
    handles = [Patch(facecolor=REGIME_COLORS[r], edgecolor='none', label=REGIME_LABELS[r])
               for r in order if r in seen]
    return handles


def plot_li_overview(t, LI, coverage, regime, force_l, force_r, N, trial_id, task_label):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # --- Subplot 1: LI with regime shading ---
    regime_handles = _shade_regimes(axes[0], t, regime)
    line_global, = axes[0].plot(t, LI, color='black', lw=1.3, zorder=5, label='LI')
    axes[0].axhline(0.40, color='dimgrey', lw=1.0, ls='--', zorder=4)
    axes[0].axhline(0.15, color='dimgrey', lw=1.0, ls='--', zorder=4)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Locomotion Index")
    axes[0].set_title(f"Global Locomotion Index (phasor CV x phase velocity) -- {trial_id} ({task_label})")
    axes[0].legend(handles=regime_handles + [line_global], loc='upper right', fontsize=8)

    # --- Subplot 2: per-channel phase coverage (CV x V) -- the per-channel
    # quantity that LI is the cross-channel mean of; there's no separate
    # "per-channel LI" in this formula the way there was for the old
    # energy-ratio one ---
    cmap = plt.get_cmap('tab10')
    for ch in range(N):
        axes[1].plot(t, coverage[:, ch], color=cmap(ch % 10), lw=0.8, alpha=0.85, label=f"ch_{ch}")
    axes[1].set_ylabel("Per-Channel Phase\nCoverage (CV x V)")
    axes[1].set_title("Per-Channel Phase Coverage Rate")
    axes[1].legend(fontsize=7, ncol=4, loc='upper right')
    axes[1].grid(alpha=0.15)

    # --- Subplot 3: insole GRF, L and R, reference only ---
    axes[2].plot(t, force_l, color='tab:blue', lw=1.0, alpha=0.85, label='L insole force')
    axes[2].plot(t, force_r, color='tab:red',  lw=1.0, alpha=0.85, label='R insole force')
    axes[2].set_ylabel("Insole GRF\n(raw sensor units)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Reference signal -- insole ground-reaction force, L+R (not part of LI)")
    axes[2].legend(fontsize=8, loc='upper right')
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_phase_diagnostics_per_channel(t, CV, V, N, trial_id):
    """Per-channel breakdown of the two quantities LI is built from: circular
    variance (how much of the phase circle gets visited) and phase velocity
    (how fast phase is moving) -- twin y-axes per channel since they're
    different units/scales (CV in [0,1], V in rad/frame, small)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axes = axes.flatten()

    for ch in range(N):
        ax = axes[ch]
        axb = ax.twinx()
        l1, = ax.plot(t, CV[:, ch], color='steelblue', lw=0.9, alpha=0.9, label='CV')
        l2, = axb.plot(t, V[:, ch], color='darkorange', lw=0.9, alpha=0.8, label='V (rad/frame)')
        ax.set_ylim(0, 1)
        ax.set_title(f"Channel {ch}", fontsize=10)
        ax.set_ylabel("Circular variance", fontsize=8, color='steelblue')
        axb.set_ylabel("Phase velocity", fontsize=8, color='darkorange')
        if ch >= N - 4:
            ax.set_xlabel("Time (s)")
        if ch == 0:
            ax.legend(handles=[l1, l2], fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        axb.tick_params(labelsize=7)
        ax.grid(alpha=0.15)

    for ax in axes[N:]:
        ax.set_visible(False)

    fig.suptitle(f"Phase Diagnostics per Channel (CV, phase velocity) -- {trial_id}",
                fontsize=12, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LI] Using device: {device}")

    all_df = _load_all_data()
    discovered = _discover_tasks(all_df)
    if TRIAL_ID not in discovered:
        raise ValueError(f"TRIAL_ID={TRIAL_ID!r} not found. Available: {sorted(discovered)}")
    task_label = discovered[TRIAL_ID]['label']
    print(f"[LI] Trial: {TRIAL_ID} ({task_label})")

    model, cfg, mean_in, std_in = load_fld_model(device, FLD_RUN)
    H = cfg["history_horizon"]
    N = cfg["latent_dim"]

    df = all_df[all_df['trial_id'] == TRIAL_ID][INPUT_COLUMNS + OUTPUT_COLUMNS].reset_index(drop=True)
    raw_in  = df[INPUT_COLUMNS].values.astype("float32")
    norm_in = torch.tensor((raw_in - mean_in) / std_in, dtype=torch.float32)

    print(f"[LI] Encoding latent series (stride={LATENT_STRIDE}, {len(df):,} rows)...")
    phase, freq_w, amp, offset, frame_indices = encode_latent_series(
        model, norm_in, H, device, stride=LATENT_STRIDE, batch_size=MAX_ENCODE_BATCH,
    )
    T = phase.shape[0]
    z = build_latent_32(phase, freq_w, amp, offset, H, SAMPLE_RATE)
    print(f"[LI] Latent series: T={T:,} windows, N={N} channels")

    # Direct z-parse, independent of whichever LI formula is active -- reused
    # by the c/f-channel stats and the SEGMENT_SLICE spot-check below.
    A_cos_all = z[:, 0::4]
    A_sin_all = z[:, 1::4]
    f_all = z[:, 2::4]
    c_all = z[:, 3::4]
    A_all = np.sqrt(A_cos_all**2 + A_sin_all**2)

    LI, CV, V, coverage, regime = compute_LI_phasor(z, N=N, window=WINDOW)
    print(f"[LI] LI (excl. {WINDOW}-frame warm-up): mean={LI[WINDOW:].mean():.3f}  "
          f"min={LI[WINDOW:].min():.3f}  max={LI[WINDOW:].max():.3f}")
    regime_stat = regime[WINDOW:]   # first WINDOW frames are a warm-up artifact (LI=0 by construction), not real 'postural'
    for r in ("locomotion", "transitional", "postural"):
        frac = np.mean(regime_stat == r)
        print(f"[LI]   {r:<13s}: {frac*100:5.1f}% of windows")

    print("c channel statistics:")
    print(f"  mean of |c|: {np.mean(np.abs(c_all)):.4f}")
    print(f"  std  of |c|: {np.std(c_all):.4f}")
    print(f"  max  of |c|: {np.max(np.abs(c_all)):.4f}")

    # "standing/squat" windows -- the closest existing proxy is the regime
    # classification itself (regime == 'postural'). TRIAL_ID=trial_3 is a
    # continuous walking trial, not a stand/squat trial, so 'postural'
    # windows here (if any) are at best brief low-oscillation moments in
    # the gait cycle (e.g. double-support), NOT true quiet standing --
    # falls back to 'transitional' with a warning if 'postural' is empty.
    regime_arr = np.array(regime)
    squat_mask = regime_arr == "postural"
    fallback_used = None
    if not squat_mask.any():
        fallback_used = "transitional"
        squat_mask = regime_arr == "transitional"
    if not squat_mask.any():
        fallback_used = "none -- ALL windows (this trial never leaves the 'locomotion' regime)"
        squat_mask = np.ones_like(regime_arr, dtype=bool)
    if fallback_used:
        print(f"[LI] NOTE: no 'postural' windows in {TRIAL_ID} (a walking trial, not stand/squat) -- "
              f"falling back to '{fallback_used}' for the f-channel stats below. Treat these as, at "
              f"best, a rough proxy for low-oscillation gait moments, not true quiet standing.")
    f_squat = f_all[squat_mask]

    print("f channel statistics during what should be standing/squat:")
    print(f"  mean f: {np.mean(f_squat):.4f}")
    print(f"  max  f: {np.max(f_squat):.4f}")

    # ---- Known-segment spot-check -- see SEGMENT_SLICE note in Config ----
    # Kept as the original energy-ratio formula (self-contained, parsed
    # directly from z) regardless of which LI formula is active above --
    # this was a separately-requested diagnostic, not tied to compute_LI_phasor.
    seg = SEGMENT_SLICE
    A_seg = A_all[seg]   # (seg_len, N)
    f_seg = f_all[seg]
    c_seg = c_all[seg]

    E_osc_seg  = A_seg**2 * f_seg**2
    E_post_seg = c_seg**2
    LI_seg     = E_osc_seg / (E_osc_seg + E_post_seg + 1e-8)

    print(f"=== SQUAT/STANDING SEGMENT DIAGNOSTICS (windows {seg.start}:{seg.stop} of {TRIAL_ID}) ===")
    print(f"A    mean={np.mean(A_seg):.4f}  std={np.std(A_seg):.4f}  max={np.max(A_seg):.4f}")
    print(f"f    mean={np.mean(f_seg):.4f}  std={np.std(f_seg):.4f}  max={np.max(f_seg):.4f}")
    print(f"c    mean={np.mean(c_seg):.4f}  std={np.std(c_seg):.4f}  max={np.max(c_seg):.4f}")
    print(f"E_osc  mean={np.mean(E_osc_seg):.4f}")
    print(f"E_post mean={np.mean(E_post_seg):.4f}")
    print(f"LI     mean={np.mean(LI_seg):.4f}")
    print(f"Ratio E_osc/E_post = {np.mean(E_osc_seg)/np.mean(E_post_seg):.2f}")

    # ---- Phase diagnostics per segment (circular variance + phase velocity) ----
    # segments reuses what's already computed above: the manually-specified
    # SEGMENT_SLICE plus the three regime masks -- lets you compare phase
    # coherence/velocity across a known window AND across regimes in one pass.
    # Boolean-mask segments that are empty (postural/transitional currently
    # are, for a walking trial) are skipped with a note instead of crashing.
    segments = {
        "known_segment": SEGMENT_SLICE,
        "locomotion":    regime_arr == "locomotion",
        "transitional":  regime_arr == "transitional",
        "postural":      regime_arr == "postural",
    }

    for name, idx in segments.items():
        if isinstance(idx, np.ndarray) and not idx.any():
            print(f"\n=== {name} === (skipped -- no windows in this regime for {TRIAL_ID})")
            continue

        A_cos_seg = z[idx, 0::4]
        A_sin_seg = z[idx, 1::4]
        phi_seg   = np.arctan2(A_sin_seg, A_cos_seg)   # (T, 8)

        # Circular variance -- named *_seg to avoid shadowing the outer
        # (T,N) CV/V arrays from compute_LI_phasor, needed below for Figure 2.
        R_seg  = np.abs(np.mean(np.exp(1j * phi_seg), axis=0))
        CV_seg = 1 - R_seg

        # Phase velocity
        phi_unwrap = np.unwrap(phi_seg, axis=0)
        dphi           = np.diff(phi_unwrap, axis=0)
        v_mean_seg     = np.abs(np.mean(dphi, axis=0))
        v_std_seg      = np.std(dphi, axis=0)

        # Amplitude stats for reference
        A_seg = np.sqrt(A_cos_seg**2 + A_sin_seg**2)

        print(f"\n=== {name} ===")
        print(f"A mean per channel:   {np.mean(A_seg, axis=0).round(4)}")
        print(f"Circular variance CV: {CV_seg.round(4)}")
        print(f"Phase velocity mean:  {v_mean_seg.round(4)}")
        print(f"Phase velocity std:   {v_std_seg.round(4)}")
        print(f"CV global mean:       {np.mean(CV_seg):.4f}")
        print(f"v_mean global mean:   {np.mean(v_mean_seg):.4f}")

    force_l = df[FORCE_COLUMNS[0]].values.astype("float64")[frame_indices]   # aligned to latent axis
    force_r = df[FORCE_COLUMNS[1]].values.astype("float64")[frame_indices]
    t = np.arange(T) / SAMPLE_RATE   # per spec: window index * dt

    fig1 = plot_li_overview(t, LI, coverage, regime, force_l, force_r, N, TRIAL_ID, task_label)
    fig2 = plot_phase_diagnostics_per_channel(t, CV, V, N, TRIAL_ID)

    print("[LI] Displaying figures (close the windows, or Ctrl+C, to exit).")
    plt.show()


if __name__ == "__main__":
    main()
