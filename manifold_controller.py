"""
manifold_controller.py

Closed-form manifold controller from the FLD-TeacherStudent latent. One
script, no module split (per the brief) -- but that constraint applies to
the RUNTIME path (Stage 3 onward), not to offline fitting: Stage 2 imports
and reuses fit_phase_torque_profile.py's Fourier-fit machinery rather than
reimplementing it (see DECISIONS #6 below).

============================================================================
DECISIONS LOCKED IN with the user after the Stage-0 exploration report --
do not silently change any of these without going back to the user:
============================================================================
  1. Encoder = the NON-GRF FLDTeacherStudent (runs/FLDTeacherStudent_*, NOT
     the "(Ang + Vel + GRF)" variant). r_star(phi)/sigma(phi)/tau(phi)
     TABLES are built from forward_teacher (privileged, offline-only).
     ALL Stage 4 and Stage 5 NUMBERS are computed from forward_student --
     the student is what actually runs. Teacher-phase results are reported
     ONLY as a privileged upper bound, explicitly labeled as such.
  2. The teacher's own biomech input already contains the joint moments
     (BIOMECH_COLUMNS = 10 angles + 10 velocities + 10 moments), so a
     teacher-phase tau(phi) fit is partly circular. Student-phase fit is
     headlined; this caveat is stated wherever a teacher-phase fit number
     is shown.
  3. Dominant latent channel = argmax(amplitude.mean(axis=0)), same
     convention as diagnostics/kphi_vs_quasistiffness.py's dom_ch. Checked
     for STABILITY across all 4 subjects x all 8 incline trials (both
     forward_teacher AND forward_student, since the shared decoder implies
     channel k should mean the same thing in both branches) before any
     Stage-1 table is built. If the argmax flips between trials, this
     script stops (SystemExit) rather than silently picking one.
  4. Frame alignment = CAUSAL-LAST (frame_index = window_start +
     history_horizon - 1) EVERYWHERE in this file -- see FRAME_ALIGNMENT
     assert below and _causal_last_frame_indices(), the ONLY place a frame
     index is ever computed. Never window-center (that convention exists
     elsewhere in this repo, e.g. plot_pca_manifold_train.py, for a
     different, non-causal diagnostic purpose -- it must not leak in here).
  5. Units, defined once (see UNITS section) and used throughout:
       phi        -- native cycle-fraction (model's own convention), NOT radians
       f_hz       -- freq_raw * sample_rate / history_horizon
       phase adv. -- phi + f_hz*dt, in CYCLE units (no 2*pi -- phi is
                     already a fraction of a cycle, not an angle)
       Fourier basis -- cos(2*pi*k*phi) / sin(2*pi*k*phi)
  6. Stage 2 imports fit_phase_torque_profile.py's _fourier_design_matrix /
     evaluate_fourier_torque rather than re-deriving the fit -- "one
     script" is about the deployed/runtime path, not offline tooling.
  7. Subject-level holdout: S04's incline trials are excluded from Stage 1
     (tables) and Stage 2 (fit) ENTIRELY -- reserved whole for Stage 5
     validation. Not a per-trial split.
  8. The annulus claim is specifically about S01_trial_13
     (incline_walking_10_deg_1_2_m_s) -- confirmed from the Stage-0 task
     table, not assumed. Flow coherence is verified quantitatively on that
     trial in Stage 1 (a global tangential-vs-radial velocity-alignment
     check); the fuller k-NN neighbor-based flow-coherence metric is
     Stage 5's job, run on transition/sit-to-stand trials.
  9. combine_data.py's SUBJECTS list: NOT edited. Stage-0 flagged it as
     hardcoded to ["S01"] -- that was stale; it currently reads
     ["S01","S02","S03","S04"], matching Data/all_data.csv already on
     disk (both dated 2026-08-25). Flagged back to the user instead of
     writing a now-false comment.
 10. The GRF-variant checkpoint (invisible to _find_checkpoint due to a
     differing filename stem) and _find_checkpoint's alphabetical-not-mtime
     run-dir sort are left alone, per instruction -- documented in
     README.md under "Known issues", not fixed here.
============================================================================

Run:
    python3 manifold_controller.py --stage 1
"""

import argparse
import os
import sys

import time

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors   # already a repo dependency, see Library/fld_plotter.py's PCA import
import matplotlib
# NOTE: deliberately NOT gated on os.environ.get("DISPLAY") -- a set DISPLAY
# does not mean a real, interactive window manager is behind it (true in
# this project's dev sandbox: DISPLAY=:1 is set but plt.show() hangs
# forever with no window ever appearing or closable). Every plot is always
# saved to disk regardless (that's the actual deliverable in this repo's
# workflow); showing a live window is opt-in only, via this env var, for
# someone running the script on their own desktop.
_INTERACTIVE = os.environ.get("MANIFOLD_CONTROLLER_INTERACTIVE") == "1"
matplotlib.use("TkAgg" if _INTERACTIVE else "Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from plot_error_across_time import _load_all_data, _discover_tasks, _find_checkpoint, _load_state_dict_flexible
from Models.FLD_teacher_student import FLDTeacherStudent
from DataLoader.teacher_student_data_loader import (
    SENSOR_DIM as TS_SENSOR_DIM, BIOMECH_COLUMNS as TS_BIOMECH_COLUMNS, SENSOR_COLUMNS as TS_SENSOR_COLUMNS,
)
from fit_phase_torque_profile import _fourier_design_matrix, evaluate_fourier_torque   # DECISION #6 -- reused, not reimplemented

# ---------------------------------------------------------------------------
# Frame alignment -- see DECISION #4. This assert exists so an accidental
# edit that reintroduces a center-frame convention fails loudly, immediately.
# ---------------------------------------------------------------------------
FRAME_ALIGNMENT = "causal_last"
assert FRAME_ALIGNMENT == "causal_last", (
    "This script only supports causal-last frame alignment "
    "(frame_index = window_start + history_horizon - 1) -- see the module "
    "docstring's DECISION #4. Do not mix in window-center alignment anywhere "
    "in this file."
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLD_TS_RUN = ""   # blank = latest NON-GRF runs/FLDTeacherStudent_* run. This
                   # resolves correctly today because the GRF-variant runs
                   # save under a DIFFERENT checkpoint filename stem
                   # ("fld_teacher_student_grf_epoch_*.pt") and are therefore
                   # invisible to this lookup -- not because the known
                   # alphabetical-sort bug in _find_checkpoint is fixed (it
                   # isn't; see README.md "Known issues"). If that ever
                   # changes, this script will fail the GRF guard below
                   # rather than silently train on the wrong encoder.

HOLDOUT_SUBJECTS = ["S04"]                          # DECISION #7 -- subject-level, reserved for Stage 5
TARGET_JOINTS = ["hip_flexion_r", "knee_angle_r"]   # hip-knee exoskeleton scope, matches kphi_vs_quasistiffness.py

NUM_PHASE_BINS = 100      # N, parameterized per brief
ENCODE_STRIDE = 5         # offline table-building stride (deployment itself runs stride=1, every new sample)
ENCODE_BATCH_SIZE = 256
SMOOTH_KERNEL_SIZE = 5    # circular moving-average window, in bins (odd)
MIN_AMPLITUDE_FOR_COHERENCE = None   # set from r_star's own scale at runtime -- see Stage 1's flow-coherence check

ANNULUS_TRIAL_ID = "S01_trial_13"   # incline_walking_10_deg_1_2_m_s -- DECISION #8

STAGE2_K_SWEEP = list(range(1, 13))   # Fourier order sweep, 1..12 -- brief expects the knee around K=4-8
STAGE2_FINAL_K = 6                    # order actually used downstream (Stage 3+) -- confirmed against the sweep in run_stage2()
STAGE2_QC_BINS = 20                   # coarser phase-bin grouping for the per-bin fit-quality report (NUM_PHASE_BINS=100 is for the r_star/sigma tables, not this)

STAGE5_QC_BINS = 20
PHASE_ADVANCE_DT_MS = [50, 100, 150]   # the brief's own three horizons
FLOW_COHERENCE_K = 5                    # k nearest neighbours in latent space, per the brief
# DECISION not explicitly asked but needed for Stage 5: this repo has no
# trial labeled just "transition" -- only "sit_stand_squat_transitions"
# (one combined task covering both squats and sit/stand transitions, 4
# trials, one per subject). Used as-is for the brief's "transition and
# sit-to-stand trials" degradation test; flagged in the write-up.
DEGRADATION_TASK_LABEL = "sit_stand_squat_transitions"

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
STAGE1_OUT_DIR = os.path.join(RUNS_DIR, "ManifoldController")
STAGE1_TABLES_PATH = os.path.join(STAGE1_OUT_DIR, "stage1_tables.npz")
STAGE1_PLOT_PATH = os.path.join(STAGE1_OUT_DIR, "stage1_annulus.png")
STAGE2_RESULTS_PATH = os.path.join(STAGE1_OUT_DIR, "stage2_torque_profile.npz")
STAGE3_PLOT_PATH = os.path.join(STAGE1_OUT_DIR, "stage3_control_law.png")
STAGE2_KSWEEP_PLOT_PATH = os.path.join(STAGE1_OUT_DIR, "stage2_k_sweep.png")
STAGE2_QC_PLOT_PATH = os.path.join(STAGE1_OUT_DIR, "stage2_per_bin_quality.png")


# ===========================================================================
# UNITS -- DECISION #5. Every phase/frequency conversion in this file goes
# through these two functions; nothing else computes them ad hoc.
# ===========================================================================

def freq_raw_to_hz(freq_raw, sample_rate, history_horizon):
    """cycles-per-window -> Hz. Display/analysis conversion ONLY -- never
    fed back into the model's own sin(2*pi*(freq_raw*args+phase)) math,
    which takes raw (cycles-per-window) frequency directly. Same fix
    locomotion_index.py documents."""
    return freq_raw * sample_rate / history_horizon


def phase_advance(phi, freq_hz, dt):
    """phi: native cycle-fraction (NOT radians). freq_hz: Hz. dt: seconds.
    Returns phi + freq_hz*dt, in the SAME cycle-fraction units as phi --
    deliberately no 2*pi here (phi is already a fraction of a cycle, not an
    angle in radians). Caller wraps to [0,1) with np.mod(..., 1.0) if a
    bounded value is needed; the raw (unwrapped) value is what a caller
    tracking multiple revolutions wants instead."""
    return phi + freq_hz * dt


# NOTE: the Fourier basis (DECISION #5: cos(2*pi*k*phi)/sin(2*pi*k*phi)) is
# NOT redefined here. Stage 2 imports fit_phase_torque_profile.py's own
# _fourier_design_matrix / evaluate_fourier_torque directly (DECISION #6) --
# Stage 1's r_star/sigma tables don't use a Fourier basis at all, so there
# was nothing here worth keeping a duplicate copy for.


# ===========================================================================
# Frame alignment -- DECISION #4. The ONLY function in this file that
# computes a frame index from a window start.
# ===========================================================================

def _window_starts(n_rows, history_horizon, stride):
    return list(range(0, n_rows - history_horizon + 1, stride))


def _causal_last_frame_indices(window_starts, history_horizon):
    """frame_index = window_start + history_horizon - 1 -- the window's own
    LAST (causal, most-recently-seen) frame. This is the ONLY frame-index
    formula in this file (see FRAME_ALIGNMENT assert at module top)."""
    return np.asarray(window_starts) + (history_horizon - 1)


# ===========================================================================
# Model loading
# ===========================================================================

def _build_model_from_ckpt(ckpt, ckpt_path, device):
    config = ckpt["config"]
    sample_rate = config["sample_rate"]
    model = FLDTeacherStudent(
        sensor_dim=TS_SENSOR_DIM, biomech_dim=len(TS_BIOMECH_COLUMNS),
        history_horizon=config["history_horizon"], latent_dim=config["latent_dim"],
        device=device, dt=1.0 / sample_rate,
        teacher_encoder_shape=config["teacher_encoder_shape"],
        student_encoder_shape=config["student_encoder_shape"],
        decoder_shape=config["decoder_shape"],
    )
    _load_state_dict_flexible(model, ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    norm_stats = ckpt["norm_stats"]
    print(f"[Manifold]   epoch={ckpt['epoch']}  history_horizon={config['history_horizon']}  "
          f"latent_dim={config['latent_dim']}  sample_rate={sample_rate}")
    return model, config, norm_stats, ckpt_path


def _assert_not_grf_variant(ckpt_path, run_dir=""):
    if "grf" in os.path.basename(ckpt_path).lower() or "grf" in run_dir.lower():
        raise SystemExit(
            f"[Manifold] Refusing to proceed: resolved checkpoint {ckpt_path!r} looks like "
            f"the GRF-variant encoder (DECISION #1 requires the NON-GRF FLDTeacherStudent). "
            f"Pin FLD_TS_RUN explicitly to a non-GRF run directory."
        )


def load_model(device, pinned_run=""):
    """Stage 1 entry point -- resolves "latest non-GRF" via _find_checkpoint
    (DECISION #10: left as-is, works today because of the stem mismatch, not
    because the sort bug is fixed)."""
    run_dir, ckpt_path = _find_checkpoint("FLDTeacherStudent", "fld_teacher_student", pinned_run)
    _assert_not_grf_variant(ckpt_path, run_dir)
    print(f"[Manifold] Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return _build_model_from_ckpt(ckpt, ckpt_path, device)


def load_model_from_path(device, ckpt_path):
    """Stage 2+ entry point -- loads the EXACT checkpoint path recorded in
    Stage 1's saved tables (stage1_tables.npz's checkpoint_path), so later
    stages, run as separate `--stage N` invocations, never silently drift
    onto a different checkpoint than the one Stage 1's tables were built
    from (e.g. if runs/ changes between invocations)."""
    _assert_not_grf_variant(ckpt_path)
    print(f"[Manifold] Checkpoint (pinned from Stage 1 tables): {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return _build_model_from_ckpt(ckpt, ckpt_path, device)


# ===========================================================================
# Data -- incline trials, subject-level holdout (DECISION #7)
# ===========================================================================

def load_incline_trials():
    """Returns (all_df, tasks, fit_trial_ids, heldout_trial_ids), all_df
    from _load_all_data() (Data/all_data.csv, moments already N*m/kg),
    restricted to trials whose task label contains "incline". fit = subjects
    NOT in HOLDOUT_SUBJECTS; heldout = subjects IN HOLDOUT_SUBJECTS."""
    print("[Manifold] Loading Data/all_data.csv...")
    all_df = _load_all_data()
    tasks = _discover_tasks(all_df)
    incline_ids = sorted(tid for tid, info in tasks.items() if "incline" in info["label"].lower())
    fit_ids = [tid for tid in incline_ids if not any(tid.startswith(s) for s in HOLDOUT_SUBJECTS)]
    heldout_ids = [tid for tid in incline_ids if tid not in fit_ids]
    print(f"[Manifold] {len(incline_ids)} incline trials found: "
          + ", ".join(f"{tid}({tasks[tid]['label']})" for tid in incline_ids))
    print(f"[Manifold] Fit set ({len(fit_ids)}, excludes {HOLDOUT_SUBJECTS}): {fit_ids}")
    print(f"[Manifold] Held-out set ({len(heldout_ids)}): {heldout_ids}")
    return all_df, tasks, fit_ids, heldout_ids


def _trial_biomech_sensor(all_df, trial_id):
    trial_df = all_df[all_df["trial_id"] == trial_id].reset_index(drop=True)
    return trial_df


# ===========================================================================
# Encoding -- generic over forward_teacher / forward_student, always
# causal-last (DECISION #4), never crosses a trial boundary.
# ===========================================================================

def encode_trial(model, encode_fn, data, history_horizon, stride, batch_size, device):
    """data: (N, dim) normalized float32 tensor for ONE trial. encode_fn:
    model.forward_teacher or model.forward_student. Returns None if the
    trial is too short, else dict with phase/frequency/amplitude/offset
    (T, K) and frame_indices (T,) -- causal-last."""
    N = data.shape[0]
    starts = _window_starts(N, history_horizon, stride)
    if not starts:
        return None
    phases, freqs, amps, offsets = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i:i + batch_size]
            wins = torch.stack([data[s:s + history_horizon].T for s in batch_starts]).to(device)
            phase, frequency, amplitude, offset = encode_fn(wins)
            phases.append(phase.cpu()); freqs.append(frequency.cpu())
            amps.append(amplitude.cpu()); offsets.append(offset.cpu())
    return {
        "phase": torch.cat(phases, dim=0).numpy(),
        "frequency": torch.cat(freqs, dim=0).numpy(),
        "amplitude": torch.cat(amps, dim=0).numpy(),
        "offset": torch.cat(offsets, dim=0).numpy(),
        "frame_indices": _causal_last_frame_indices(starts, history_horizon),
    }


def _normalize_biomech(trial_df, norm_stats):
    raw = trial_df[TS_BIOMECH_COLUMNS].values.astype("float32")
    return torch.tensor((raw - norm_stats["mean_biomech"]) / norm_stats["std_biomech"], dtype=torch.float32)


def _normalize_sensor(trial_df, norm_stats):
    raw = trial_df[TS_SENSOR_COLUMNS].values.astype("float32")
    return torch.tensor((raw - norm_stats["mean_sensor"]) / norm_stats["std_sensor"], dtype=torch.float32)


# ===========================================================================
# DECISION #3 -- dominant-channel stability check across ALL 4 subjects /
# ALL 8 incline trials, both branches. Halts (SystemExit) if unstable.
# ===========================================================================

NEAR_TIE_MAX_RATIO = 1.02   # top-2 mean-amplitude ratio below which a per-trial argmax
                            # disagreement with the POOLED dominant channel is treated as a
                            # near-tie (noise), not real instability. See RESOLUTION note below.

# ---------------------------------------------------------------------------
# RESOLUTION (2026-08-27, after Stage 1 first halted here): the strict
# per-trial-unanimity check below DID flip once -- S01_trial_13 (the annulus
# trial itself, 10-degree incline) has teacher channel 0 edge out channel 5
# by 0.6% mean amplitude, a genuine coin-flip. Every other trial (7/8) has
# channel 5 winning by 10-27+%, and the STUDENT branch is unanimous at
# channel 5 across all 8 trials with no exceptions. Investigated with the
# user before proceeding (not decided unilaterally): resolved to use the
# POOLED dominant channel (argmax of mean amplitude over ALL windows from
# ALL trials, not per-trial argmax unanimity) as the stability criterion --
# same "pool first, then argmax once" convention fit_phase_torque_profile.py
# already uses. A per-trial disagreement is only a hard stop if its own
# top-2 amplitude ratio exceeds NEAR_TIE_MAX_RATIO (i.e. NOT explainable as
# noise near a genuine tie).
# ---------------------------------------------------------------------------

def check_dominant_channel_stability(model, config, norm_stats, all_df, incline_ids, device):
    H = config["history_horizon"]
    print(f"\n[Manifold] === Dominant-channel stability check ({len(incline_ids)} incline trials) ===")
    rows = []
    teacher_amps, student_amps = [], []
    for tid in incline_ids:
        trial_df = _trial_biomech_sensor(all_df, tid)
        if len(trial_df) < H:
            print(f"[Manifold]   {tid}: skipped, {len(trial_df)} rows < history_horizon={H}")
            continue
        teacher_enc = encode_trial(model, model.forward_teacher, _normalize_biomech(trial_df, norm_stats),
                                    H, ENCODE_STRIDE, ENCODE_BATCH_SIZE, device)
        student_enc = encode_trial(model, model.forward_student, _normalize_sensor(trial_df, norm_stats),
                                    H, ENCODE_STRIDE, ENCODE_BATCH_SIZE, device)
        teacher_amps.append(teacher_enc["amplitude"])
        student_amps.append(student_enc["amplitude"])
        t_mean, s_mean = teacher_enc["amplitude"].mean(axis=0), student_enc["amplitude"].mean(axis=0)
        teacher_dom, student_dom = int(t_mean.argmax()), int(s_mean.argmax())
        t_order = t_mean.argsort()[::-1]
        t_ratio = float(t_mean[t_order[0]] / max(t_mean[t_order[1]], 1e-12))
        rows.append((tid, teacher_dom, student_dom, t_ratio))
        flag = "" if teacher_dom == student_dom else "  <-- teacher/student DISAGREE"
        print(f"[Manifold]   {tid:15s} teacher_dom_ch={teacher_dom} (top2 ratio={t_ratio:.3f})  "
              f"student_dom_ch={student_dom}{flag}")

    pooled_teacher_amp = np.concatenate(teacher_amps, axis=0).mean(axis=0)
    pooled_student_amp = np.concatenate(student_amps, axis=0).mean(axis=0)
    pooled_teacher_dom, pooled_student_dom = int(pooled_teacher_amp.argmax()), int(pooled_student_amp.argmax())
    print(f"[Manifold] Pooled (all trials, all windows) teacher dom_ch={pooled_teacher_dom}, "
          f"student dom_ch={pooled_student_dom}")

    if pooled_teacher_dom != pooled_student_dom:
        raise SystemExit(
            f"[Manifold] STOP -- pooled teacher dom_ch ({pooled_teacher_dom}) != pooled student dom_ch "
            f"({pooled_student_dom}). This is a disagreement between branches on the POOLED statistic, "
            f"not a per-trial near-tie -- report this back before proceeding."
        )

    dom_ch = pooled_teacher_dom
    real_instability = []
    for tid, teacher_dom, student_dom, t_ratio in rows:
        if teacher_dom != dom_ch:
            if t_ratio <= NEAR_TIE_MAX_RATIO:
                print(f"[Manifold]   NOTE: {tid}'s local teacher argmax is ch{teacher_dom}, not the pooled "
                      f"ch{dom_ch} -- top-2 ratio={t_ratio:.3f} <= {NEAR_TIE_MAX_RATIO} (near-tie, "
                      f"<{(NEAR_TIE_MAX_RATIO - 1) * 100:.0f}% gap) -- NOT treated as real instability.")
            else:
                real_instability.append((tid, teacher_dom, t_ratio))

    if real_instability:
        raise SystemExit(
            f"[Manifold] STOP -- teacher dominant channel is unstable BEYOND near-tie tolerance on: "
            f"{real_instability}. Per DECISION #3, halting before Stage 1 tables are built."
        )

    print(f"[Manifold] STABLE (pooled). Using dominant channel = {dom_ch} for everything downstream "
          f"(Stage 1 tables, Stage 2 fit, Stage 3 control law). See any near-tie NOTE above for caveats.")
    return dom_ch, rows


# ===========================================================================
# Stage 1 -- r_star(phi) / sigma(phi) tables, circular
# ===========================================================================

def _circular_bin_index(phi_wrapped, n_bins):
    """phi_wrapped in [0,1). Returns integer bin in [0, n_bins)."""
    idx = np.floor(phi_wrapped * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def _circular_smooth(values, kernel_size):
    """Circular moving average -- wraps around, so bin N-1 and bin 0 are
    genuinely adjacent (no seam). kernel_size must be odd."""
    assert kernel_size % 2 == 1, "kernel_size must be odd for a symmetric circular window"
    half = kernel_size // 2
    padded = np.concatenate([values[-half:], values, values[:half]]) if half > 0 else values
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def build_phase_binned_table(phi_all, radius_all, n_bins, smooth_kernel_size):
    """Bins radius_all by phi_all (circular, N=n_bins) and returns a dict:
      'bin_centers' (N,), 'r_star' (N,) median radius per bin (circularly
      smoothed, empty bins filled), 'sigma' (N,) robust spread (MAD, scaled
      to be a robust-std-equivalent: 1.4826*median(|x-median|)), 'counts'
      (N,) raw sample count per bin, 'empty_mask' (N,) bool -- bins with
      ZERO samples BEFORE fill, so thin/absent support stays visible even
      though r_star/sigma are never left as NaN for them (brief: "no silent
      NaN").

    Circularity: bin N-1 is adjacent to bin 0 throughout -- fill uses
    np.interp(..., period=1.0) (circular interpolation, same convention
    diagnostics/kphi_vs_quasistiffness.py already uses for its K(phi)
    lookup) and smoothing wraps via _circular_smooth(). A query at phi=0.999
    and phi=0.001 therefore reads two adjacent, similar table values -- no
    wrap discontinuity.
    """
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    bin_idx = _circular_bin_index(np.mod(phi_all, 1.0), n_bins)

    r_raw = np.full(n_bins, np.nan)
    sigma_raw = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        vals = radius_all[bin_idx == b]
        counts[b] = len(vals)
        if len(vals) == 0:
            continue
        med = np.median(vals)
        r_raw[b] = med
        sigma_raw[b] = 1.4826 * np.median(np.abs(vals - med))   # MAD -> robust-std-equivalent

    empty_mask = counts == 0
    n_empty = int(empty_mask.sum())
    if n_empty > 0:
        print(f"[Manifold]   {n_empty}/{n_bins} phase bins have ZERO samples -- filling via circular "
              f"interpolation (period=1.0), not leaving as NaN. Empty bins: "
              f"{np.flatnonzero(empty_mask).tolist()}")
        valid = ~empty_mask
        if valid.sum() < 2:
            raise SystemExit(f"[Manifold] STOP -- only {valid.sum()} non-empty phase bins, cannot "
                              f"interpolate a usable table. Check ENCODE_STRIDE / NUM_PHASE_BINS / input data.")
        r_raw = np.interp(bin_centers, bin_centers[valid], r_raw[valid], period=1.0)
        sigma_raw = np.interp(bin_centers, bin_centers[valid], sigma_raw[valid], period=1.0)
    else:
        print(f"[Manifold]   All {n_bins} phase bins have support (min count={counts.min()}, "
              f"max count={counts.max()}, mean count={counts.mean():.1f}) -- no empty-bin fill needed.")

    r_star = _circular_smooth(r_raw, smooth_kernel_size)
    sigma = _circular_smooth(sigma_raw, smooth_kernel_size)

    return {
        "bin_centers": bin_centers,
        "r_star": r_star,
        "sigma": sigma,
        "counts": counts,
        "empty_mask": empty_mask,
    }


def circular_interp(phi_query, bin_centers, values):
    """Query-time lookup for an r_star/sigma table -- circular (period=1.0),
    so phi just below 1.0 and phi just above 0.0 interpolate smoothly across
    the wrap. Used by Stage 3's k(phi) control law (not yet implemented)."""
    return np.interp(np.mod(phi_query, 1.0), bin_centers, values, period=1.0)


# ===========================================================================
# Stage 1 -- quantitative flow-coherence check on the annulus trial
# (DECISION #8: global tangential-vs-radial velocity alignment, NOT the
# fuller k-NN neighbor metric -- that's Stage 5, run on OTHER trial types).
# ===========================================================================

def flow_coherence_on_annulus(phase, amplitude, dom_ch, r_star_scale):
    """phase/amplitude: (T, K) from ONE trial's teacher encoding.
    r_star_scale: a representative radius (e.g. median of r_star table)
    used to exclude near-origin points, per the "phase ill-conditioned near
    A=0" caveat -- excluded points are reported, not silently dropped.

    Embedding: x = A*cos(2*pi*phi), y = A*sin(2*pi*phi) (dominant channel
    only). Velocity: finite difference between consecutive ENCODED samples
    (ENCODE_STRIDE frames apart in raw time -- direction only matters here,
    not the timestep's absolute scale). For each interior sample, computes
    cos-similarity of velocity with the LOCAL radial direction and with the
    LOCAL tangential direction (radial rotated 90 degrees) at that point.
    "Coherent tangential flow" predicts: tangential alignment ~ high
    (near +-1), radial alignment ~ low (near 0).

    Returns dict with mean |cos| tangential/radial alignment and how many
    points were excluded for being too close to the origin.
    """
    A = amplitude[:, dom_ch].astype(np.float64)
    phi = phase[:, dom_ch].astype(np.float64)
    x = A * np.cos(2.0 * np.pi * phi)
    y = A * np.sin(2.0 * np.pi * phi)
    points = np.stack([x, y], axis=1)

    amp_threshold = 0.1 * r_star_scale
    keep = A[:-1] > amp_threshold   # exclude points whose STARTING amplitude is near-origin
    n_excluded = int((~keep).sum())

    velocity = points[1:] - points[:-1]
    origin_pts = points[:-1]
    radius = np.linalg.norm(origin_pts, axis=1, keepdims=True)
    radius_safe = np.where(radius > 1e-12, radius, 1.0)
    r_hat = origin_pts / radius_safe
    t_hat = np.stack([-r_hat[:, 1], r_hat[:, 0]], axis=1)   # rotate r_hat by +90 degrees

    v_norm = np.linalg.norm(velocity, axis=1, keepdims=True)
    v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1.0)
    v_hat = velocity / v_norm_safe

    cos_tangential = np.sum(v_hat * t_hat, axis=1)
    cos_radial = np.sum(v_hat * r_hat, axis=1)

    cos_tangential, cos_radial = cos_tangential[keep], cos_radial[keep]
    return {
        "mean_abs_cos_tangential": float(np.mean(np.abs(cos_tangential))),
        "mean_abs_cos_radial": float(np.mean(np.abs(cos_radial))),
        "n_points": int(keep.sum()),
        "n_excluded_near_origin": n_excluded,
        "amp_threshold": float(amp_threshold),
    }


# ===========================================================================
# Stage 1 -- entry point
# ===========================================================================

def run_stage1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Manifold] Using device: {device}")

    model, config, norm_stats, ckpt_path = load_model(device, FLD_TS_RUN)
    H = config["history_horizon"]
    sample_rate = config["sample_rate"]

    all_df, tasks, fit_ids, heldout_ids = load_incline_trials()
    incline_ids = fit_ids + heldout_ids

    dom_ch, stability_rows = check_dominant_channel_stability(model, config, norm_stats, all_df, incline_ids, device)

    # ------------------------------------------------------------------
    # Encode FIT trials only (teacher branch -- DECISION #1) for the tables.
    # ------------------------------------------------------------------
    print(f"\n[Manifold] === Encoding {len(fit_ids)} fit trials (teacher branch) for Stage 1 tables ===")
    per_trial_teacher = {}
    for tid in fit_ids:
        trial_df = _trial_biomech_sensor(all_df, tid)
        enc = encode_trial(model, model.forward_teacher, _normalize_biomech(trial_df, norm_stats),
                            H, ENCODE_STRIDE, ENCODE_BATCH_SIZE, device)
        if enc is None:
            print(f"[Manifold]   {tid}: skipped, too short")
            continue
        print(f"[Manifold]   {tid} ({tasks[tid]['label']}): {len(trial_df):,} rows -> "
              f"{len(enc['frame_indices']):,} windows")
        per_trial_teacher[tid] = enc

    phi_all = np.concatenate([np.mod(enc["phase"][:, dom_ch], 1.0) for enc in per_trial_teacher.values()])
    radius_all = np.concatenate([np.abs(enc["amplitude"][:, dom_ch]) for enc in per_trial_teacher.values()])
    print(f"\n[Manifold] Pooled {len(phi_all):,} windows from {len(per_trial_teacher)} fit trials "
          f"for phase-binned tables.")

    table = build_phase_binned_table(phi_all, radius_all, NUM_PHASE_BINS, SMOOTH_KERNEL_SIZE)
    print(f"[Manifold] r_star range: [{table['r_star'].min():.4g}, {table['r_star'].max():.4g}]  "
          f"sigma range: [{table['sigma'].min():.4g}, {table['sigma'].max():.4g}]")

    # ------------------------------------------------------------------
    # Wrap-continuity sanity print (also covered by --stage test)
    # ------------------------------------------------------------------
    seam_gap_r = abs(table["r_star"][0] - table["r_star"][-1])
    seam_gap_sigma = abs(table["sigma"][0] - table["sigma"][-1])
    print(f"[Manifold] Wrap-seam check: |r_star[0]-r_star[-1]|={seam_gap_r:.4g}  "
          f"|sigma[0]-sigma[-1]|={seam_gap_sigma:.4g} (should be small relative to the tables' own range)")

    # ------------------------------------------------------------------
    # Flow coherence on the annulus trial (DECISION #8)
    # ------------------------------------------------------------------
    if ANNULUS_TRIAL_ID not in per_trial_teacher:
        print(f"[Manifold] WARNING -- ANNULUS_TRIAL_ID={ANNULUS_TRIAL_ID!r} not in the fit set "
              f"({list(per_trial_teacher.keys())}), skipping flow-coherence check.")
        coherence = None
    else:
        enc = per_trial_teacher[ANNULUS_TRIAL_ID]
        coherence = flow_coherence_on_annulus(enc["phase"], enc["amplitude"], dom_ch,
                                               r_star_scale=float(np.median(table["r_star"])))
        print(f"\n[Manifold] === Flow coherence on {ANNULUS_TRIAL_ID} ({tasks[ANNULUS_TRIAL_ID]['label']}) ===")
        print(f"[Manifold]   mean |cos(velocity, tangential)| = {coherence['mean_abs_cos_tangential']:.3f}  "
              f"(near 1.0 = coherent tangential/angular flow)")
        print(f"[Manifold]   mean |cos(velocity, radial)|     = {coherence['mean_abs_cos_radial']:.3f}  "
              f"(near 0.0 = little radial/noisy motion)")
        print(f"[Manifold]   {coherence['n_points']:,} points used, {coherence['n_excluded_near_origin']:,} "
              f"excluded (A < {coherence['amp_threshold']:.4g}, near-origin -- phase ill-conditioned there)")

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------
    os.makedirs(STAGE1_OUT_DIR, exist_ok=True)
    np.savez(
        STAGE1_TABLES_PATH,
        dom_ch=dom_ch, num_phase_bins=NUM_PHASE_BINS,
        bin_centers=table["bin_centers"], r_star=table["r_star"], sigma=table["sigma"],
        counts=table["counts"], empty_mask=table["empty_mask"],
        fit_trial_ids=np.array(fit_ids), heldout_trial_ids=np.array(heldout_ids),
        checkpoint_path=ckpt_path, sample_rate=sample_rate, history_horizon=H,
        encode_stride=ENCODE_STRIDE,
    )
    print(f"\n[Manifold] Saved Stage 1 tables: {STAGE1_TABLES_PATH}")

    # ------------------------------------------------------------------
    # Plot: annulus (S01_trial_13) overlaid with r_star(phi) ring + sigma band
    # ------------------------------------------------------------------
    plot_annulus(per_trial_teacher, dom_ch, table, tasks)

    return {
        "dom_ch": dom_ch, "table": table, "coherence": coherence,
        "fit_ids": fit_ids, "heldout_ids": heldout_ids, "stability_rows": stability_rows,
    }


def plot_annulus(per_trial_teacher, dom_ch, table, tasks):
    fig, ax = plt.subplots(figsize=(8, 8))

    if ANNULUS_TRIAL_ID in per_trial_teacher:
        enc = per_trial_teacher[ANNULUS_TRIAL_ID]
        A = enc["amplitude"][:, dom_ch]
        phi = enc["phase"][:, dom_ch]
        x, y = A * np.cos(2 * np.pi * phi), A * np.sin(2 * np.pi * phi)
        ax.scatter(x, y, s=3, alpha=0.15, color="tab:gray",
                   label=f"{ANNULUS_TRIAL_ID} ({tasks[ANNULUS_TRIAL_ID]['label']}), ch{dom_ch}")

    phi_ring = table["bin_centers"]
    r = table["r_star"]
    sig = table["sigma"]
    ring_x, ring_y = r * np.cos(2 * np.pi * phi_ring), r * np.sin(2 * np.pi * phi_ring)
    outer_x, outer_y = (r + sig) * np.cos(2 * np.pi * phi_ring), (r + sig) * np.sin(2 * np.pi * phi_ring)
    inner_x, inner_y = np.clip(r - sig, 0, None) * np.cos(2 * np.pi * phi_ring), \
                        np.clip(r - sig, 0, None) * np.sin(2 * np.pi * phi_ring)

    # close the loop for plotting (bin N-1 -> bin 0)
    def _closed(a, b):
        return np.append(a, a[0]), np.append(b, b[0])

    ax.plot(*_closed(ring_x, ring_y), color="tab:red", lw=2.0, label="r_star(phi) (pooled fit trials)")
    ax.plot(*_closed(outer_x, outer_y), color="tab:red", lw=0.8, ls="--", alpha=0.7, label="r_star +/- sigma")
    ax.plot(*_closed(inner_x, inner_y), color="tab:red", lw=0.8, ls="--", alpha=0.7)
    if table["empty_mask"].any():
        empty_phi = phi_ring[table["empty_mask"]]
        ex = r[table["empty_mask"]] * np.cos(2 * np.pi * empty_phi)
        ey = r[table["empty_mask"]] * np.sin(2 * np.pi * empty_phi)
        ax.scatter(ex, ey, marker="x", color="black", s=60, zorder=5, label="empty bin (filled by interp)")

    ax.set_aspect("equal")
    ax.set_xlabel(f"A*cos(2*pi*phi), channel {dom_ch}")
    ax.set_ylabel(f"A*sin(2*pi*phi), channel {dom_ch}")
    ax.set_title("Stage 1 -- Latent Annulus vs. r_star(phi) +/- sigma(phi)\n"
                 "(annulus = one trial; r_star/sigma pooled over ALL fit trials)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    fig.savefig(STAGE1_PLOT_PATH, dpi=150)
    print(f"[Manifold] Saved plot: {STAGE1_PLOT_PATH}")
    if _INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)


# ===========================================================================
# Shared helpers -- Stage 2 onward. Loading Stage 1's saved tables,
# multi-trial multi-branch encoding, target-moment extraction.
# ===========================================================================

def load_stage1_tables():
    if not os.path.exists(STAGE1_TABLES_PATH):
        raise SystemExit(f"[Manifold] {STAGE1_TABLES_PATH} not found -- run `--stage 1` first.")
    data = np.load(STAGE1_TABLES_PATH, allow_pickle=True)
    return {
        "dom_ch": int(data["dom_ch"]), "num_phase_bins": int(data["num_phase_bins"]),
        "bin_centers": data["bin_centers"], "r_star": data["r_star"], "sigma": data["sigma"],
        "counts": data["counts"], "empty_mask": data["empty_mask"],
        "fit_trial_ids": list(data["fit_trial_ids"]), "heldout_trial_ids": list(data["heldout_trial_ids"]),
        "checkpoint_path": str(data["checkpoint_path"]), "sample_rate": float(data["sample_rate"]),
        "history_horizon": int(data["history_horizon"]), "encode_stride": int(data["encode_stride"]),
    }


def encode_trials_both_branches(model, all_df, trial_ids, norm_stats, H, stride, batch_size, device):
    """Encodes EVERY trial in trial_ids with BOTH branches (teacher +
    student), never crossing a trial boundary. Returns
    {trial_id: {'teacher': enc_dict, 'student': enc_dict, 'trial_df': df}}.
    A trial shorter than H is skipped (reported, not silently dropped)."""
    out = {}
    for tid in trial_ids:
        trial_df = _trial_biomech_sensor(all_df, tid)
        if len(trial_df) < H:
            print(f"[Manifold]   {tid}: skipped, {len(trial_df)} rows < history_horizon={H}")
            continue
        teacher_enc = encode_trial(model, model.forward_teacher, _normalize_biomech(trial_df, norm_stats),
                                    H, stride, batch_size, device)
        student_enc = encode_trial(model, model.forward_student, _normalize_sensor(trial_df, norm_stats),
                                    H, stride, batch_size, device)
        out[tid] = {"teacher": teacher_enc, "student": student_enc, "trial_df": trial_df}
    return out


def target_moment(trial_df, frame_indices, joint):
    """Ground-truth teacher moment (N*m/kg) at each causal-last frame_index --
    the SAME target regardless of which branch's phase is the independent
    variable (DECISION #2: only the teacher-PHASE fit is circular, since the
    teacher's own input window contains this column; the student never sees
    it -- a student-phase fit against this target is a genuine regression)."""
    return trial_df[f"{joint}_moment"].values.astype("float64")[frame_indices]


def _load_right_grf_for_trial(trial_id, n_rows_expected):
    """Reads R_plate_Fz directly from the trial's raw per-subject CSV (NOT
    one of _load_all_data()'s columns) -- used ONLY for Stage 2's stance/
    swing QC overlay. Returns None (with a warning, not a crash) if the row
    count doesn't line up with what the caller expected."""
    subject, _, trial_num = trial_id.partition("_trial_")
    path = os.path.join(os.path.dirname(__file__), "Data", subject, f"trial_{trial_num}_all_data.csv")
    if not os.path.exists(path):
        print(f"[Manifold]   WARNING: {path} not found, skipping GRF QC overlay for {trial_id}.")
        return None
    grf = pd.read_csv(path, usecols=["R_plate_Fz"])["R_plate_Fz"].values.astype("float32")
    if len(grf) != n_rows_expected:
        print(f"[Manifold]   WARNING: {trial_id} GRF row count {len(grf)} != trial_df rows "
              f"{n_rows_expected}, skipping GRF QC overlay for this trial.")
        return None
    return grf


# ===========================================================================
# Stage 2 -- torque profile fit. Fourier basis/design matrix is IMPORTED
# from fit_phase_torque_profile.py (DECISION #6), not redefined.
# ===========================================================================

def _fit_and_score(design, target):
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    pred = design @ coeffs
    resid = target - pred
    ss_res, ss_tot = float(np.sum(resid ** 2)), float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return coeffs, r2, rmse


def sweep_fourier_order(phi_all, target_all, k_values):
    """{k: {'r2':.., 'rmse':..}} for each k, least-squares fit via
    fit_phase_torque_profile.py's _fourier_design_matrix (DECISION #6)."""
    return {k: dict(zip(("r2", "rmse"), _fit_and_score(_fourier_design_matrix(phi_all, k), target_all)[1:]))
            for k in k_values}


def detect_knee(sweep_results, k_values, rel_improvement_threshold=0.02):
    """First K beyond which each further +1 order improves RMSE by less
    than rel_improvement_threshold (relative). Falls back to the largest
    swept K if RMSE is still improving substantially throughout."""
    rmses = [sweep_results[k]["rmse"] for k in k_values]
    for i in range(1, len(k_values)):
        prev, curr = rmses[i - 1], rmses[i]
        rel_improve = (prev - curr) / prev if prev > 0 else 0.0
        if rel_improve < rel_improvement_threshold:
            return k_values[i - 1]
    return k_values[-1]


def per_bin_fit_quality(phi_all, target_all, coeffs, n_bins):
    """RMSE/R^2 of a FIXED (already-fit) coeffs vector, broken out per phase
    bin -- so "good in swing, poor in stance" (the brief's own example) is
    visible instead of hiding behind one aggregate number. Bins with fewer
    than 5 samples are left NaN (too few to trust an R^2/RMSE from) but
    their count is still reported, same "no silent gap" spirit as Stage 1's
    empty-bin handling (just not filled here -- this is a diagnostic report,
    not a lookup table that must always return something)."""
    order = (len(coeffs) - 1) // 2
    pred_all = _fourier_design_matrix(phi_all, order) @ coeffs
    resid_all = target_all - pred_all
    bin_idx = _circular_bin_index(np.mod(phi_all, 1.0), n_bins)
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    rmse, r2, counts = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] < 5:
            continue
        r, t = resid_all[mask], target_all[mask]
        rmse[b] = float(np.sqrt(np.mean(r ** 2)))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2[b] = 1.0 - float(np.sum(r ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"bin_centers": bin_centers, "rmse": rmse, "r2": r2, "counts": counts}


def run_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Manifold] Using device: {device}")

    s1 = load_stage1_tables()
    dom_ch, H, stride = s1["dom_ch"], s1["history_horizon"], s1["encode_stride"]
    model, config, norm_stats, ckpt_path = load_model_from_path(device, s1["checkpoint_path"])

    all_df, tasks, fit_ids, heldout_ids = load_incline_trials()
    if fit_ids != s1["fit_trial_ids"]:
        raise SystemExit(f"[Manifold] STOP -- fit trial set changed since Stage 1 "
                          f"({fit_ids} vs {s1['fit_trial_ids']}) -- rerun `--stage 1` first.")

    print(f"\n[Manifold] === Stage 2: encoding {len(fit_ids)} fit trials (both branches) ===")
    encoded = encode_trials_both_branches(model, all_df, fit_ids, norm_stats, H, stride, ENCODE_BATCH_SIZE, device)

    results_by_joint = {}
    for joint in TARGET_JOINTS:
        phi_teacher_all, phi_student_all, target_all = [], [], []
        for tid, enc in encoded.items():
            frame_idx = enc["teacher"]["frame_indices"]
            assert np.array_equal(frame_idx, enc["student"]["frame_indices"]), \
                f"{tid}: teacher/student frame_indices disagree -- alignment bug (DECISION #4 violated)"
            phi_teacher_all.append(np.mod(enc["teacher"]["phase"][:, dom_ch], 1.0))
            phi_student_all.append(np.mod(enc["student"]["phase"][:, dom_ch], 1.0))
            target_all.append(target_moment(enc["trial_df"], frame_idx, joint))
        phi_teacher_all, phi_student_all = np.concatenate(phi_teacher_all), np.concatenate(phi_student_all)
        target_all = np.concatenate(target_all)

        print(f"\n[Manifold] --- {joint}: K sweep ({len(target_all):,} pooled windows) ---")
        teacher_sweep = sweep_fourier_order(phi_teacher_all, target_all, STAGE2_K_SWEEP)
        student_sweep = sweep_fourier_order(phi_student_all, target_all, STAGE2_K_SWEEP)
        for k in STAGE2_K_SWEEP:
            print(f"[Manifold]   K={k:2d}  teacher(privileged): RMSE={teacher_sweep[k]['rmse']:.4g} "
                  f"R^2={teacher_sweep[k]['r2']:.3f}   student(headline): RMSE={student_sweep[k]['rmse']:.4g} "
                  f"R^2={student_sweep[k]['r2']:.3f}")

        teacher_knee, student_knee = detect_knee(teacher_sweep, STAGE2_K_SWEEP), detect_knee(student_sweep, STAGE2_K_SWEEP)
        print(f"[Manifold]   Detected knee: teacher K={teacher_knee}, student K={student_knee} "
              f"(brief expects ~4-8); STAGE2_FINAL_K={STAGE2_FINAL_K} used downstream for both.")

        coeffs_teacher, r2_teacher, rmse_teacher = _fit_and_score(_fourier_design_matrix(phi_teacher_all, STAGE2_FINAL_K), target_all)
        coeffs_student, r2_student, rmse_student = _fit_and_score(_fourier_design_matrix(phi_student_all, STAGE2_FINAL_K), target_all)
        print(f"[Manifold]   FINAL (K={STAGE2_FINAL_K}): teacher(privileged upper bound, PARTLY CIRCULAR -- "
              f"DECISION #2) R^2={r2_teacher:.3f} RMSE={rmse_teacher:.4g}")
        print(f"[Manifold]   FINAL (K={STAGE2_FINAL_K}): student(HEADLINE, deployable, NOT circular) "
              f"R^2={r2_student:.3f} RMSE={rmse_student:.4g}")

        qc_teacher = per_bin_fit_quality(phi_teacher_all, target_all, coeffs_teacher, STAGE2_QC_BINS)
        qc_student = per_bin_fit_quality(phi_student_all, target_all, coeffs_student, STAGE2_QC_BINS)

        # Stance/swing QC overlay -- best-effort join against raw per-trial R_plate_Fz
        # (not part of _load_all_data()'s columns), same GRF_STANCE_THRESHOLD_N=40N
        # convention plot_pca_manifold_train.py already established.
        grf_sum, grf_count = np.zeros(STAGE2_QC_BINS), np.zeros(STAGE2_QC_BINS)
        for tid, enc in encoded.items():
            grf = _load_right_grf_for_trial(tid, len(enc["trial_df"]))
            if grf is None:
                continue
            frame_idx = enc["student"]["frame_indices"]
            bin_idx = _circular_bin_index(np.mod(enc["student"]["phase"][:, dom_ch], 1.0), STAGE2_QC_BINS)
            fz_abs = np.abs(grf[frame_idx])
            for b in range(STAGE2_QC_BINS):
                mask = bin_idx == b
                grf_sum[b] += fz_abs[mask].sum()
                grf_count[b] += mask.sum()
        with np.errstate(invalid="ignore"):
            mean_abs_fz = grf_sum / np.where(grf_count > 0, grf_count, np.nan)
        stance_by_bin = mean_abs_fz > 40.0

        print(f"[Manifold]   Per-bin ({STAGE2_QC_BINS} bins) student-phase fit quality:")
        for b in range(STAGE2_QC_BINS):
            label = "stance" if stance_by_bin[b] else ("swing" if not np.isnan(mean_abs_fz[b]) else "?")
            print(f"[Manifold]     phi={qc_student['bin_centers'][b]:.3f}  RMSE={qc_student['rmse'][b]:.4g}  "
                  f"R^2={qc_student['r2'][b]:.3f}  n={qc_student['counts'][b]}  [{label}]")

        stance_mask, swing_mask = stance_by_bin, (~stance_by_bin & ~np.isnan(mean_abs_fz))
        stance_rmse = float(np.nanmean(qc_student["rmse"][stance_mask])) if stance_mask.any() else float("nan")
        swing_rmse = float(np.nanmean(qc_student["rmse"][swing_mask])) if swing_mask.any() else float("nan")
        print(f"[Manifold]   Stance-bin mean RMSE={stance_rmse:.4g}  Swing-bin mean RMSE={swing_rmse:.4g} "
              f"(student-phase fit, K={STAGE2_FINAL_K})")

        results_by_joint[joint] = {
            "coeffs_teacher": coeffs_teacher, "r2_teacher": r2_teacher, "rmse_teacher": rmse_teacher,
            "coeffs_student": coeffs_student, "r2_student": r2_student, "rmse_student": rmse_student,
            "teacher_sweep": teacher_sweep, "student_sweep": student_sweep,
            "teacher_knee": teacher_knee, "student_knee": student_knee,
            "qc_teacher": qc_teacher, "qc_student": qc_student,
            "stance_rmse": stance_rmse, "swing_rmse": swing_rmse,
        }

    save_kwargs = {"dom_ch": dom_ch, "fourier_order": STAGE2_FINAL_K, "joints": np.array(TARGET_JOINTS),
                    "checkpoint_path": ckpt_path, "k_sweep": np.array(STAGE2_K_SWEEP)}
    for joint in TARGET_JOINTS:
        r = results_by_joint[joint]
        save_kwargs.update({
            f"coeffs_student_{joint}": r["coeffs_student"], f"coeffs_teacher_{joint}": r["coeffs_teacher"],
            f"r2_student_{joint}": r["r2_student"], f"rmse_student_{joint}": r["rmse_student"],
            f"r2_teacher_{joint}": r["r2_teacher"], f"rmse_teacher_{joint}": r["rmse_teacher"],
        })
    np.savez(STAGE2_RESULTS_PATH, **save_kwargs)
    print(f"\n[Manifold] Saved Stage 2 results: {STAGE2_RESULTS_PATH}")

    plot_stage2(results_by_joint)
    return results_by_joint


def plot_stage2(results_by_joint):
    fig, axes = plt.subplots(1, len(TARGET_JOINTS), figsize=(7 * len(TARGET_JOINTS), 5))
    axes = np.atleast_1d(axes)
    for ax, joint in zip(axes, TARGET_JOINTS):
        r = results_by_joint[joint]
        ks = STAGE2_K_SWEEP
        ax.plot(ks, [r["teacher_sweep"][k]["rmse"] for k in ks], "o-", color="tab:green",
                label="Teacher (privileged, partly circular)")
        ax.plot(ks, [r["student_sweep"][k]["rmse"] for k in ks], "o-", color="tab:red", label="Student (HEADLINE)")
        ax.axvline(STAGE2_FINAL_K, color="black", ls="--", lw=1, alpha=0.6, label=f"FINAL K={STAGE2_FINAL_K}")
        ax.set_xlabel("Fourier order K")
        ax.set_ylabel("RMSE (N*m/kg)")
        ax.set_title(f"{joint}: RMSE vs. K")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Stage 2 -- Fourier Order Sweep", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(STAGE2_KSWEEP_PLOT_PATH, dpi=150)
    print(f"[Manifold] Saved plot: {STAGE2_KSWEEP_PLOT_PATH}")
    plt.show() if _INTERACTIVE else plt.close(fig)

    fig2, axes2 = plt.subplots(1, len(TARGET_JOINTS), figsize=(7 * len(TARGET_JOINTS), 5))
    axes2 = np.atleast_1d(axes2)
    for ax, joint in zip(axes2, TARGET_JOINTS):
        r = results_by_joint[joint]
        qc = r["qc_student"]
        ax.bar(qc["bin_centers"], qc["rmse"], width=1.0 / STAGE2_QC_BINS * 0.9, color="tab:red", alpha=0.7)
        ax.set_xlabel("phi (student phase, cycle fraction)")
        ax.set_ylabel("RMSE (N*m/kg)")
        ax.set_title(f"{joint}: per-bin RMSE (student-phase fit, K={STAGE2_FINAL_K})\n"
                     f"stance mean={r['stance_rmse']:.3g}  swing mean={r['swing_rmse']:.3g}")
        ax.grid(alpha=0.2)
    fig2.suptitle("Stage 2 -- Per-Phase-Bin Fit Quality", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(STAGE2_QC_PLOT_PATH, dpi=150)
    print(f"[Manifold] Saved plot: {STAGE2_QC_PLOT_PATH}")
    plt.show() if _INTERACTIVE else plt.close(fig2)


# ===========================================================================
# Stage 3 -- control law. NUMPY-ONLY from here down: this is the "one
# script, closed-form runtime path" the brief means -- no torch import
# appears anywhere below this point. Encoding (torch, Stage 4's "encoder"
# half of the latency benchmark) stays strictly upstream of this boundary.
#
#   gain(A) = min(1, A / A_threshold)
#   k(phi)  = k_base * sigma_ref / sigma(phi)      -- stiff where the manifold is thin
#   tau_cmd = gain(A) * scale * tau(phi)
#
# The four named constants the brief calls for: CTRL_A_THRESHOLD,
# CTRL_K_BASE, CTRL_SIGMA_REF, CTRL_SCALE. CTRL_K_MIN/MAX (clamp) and
# CTRL_RATE_LIMIT are additional tunables the brief also asks for, kept as
# named config alongside them rather than magic numbers.
# ===========================================================================

CTRL_A_THRESHOLD = 0.20   # gain(A) saturates to 1 above this amplitude. Picked well below
                          # Stage 1's observed r_star range (~0.60-0.76 across incline walking),
                          # so normal locomotion reaches full gain quickly; low-amplitude
                          # regimes (e.g. A collapsing during sit-to-stand, DECISION #8's
                          # known-weakness) de-rate smoothly through this instead of a cliff.
CTRL_K_BASE = 1.0         # baseline stiffness gain (dimensionless multiplier)
CTRL_SIGMA_REF = 0.07     # reference sigma: k(phi) == k_base exactly where sigma(phi) == this.
                          # Picked near the middle of Stage 1's observed sigma range (~0.045-0.115).
CTRL_SCALE = 1.0          # overall torque scale. tau(phi) is fit directly in N*m/kg units
                          # matching teacher moments, so 1.0 trusts the fit's own calibration.

CTRL_K_MIN = 0.2 * CTRL_K_BASE   # clamp -- sigma -> 0 would otherwise blow k(phi) up
CTRL_K_MAX = 5.0 * CTRL_K_BASE
CTRL_RATE_LIMIT = 20.0    # max |d(tau_cmd)/dt|, N*m/kg per second, per joint


def gain_of_amplitude(A, a_threshold=CTRL_A_THRESHOLD):
    """gain(A) = min(1, A/A_threshold). A is a Fourier amplitude (always
    >= 0, see FLD.fft()'s 2*sqrt(sum(power)) formula), so this is always in
    [0, 1] with no separate lower clip needed."""
    return np.minimum(1.0, np.asarray(A, dtype=np.float64) / a_threshold)


def stiffness_of_phase(phi, bin_centers, sigma_table, k_base=CTRL_K_BASE,
                        sigma_ref=CTRL_SIGMA_REF, k_min=CTRL_K_MIN, k_max=CTRL_K_MAX):
    """k(phi) = k_base * sigma_ref / sigma(phi), clamped to [k_min, k_max].
    sigma(phi) comes from Stage 1's circularly-smoothed table via
    circular_interp() -- continuous across the phase wrap by construction
    (same np.interp(..., period=1.0) mechanism Stage 1's own tests verify)."""
    sigma = circular_interp(phi, bin_centers, sigma_table)
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)   # sigma->0 guard, belt-and-suspenders under the clamp below
    return np.clip(k_base * sigma_ref / sigma_safe, k_min, k_max)


def torque_of_phase(phi, coeffs):
    """tau(phi) -- DECISION #6: imported evaluate_fourier_torque, not
    reimplemented. Continuous across the phase wrap for free -- cos/sin of
    2*pi*k*phi are exactly periodic in phi regardless of how phi is scaled
    or offset going in."""
    return evaluate_fourier_torque(phi, coeffs)


class RateLimiter:
    """Per-signal rate limiter: |output[t] - output[t-1]| <= rate_limit*dt.
    Stateful (holds the last output across calls), numpy/python-only."""

    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self._last = None

    def reset(self, initial_value=0.0):
        self._last = float(initial_value)

    def step(self, value, dt):
        value = float(value)
        if self._last is None:
            self._last = value
            return value
        max_step = self.rate_limit * dt
        delta = float(np.clip(value - self._last, -max_step, max_step))
        self._last = self._last + delta
        return self._last


class ManifoldControlLaw:
    """The closed-form, numpy-only, post-latent control law: Stage 1's two
    lookup tables (bin_centers/sigma_table) plus Stage 2's truncated
    Fourier series (tau_coeffs), evaluated on whatever phi/A the CALLER
    supplies. DECISION #1: tables/coeffs are FIT from forward_teacher, but
    Stage 4/5 always CALL this with the STUDENT's own phi/A -- this class
    itself has no opinion on which branch produced its inputs.

    One instance per joint (each has its own tau(phi) coefficients;
    bin_centers/sigma_table are shared -- they describe the SAME latent
    channel's geometry, not a joint-specific quantity).
    """

    def __init__(self, bin_centers, sigma_table, tau_coeffs, rate_limit=CTRL_RATE_LIMIT):
        self.bin_centers = bin_centers
        self.sigma_table = sigma_table
        self.tau_coeffs = tau_coeffs
        self._rate_limiter = RateLimiter(rate_limit)

    def reset(self):
        self._rate_limiter.reset(0.0)

    def step(self, phi, A, dt):
        """phi: cycle-fraction (native, need not be pre-wrapped to [0,1)).
        A: latent amplitude (dominant channel). dt: seconds since the last
        step (for rate limiting). Returns (tau_cmd, k_phi, gain) -- all
        closed-form, numpy-only, no model forward pass."""
        gain = float(gain_of_amplitude(A))
        k_phi = float(stiffness_of_phase(np.asarray([phi]), self.bin_centers, self.sigma_table)[0])
        tau_raw = gain * CTRL_SCALE * float(torque_of_phase(np.asarray([phi]), self.tau_coeffs)[0])
        tau_cmd = self._rate_limiter.step(tau_raw, dt)
        return tau_cmd, k_phi, gain


def load_control_laws():
    """Builds one ManifoldControlLaw per TARGET_JOINTS entry from Stage 1's
    saved tables + Stage 2's saved student-phase coefficients (the HEADLINE
    fit, DECISION #1/#2 -- teacher coefficients are not used here)."""
    s1 = load_stage1_tables()
    if not os.path.exists(STAGE2_RESULTS_PATH):
        raise SystemExit(f"[Manifold] {STAGE2_RESULTS_PATH} not found -- run `--stage 2` first.")
    s2 = np.load(STAGE2_RESULTS_PATH, allow_pickle=True)
    laws = {}
    for joint in TARGET_JOINTS:
        coeffs = s2[f"coeffs_student_{joint}"]
        laws[joint] = ManifoldControlLaw(s1["bin_centers"], s1["sigma"], coeffs)
    return laws, s1, s2


def run_stage3():
    laws, s1, s2 = load_control_laws()
    print("[Manifold] === Stage 3: control law demonstration (numpy-only, no model forward pass) ===")
    phi_curve = np.linspace(0.0, 1.0, 300)
    A_nominal = float(np.median(s1["r_star"]))
    print(f"[Manifold] Nominal amplitude A={A_nominal:.4g} (median of Stage 1's r_star table) used for the "
          f"tau_cmd demonstration curve below (gain saturates to 1 well before this, since "
          f"CTRL_A_THRESHOLD={CTRL_A_THRESHOLD}).")

    k_curve = stiffness_of_phase(phi_curve, s1["bin_centers"], s1["sigma"])
    gain_curve = gain_of_amplitude(np.full_like(phi_curve, A_nominal))
    print(f"[Manifold] k(phi) range: [{k_curve.min():.3g}, {k_curve.max():.3g}] "
          f"(clamped to [{CTRL_K_MIN:.3g}, {CTRL_K_MAX:.3g}])  gain={float(gain_curve[0]):.3f} (constant, A fixed)")

    fig, axes = plt.subplots(1, len(TARGET_JOINTS) + 1, figsize=(6 * (len(TARGET_JOINTS) + 1), 5))
    axes[0].plot(phi_curve, k_curve, color="tab:blue")
    axes[0].set_title("k(phi) -- shared across joints")
    axes[0].set_xlabel("phi"); axes[0].set_ylabel("k(phi)")
    axes[0].grid(alpha=0.2)

    for ax, joint in zip(axes[1:], TARGET_JOINTS):
        law = laws[joint]
        tau_curve = torque_of_phase(phi_curve, law.tau_coeffs)
        tau_cmd_curve = gain_curve * CTRL_SCALE * tau_curve
        ax.plot(phi_curve, tau_curve, color="tab:gray", ls="--", label="tau(phi), raw fit")
        ax.plot(phi_curve, tau_cmd_curve, color="tab:red", label=f"gain*scale*tau(phi), A={A_nominal:.3g}")
        ax.set_title(f"{joint}: commanded torque shape")
        ax.set_xlabel("phi"); ax.set_ylabel("N*m/kg")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        print(f"[Manifold] {joint}: tau(phi) range [{tau_curve.min():.4g}, {tau_curve.max():.4g}] N*m/kg")

    fig.suptitle("Stage 3 -- Control Law: k(phi) and gain*scale*tau(phi)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(STAGE3_PLOT_PATH, dpi=150)
    print(f"[Manifold] Saved plot: {STAGE3_PLOT_PATH}")
    plt.show() if _INTERACTIVE else plt.close(fig)

    print("[Manifold] Running the closed-form wrap-continuity + rate-limit tests inline:")
    _test_control_law_wrap_continuity()
    _test_rate_limit()
    return laws, s1, s2


# ===========================================================================
# Stage 4 -- latency. Encoder (torch, forward_student -- DECISION #1: the
# student is what runs) and the post-latent closed-form path (numpy-only,
# Stage 3's ManifoldControlLaw) are benchmarked SEPARATELY, >=10k calls
# each, headlining p99 (not mean) per the brief.
# ===========================================================================

STAGE4_N_CALLS = 10000
STAGE4_RESULTS_PATH = os.path.join(STAGE1_OUT_DIR, "stage4_latency.npz")
SUB_MILLISECOND_THRESHOLD_MS = 1.0


def _latency_stats(times_s):
    ms = np.asarray(times_s) * 1000.0
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)),
            "p99": float(np.percentile(ms, 99)), "max": float(ms.max())}


def _print_latency(label, stats):
    print(f"[Manifold]   {label}: mean={stats['mean']:.4f}ms  p50={stats['p50']:.4f}ms  "
          f"p99={stats['p99']:.4f}ms  max={stats['max']:.4f}ms")


def _benchmark_encoder(device, ckpt_path, H, n_calls):
    model, config, norm_stats, _ = load_model_from_path(device, ckpt_path)
    dummy_win = torch.randn(1, TS_SENSOR_DIM, H, device=device)
    with torch.no_grad():
        for _ in range(50):   # warmup -- CUDA context/kernel warmup, first-call overhead
            model.forward_student(dummy_win)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = np.zeros(n_calls)
        for i in range(n_calls):
            t0 = time.perf_counter()
            model.forward_student(dummy_win)
            if device.type == "cuda":
                torch.cuda.synchronize()   # otherwise we'd time kernel LAUNCH, not completion
            times[i] = time.perf_counter() - t0
    return _latency_stats(times)


def _profile_post_latent_components(law, phi_vals, A_vals, dt, n_calls):
    """Breaks the post-latent path into its 4 pieces so a not-sub-ms result
    can be attributed to one of them, per the brief ("if it isn't
    comfortably sub-millisecond, say so and profile it")."""
    def _time_us(fn):
        t0 = time.perf_counter()
        for i in range(n_calls):
            fn(i)
        return (time.perf_counter() - t0) / n_calls * 1e6

    us_gain = _time_us(lambda i: gain_of_amplitude(A_vals[i]))
    us_k = _time_us(lambda i: stiffness_of_phase(np.asarray([phi_vals[i]]), law.bin_centers, law.sigma_table))
    us_tau = _time_us(lambda i: torque_of_phase(np.asarray([phi_vals[i]]), law.tau_coeffs))
    limiter = RateLimiter(CTRL_RATE_LIMIT); limiter.reset(0.0)
    us_limiter = _time_us(lambda i: limiter.step(1.0, dt))
    print(f"[Manifold]     gain_of_amplitude:                {us_gain:7.2f} us/call")
    print(f"[Manifold]     stiffness_of_phase (circular_interp): {us_k:7.2f} us/call")
    print(f"[Manifold]     torque_of_phase (Fourier + dot):   {us_tau:7.2f} us/call")
    print(f"[Manifold]     RateLimiter.step:                 {us_limiter:7.2f} us/call")
    print(f"[Manifold]     sum of parts (x{len(TARGET_JOINTS)} joints, k(phi) shared): "
          f"{(us_gain + us_tau + us_limiter) * len(TARGET_JOINTS) + us_k:7.2f} us/call (vs. measured full-step time above)")


def run_stage4():
    s1 = load_stage1_tables()
    laws, _, _ = load_control_laws()
    H, sample_rate, ckpt_path = s1["history_horizon"], s1["sample_rate"], s1["checkpoint_path"]
    dt = 1.0 / sample_rate

    print("[Manifold] === Stage 4: latency ===")
    hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"[Manifold] Hardware: {hw}")
    print(f"[Manifold] torch=={torch.__version__}  numpy=={np.__version__}  python=={sys.version.split()[0]}")
    print(f"[Manifold] N_CALLS={STAGE4_N_CALLS:,} for every benchmark below.")

    results = {}
    print(f"\n[Manifold] --- ENCODER (forward_student, B=1, single window) ---")
    for device in ([torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])):
        stats = _benchmark_encoder(device, ckpt_path, H, STAGE4_N_CALLS)
        _print_latency(f"forward_student [{device.type}]", stats)
        results[f"encoder_{device.type}"] = stats

    print(f"\n[Manifold] --- POST-LATENT closed-form path (numpy-only, {len(TARGET_JOINTS)} joints/step) ---")
    rng = np.random.default_rng(0)
    phi_vals = rng.uniform(0.0, 1.0, STAGE4_N_CALLS)
    A_vals = rng.uniform(0.05, 1.0, STAGE4_N_CALLS)
    for joint in TARGET_JOINTS:
        laws[joint].reset()
    times = np.zeros(STAGE4_N_CALLS)
    for i in range(STAGE4_N_CALLS):
        t0 = time.perf_counter()
        for joint in TARGET_JOINTS:
            laws[joint].step(phi_vals[i], A_vals[i], dt)
        times[i] = time.perf_counter() - t0
    post_latent_stats = _latency_stats(times)
    _print_latency("post-latent control law (both joints)", post_latent_stats)
    results["post_latent"] = post_latent_stats

    is_sub_ms = post_latent_stats["p99"] < SUB_MILLISECOND_THRESHOLD_MS
    print(f"\n[Manifold] Post-latent p99 = {post_latent_stats['p99']:.4f}ms -- "
          f"{'COMFORTABLY sub-millisecond' if is_sub_ms else 'NOT sub-millisecond'} "
          f"(threshold={SUB_MILLISECOND_THRESHOLD_MS}ms).")
    if not is_sub_ms:
        print("[Manifold] Profiling the post-latent path's components:")
        _profile_post_latent_components(laws[TARGET_JOINTS[0]], phi_vals, A_vals, dt, STAGE4_N_CALLS)
    else:
        print("[Manifold] (Per-component profile skipped -- already comfortably sub-millisecond. "
              "Set FORCE_STAGE4_PROFILE=1 to see the breakdown anyway.)")
        if os.environ.get("FORCE_STAGE4_PROFILE") == "1":
            _profile_post_latent_components(laws[TARGET_JOINTS[0]], phi_vals, A_vals, dt, STAGE4_N_CALLS)

    np.savez(STAGE4_RESULTS_PATH, **{f"{k}_{stat}": v[stat] for k, v in results.items() for stat in ("mean", "p50", "p99", "max")})
    print(f"\n[Manifold] Saved: {STAGE4_RESULTS_PATH}")
    return results


# ===========================================================================
# Stage 5 -- validation. Held-out incline (S04, DECISION #7) first, then the
# UNCHANGED controller run on sit_stand_squat_transitions (this repo's only
# task combining "transition" and "sit-to-stand" -- there is no separately
# labeled "transition" trial, see DEGRADATION_TASK_LABEL's comment). ALL
# numbers here use the STUDENT branch (DECISION #1).
# ===========================================================================

def circular_diff(a, b):
    """Smallest signed difference a-b on the unit circle (cycle-fraction
    units), in (-0.5, 0.5]. E.g. circular_diff(0.01, 0.99) = 0.02, not -0.98."""
    return (np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5


def per_bin_error_stats(phi_all, pred_all, actual_all, n_bins):
    """Generic per-phase-bin RMSE + Pearson r between two raw series
    (unlike Stage 2's per_bin_fit_quality, this doesn't assume a Fourier
    coeffs vector -- used for validating tau_cmd, which passed through
    gain/scale/rate-limit, not just the raw Fourier fit)."""
    bin_idx = _circular_bin_index(np.mod(phi_all, 1.0), n_bins)
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins
    rmse, corr, counts = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] < 5:
            continue
        p, a = pred_all[mask], actual_all[mask]
        rmse[b] = float(np.sqrt(np.mean((p - a) ** 2)))
        if np.std(p) > 1e-9 and np.std(a) > 1e-9:
            corr[b] = float(np.corrcoef(p, a)[0, 1])
    return {"bin_centers": bin_centers, "rmse": rmse, "corr": corr, "counts": counts}


def run_control_law_over_series(law, phi_series, A_series, sample_rate, encode_stride):
    """Steps law.step() sequentially across an ENTIRE encoded trial series,
    dt = encode_stride/sample_rate between consecutive samples (the real
    cadence these samples were taken at) -- so the rate limiter sees
    realistic timing, not an arbitrary one. law.reset() is called first so
    a previous trial's rate-limiter state never leaks into this one."""
    law.reset()
    dt = encode_stride / sample_rate
    tau_cmd = np.zeros(len(phi_series))
    gains = np.zeros(len(phi_series))
    for i in range(len(phi_series)):
        tau_cmd[i], _k, gains[i] = law.step(phi_series[i], A_series[i], dt)
    return tau_cmd, gains


def compute_phase_advance_error(phase_col, freq_raw_col, sample_rate, history_horizon,
                                 encode_stride, dt_ms):
    """phase_col/freq_raw_col: (T,) dominant-channel phase/RAW frequency for
    ONE trial's encoded series (uniformly spaced by encode_stride frames).
    dt_ms: requested look-ahead in milliseconds.

    Converts freq_raw -> Hz (DECISION #5's freq_raw_to_hz), predicts
    phi(t+dt) = phi(t) + f_hz(t)*dt_actual (DECISION #5's cycle-unit
    phase_advance -- NOT the brief's literal 2*pi*f*dt, which assumes
    radians; phi here is a native cycle-fraction), compares against the
    ACTUAL phase measured dt_actual later (dt_actual is dt_ms rounded to
    the nearest whole encode_stride step -- reported explicitly, since it
    will not exactly equal dt_ms unless dt_ms*sample_rate/1000 is itself a
    multiple of encode_stride).

    Returns dict: 'dt_actual_ms', 'index_offset', 'error' (T-offset,)
    circular signed error in CYCLE-FRACTION units (multiply by 1000/f_hz or
    similar to convert to a time error if wanted), 'rmse_cycles'.
    """
    freq_hz_col = freq_raw_to_hz(freq_raw_col, sample_rate, history_horizon)
    index_offset = max(1, round((dt_ms / 1000.0) * sample_rate / encode_stride))
    dt_actual = index_offset * encode_stride / sample_rate
    if index_offset >= len(phase_col):
        return None
    phi_now = phase_col[:-index_offset]
    freq_now = freq_hz_col[:-index_offset]
    phi_actual_future = phase_col[index_offset:]
    phi_pred_future = phase_advance(phi_now, freq_now, dt_actual)
    error = circular_diff(phi_pred_future, phi_actual_future)
    return {
        "dt_requested_ms": dt_ms, "dt_actual_ms": dt_actual * 1000.0, "index_offset": index_offset,
        "error": error, "rmse_cycles": float(np.sqrt(np.mean(error ** 2))),
        "mean_abs_cycles": float(np.mean(np.abs(error))),
    }


def flow_coherence_series(points_2d, k=FLOW_COHERENCE_K):
    """points_2d: (T, 2) -- (A*cos(2*pi*phi), A*sin(2*pi*phi)) for ONE
    trial's dominant-channel embedding (the SAME 2D projection this whole
    controller operates on -- deliberately NOT the full 8-channel latent
    space, since the controller itself never sees more than this one
    channel either; see module docstring's DECISION #3).

    velocity[t] = points[t+1] - points[t], t in [0, T-2). For each t, finds
    its k nearest neighbours (by Euclidean distance in points_2d[:-1], so
    every neighbour has a defined velocity too) and computes cosine
    similarity between velocity[t] and the MEAN velocity of those
    neighbours -- "local flow coherence" exactly as specified.

    Returns (T-1,) array (one value per velocity sample, i.e. per t in
    [0, T-2)); NaN where velocity is exactly zero (undefined direction).
    """
    origins = points_2d[:-1]
    velocity = points_2d[1:] - points_2d[:-1]
    n = len(origins)
    if n <= k:
        return np.full(n, np.nan)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(origins)   # +1 -- a point is its own nearest neighbour
    _dist, idx = nn.kneighbors(origins)
    idx = idx[:, 1:]   # drop self
    mean_neighbor_v = velocity[idx].mean(axis=1)   # (n, 2)

    v_norm = np.linalg.norm(velocity, axis=1)
    mv_norm = np.linalg.norm(mean_neighbor_v, axis=1)
    safe = (v_norm > 1e-12) & (mv_norm > 1e-12)
    coherence = np.full(n, np.nan)
    coherence[safe] = np.sum(velocity[safe] * mean_neighbor_v[safe], axis=1) / (v_norm[safe] * mv_norm[safe])
    return coherence


def _dom_ch_series(enc, dom_ch):
    """phi (wrapped [0,1)), raw phase (unwrapped, for phase-advance), A,
    freq_raw -- all (T,) for the dominant channel of one branch's encoding."""
    return {
        "phi": np.mod(enc["phase"][:, dom_ch], 1.0),
        "phase_raw": enc["phase"][:, dom_ch],
        "A": enc["amplitude"][:, dom_ch],
        "freq_raw": enc["frequency"][:, dom_ch],
        "offset": enc["offset"][:, dom_ch],
    }


def run_stage5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1 = load_stage1_tables()
    dom_ch, H, stride, sample_rate = s1["dom_ch"], s1["history_horizon"], s1["encode_stride"], s1["sample_rate"]
    laws, _, _ = load_control_laws()
    model, config, norm_stats, ckpt_path = load_model_from_path(device, s1["checkpoint_path"])

    all_df, tasks, fit_ids, heldout_ids = load_incline_trials()
    if heldout_ids != list(s1["heldout_trial_ids"]):
        print(f"[Manifold] NOTE: held-out trial set changed since Stage 1 ({heldout_ids} vs "
              f"{list(s1['heldout_trial_ids'])}) -- HOLDOUT_SUBJECTS/data must have changed; proceeding "
              f"with the CURRENT set since it's fixed by HOLDOUT_SUBJECTS, not by Stage 1's saved file.")

    # =======================================================================
    # Part A -- held-out INCLINE walking (S04)
    # =======================================================================
    print(f"\n[Manifold] ################ STAGE 5A: held-out incline walking ({heldout_ids}) ################")
    heldout_enc = encode_trials_both_branches(model, all_df, heldout_ids, norm_stats, H, stride, ENCODE_BATCH_SIZE, device)

    for joint in TARGET_JOINTS:
        print(f"\n[Manifold] --- {joint} ---")
        phi_all, pred_all, actual_all = [], [], []
        for tid, enc in heldout_enc.items():
            s = _dom_ch_series(enc["student"], dom_ch)
            tau_cmd, gains = run_control_law_over_series(laws[joint], s["phi"], s["A"], sample_rate, stride)
            actual = target_moment(enc["trial_df"], enc["student"]["frame_indices"], joint)
            phi_all.append(s["phi"]); pred_all.append(tau_cmd); actual_all.append(actual)
        phi_all, pred_all, actual_all = np.concatenate(phi_all), np.concatenate(pred_all), np.concatenate(actual_all)

        rmse = float(np.sqrt(np.mean((pred_all - actual_all) ** 2)))
        corr = float(np.corrcoef(pred_all, actual_all)[0, 1])
        print(f"[Manifold]   Commanded tau_cmd vs. measured moment: RMSE={rmse:.4g} N*m/kg  corr={corr:.3f} "
              f"(n={len(phi_all):,} pooled windows, {len(heldout_enc)} trials)")
        qc = per_bin_error_stats(phi_all, pred_all, actual_all, STAGE5_QC_BINS)
        worst_bins = np.argsort(np.nan_to_num(qc["rmse"], nan=-1))[::-1][:3]
        print(f"[Manifold]   Worst 3 phase bins (phi, RMSE): "
              + ", ".join(f"({qc['bin_centers'][b]:.2f}, {qc['rmse'][b]:.3g})" for b in worst_bins))

    print(f"\n[Manifold] --- Phase-advance error (student, dominant channel), held-out incline ---")
    for dt_ms in PHASE_ADVANCE_DT_MS:
        errs = []
        for tid, enc in heldout_enc.items():
            s = _dom_ch_series(enc["student"], dom_ch)
            r = compute_phase_advance_error(s["phase_raw"], s["freq_raw"], sample_rate, H, stride, dt_ms)
            if r is not None:
                errs.append(r["error"])
        errs = np.concatenate(errs)
        rmse_cycles = float(np.sqrt(np.mean(errs ** 2)))
        print(f"[Manifold]   dt={dt_ms}ms: RMSE={rmse_cycles:.4g} cycles ({rmse_cycles*360:.2f} deg, "
              f"{rmse_cycles*100:.2f}% of a full cycle), mean|err|={float(np.mean(np.abs(errs))):.4g} cycles "
              f"(n={len(errs):,})")

    all_A_heldout = np.concatenate([_dom_ch_series(enc["student"], dom_ch)["A"] for enc in heldout_enc.values()])
    gain_heldout = gain_of_amplitude(all_A_heldout)
    print(f"\n[Manifold] --- gain(A) distribution, held-out incline ---")
    print(f"[Manifold]   mean={gain_heldout.mean():.3f}  median={np.median(gain_heldout):.3f}  "
          f"p5={np.percentile(gain_heldout,5):.3f}  p95={np.percentile(gain_heldout,95):.3f}  "
          f"fraction saturated at 1.0: {float(np.mean(gain_heldout >= 0.999)):.3f}")

    plot_stage5_annulus(heldout_enc, dom_ch, s1)

    # =======================================================================
    # Part B -- UNCHANGED controller on sit_stand_squat_transitions
    # =======================================================================
    print(f"\n[Manifold] ################ STAGE 5B: sit_stand_squat_transitions (degradation) ################")
    degradation_ids = sorted(tid for tid, info in _discover_tasks(all_df).items()
                              if info["label"] == DEGRADATION_TASK_LABEL)
    print(f"[Manifold] {len(degradation_ids)} trials with task=={DEGRADATION_TASK_LABEL!r}: {degradation_ids}")
    degr_enc = encode_trials_both_branches(model, all_df, degradation_ids, norm_stats, H, stride, ENCODE_BATCH_SIZE, device)

    print(f"\n[Manifold] --- Phase-advance error, sit_stand_squat_transitions (vs. held-out incline above) ---")
    for dt_ms in PHASE_ADVANCE_DT_MS:
        errs = []
        for tid, enc in degr_enc.items():
            s = _dom_ch_series(enc["student"], dom_ch)
            r = compute_phase_advance_error(s["phase_raw"], s["freq_raw"], sample_rate, H, stride, dt_ms)
            if r is not None:
                errs.append(r["error"])
        errs = np.concatenate(errs)
        print(f"[Manifold]   dt={dt_ms}ms: RMSE={float(np.sqrt(np.mean(errs ** 2))):.4g} cycles  "
              f"mean|err|={float(np.mean(np.abs(errs))):.4g} cycles  (n={len(errs):,})")

    all_A_degr = np.concatenate([_dom_ch_series(enc["student"], dom_ch)["A"] for enc in degr_enc.values()])
    gain_degr = gain_of_amplitude(all_A_degr)
    frac_below_threshold_heldout = float(np.mean(all_A_heldout < CTRL_A_THRESHOLD))
    frac_below_threshold_degr = float(np.mean(all_A_degr < CTRL_A_THRESHOLD))
    print(f"\n[Manifold] --- Amplitude collapse check (does A -> origin de-rate gain during sit-to-stand?) ---")
    print(f"[Manifold]   Held-out incline: mean A={all_A_heldout.mean():.4g}  "
          f"fraction A < CTRL_A_THRESHOLD({CTRL_A_THRESHOLD})={frac_below_threshold_heldout:.3f}  "
          f"mean gain={gain_heldout.mean():.3f}")
    print(f"[Manifold]   sit_stand_squat_transitions: mean A={all_A_degr.mean():.4g}  "
          f"fraction A < CTRL_A_THRESHOLD({CTRL_A_THRESHOLD})={frac_below_threshold_degr:.3f}  "
          f"mean gain={gain_degr.mean():.3f}")

    # offset ("c") vs (A, phi)-plane variability check -- known weakness #3
    print(f"\n[Manifold] --- offset ('c') vs. (A,phi)-plane variability (known weakness: sit-to-stand may live in c) ---")
    for label, enc_dict in (("held-out incline", heldout_enc), ("sit_stand_squat_transitions", degr_enc)):
        offsets, radii = [], []
        for enc in enc_dict.values():
            s = _dom_ch_series(enc["student"], dom_ch)
            offsets.append(s["offset"]); radii.append(s["A"])
        offsets, radii = np.concatenate(offsets), np.concatenate(radii)
        print(f"[Manifold]   {label}: std(offset)={offsets.std():.4g}  std(A)={radii.std():.4g}  "
              f"ratio std(offset)/std(A)={offsets.std() / max(radii.std(), 1e-9):.3f}")

    # =======================================================================
    # Flow coherence vs. phase-advance error -- prioritized per the brief
    # =======================================================================
    print(f"\n[Manifold] ################ Flow coherence vs. phase-advance error (PRIORITY RESULT) ################")
    dt_ms_primary = 100
    coherence_all, error_all = [], []
    for tid, enc in degr_enc.items():
        s = _dom_ch_series(enc["student"], dom_ch)
        points_2d = np.stack([s["A"] * np.cos(2 * np.pi * s["phi"]), s["A"] * np.sin(2 * np.pi * s["phi"])], axis=1)
        coherence = flow_coherence_series(points_2d, FLOW_COHERENCE_K)   # (T-1,)
        pa = compute_phase_advance_error(s["phase_raw"], s["freq_raw"], sample_rate, H, stride, dt_ms_primary)
        if pa is None:
            continue
        # align: coherence[t] describes the step FROM sample t (length T-1);
        # pa["error"][i] describes the look-ahead FROM sample i (length T-offset).
        # Both are indexed from sample 0 -- truncate to the shorter common length.
        n_common = min(len(coherence), len(pa["error"]))
        coherence_all.append(coherence[:n_common])
        error_all.append(np.abs(pa["error"][:n_common]))
    coherence_all, error_all = np.concatenate(coherence_all), np.concatenate(error_all)
    valid = ~np.isnan(coherence_all)
    pearson_r = float(np.corrcoef(coherence_all[valid], error_all[valid])[0, 1])
    print(f"[Manifold] Pearson r(flow_coherence, |phase_advance_error @ {dt_ms_primary}ms|) = {pearson_r:.3f} "
          f"(n={int(valid.sum()):,}, sit_stand_squat_transitions trials)")
    print(f"[Manifold] {'NEGATIVE correlation as expected' if pearson_r < -0.05 else 'no clear / unexpected-sign correlation'} "
          f"-- {'low flow coherence DOES predict high phase-advance error here' if pearson_r < -0.05 else 'flow coherence is NOT a reliable predictor of phase-advance error on this data'}.")

    plot_stage5_coherence(coherence_all[valid], error_all[valid], pearson_r)

    results = {
        "gain_heldout_mean": float(gain_heldout.mean()), "gain_degr_mean": float(gain_degr.mean()),
        "frac_below_threshold_heldout": frac_below_threshold_heldout, "frac_below_threshold_degr": frac_below_threshold_degr,
        "coherence_error_pearson_r": pearson_r,
    }
    np.savez(os.path.join(STAGE1_OUT_DIR, "stage5_validation.npz"), **results)
    print(f"\n[Manifold] Saved: {os.path.join(STAGE1_OUT_DIR, 'stage5_validation.npz')}")
    return results


def plot_stage5_annulus(heldout_enc, dom_ch, s1):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(heldout_enc)))
    for (tid, enc), color in zip(heldout_enc.items(), colors):
        s = _dom_ch_series(enc["student"], dom_ch)
        x, y = s["A"] * np.cos(2 * np.pi * s["phi"]), s["A"] * np.sin(2 * np.pi * s["phi"])
        ax.scatter(x, y, s=3, alpha=0.15, color=color, label=f"{tid} (student, held-out)")

    phi_ring, r, sig = s1["bin_centers"], s1["r_star"], s1["sigma"]
    def _closed(a): return np.append(a, a[0])
    ring_x, ring_y = r * np.cos(2 * np.pi * phi_ring), r * np.sin(2 * np.pi * phi_ring)
    outer = np.clip(r + sig, 0, None); inner = np.clip(r - sig, 0, None)
    ax.plot(_closed(ring_x), _closed(ring_y), color="tab:red", lw=2.0, label="r_star(phi) (teacher, fit trials)")
    ax.plot(_closed(outer * np.cos(2 * np.pi * phi_ring)), _closed(outer * np.sin(2 * np.pi * phi_ring)),
            color="tab:red", lw=0.8, ls="--", alpha=0.7, label="r_star +/- sigma")
    ax.plot(_closed(inner * np.cos(2 * np.pi * phi_ring)), _closed(inner * np.sin(2 * np.pi * phi_ring)),
            color="tab:red", lw=0.8, ls="--", alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel(f"A*cos(2*pi*phi), channel {dom_ch}"); ax.set_ylabel(f"A*sin(2*pi*phi), channel {dom_ch}")
    ax.set_title("Stage 5 -- Held-out (S04) STUDENT annulus vs. teacher-fit r_star(phi) +/- sigma(phi)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(STAGE1_OUT_DIR, "stage5_heldout_annulus.png")
    fig.savefig(path, dpi=150)
    print(f"[Manifold] Saved plot: {path}")
    plt.show() if _INTERACTIVE else plt.close(fig)


def plot_stage5_coherence(coherence, abs_error, pearson_r):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coherence, abs_error, s=4, alpha=0.15, color="tab:purple")
    ax.set_xlabel("local flow coherence (cosine sim. with k-NN mean velocity)")
    ax.set_ylabel("|phase-advance error| @ 100ms (cycles)")
    ax.set_title(f"Flow Coherence vs. Phase-Advance Error\nsit_stand_squat_transitions, Pearson r={pearson_r:.3f}",
                 fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(STAGE1_OUT_DIR, "stage5_coherence_vs_error.png")
    fig.savefig(path, dpi=150)
    print(f"[Manifold] Saved plot: {path}")
    plt.show() if _INTERACTIVE else plt.close(fig)


# ===========================================================================
# Tests -- phase-wrap continuity, empty-bin handling, control-law wrap
# continuity, and rate limiting.
# ===========================================================================

def _test_wrap_continuity():
    rng = np.random.default_rng(0)
    n_bins = 50
    phi_all = rng.uniform(0, 1, 20000)
    true_r = 1.0 + 0.3 * np.sin(2 * np.pi * phi_all)   # smooth ground truth, no seam
    radius_all = true_r + rng.normal(0, 0.02, size=phi_all.shape)
    table = build_phase_binned_table(phi_all, radius_all, n_bins, SMOOTH_KERNEL_SIZE)

    # 1) direct table seam: bin[-1] and bin[0] must be close (adjacent bins)
    seam = abs(table["r_star"][0] - table["r_star"][-1])
    typical_step = np.mean(np.abs(np.diff(table["r_star"])))
    assert seam < 5 * typical_step + 1e-6, f"wrap-seam gap {seam:.4g} >> typical bin-to-bin step {typical_step:.4g}"

    # 2) circular_interp continuity: query just below 1.0 and just above 0.0
    #    must be close to each other (they're ~1/n_bins apart on the circle)
    left = circular_interp(np.array([0.999]), table["bin_centers"], table["r_star"])[0]
    right = circular_interp(np.array([0.001]), table["bin_centers"], table["r_star"])[0]
    gap = abs(left - right)
    assert gap < 5 * typical_step + 1e-6, f"circular_interp wrap discontinuity: {gap:.4g}"
    print(f"[Test] wrap_continuity: PASS (seam={seam:.4g}, interp_gap={gap:.4g}, typical_step={typical_step:.4g})")


def _test_empty_bins():
    rng = np.random.default_rng(1)
    n_bins = 20
    # Deliberately leave a gap: no samples in [0.5, 0.6)
    phi_all = rng.uniform(0, 1, 20000)
    phi_all = phi_all[~((phi_all >= 0.5) & (phi_all < 0.6))]
    radius_all = 1.0 + 0.1 * np.sin(2 * np.pi * phi_all) + rng.normal(0, 0.01, size=phi_all.shape)
    table = build_phase_binned_table(phi_all, radius_all, n_bins, SMOOTH_KERNEL_SIZE)

    assert table["empty_mask"].any(), "test setup should have produced at least one empty bin"
    assert not np.isnan(table["r_star"]).any(), "r_star must never contain NaN (brief: no silent NaN)"
    assert not np.isnan(table["sigma"]).any(), "sigma must never contain NaN"
    # filled value should be a plausible interpolation, not some default like 0
    empty_idx = np.flatnonzero(table["empty_mask"])
    assert np.all(table["r_star"][empty_idx] > 0.5), "filled r_star in the empty gap looks implausible"
    print(f"[Test] empty_bins: PASS ({table['empty_mask'].sum()} empty bins detected and filled, no NaN)")


def _test_control_law_wrap_continuity():
    """Self-contained (no dependency on Stage 1's saved tables): builds a
    synthetic sigma(phi) table with a phi-dependent spread and checks
    stiffness_of_phase() has no seam at phi=0/1. Also checks torque_of_phase
    (though that one is continuous by construction -- cos/sin(2*pi*k*phi)
    are exactly periodic regardless of input scaling -- the real risk is in
    the interpolated k(phi) table, not the analytic Fourier series)."""
    rng = np.random.default_rng(2)
    n_bins = 50
    phi_all = rng.uniform(0, 1, 40000)
    local_scale = 0.07 + 0.02 * np.cos(2 * np.pi * phi_all)   # smooth spread, no seam by construction
    radius_all = 1.0 + local_scale * rng.normal(0, 1, size=phi_all.shape)
    table = build_phase_binned_table(phi_all, radius_all, n_bins, SMOOTH_KERNEL_SIZE)

    k_curve = stiffness_of_phase(table["bin_centers"], table["bin_centers"], table["sigma"])
    typical_k_step = np.mean(np.abs(np.diff(k_curve)))
    k_left = stiffness_of_phase(np.array([0.999]), table["bin_centers"], table["sigma"])[0]
    k_right = stiffness_of_phase(np.array([0.001]), table["bin_centers"], table["sigma"])[0]
    k_gap = abs(k_left - k_right)
    assert k_gap < 5 * typical_k_step + 1e-6, f"k(phi) wrap discontinuity {k_gap:.4g} >> typical step {typical_k_step:.4g}"

    # tau(phi) check uses phi=0.0 vs phi=1.0 EXACTLY (mathematically the same
    # point on the circle for a 1-periodic function), not two nearby-but-
    # distinct points like 0.999/0.001 (which are 0.002 apart and, with
    # random O(1) coefficients up to k=STAGE2_FINAL_K, can legitimately
    # differ by more than a tiny tolerance -- that's normal local variation,
    # not a wrap bug). This exact-invariance check would catch a real
    # regression (e.g. someone swapping cos(2*pi*k*phi) for cos(k*phi),
    # which is NOT 1-periodic) without being fooled by coefficient scale.
    rng2 = np.random.default_rng(3)
    coeffs = rng2.normal(0, 1, size=2 * STAGE2_FINAL_K + 1)
    tau_at_0 = torque_of_phase(np.array([0.0]), coeffs)[0]
    tau_at_1 = torque_of_phase(np.array([1.0]), coeffs)[0]
    tau_at_2 = torque_of_phase(np.array([2.0]), coeffs)[0]   # two full cycles out -- still must match
    tau_gap = max(abs(tau_at_0 - tau_at_1), abs(tau_at_0 - tau_at_2))
    assert tau_gap < 1e-9, f"tau(phi) wrap discontinuity: {tau_gap:.4g} (should be ~0, analytically periodic)"
    print(f"[Test] control_law_wrap_continuity: PASS (k gap={k_gap:.4g} vs typical step={typical_k_step:.4g}; "
          f"tau gap={tau_gap:.2g}, analytically periodic)")


def _test_rate_limit():
    limiter = RateLimiter(rate_limit=10.0)   # 10 units/sec
    limiter.reset(0.0)
    dt = 0.01

    out = limiter.step(1000.0, dt)   # command a huge instantaneous jump
    max_allowed = 10.0 * dt
    assert abs(out - 0.0) <= max_allowed + 1e-9, f"rate limiter allowed a jump of {out}, expected <= {max_allowed}"

    for _ in range(500):
        out = limiter.step(1000.0, dt)   # 500 steps * dt=0.01s = 5s * 10 units/s = up to 50
    assert out > 40.0, f"rate limiter should have ramped up substantially after 500 steps, got {out}"
    print(f"[Test] rate_limit: PASS (single-step clamp <= {max_allowed}, ramped to {out:.2f} after 500 steps)")


def run_tests():
    print("[Manifold] Running tests (wrap continuity, empty bins, control-law wrap, rate limit)...")
    _test_wrap_continuity()
    _test_empty_bins()
    _test_control_law_wrap_continuity()
    _test_rate_limit()
    print("[Manifold] All tests passed.")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["1", "2", "3", "4", "5", "test"],
                         help="Which stage to run.")
    args = parser.parse_args()

    if args.stage == "1":
        run_stage1()
    elif args.stage == "2":
        run_stage2()
    elif args.stage == "3":
        run_stage3()
    elif args.stage == "4":
        run_stage4()
    elif args.stage == "5":
        run_stage5()
    elif args.stage == "test":
        run_tests()
    else:
        raise NotImplementedError(
            f"Stage {args.stage} is not implemented yet. "
            f"See the module docstring's DECISIONS list for what's locked in so far."
        )


if __name__ == "__main__":
    main()
