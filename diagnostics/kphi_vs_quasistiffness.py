"""
diagnostics/kphi_vs_quasistiffness.py

Throwaway diagnostic: does K = J*diag(m*omega^2)*J^T (analytic, from the
FLD-TeacherStudent student latent) match measured quasi-stiffness over the
gait cycle, for ONE subject/trial, for hip flexion and knee flexion (right
side) -- a hip-knee exoskeleton, so ankle was dropped.

Also sweeps theta_ref's forecast lookahead step (0-50, the checkpoint's own
trained forecast_horizon) to find which step minimizes RMSE(tau_hat, measured
moment) over the whole trial, per joint -- see the OPTIMIZATION SWEEP section.

`impedance_from_latents.py` / `decoder_jacobian_at_bin` / `compute_joint_space_stiffness`
were named in the original brief but do not exist anywhere in this repo (not
in the working tree, not in git history) -- confirmed by search, not assumed.
So the modal-stiffness math below is written fresh, directly from the
formula in the brief, NOT reverse-engineered from a file that isn't there.
What IS reused, imported (not reimplemented): FLDTeacherStudent.decode() (the
propagation function) and encode_student_series() (the student-only encode
loop, from ../analyze_student_latents.py).

CURVE A (predicted): amplitude is treated as each latent channel's SHM modal
coordinate. J = d(decoded angle_j, normalized)/d(amplitude_i), autograd,
evaluated at the causal (last) time-bin of the decoder's window -- "at bin"
in decoder_jacobian_at_bin's name. m_i = J_i^2 * I_joint (single-joint case
of the brief's m_i = sum_j J_ji^2*I_j). K = sum_i J_i^2 * m_i * omega_i^2.
Phase is swept across one representative stride by calling decode() with a
pre-advanced phase (phase0 + freq_raw*dt*h -- decode()'s own phase_dynamics
formula for step h) and horizon_steps=1, instead of horizon_steps=h+1 growing
each time -- same values decode() would produce, O(n) calls instead of O(n^2).

CURVE B (measured): overlapping sliding-window least-squares fit of
M = -K*theta - B*theta_dot + c per the brief; ill-conditioned/low-excursion
windows are left as NaN gaps, not fabricated numbers.

Curve A's absolute scale is a function of the encoder's own arbitrary
amplitude units -- meaningless on its own, per the brief. ONE global scalar
is fit to Curve A per joint and reported; shape/timing is what's evaluated.

Run:
    python3 diagnostics/kphi_vs_quasistiffness.py
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib
_HAS_DISPLAY = bool(os.environ.get("DISPLAY")) or os.name == "nt"
matplotlib.use("TkAgg" if _HAS_DISPLAY else "Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from Models.FLD_teacher_student import FLDTeacherStudent               # decode() reused, not reimplemented
from DataLoader.teacher_student_data_loader import SENSOR_COLUMNS, SENSOR_DIM, BIOMECH_COLUMNS, BIOMECH_DIM
from plot_error_across_time import _find_checkpoint, _load_state_dict_flexible
from analyze_student_latents import encode_student_series               # reused, not reimplemented

# ---------------------------------------------------------------------------
# Subject / trial -- hardcoded, no auto-selection. Printed at runtime.
# ---------------------------------------------------------------------------
CHECKPOINT_RUN = "FLDTeacherStudent_20260811_180417"   # pinned, trained (epoch 20) -- see analyze_student_latents.py
SUBJECT_NAME = "S01"
TRIAL_NUMBER = "2"
DATA_DIR = os.path.join(REPO_ROOT, "Data")

TARGET_JOINTS = ["hip_flexion_r", "knee_angle_r"]   # right side -- hip-knee exoskeleton, ankle dropped
JOINT_CHAIN = {   # segments (proximal->distal) whose inertia contributes to that joint's swing
    "hip_flexion_r": ["thigh", "shank", "foot"],
    "knee_angle_r":  ["shank", "foot"],
}
# Winter (1990) anthropometric regressions: mass fraction, length fraction of
# height, radius of gyration about the segment's OWN proximal joint (fraction
# of segment length). Population defaults, applied to this subject's own
# mass/height (or population mass/height if those columns are missing).
WINTER_TABLE = {"thigh": (0.100, 0.245, 0.540), "shank": (0.0465, 0.246, 0.528), "foot": (0.0145, 0.152, 0.690)}
POP_MASS_KG, POP_HEIGHT_M = 70.0, 1.70

LATENT_STRIDE = 1
MAX_ENCODE_BATCH = 256
WINDOW_FRAC, STEP_FRAC = 0.12, 0.01     # Curve B sliding window: 12% of stride, 1% step, overlapping
COND_THRESHOLD = 1e3
MIN_EXCURSION_DEG = 3.0

THETA_REF_STEP = 49   # from the RMSE sweep printout: lowest-error step for both hip_flexion_r and knee_angle_r

MAX_FORECAST_STEP = 50   # sweep theta_ref's lookahead step over 0..MAX_FORECAST_STEP (checkpoint's own trained forecast_horizon)


def compute_segment_inertias(mass_kg, height_m):
    """segment -> (mass_kg, length_m, I_about_own_proximal_joint [kg*m^2])."""
    out = {}
    for seg, (mf, lf, rf) in WINTER_TABLE.items():
        m, L = mf * mass_kg, lf * height_m
        out[seg] = (m, L, m * (rf * L) ** 2)
    return out


def joint_inertia(joint_key, seg_data):
    """Effective swing inertia about `joint_key`: the first (proximal-most)
    segment's own inertia about its proximal end, plus each more-distal
    segment lumped as a point mass at ITS OWN proximal end (parallel-axis) --
    a standard simplified serial-chain approximation, not exact rigid-body
    inertia (this is a throwaway diagnostic, not a full inverse-dynamics
    model)."""
    chain = JOINT_CHAIN[joint_key]
    m0, L0, I0 = seg_data[chain[0]]
    I_total, cum_L = I0, L0
    for seg in chain[1:]:
        m, L, _ = seg_data[seg]
        I_total += m * cum_L ** 2
        cum_L += L
    return I_total


def load_checkpoint(device):
    run_dir, ckpt_path = _find_checkpoint("FLDTeacherStudent", "fld_teacher_student", CHECKPOINT_RUN)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config, norm_stats = ckpt["config"], ckpt["norm_stats"]
    model = FLDTeacherStudent(
        sensor_dim=SENSOR_DIM, biomech_dim=BIOMECH_DIM, history_horizon=config["history_horizon"],
        latent_dim=config["latent_dim"], device=device, dt=1.0 / config["sample_rate"],
        teacher_encoder_shape=config["teacher_encoder_shape"], student_encoder_shape=config["student_encoder_shape"],
        decoder_shape=config["decoder_shape"],
    ).to(device)
    _load_state_dict_flexible(model, ckpt["model_state_dict"])
    model.eval()
    print(f"[Kphi] Checkpoint: {ckpt_path} (epoch={ckpt['epoch']}, latent_dim={config['latent_dim']})")
    return model, config, norm_stats


def load_trial(sample_rate):
    path = os.path.join(DATA_DIR, SUBJECT_NAME, f"trial_{TRIAL_NUMBER}_all_data.csv")
    print(f"[Kphi] Subject={SUBJECT_NAME}  Trial={TRIAL_NUMBER}  file={path}")
    if not os.path.exists(path):
        raise SystemExit(f"[Kphi] {path} does not exist -- refusing to substitute another trial.")
    needed = list(dict.fromkeys(SENSOR_COLUMNS + BIOMECH_COLUMNS + ["subject_weight", "subject_height", "task"]))
    df = pd.read_csv(path, usecols=needed)
    task = str(df["task"].iloc[0])
    print(f"[Kphi] task={task!r}  {len(df):,} rows ({len(df) / sample_rate:.1f}s @ {sample_rate:.0f}Hz)")
    if "walk" not in task.lower():
        raise SystemExit(f"[Kphi] task={task!r} is not steady walking -- refusing to substitute another trial.")
    moment_cols = [c for c in BIOMECH_COLUMNS if c.endswith("_moment")]
    df[moment_cols] = df[moment_cols].div(df["subject_weight"], axis=0)   # -> N*m/kg, same convention as everywhere else
    return df


def decode_angle_at(model, phase_t, freq_t, amp_t, offset_t):
    """decoder_jacobian_at_bin's target function: the shared decoder's
    ANGLE-channel outputs (BIOMECH_COLUMNS[:10]) at the causal (last) time-bin
    of its reconstructed window, for one propagation step -- normalized
    biomech space, un-normalized by the caller."""
    decoded = model.decode((phase_t, freq_t, amp_t, offset_t), horizon_steps=1)
    return decoded[0, 0, :10, -1]


def batched_decode_all_steps(model, phase, freq, amp, offset, max_step, angle_idx,
                              mean_biomech, std_biomech, device, batch_size):
    """theta_ref candidates for EVERY lookahead step 0..max_step at once, for
    the given angle channels, over an already-encoded (T,K) series. Calls
    model.decode() directly (horizon_steps=max_step+1) rather than
    model.forecast() in a loop per step -- decode() is exactly what
    forecast() itself calls internally (forecast(params,h) ==
    decode(params,h+1)[1:]), so this is the SAME propagation function, just
    read once per batch instead of re-run from step 0 for every candidate
    step (O(steps) decode() calls total instead of O(steps) separate full
    per-step passes). Step 0 is the model's own CURRENT reconstruction (not
    literal ground truth), kept consistent with steps 1..max_step rather
    than special-cased. Returns (T, max_step+1, len(angle_idx)) physical
    degrees."""
    T = phase.shape[0]
    out = np.zeros((T, max_step + 1, len(angle_idx)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, T, batch_size):
            sl = slice(i, i + batch_size)
            params = (
                torch.tensor(phase[sl], dtype=torch.float32, device=device),
                torch.tensor(freq[sl], dtype=torch.float32, device=device),
                torch.tensor(amp[sl], dtype=torch.float32, device=device),
                torch.tensor(offset[sl], dtype=torch.float32, device=device),
            )
            decoded = model.decode(params, horizon_steps=max_step + 1)         # (max_step+1, b, biomech_dim, H), normalized
            vals = decoded[:, :, angle_idx, -1].permute(1, 0, 2).cpu().numpy()  # (b, max_step+1, len(angle_idx))
            out[sl] = vals * std_biomech[angle_idx] + mean_biomech[angle_idx]
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Kphi] Using device: {device}")
    model, config, norm_stats = load_checkpoint(device)
    H, sample_rate, dt = config["history_horizon"], config["sample_rate"], 1.0 / config["sample_rate"]

    df = load_trial(sample_rate)

    if "subject_weight" in df.columns and "subject_height" in df.columns:
        mass_kg, height_m = float(df["subject_weight"].iloc[0]), float(df["subject_height"].iloc[0]) / 100.0
        print(f"[Kphi] Inertia source: MEASURED subject_weight={mass_kg:.1f}kg, subject_height={height_m*100:.0f}cm")
    else:
        mass_kg, height_m = POP_MASS_KG, POP_HEIGHT_M
        print(f"[Kphi] Inertia source: POPULATION DEFAULT mass={mass_kg}kg, height={height_m*100:.0f}cm "
              f"(subject_weight/height missing)")
    seg_data = compute_segment_inertias(mass_kg, height_m)
    I_joint = {j: joint_inertia(j, seg_data) for j in TARGET_JOINTS}
    for j in TARGET_JOINTS:
        print(f"[Kphi]   I_{j} = {I_joint[j]:.4f} kg*m^2 (Winter regression, chain={JOINT_CHAIN[j]})")

    raw_sensor = df[SENSOR_COLUMNS].values.astype("float32")
    norm_sensor = (raw_sensor - norm_stats["mean_sensor"]) / norm_stats["std_sensor"]
    sensor_t = torch.tensor(norm_sensor, dtype=torch.float32)
    print(f"[Kphi] Encoding student latents (LATENT_STRIDE={LATENT_STRIDE}, {len(df):,} rows -- this takes a while)...")
    series = encode_student_series(model, sensor_t, H, LATENT_STRIDE, MAX_ENCODE_BATCH, device)
    phase, freq_raw, amp, offset, frame_idx = (series["phase"], series["frequency"], series["amplitude"],
                                                series["offset"], series["frame_indices"])
    K = phase.shape[1]
    freq_hz = freq_raw * sample_rate / H

    dom_ch = int(amp.mean(axis=0).argmax())
    print(f"[Kphi] No gait-event logic exists in this repo (confirmed by search) -- falling back to "
          f"zero-crossings of channel {dom_ch}'s phase (highest mean amplitude) for stride segmentation.")
    ph = phase[:, dom_ch]
    crossings = np.flatnonzero((ph[:-1] < 0) & (ph[1:] >= 0)) + 1
    if len(crossings) < 3:
        raise SystemExit("[Kphi] Not enough phase cycles detected to pick a representative stride.")
    i_mid = len(crossings) // 2
    s0, s1 = crossings[i_mid], crossings[i_mid + 1]
    frame_s0, frame_s1 = int(frame_idx[s0]), int(frame_idx[s1])
    stride_frames = frame_s1 - frame_s0
    print(f"[Kphi] Representative stride: raw frames [{frame_s0},{frame_s1}] "
          f"({frame_s0/sample_rate:.2f}s-{frame_s1/sample_rate:.2f}s, {stride_frames} frames)")

    freq_hz_stride = freq_hz[s0:s1].mean(axis=0)     # "within one stride f barely changes" -- per brief
    freq_raw_stride = freq_hz_stride * H / sample_rate
    amp_stride, offset_stride, phase0 = amp[s0:s1].mean(axis=0), offset[s0:s1].mean(axis=0), phase[s0]
    print(f"[Kphi] f_i range across stride (min, max, %variation): " +
          "; ".join(f"ch{i}: {freq_hz[s0:s1,i].min():.3f}-{freq_hz[s0:s1,i].max():.3f}Hz "
                    f"({(freq_hz[s0:s1,i].max()-freq_hz[s0:s1,i].min())/max(freq_hz_stride[i],1e-6)*100:.1f}%)"
                    for i in range(K)))

    phase0_t = torch.tensor(phase0, dtype=torch.float32, device=device).unsqueeze(0)
    freq_t = torch.tensor(freq_raw_stride, dtype=torch.float32, device=device).unsqueeze(0)
    amp_t = torch.tensor(amp_stride, dtype=torch.float32, device=device).unsqueeze(0)
    offset_t = torch.tensor(offset_stride, dtype=torch.float32, device=device).unsqueeze(0)

    std_biomech, angle_idx = norm_stats["std_biomech"], [BIOMECH_COLUMNS.index(j) for j in TARGET_JOINTS]
    omega2 = (2.0 * np.pi * freq_hz_stride) ** 2
    n_bins = stride_frames
    K_pred = {j: np.zeros(n_bins) for j in TARGET_JOINTS}

    for h in range(n_bins):
        phase_h = phase0_t + freq_t * dt * h                       # decode()'s own phase_dynamics formula for step h
        func = lambda a: decode_angle_at(model, phase_h, freq_t, a, offset_t)
        J_norm = torch.autograd.functional.jacobian(func, amp_t).squeeze(1).detach().cpu().numpy()  # (10, K)
        J_rad = J_norm * std_biomech[:10][:, None] * (math.pi / 180.0)   # un-normalize, deg->rad
        for j_name, a_idx in zip(TARGET_JOINTS, angle_idx):
            row = J_rad[a_idx]                                      # (K,)
            m_i = row ** 2 * I_joint[j_name]
            K_pred[j_name][h] = float(np.sum(row ** 2 * m_i * omega2))

    x_pred = np.arange(n_bins) / (n_bins - 1) * 100.0

    fig, axes = plt.subplots(len(TARGET_JOINTS), 1, figsize=(11, 3.2 * len(TARGET_JOINTS)), sharex=True)
    print("\n" + "=" * 78 + "\nPREDICTED vs MEASURED QUASI-STIFFNESS\n" + "=" * 78)
    for ax, j_name in zip(np.atleast_1d(axes), TARGET_JOINTS):
        theta = np.radians(df[j_name].values[frame_s0:frame_s1])
        theta_dot = np.radians(df[f"{j_name}_velocity"].values[frame_s0:frame_s1])
        moment = df[f"{j_name}_moment"].values[frame_s0:frame_s1]
        n = len(theta)
        width, step = max(5, round(WINDOW_FRAC * n)), max(1, round(STEP_FRAC * n))
        x_meas, K_meas, n_cond, n_exc = [], [], 0, 0
        for s in range(0, n - width + 1, step):
            e = s + width
            th, thd, m = theta[s:e], theta_dot[s:e], moment[s:e]
            design = np.column_stack([th, thd, np.ones_like(th)])
            excursion_deg = np.degrees(th.max() - th.min())
            cond = np.linalg.cond(design)
            x_meas.append((s + width / 2.0) / n * 100.0)
            if cond > COND_THRESHOLD or excursion_deg < MIN_EXCURSION_DEG:
                K_meas.append(np.nan)
                n_cond += cond > COND_THRESHOLD
                n_exc += (cond <= COND_THRESHOLD) and (excursion_deg < MIN_EXCURSION_DEG)
                continue
            coeffs, *_ = np.linalg.lstsq(design, m, rcond=None)
            K_meas.append(-coeffs[0])
        x_meas, K_meas = np.array(x_meas), np.array(K_meas)

        pred_at_meas = np.interp(x_meas, x_pred, K_pred[j_name])
        valid = ~np.isnan(K_meas)
        scale = float(np.sum(pred_at_meas[valid] * K_meas[valid]) / np.sum(pred_at_meas[valid] ** 2)) if valid.any() else float("nan")
        r = float(np.corrcoef(pred_at_meas[valid], K_meas[valid])[0, 1]) if valid.sum() > 1 else float("nan")
        cv_pred = float(np.std(K_pred[j_name]) / np.mean(K_pred[j_name])) if np.mean(K_pred[j_name]) else float("nan")

        print(f"\n-- {j_name} --")
        print(f"  Pearson r (scaled predicted vs measured, valid bins): {r:.3f}")
        print(f"  Peak location: predicted={x_pred[np.argmax(K_pred[j_name])]:.1f}%stride  "
              f"measured={(x_meas[np.nanargmax(K_meas)] if valid.any() else float('nan')):.1f}%stride")
        print(f"  Predicted (unscaled): min={K_pred[j_name].min():.4g} max={K_pred[j_name].max():.4g} "
              f"mean={K_pred[j_name].mean():.4g}  CV={cv_pred:.3f}")
        if valid.any():
            print(f"  Measured: min={np.nanmin(K_meas):.4g} max={np.nanmax(K_meas):.4g} mean={np.nanmean(K_meas):.4g} N*m/(kg*rad)")
        print(f"  Ground-truth bins dropped: {n_cond} (condition number > {COND_THRESHOLD:g}), "
              f"{n_exc} (angular excursion < {MIN_EXCURSION_DEG} deg), of {len(K_meas)} total")

        ax.plot(x_pred, K_pred[j_name] * scale, color="tab:blue", lw=1.4, ls="-",
                label=f"Predicted (J*diag(m*omega^2)*J^T), scale x{scale:.3g}")
        ax.plot(x_meas, K_meas, color="tab:red", lw=1.4, ls="--", marker="o", ms=3,
                label="Measured (sliding-window fit, gaps = ill-conditioned)")
        ax.set_ylabel(f"{j_name}\nK (N*m/(kg*rad))", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("% stride")
    fig.suptitle(f"Predicted vs Measured Quasi-Stiffness -- {SUBJECT_NAME}/trial_{TRIAL_NUMBER} "
                 f"({frame_s0/sample_rate:.2f}s-{frame_s1/sample_rate:.2f}s stride)", fontsize=10, fontweight="bold")
    fig.tight_layout()

    # ------------------------------------------------------------------
    # OPTIMIZATION SWEEP: tau_hat(t) = K(phi(t)) * (theta_ref(t) - theta(t)),
    # no damping. K(phi) = the representative-stride curve above (K_pred),
    # looked up by the dominant channel's own phase at every t -- i.e.
    # ASSUMED STATIONARY across the whole trial (learned from one stride,
    # reused everywhere). Sweep theta_ref's forecast lookahead step over
    # 0..MAX_FORECAST_STEP, compute RMSE(tau_hat, measured moment) over the
    # WHOLE TRIAL for each step, per joint -- find which step minimizes it.
    # ------------------------------------------------------------------
    print(f"\n[Kphi] Optimization sweep: theta_ref lookahead step 0-{MAX_FORECAST_STEP}, "
          f"RMSE(tau_hat, measured moment) per step, whole trial. K(phi) assumed stationary "
          f"across the trial (learned from the representative stride above). No damping.")
    phi_pred_cycle = np.mod(phase0[dom_ch] + freq_raw_stride[dom_ch] * dt * np.arange(n_bins), 1.0)
    order = np.argsort(phi_pred_cycle)
    phi_sorted = phi_pred_cycle[order]
    phi_all = np.mod(phase[:, dom_ch], 1.0)
    t_full = frame_idx / sample_rate

    theta_ref_all_steps = batched_decode_all_steps(model, phase, freq_raw, amp, offset, MAX_FORECAST_STEP,
                                                    angle_idx, norm_stats["mean_biomech"], std_biomech,
                                                    device, MAX_ENCODE_BATCH)   # (T, MAX_FORECAST_STEP+1, n_joints)

    steps = np.arange(MAX_FORECAST_STEP + 1)
    rmse_by_joint, best_step, tau_hat_fixed = {}, {}, {}
    fig3, axes3 = plt.subplots(1, 1, figsize=(9, 5))
    for j_name in TARGET_JOINTS:
        K_of_phi = np.interp(phi_all, phi_sorted, K_pred[j_name][order], period=1.0)
        theta_now = df[j_name].values[frame_idx]
        moment_gt = df[f"{j_name}_moment"].values[frame_idx]
        j_pos = TARGET_JOINTS.index(j_name)

        rmse = np.array([
            np.sqrt(np.mean((K_of_phi * np.radians(theta_ref_all_steps[:, h, j_pos] - theta_now) - moment_gt) ** 2))
            for h in steps
        ])
        rmse_by_joint[j_name] = rmse
        best_step[j_name] = int(steps[np.argmin(rmse)])
        # Figure 2 uses THETA_REF_STEP (fixed, set from a prior sweep's printout) rather than
        # each fresh run's own best_step -- see THETA_REF_STEP's own comment.
        theta_ref_fixed = theta_ref_all_steps[:, THETA_REF_STEP, j_pos]
        tau_hat_fixed[j_name] = K_of_phi * np.radians(theta_ref_fixed - theta_now)

        print(f"  {j_name}: best step={best_step[j_name]} ({best_step[j_name]/sample_rate*1000:.0f}ms)  "
              f"RMSE={rmse.min():.4g}  (step 0 RMSE={rmse[0]:.4g}, step {MAX_FORECAST_STEP} RMSE={rmse[-1]:.4g}, "
              f"THETA_REF_STEP={THETA_REF_STEP} RMSE={rmse[THETA_REF_STEP]:.4g})")

        axes3.plot(steps, rmse, lw=1.4, marker=".", ms=3, label=j_name)
        axes3.plot(best_step[j_name], rmse.min(), marker="*", ms=14, color="black", zorder=5)
    axes3.set_xlabel("theta_ref forecast lookahead step")
    axes3.set_ylabel("RMSE(tau_hat, measured moment)  N*m/kg")
    axes3.set_title(f"theta_ref Lookahead Sweep -- {SUBJECT_NAME}/trial_{TRIAL_NUMBER} (* = best step per joint)")
    axes3.legend(fontsize=8)
    axes3.grid(alpha=0.2)
    fig3.tight_layout()

    # ---- Figure 2: predicted torque (THETA_REF_STEP, fixed) vs measured moment, whole trial ----
    fig2, axes2 = plt.subplots(len(TARGET_JOINTS), 1, figsize=(14, 3.0 * len(TARGET_JOINTS)), sharex=True)
    for ax2, j_name in zip(np.atleast_1d(axes2), TARGET_JOINTS):
        moment_gt = df[f"{j_name}_moment"].values[frame_idx]
        ax2.plot(t_full, moment_gt, color="tab:blue", lw=0.8, alpha=0.85, label="Ground truth moment")
        ax2.plot(t_full, tau_hat_fixed[j_name], color="tab:red", lw=0.8, alpha=0.85, ls="--",
                 label=f"Predicted tau_hat = K(phi)*(theta_ref-theta), theta_ref=step {THETA_REF_STEP}, B=0")
        ax2.set_ylabel(f"{j_name}\n(N*m/kg)", fontsize=8)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.grid(alpha=0.2)
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle(f"Predicted Impedance Torque (theta_ref = step {THETA_REF_STEP}) vs "
                  f"Ground-Truth Moment -- whole trial -- {SUBJECT_NAME}/trial_{TRIAL_NUMBER}",
                  fontsize=10, fontweight="bold")
    fig2.tight_layout()

    print("\n[Kphi] Displaying figures. Close the window(s), or Ctrl+C, to exit.")
    plt.show()


if __name__ == "__main__":
    main()
