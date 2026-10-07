# Motion-Prediction — Project Summary

**Goal:** Learn a wearable-sensor-only latent motion representation that can drive a real-time hip-knee exoskeleton impedance controller (stiffness `K`, damping `B`, equilibrium angle `theta_eq`), trained/validated against lab-grade biomechanics data (motion capture-derived joint angles/moments, EMG) but deployable from IMUs + insoles alone.

This is a research codebase (no packaging, no tests-as-CI) — a chain of PyTorch models, training scripts, and diagnostic/plotting scripts, iterated on quickly, with prior scripts kept side-by-side rather than replaced.

## Pipeline, in order

1. **Data** — 4 subjects (`S01`–`S04`), ~12 trials each, in `Data/<subject>/trial_<n>_all_data.csv`. Each row is a time sample from a full-body IMU rig (pelvis/thigh/shank ×2 legs: accel, gyro, mag, quaternion), instrumented insoles (force + COP), joint angles/velocities, joint moments, and EMG. An `AMASS Dataset/` (CMU mocap) folder is also used for large-scale pretraining (`train_amass_fld.py`).
2. **`combine_data.py`** — stacks all subjects' per-trial CSVs into `Data/all_data.csv`, tagging rows with a `trial_id` so trial boundaries survive concatenation.
3. **`build_train_val_index.py`** — builds an 85/15 train/val split of window-start indices *within* each trial (several spread-out val segments per trial, not just the tail), guaranteeing no window crosses a trial boundary.
4. **Models** (`Models/`):
   - `FLD.py` — Fourier Latent Dynamics autoencoder: encodes a history window into a small set of latent channels, each parameterized as a sinusoid (phase/frequency/amplitude/offset), decodes back to (or forecasts forward) the observation space.
   - `FLD_teacher_student.py` — two encoders (privileged **teacher**, trained on biomechanics; deployable **student**, trained on wearable sensors only) writing into one *shared* Fourier latent space, decoded by one shared decoder. Built by composing real `FLD` instances rather than reimplementing their math.
   - `TCNN.py` / `PAE.py` — baseline temporal-convolutional and periodic-autoencoder models for comparison.
5. **Training scripts** (repo root, `train_*.py`) — `train_fld.py` (plain FLD), `train_tcnn.py` (TCN baseline), `train_fld_teacher_student.py` (teacher/student FLD with a teacher-reconstruction + student-reconstruction + latent-alignment loss), `train_impedance_head.py` / `_v2.py` (small MLP heads on top of a **frozen** teacher-student model that map the student latent + subject mass/height to per-joint `K`, `B`, `theta_eq`, supervised only through matching predicted vs. measured joint moments — no direct K/B labels exist). `_v2` additionally lets the head read off `theta_eq` from the model's own decoded future trajectory at a learned time index, instead of a heuristic residual around a fixed neutral angle.
6. **`impedance_controller_skeleton.py`** — end-to-end scaffold for the deployed controller (continuous latent extraction → angle forecast → impedance computation → plotting); the actual `K(t)`/`B(t)` derivation is still a stubbed TODO, everything around it is wired up.
7. **Diagnostics & plots** (root + `diagnostics/`) — a large set of one-off analysis scripts: latent-space PCA manifolds across FLD/TeacherStudent/TCNN (`plot_pca_manifold_train.py`), stiffness-vs-angle curves (`plot_stiffness_vs_angle*.py`), predicted-vs-measured impedance head outputs (`plot_impedance_head_predictions*.py`), error growth over forecast horizon (`plot_error_across_time.py`), EMG vs. joint moments (`plot_emg_vs_moments.py`), a "Locomotion Index" derived from latent phasor coherence (`locomotion_index.py`), and a check of whether an analytic `K = J·diag(mω²)·Jᵀ` stiffness estimate from the student latent matches measured quasi-stiffness (`diagnostics/kphi_vs_quasistiffness.py`).
8. **`runs/`** — timestamped checkpoint/log directories from past training and diagnostic runs (FLD, FLDTeacherStudent, TCNN, ImpedanceHead, ImpedanceHeadV2, ErrorAcrossTime), most recent dated 2026-08-16.

## Collected data, in detail

Per-subject IMU + insole + mocap + EMG recordings, ~150 Hz (median dt ≈ 0.00667 s), 181 columns per trial CSV (full-body IMU × 5 segments, bilateral instrumented insoles with force/COP/accel/gyro/quat, 8-channel EMG, force plates, 10 joint angles + 10 joint velocities + 10 joint moments, plus `subject_name`/`subject_weight`/`subject_height`/`task` metadata columns).

| Subject | Weight (kg) | Height (cm) | Trials | Total recorded time |
|---|---|---|---|---|
| S01 | 83.0 | 172 | 12 | ~49.2 min |
| S02 | 88.3 | 176 | 12 | ~65.1 min |
| S03 | 63.4 | 159 | 11 | ~61.3 min |
| S04 | 80.15 | 178 | 11 | ~55.2 min |

**~230.8 minutes (~3.8 hours) of recording across all 4 subjects.**

Each trial is labeled with a `task`. Tasks recorded per subject (S01's naming/coverage differs slightly from S02–S04, who follow a matched protocol):

- `standing` — quiet standing baseline (S01 has it twice: `trial_1` and `trial_1_2`)
- `walking_0_8_m_s`, `walking_1_2_m_s`, `walking_1_4_m_s` — level walking at fixed speeds
- `random` — mixed/self-selected speed walking (S02–S04 only; S01 instead has one `weighted_walk_1_0_m_s_10_lbs` trial)
- `squats`
- `sit_stand_squat_transitions`
- `backward_walking_0_6_m_s` (S01) / `backward_walking_mix_speed` (S02–S04)
- `incline_walking_5_deg_1_2_m_s`, `incline_walking_10_deg_1_2_m_s` (S01) / `incline_walking_5_deg_mix_speed`, `incline_walking_10_mix_speed` (S02–S04)
- `side_sway` (S01 only)

So the dataset spans quasi-static posture, level walking across a speed range, backward walking, incline walking, squatting/sit-to-stand transitions, and (S01 only) a loaded-walking and a side-sway condition — the variety `plot_pca_manifold_train.py`, `locomotion_index.py`, etc. rely on to check that the latent representation generalizes across gait regimes, not just steady-state level walking.

Note (corrected 2026-08-27): `combine_data.py`'s `SUBJECTS` list was briefly narrowed to `["S01"]` at one point; it currently reads `SUBJECTS = ["S01","S02","S03","S04"]`, matching `Data/all_data.csv` on disk (both last touched 2026-08-25). No action needed — just don't assume the old narrowed-to-S01 state without checking the file first, since it's changed at least once already.

## Repo conventions worth knowing
- **Reuse by composition, not reimplementation** — new scripts explicitly import and call into `Models/FLD.py`, `train_fld.py`'s loss functions, etc., rather than duplicating logic. Docstrings call this out and name exactly which lines are reused.
- **Additive, non-destructive iteration** — new variants (`train_fld_no_input_norm.py`, `_v2` scripts, `FLD_teacher_student.py`) are added alongside the originals rather than editing them in place, so older experiments stay runnable/comparable.
- **Every script's module docstring is a mini design doc** — explaining tensor shapes, latent conventions, and *why*, not just what.

## Known issues
- **`_find_checkpoint()` (defined in `plot_error_across_time.py`, reused by every diagnostic script including `manifold_controller.py`) sorts `runs/<prefix>_*` directories alphabetically, not by modification time.** In practice this hasn't caused a wrong-checkpoint pick yet: `runs/FLDTeacherStudent_20260814_151352` sorts after every `runs/FLDTeacherStudent_2026081*` sibling (correctly the newest by date, since all those names share the same `YYYYMMDD_HHMMSS` format), and the two newer `runs/FLDTeacherStudent_(Ang + Vel + GRF)_2026082*` runs sort *before* it regardless of date (`(` < `2` in ASCII) — but those are also invisible for an unrelated reason (see next item), so the bug hasn't bitten in practice. It would if two runs ever shared the plain `FLDTeacherStudent_YYYYMMDD_HHMMSS` naming convention out of date order, or if a differently-prefixed run sorted higher. Left as-is per explicit instruction when found (2026-08-27) — not fixed.
- **The GRF-variant checkpoints (`train_fld_teacher_student_grf.py`, `runs/FLDTeacherStudent_(Ang + Vel + GRF)_*`) save under a different checkpoint filename stem** (`fld_teacher_student_grf_epoch_*.pt` vs. the standard `fld_teacher_student_epoch_*.pt`), so `_find_checkpoint("FLDTeacherStudent", "fld_teacher_student", ...)` never finds them, at any sort order — they're effectively invisible to every script that uses the standard lookup. That variant's TEACHER encoder swaps the 10 joint-moment columns for 10 GRF/COP columns (see `train_fld_teacher_student_grf.py`'s own docstring), which matters if anyone ever wants "the encoder trained without GRF supervision" — that claim is only true of the *original* `FLDTeacherStudent`'s teacher branch, not this variant. Left alone per explicit instruction (2026-08-27) — not fixed, not merged into the standard lookup.

## Current git state (uncommitted, on `dev`)
Working changes to `Models/FLD.py`, `train_fld.py`, `train_tcnn.py`, `combine_data.py`, `build_train_val_index.py`, `locomotion_index.py`, plus many new untracked files (teacher-student model/loader/training/test, impedance-head training + plotting scripts, controller skeleton, diagnostics). Nothing has been committed since `90f32ae`.

## Ideas for "something interesting" with this
- A write-up/dashboard artifact walking through the FLD → teacher-student → impedance-head pipeline with the actual latent/PCA/stiffness plots pulled in.
- A diagram of the data → model → controller pipeline (useful for a thesis/paper figure).
- Summarizing what the `runs/` history shows across iterations (e.g. did error-across-time improve run over run?).
- Turning the impedance-head RMSE / stiffness-vs-quasistiffness diagnostics into a single comparison view across model variants (FLD vs TeacherStudent vs TCNN).
