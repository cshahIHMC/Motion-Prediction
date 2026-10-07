"""
diagnostic_plotter.py

Diagnostic plots and analysis for FLD latent vectors.

The FLD model represents each latent channel i as a sinusoid:
    x_i(t) = A_i * sin(2*pi*f_i*t + phi_i) + c_i

The latent vector stores this in phasor form (to avoid phase-wrapping
discontinuities). For N channels, z has shape (T, 4N), laid out per channel as:
    [A_i*cos(phi_i), A_i*sin(phi_i), f_i, c_i]

All functions below take this z / N convention. Auxiliary inputs (events,
torque, gait_speed, joint_angles) are optional where noted -- functions skip
the parts of the plot that need them if they're not provided.

Every plot function returns the matplotlib Figure (for compositing); most
also return whatever they computed, for reuse outside the plot itself.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import circmean, circstd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Helper: latent vector parser
# ---------------------------------------------------------------------------

def parse_latent(z, N):
    """
    z: (T, 4N) array, layout per channel = [A*cos(phi), A*sin(phi), f, c]
    N: number of channels

    Returns dict with keys 'A_cos', 'A_sin', 'f', 'c', 'A', 'phi', each (T, N).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.shape[1] != 4 * N:
        raise ValueError(f"parse_latent: expected z.shape[1] == 4*N == {4*N}, got {z.shape[1]}")
    A_cos = z[:, 0::4]
    A_sin = z[:, 1::4]
    f     = z[:, 2::4]
    c     = z[:, 3::4]
    A   = np.sqrt(A_cos ** 2 + A_sin ** 2)
    phi = np.arctan2(A_sin, A_cos)
    return {'A_cos': A_cos, 'A_sin': A_sin, 'f': f, 'c': c, 'A': A, 'phi': phi}


def _wrap_pi(angle):
    """Wrap angle(s) in radians to (-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def _chan_name(channel_names, idx):
    return channel_names[idx] if channel_names is not None else f"ch{idx}"


def _stride_windows(events, T):
    """Consecutive heel-strike frames -> list of (start, end) windows."""
    if not events or not events.get('heel_strike'):
        return []
    hs = sorted(i for i in events['heel_strike'] if 0 <= i < T)
    return [(hs[i], hs[i + 1]) for i in range(len(hs) - 1) if hs[i + 1] > hs[i]]


# ---------------------------------------------------------------------------
# Function 1: Phasor portrait (single channel)
# ---------------------------------------------------------------------------

def plot_phasor_portrait(z, N, channel_idx, events=None, channel_names=None):
    parsed = parse_latent(z, N)
    x = parsed['A_cos'][:, channel_idx]
    y = parsed['A_sin'][:, channel_idx]
    T = len(x)

    colors = np.full(T, "grey", dtype=object)
    if events:
        for idx in events.get('heel_strike', []):
            if 0 <= idx < T:
                colors[idx] = "red"
        for idx in events.get('toe_off', []):
            if 0 <= idx < T:
                colors[idx] = "blue"

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(x, y, "-", color="lightgrey", lw=0.5, alpha=0.6, zorder=1)
    ax.scatter(x, y, c=colors, s=8, alpha=0.75, zorder=2, edgecolors="none")

    mean_amp = parsed['A'][:, channel_idx].mean()
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mean_amp * np.cos(theta), mean_amp * np.sin(theta),
            "k--", lw=1.2, alpha=0.7)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='grey', label='other'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='red', label='heel strike'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='blue', label='toe off'),
        Line2D([0], [0], color='k', ls='--', label=f'mean |A|={mean_amp:.3f}'),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right")
    ax.set_xlabel("A·cosφ")
    ax.set_ylabel("A·sinφ")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Phasor Portrait — channel {channel_idx} ({_chan_name(channel_names, channel_idx)})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Function 2: Amplitude vs torque
# ---------------------------------------------------------------------------

def _norm01(x):
    x = np.asarray(x, dtype=np.float64)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def plot_amplitude_vs_torque(z, N, channel_idx, torque, events=None, channel_names=None):
    parsed = parse_latent(z, N)
    A = parsed['A'][:, channel_idx]
    T = len(A)
    A_n = _norm01(A)
    torque_n = _norm01(torque)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, A_n, color="steelblue", lw=1.3, label="Amplitude A(t) (norm.)")
    ax.plot(t, torque_n, color="tomato", lw=1.3, ls="--", label="Torque (norm.)")
    if events:
        for i, idx in enumerate(events.get('heel_strike', [])):
            if 0 <= idx < T:
                ax.axvline(idx, color="grey", lw=0.5, ls=":", alpha=0.5,
                           label="heel strike" if i == 0 else None)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalised [0, 1]")
    ax.set_title(f"Amplitude vs Torque — channel {channel_idx} ({_chan_name(channel_names, channel_idx)})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Function 3: Frequency vs gait speed (per stride)
# ---------------------------------------------------------------------------

def _linfit_r2(x, y):
    """Least-squares line + R^2. Returns (coeffs, r2) or (None, nan) if degenerate."""
    if len(x) < 2 or np.std(x) < 1e-9:
        return None, float("nan")
    coeffs = np.polyfit(x, y, 1)
    pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return coeffs, r2


def plot_frequency_vs_speed(z, N, channel_idx, gait_speed, events=None, channel_names=None):
    parsed = parse_latent(z, N)
    f = parsed['f'][:, channel_idx]
    gait_speed = np.asarray(gait_speed, dtype=np.float64)
    T = len(f)

    windows = _stride_windows(events, T)
    stride_f     = np.array([f[s:e].mean() for s, e in windows])
    stride_speed = np.array([gait_speed[s:e].mean() for s, e in windows])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(stride_speed, stride_f, color="steelblue", s=24, alpha=0.75)
    coeffs, r2 = _linfit_r2(stride_speed, stride_f)
    if coeffs is not None:
        xs = np.linspace(stride_speed.min(), stride_speed.max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), "r--", lw=1.5)
    ax.set_xlabel("Mean gait speed (per stride)")
    ax.set_ylabel("Mean frequency (per stride)")
    ax.set_title(f"Frequency vs Speed — channel {channel_idx} "
                 f"({_chan_name(channel_names, channel_idx)})  R²={r2:.3f}  n={len(windows)} strides")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Function 4: Offset vs mean joint angle (per stride)
# ---------------------------------------------------------------------------

def plot_offset_vs_mean_angle(z, N, channel_idx, joint_angles, events=None, channel_names=None):
    parsed = parse_latent(z, N)
    c = parsed['c'][:, channel_idx]
    joint_angles = np.asarray(joint_angles, dtype=np.float64)
    T = len(c)

    windows = _stride_windows(events, T)
    stride_c     = np.array([c[s:e].mean() for s, e in windows])
    stride_angle = np.array([joint_angles[s:e].mean() for s, e in windows])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(stride_c, stride_angle, color="seagreen", s=24, alpha=0.75)
    coeffs, r2 = _linfit_r2(stride_c, stride_angle)
    if coeffs is not None:
        xs = np.linspace(stride_c.min(), stride_c.max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), "r--", lw=1.5)
    ax.set_xlabel("Offset c")
    ax.set_ylabel("Mean joint angle (per stride)")
    ax.set_title(f"Offset vs Mean Joint Angle — channel {channel_idx} "
                 f"({_chan_name(channel_names, channel_idx)})  R²={r2:.3f}  n={len(windows)} strides")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Function 5: Inter-channel phase-difference matrix
# ---------------------------------------------------------------------------

def compute_phase_difference_matrix(z, N):
    """
    Returns (mean_phase_diff, std_phase_diff), each (N, N) in degrees.
    mean/std computed circularly (handles phase wraparound correctly).
    """
    parsed = parse_latent(z, N)
    phi = parsed['phi']  # (T, N) radians

    mean_phase_diff = np.zeros((N, N))
    std_phase_diff  = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dphi = _wrap_pi(phi[:, i] - phi[:, j])
            mean_phase_diff[i, j] = np.degrees(circmean(dphi, low=-np.pi, high=np.pi))
            std_phase_diff[i, j]  = np.degrees(circstd(dphi, low=-np.pi, high=np.pi))
    return mean_phase_diff, std_phase_diff


def plot_phase_difference_matrix(z, N, channel_names=None):
    mean_pd, std_pd = compute_phase_difference_matrix(z, N)
    names = channel_names if channel_names is not None else [f"ch{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(1.1 * N + 3, 1.0 * N + 2.5))
    im = ax.imshow(mean_pd, cmap="RdBu_r", vmin=-180, vmax=180)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{mean_pd[i, j]:.0f}°\n±{std_pd[i, j]:.0f}",
                    ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(N)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(names)
    fig.colorbar(im, ax=ax, label="Mean phase lag (deg)")
    ax.set_title("Inter-channel Phase Difference\n(mean ± circular std, degrees)")
    fig.tight_layout()
    return fig, mean_pd, std_pd


# ---------------------------------------------------------------------------
# Function 6: Amplitude ratio matrix
# ---------------------------------------------------------------------------

def compute_amplitude_ratio_matrix(z, N):
    """R_ij = mean(A_i) / mean(A_j) over the whole trial. Returns (N, N)."""
    parsed = parse_latent(z, N)
    mean_A = parsed['A'].mean(axis=0)
    return mean_A[:, None] / np.clip(mean_A[None, :], 1e-9, None)


def compute_amplitude_ratio_over_time(z, N, events=None, window=None):
    """
    R_ij per stride (if events with heel_strike given) else per fixed window.
    Returns (n_windows, N, N).
    """
    parsed = parse_latent(z, N)
    A = parsed['A']
    T = A.shape[0]

    windows = _stride_windows(events, T)
    if not windows:
        w = window or max(T // 20, 1)
        windows = [(s, min(s + w, T)) for s in range(0, T, w) if min(s + w, T) > s]

    R_t = np.array([
        (A[s:e].mean(axis=0)[:, None] / np.clip(A[s:e].mean(axis=0)[None, :], 1e-9, None))
        for s, e in windows
    ])
    return R_t


def plot_amplitude_ratio_matrix(z, N, channel_names=None):
    R = compute_amplitude_ratio_matrix(z, N)
    names = channel_names if channel_names is not None else [f"ch{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(1.1 * N + 3, 1.0 * N + 2.5))
    im = ax.imshow(R, cmap="viridis")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(N)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(names)
    fig.colorbar(im, ax=ax, label="R_ij = mean(A_i) / mean(A_j)")
    ax.set_title("Amplitude Ratio Matrix (whole trial)")
    fig.tight_layout()
    return fig, R


def plot_amplitude_ratio_over_time(z, N, channel_names=None, events=None, pairs=None):
    """pairs: list of (i, j) tuples; defaults to every i<j pair."""
    R_t = compute_amplitude_ratio_over_time(z, N, events=events)
    names = channel_names if channel_names is not None else [f"ch{i}" for i in range(N)]
    if pairs is None:
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    fig, ax = plt.subplots(figsize=(14, 5))
    for (i, j) in pairs:
        ax.plot(R_t[:, i, j], lw=1.0, alpha=0.85, label=f"{names[i]}/{names[j]}")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Stride / window index")
    ax.set_ylabel("Amplitude ratio")
    ax.set_title("Amplitude Ratio Over Time — watch pairs involving distal channels\n"
                 "(e.g. ankle) trend down relative to proximal channels as a fatigue signature")
    ax.legend(fontsize=6.5, ncol=min(len(pairs), 4), loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, R_t


# ---------------------------------------------------------------------------
# Function 7: Coordination quality (2D subspace via PCA on phasor coords)
# ---------------------------------------------------------------------------

def compute_coordination_quality(z, N):
    """
    Returns (coordination_quality, explained_variance_ratio).
    coordination_quality = fraction of phasor-space variance explained by
    the first 2 principal components. 1.0 = perfectly coordinated (planar).
    """
    parsed = parse_latent(z, N)
    Z_phasor = np.concatenate([parsed['A_cos'], parsed['A_sin']], axis=1)  # (T, 2N)
    n_components = min(2 * N, Z_phasor.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(Z_phasor - Z_phasor.mean(axis=0, keepdims=True))
    evr = pca.explained_variance_ratio_
    coordination_quality = float(evr[:2].sum())
    return coordination_quality, evr


def compute_coordination_quality_over_time(z, N, window=None, stride=None):
    T = z.shape[0]
    window = window or max(T // 10, 4 * N + 1)
    stride = stride or max(window // 2, 1)
    idxs, cq_t = [], []
    for s in range(0, max(T - window, 1), stride):
        cq, _ = compute_coordination_quality(z[s:s + window], N)
        idxs.append(s + window // 2)
        cq_t.append(cq)
    return np.array(idxs), np.array(cq_t)


def plot_coordination_quality(z, N):
    cq, evr = compute_coordination_quality(z, N)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(evr))
    colors = ["seagreen" if i < 2 else "lightgrey" for i in x]
    ax.bar(x, evr, color=colors)
    ax.axvline(1.5, color="k", ls="--", lw=1)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"Coordination Quality Scree Plot — quality (PC1+PC2) = {cq:.3f}")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    return fig, cq, evr


def plot_coordination_quality_over_time(z, N, window=None, stride=None):
    idxs, cq_t = compute_coordination_quality_over_time(z, N, window, stride)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(idxs, cq_t, color="seagreen", lw=1.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Coordination quality")
    ax.set_title("Coordination Quality Over Time")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, idxs, cq_t


# ---------------------------------------------------------------------------
# Function 8: Per-channel energy
# ---------------------------------------------------------------------------

def compute_channel_energy(z, N):
    """E_i(t) = A_i(t)^2 * f_i(t)^2. Returns (T, N)."""
    parsed = parse_latent(z, N)
    return (parsed['A'] ** 2) * (parsed['f'] ** 2)


def plot_channel_energy(z, N, events=None, channel_names=None):
    E = compute_channel_energy(z, N)
    names = channel_names if channel_names is not None else [f"ch{i}" for i in range(N)]
    T = E.shape[0]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.stackplot(t, E.T, labels=names, alpha=0.85)
    if events:
        for i, idx in enumerate(events.get('heel_strike', [])):
            if 0 <= idx < T:
                ax.axvline(idx, color="k", lw=0.4, ls=":", alpha=0.4,
                           label="heel strike" if i == 0 else None)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy (A²f²)")
    ax.set_title("Channel Energy Over Time")
    ax.legend(fontsize=7, loc="upper right", ncol=min(N, 4))
    fig.tight_layout()
    return fig, E


# ---------------------------------------------------------------------------
# Function 9: Cross-channel interaction energy (HKB coupling term)
# ---------------------------------------------------------------------------

def compute_cross_channel_interaction_energy(z, N):
    """
    E_cross(i,j)(t) = A_i(t)*A_j(t)*cos(phi_i(t) - phi_j(t)).
    Returns (E_cross (T,N,N), E_cross_total (T,) summed over i != j).
    """
    parsed = parse_latent(z, N)
    A, phi = parsed['A'], parsed['phi']
    T = A.shape[0]

    dphi = phi[:, :, None] - phi[:, None, :]          # (T, N, N)
    E_cross = A[:, :, None] * A[:, None, :] * np.cos(dphi)  # (T, N, N)

    off_diag = ~np.eye(N, dtype=bool)
    E_cross_total = E_cross[:, off_diag].sum(axis=1)  # (T,)
    return E_cross, E_cross_total


def plot_cross_channel_interaction(z, N, events=None, channel_names=None):
    E_cross, E_cross_total = compute_cross_channel_interaction_energy(z, N)
    names = channel_names if channel_names is not None else [f"ch{i}" for i in range(N)]
    mean_E = E_cross.mean(axis=0)

    fig1, ax1 = plt.subplots(figsize=(1.1 * N + 3, 1.0 * N + 2.5))
    im = ax1.imshow(mean_E, cmap="coolwarm")
    for i in range(N):
        for j in range(N):
            ax1.text(j, i, f"{mean_E[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax1.set_xticks(range(N)); ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_yticks(range(N)); ax1.set_yticklabels(names)
    fig1.colorbar(im, ax=ax1, label="Mean E_cross(i,j)")
    ax1.set_title("Mean Cross-Channel Interaction Energy\n(HKB coupling term)")
    fig1.tight_layout()

    T = E_cross_total.shape[0]
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(E_cross_total, color="darkorange", lw=1.1)
    if events:
        for i, idx in enumerate(events.get('heel_strike', [])):
            if 0 <= idx < T:
                ax2.axvline(idx, color="k", lw=0.4, ls=":", alpha=0.4,
                            label="heel strike" if i == 0 else None)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("E_cross_total(t)")
    ax2.set_title("Total Cross-Channel Interaction Energy Over Time\n"
                  "(peaks = mechanically expensive transitions)")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)
    fig2.tight_layout()

    return fig1, fig2, E_cross, E_cross_total


# ---------------------------------------------------------------------------
# Function 10: Total system energy
# ---------------------------------------------------------------------------

def compute_total_system_energy(z, N, lam=0.5):
    """E_total(t) = sum_i A_i^2 f_i^2 + lam * sum_{i!=j} A_i A_j cos(dphi_ij)."""
    E_kinetic = compute_channel_energy(z, N).sum(axis=1)
    _, E_cross_total = compute_cross_channel_interaction_energy(z, N)
    return E_kinetic + lam * E_cross_total


def plot_total_system_energy(z, N, lam=0.5, events=None):
    E_total = compute_total_system_energy(z, N, lam)
    T = len(E_total)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(E_total, color="purple", lw=1.3)
    if events:
        for i, idx in enumerate(events.get('heel_strike', [])):
            if 0 <= idx < T:
                ax.axvline(idx, color="red", lw=0.5, ls=":", alpha=0.5,
                           label="heel strike" if i == 0 else None)
        for i, idx in enumerate(events.get('toe_off', [])):
            if 0 <= idx < T:
                ax.axvline(idx, color="blue", lw=0.5, ls=":", alpha=0.5,
                           label="toe off" if i == 0 else None)
    ax.set_xlabel("Frame")
    ax.set_ylabel("E_total(t)")
    ax.set_title(f"Total System Energy (λ={lam}) — expect periodicity, "
                 f"peaks at loading, troughs at mid-swing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, E_total
