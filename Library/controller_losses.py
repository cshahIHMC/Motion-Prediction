"""
controller_losses.py

Differentiable (torch) training losses on FLD latent vectors, for training an
exoskeleton controller against energy/regularity/coordination objectives.

Not diagnostic plots -- see Library/diagnostic_plotter.py for the numpy-based
analysis/plotting side of the same latent representation. Kept in a separate
module because these are loss functions meant to sit inside a training loop
with gradients flowing back through z, not visualization tools.

Same latent convention as diagnostic_plotter: z has shape (T, 4N) for N
channels, laid out per channel as [A*cos(phi), A*sin(phi), f, c].
Every function here expects a 2D (T, 4N) torch tensor (unbatched) with
requires_grad as needed by the caller -- "time" is always dim 0.

Unit note: phase angles here are in RADIANS internally (torch.atan2 output).
Library/diagnostic_plotter.compute_phase_difference_matrix returns degrees
for readability in plots -- convert with np.radians(...) before passing a
diagnostic-derived reference_phase_diffs into compute_coordination_loss.
"""

import torch


def _parse_latent_torch(z, N):
    """z: (T, 4N) torch tensor. Returns (A_cos, A_sin, f, c, A, phi), each (T, N)."""
    if z.shape[-1] != 4 * N:
        raise ValueError(f"_parse_latent_torch: expected last dim 4*N={4*N}, got {z.shape[-1]}")
    A_cos = z[..., 0::4]
    A_sin = z[..., 1::4]
    f     = z[..., 2::4]
    c     = z[..., 3::4]
    A   = torch.sqrt(A_cos ** 2 + A_sin ** 2 + 1e-12)
    phi = torch.atan2(A_sin, A_cos)
    return A_cos, A_sin, f, c, A, phi


# ---------------------------------------------------------------------------
# Function 11: Energy loss
# ---------------------------------------------------------------------------

def compute_energy_loss(z, N, lambda_cross=0.5):
    """
    L_energy = mean_t[ sum_i A_i(t)^2*f_i(t)^2 + lambda * sum_{i!=j} A_i(t)*A_j(t)*cos(dphi_ij(t)) ]

    z: (T, 4N) torch tensor, grad-enabled as needed.
    Returns a scalar tensor.
    """
    _, _, f, _, A, phi = _parse_latent_torch(z, N)

    E_kinetic = (A ** 2 * f ** 2).sum(dim=-1)                    # (T,)

    dphi = phi.unsqueeze(-1) - phi.unsqueeze(-2)                 # (T, N, N)
    E_cross_mat = A.unsqueeze(-1) * A.unsqueeze(-2) * torch.cos(dphi)  # (T, N, N)
    off_diag = 1.0 - torch.eye(N, device=z.device, dtype=z.dtype)
    E_cross = (E_cross_mat * off_diag).sum(dim=(-2, -1))         # (T,)

    return (E_kinetic + lambda_cross * E_cross).mean()


# ---------------------------------------------------------------------------
# Function 12: Regularity loss
# ---------------------------------------------------------------------------

def compute_regularity_loss(z, N):
    """
    L_regularity = mean_i[ Var_t(A_i) + Var_t(f_i) ]

    Penalises amplitude/frequency variation within the window -- a healthy
    stride has near-constant amplitude and frequency.
    z: (T, 4N) torch tensor, time is dim 0.
    Returns a scalar tensor.
    """
    _, _, f, _, A, _ = _parse_latent_torch(z, N)
    var_A = A.var(dim=0, unbiased=False)   # (N,)
    var_f = f.var(dim=0, unbiased=False)   # (N,)
    return (var_A + var_f).mean()


# ---------------------------------------------------------------------------
# Function 13: Coordination loss
# ---------------------------------------------------------------------------

def compute_coordination_loss(z, N, reference_phase_diffs):
    """
    L_coordination = sum_{i,j} mean_t[ circular_diff(dphi_ij(t), dphi_ij_ref) ^2 ]

    reference_phase_diffs: (N, N) array/tensor of target phase lags, in
    RADIANS (see module docstring re: converting from diagnostic_plotter's
    degree-valued output).
    z: (T, 4N) torch tensor.
    Returns a scalar tensor.
    """
    _, _, _, _, _, phi = _parse_latent_torch(z, N)          # phi: (T, N)
    dphi = phi.unsqueeze(-1) - phi.unsqueeze(-2)             # (T, N, N)

    ref = torch.as_tensor(reference_phase_diffs, dtype=z.dtype, device=z.device)  # (N, N)
    diff = dphi - ref.unsqueeze(0)                           # (T, N, N)
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))  # circular difference

    return (diff_wrapped ** 2).mean(dim=0).sum()


# ---------------------------------------------------------------------------
# Function 14: Combined controller loss
# ---------------------------------------------------------------------------

def compute_total_controller_loss(z, N, reference_phase_diffs,
                                   alpha=1.0, beta=1.0, lambda_cross=0.5):
    """
    L_total = L_energy + alpha * L_regularity + beta * L_coordination

    Returns (L_total, components) where components is a dict with keys
    'energy', 'regularity', 'coordination' (each a scalar tensor) for logging.
    """
    L_energy       = compute_energy_loss(z, N, lambda_cross=lambda_cross)
    L_regularity   = compute_regularity_loss(z, N)
    L_coordination = compute_coordination_loss(z, N, reference_phase_diffs)

    L_total = L_energy + alpha * L_regularity + beta * L_coordination
    components = {
        "energy": L_energy,
        "regularity": L_regularity,
        "coordination": L_coordination,
    }
    return L_total, components
