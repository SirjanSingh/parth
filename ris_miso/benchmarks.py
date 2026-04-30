"""
Benchmark methods:
  1. WMMSE  – alternating: WMMSE for G (fixed Phi) + gradient ascent for Phi (fixed G)
  2. FP+ZF  – alternating: ZF for G (fixed Phi)    + gradient ascent for Phi (fixed G)

Phase optimization uses PyTorch gradient ascent (runs on CPU, GPU if available).
"""

import numpy as np
import torch

from .metrics import (
    build_phi,
    compute_effective_channel,
    compute_sum_rate,
)


# ---------------------------------------------------------------------------
# Beamformer helpers
# ---------------------------------------------------------------------------

def _normalize_beamformer(G: np.ndarray, Pt: float) -> np.ndarray:
    power = np.real(np.trace(G @ G.conj().T))
    return G * np.sqrt(Pt / (power + 1e-12))


def zf_beamformer(H_eff: np.ndarray, Pt: float) -> np.ndarray:
    """Zero-Forcing: G = H_eff^H (H_eff H_eff^H)^{-1}, normalized."""
    try:
        G = H_eff.conj().T @ np.linalg.inv(H_eff @ H_eff.conj().T)   # (M, K)
    except np.linalg.LinAlgError:
        M, K = H_eff.shape[1], H_eff.shape[0]
        G = (np.random.randn(M, K) + 1j * np.random.randn(M, K)) / np.sqrt(2)
    return _normalize_beamformer(G, Pt)


def _solve_with_mu(lam: np.ndarray, U: np.ndarray, Uhr: np.ndarray,
                   norms_sq: np.ndarray, Pt: float) -> np.ndarray:
    """
    Bisect on mu using the eigendecomposition of A (precomputed once per WMMSE iter).
    A = U @ diag(lam) @ U^H  →  (A + mu*I)^{-1} = U @ diag(1/(lam+mu)) @ U^H

    tr(G G^H) = sum_i  norms_sq[i] / (lam[i] + mu)^2

    This replaces 60 × np.linalg.solve calls with 60 scalar evaluations.
    Returns G = (A + mu*I)^{-1} @ rhs.
    """
    lo, hi = 0.0, 1e8
    for _ in range(60):
        mu = (lo + hi) / 2.0
        power = np.sum(norms_sq / (lam + mu) ** 2)
        if power > Pt:
            lo = mu
        else:
            hi = mu
        if hi - lo < 1e-4:
            break
    mu_star = (lo + hi) / 2.0
    return U @ (Uhr / np.maximum(lam + mu_star, 1e-12)[:, None])


def wmmse_beamformer(H_eff: np.ndarray, Pt: float, sigma2: float,
                     num_iter: int = 50) -> np.ndarray:
    """
    WMMSE algorithm for downlink MU-MISO, fixed H_eff.
    Returns G (M, K) satisfying tr(GG^H) <= Pt.
    """
    K, M = H_eff.shape

    # Init with ZF (already power-normalized)
    G = zf_beamformer(H_eff, Pt)

    for _ in range(num_iter):
        HG = H_eff @ G                                    # (K, K)
        total_rx = np.sum(np.abs(HG) ** 2, axis=1) + sigma2  # (K,)

        # MMSE receiver scalars
        u = np.diag(HG) / (total_rx + 1e-12)             # (K,)

        # MSE weights
        e = np.real(1.0 - np.conj(u) * np.diag(HG))
        e = np.maximum(e, 1e-8)
        w = 1.0 / e                                       # (K,)

        # Build Hessian A and RHS — vectorised (no Python loop over K)
        wu2 = w * np.abs(u) ** 2                                  # (K,)
        A   = (H_eff * wu2[:, None]).conj().T @ H_eff             # (M, M)
        rhs = H_eff.conj().T * (w * np.conj(u))[np.newaxis, :]   # (M, K)

        # Eigendecompose A once; bisect in O(M) per iter instead of O(M^3)
        lam, U = np.linalg.eigh(A)                               # (M,), (M,M)
        lam    = np.maximum(lam, 1e-12)                          # numerical safety
        Uhr    = U.conj().T @ rhs                                # (M, K)
        norms_sq = np.sum(np.abs(Uhr) ** 2, axis=1)             # (M,)
        G = _solve_with_mu(lam, U, Uhr, norms_sq, Pt)

    return G


# ---------------------------------------------------------------------------
# Phase optimization via gradient ascent (PyTorch)
# ---------------------------------------------------------------------------

def _optimize_phases_gradient(
    H1: np.ndarray,
    H2: np.ndarray,
    G: np.ndarray,
    sigma2: float,
    N: int,
    num_steps: int = 200,
    lr: float = 0.05,
    device: str = "cpu",
) -> np.ndarray:
    """
    Maximize sum rate over RIS phase angles theta_n using Adam gradient ascent.
    Returns phases (N,) in radians.
    """
    dev = torch.device(device)

    H1_t = torch.tensor(H1, dtype=torch.complex64, device=dev)  # (N, M)
    H2_t = torch.tensor(H2, dtype=torch.complex64, device=dev)  # (N, K)
    G_t  = torch.tensor(G,  dtype=torch.complex64, device=dev)  # (M, K)
    s2   = torch.tensor(sigma2, dtype=torch.float32, device=dev)

    # Learnable phases
    theta = torch.nn.Parameter(
        torch.rand(N, device=dev) * 2 * np.pi
    )
    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        phi = torch.exp(1j * theta.to(torch.complex64))   # (N,)
        Phi = torch.diag(phi)                              # (N, N)
        H_eff = H2_t.T @ Phi @ H1_t                       # (K, M)
        HG = H_eff @ G_t                                   # (K, K)

        sig_pwr  = torch.abs(torch.diagonal(HG)) ** 2     # (K,)
        tot_pwr  = torch.sum(torch.abs(HG) ** 2, dim=1)   # (K,)
        int_n    = tot_pwr - sig_pwr + s2
        sinr     = sig_pwr / int_n
        rate     = torch.sum(torch.log2(1.0 + sinr))

        (-rate).backward()                                 # ascent -> negate
        optimizer.step()

    return theta.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Main benchmark pipelines
# ---------------------------------------------------------------------------

def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_wmmse(H1: np.ndarray, H2: np.ndarray, Pt: float, sigma2: float,
              num_outer: int = 5, wmmse_iter: int = 50,
              phase_steps: int = 200, phase_lr: float = 0.05) -> float:
    """
    Alternating: WMMSE for G  ↔  gradient ascent for phases.
    Returns final sum rate.
    """
    N = H1.shape[0]
    device = _get_device()
    phases = np.random.uniform(0, 2 * np.pi, N)

    for _ in range(num_outer):
        Phi   = build_phi(phases)
        H_eff = compute_effective_channel(H1, H2, Phi)
        G     = wmmse_beamformer(H_eff, Pt, sigma2, num_iter=wmmse_iter)
        phases = _optimize_phases_gradient(H1, H2, G, sigma2, N,
                                           num_steps=phase_steps, lr=phase_lr,
                                           device=device)

    Phi   = build_phi(phases)
    H_eff = compute_effective_channel(H1, H2, Phi)
    return compute_sum_rate(H_eff, G, sigma2)


def run_fp_zf(H1: np.ndarray, H2: np.ndarray, Pt: float, sigma2: float,
              num_outer: int = 5, phase_steps: int = 200,
              phase_lr: float = 0.05) -> float:
    """
    Alternating: ZF for G  ↔  gradient ascent for phases (FP-style).
    Returns final sum rate.
    """
    N = H1.shape[0]
    device = _get_device()
    phases = np.random.uniform(0, 2 * np.pi, N)

    for _ in range(num_outer):
        Phi   = build_phi(phases)
        H_eff = compute_effective_channel(H1, H2, Phi)
        G     = zf_beamformer(H_eff, Pt)
        phases = _optimize_phases_gradient(H1, H2, G, sigma2, N,
                                           num_steps=phase_steps, lr=phase_lr,
                                           device=device)

    Phi   = build_phi(phases)
    H_eff = compute_effective_channel(H1, H2, Phi)
    return compute_sum_rate(H_eff, G, sigma2)
