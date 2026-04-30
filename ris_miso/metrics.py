"""Core signal model and rate computation.

Signal model (no direct path):
    y_k = h_{k,2}^T @ Phi @ H1 @ G @ x + w_k

Effective channel (stacked):
    H_eff = H2^T @ Phi @ H1    shape: (K, M)

SINR_k = |h_k_eff @ g_k|^2 / (sum_{j!=k} |h_k_eff @ g_j|^2 + sigma2)
"""

import numpy as np


def build_phi(phases: np.ndarray) -> np.ndarray:
    """phases: (N,) radians. Returns (N, N) diagonal unit-modulus matrix."""
    return np.diag(np.exp(1j * phases))


def compute_effective_channel(H1: np.ndarray, H2: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """
    H1: (N, M)  H2: (N, K)  Phi: (N, N)
    Returns H_eff: (K, M)
    """
    return H2.T @ Phi @ H1          # (K,N)@(N,N)@(N,M) -> (K,M)


def compute_sum_rate(H_eff: np.ndarray, G: np.ndarray, sigma2: float) -> float:
    """
    H_eff: (K, M)   G: (M, K)   sigma2: scalar
    Returns sum rate in bits/s/Hz.
    """
    HG = H_eff @ G                              # (K, K): entry (k,j) = h_k_eff @ g_j
    signal_power = np.abs(np.diag(HG)) ** 2    # (K,)
    total_power   = np.sum(np.abs(HG) ** 2, axis=1)  # (K,)
    int_plus_noise = total_power - signal_power + sigma2
    sinr = signal_power / int_plus_noise
    return float(np.sum(np.log2(1.0 + sinr)))


def compute_sum_rate_from_parts(H1, H2, phases, G, sigma2):
    """Convenience wrapper used throughout benchmarks and training."""
    Phi = build_phi(phases)
    H_eff = compute_effective_channel(H1, H2, Phi)
    return compute_sum_rate(H_eff, G, sigma2)
