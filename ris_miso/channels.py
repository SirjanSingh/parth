import numpy as np


def generate_bs_ris_channel(N: int, M: int) -> np.ndarray:
    """H1: BS-to-RIS channel, shape (N, M), i.i.d. CN(0,1)."""
    return (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2)


def generate_ris_user_channel(N: int, K: int) -> np.ndarray:
    """H2: RIS-to-users channel, shape (N, K), i.i.d. CN(0,1)."""
    return (np.random.randn(N, K) + 1j * np.random.randn(N, K)) / np.sqrt(2)


def generate_channels(N: int, M: int, K: int):
    """One channel snapshot. Returns (H1, H2)."""
    H1 = generate_bs_ris_channel(N, M)   # (N, M)
    H2 = generate_ris_user_channel(N, K)  # (N, K)
    return H1, H2
