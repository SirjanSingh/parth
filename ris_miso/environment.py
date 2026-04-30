"""
RL environment for the RIS-MISO system.

Episode:  one fixed channel realization (H1, H2).
Step:     agent outputs (G, phases); env returns sum rate as reward.
State:    [H1_re, H1_im, H2_re, H2_im, G_re, G_im, phase_re, phase_im]
Action:   [G_re, G_im, phase_re, phase_im]  (after actor normalisation)
"""

import numpy as np
from .channels import generate_channels
from .metrics import build_phi, compute_effective_channel, compute_sum_rate


class RIS_MISO_Env:
    def __init__(self, M: int, N: int, K: int, Pt: float, sigma2: float):
        self.M, self.N, self.K = M, N, K
        self.Pt, self.sigma2 = Pt, sigma2

        # state: 2*(N*M + N*K + M*K + N)
        self.state_dim  = 2 * (N * M + N * K + M * K + N)
        # action: 2*M*K (G) + 2*N (Phi real+imag on unit circle)
        self.action_dim = 2 * M * K + 2 * N

        self.H1 = self.H2 = self.G = self.phases = None

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.H1, self.H2 = generate_channels(self.N, self.M, self.K)
        # Initialise G (small random, power-normalised) and uniform phases
        G_raw = (np.random.randn(self.M, self.K) +
                 1j * np.random.randn(self.M, self.K)) / np.sqrt(2)
        power = np.real(np.trace(G_raw @ G_raw.conj().T))
        self.G = G_raw * np.sqrt(self.Pt / (power + 1e-12))
        self.phases = np.random.uniform(0, 2 * np.pi, self.N)
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """action: flat numpy (action_dim,)"""
        action = action.flatten()
        MK = self.M * self.K

        # Parse G
        G_re = action[:MK].reshape(self.M, self.K)
        G_im = action[MK:2 * MK].reshape(self.M, self.K)
        self.G = G_re + 1j * G_im

        # Parse phases from (cos, sin) representation
        ph_re = action[2 * MK:2 * MK + self.N]
        ph_im = action[2 * MK + self.N:]
        self.phases = np.arctan2(ph_im, ph_re)

        reward = self._compute_reward()
        return self._get_state(), reward, False, {}

    # ------------------------------------------------------------------
    def _compute_reward(self) -> float:
        Phi   = build_phi(self.phases)
        H_eff = compute_effective_channel(self.H1, self.H2, Phi)
        return compute_sum_rate(H_eff, self.G, self.sigma2)

    # ------------------------------------------------------------------
    def _get_state(self) -> np.ndarray:
        def split(X):
            return np.real(X).flatten(), np.imag(X).flatten()

        H1_re, H1_im = split(self.H1)
        H2_re, H2_im = split(self.H2)
        G_re,  G_im  = split(self.G)
        ph_re = np.cos(self.phases)
        ph_im = np.sin(self.phases)

        state = np.concatenate([H1_re, H1_im, H2_re, H2_im,
                                 G_re,  G_im,  ph_re, ph_im])
        # Whiten (helps network training)
        std = state.std()
        if std > 1e-8:
            state = (state - state.mean()) / std
        return state.astype(np.float32)
