# ============================================================
# All parameters in one place. Modify here to change experiment.
# ============================================================

import numpy as np

# --- System (Part A: reproduce Fig 5) ---
M = 64          # BS antennas
K = 64          # Users  (change both to 256 for Part B)
Pt_dB = 20.0    # Transmit power in dB
sigma2 = 1.0    # Noise variance (normalized)
Pt = 10 ** (Pt_dB / 10)   # Linear transmit power = 100

# --- RIS sweep ---
N_values = [4, 8, 16, 32, 64]   # extend to 128 if compute allows

# --- Simulation ---
num_realizations = 500           # channel realizations for averaging
seed = 42

# --- Benchmark optimization ---
wmmse_max_iter = 50              # WMMSE inner iterations
phase_opt_steps = 200            # gradient steps for phase optimization
phase_opt_lr = 0.05              # Adam lr for phase optimization
num_outer_alternating = 5        # outer alternating iterations (WMMSE / FP+ZF)

# --- DDPG hyperparameters (from paper) ---
discount = 0.99
tau = 1e-3
buffer_size = 100_000
batch_size = 16
num_episodes = 5000              # paper; reduce to 200 for quick test
steps_per_episode = 20_000       # paper; reduce to 1000 for quick test
actor_lr = 1e-3
critic_lr = 1e-3
actor_decay = 1e-5
critic_decay = 1e-5
hidden_dim = 256                 # network width (fixed, not state-dim-dependent)
exploration_noise = 0.1          # Gaussian noise std added to actions during training
