"""
Generate Figure 5: sum rate vs N for all three methods.

Reads:
  results/benchmarks_M{M}_K{K}.npz   (WMMSE, FP+ZF)
  results/drl_M{M}_K{K}.npz          (DDPG)   [optional]

Usage:
    python plot_fig5.py                   # M=K=64 from config
    python plot_fig5.py --M 256 --K 256  # Part B
    python plot_fig5.py --no_drl         # benchmarks only (before DRL is trained)
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ris_miso import config as cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M",      type=int, default=cfg.M)
    p.add_argument("--K",      type=int, default=cfg.K)
    p.add_argument("--no_drl", action="store_true",
                   help="Plot benchmarks only (DRL not yet trained)")
    return p.parse_args()


def main():
    args = parse_args()
    M, K = args.M, args.K

    bench_path = f"results/benchmarks_M{M}_K{K}.npz"
    drl_path   = f"results/drl_M{M}_K{K}.npz"

    if not os.path.exists(bench_path):
        print(f"[ERROR] Benchmark file not found: {bench_path}")
        print("Run  python run_benchmarks.py  first.")
        return

    bench = np.load(bench_path)
    N_b   = bench["N_values"]
    wmmse = bench["wmmse"]
    fpzf  = bench["fpzf"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(N_b, wmmse, "s--", color="tab:blue",   label="WMMSE",     linewidth=1.8, markersize=7)
    ax.plot(N_b, fpzf,  "^--", color="tab:orange", label="FP with ZF",linewidth=1.8, markersize=7)

    if not args.no_drl and os.path.exists(drl_path):
        drl  = np.load(drl_path)
        N_d  = drl["N_values"]
        rate = drl["drl"]
        # Align to common N values
        common = np.intersect1d(N_b, N_d)
        idx_d  = np.isin(N_d, common)
        idx_b  = np.isin(N_b, common)
        ax.plot(N_d[idx_d], rate[idx_d], "o-", color="tab:green",
                label="DRL (DDPG)", linewidth=1.8, markersize=7)

    ax.set_xlabel("Number of RIS Elements $N$", fontsize=13)
    ax.set_ylabel("Sum Rate (bits/s/Hz)", fontsize=13)
    ax.set_title(f"Fig. 5 — Sum Rate vs $N$\n$M={M}$, $K={K}$, $P_t={cfg.Pt_dB}$ dB",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    os.makedirs("results", exist_ok=True)
    out = f"results/fig5_M{M}_K{K}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
