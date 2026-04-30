"""
Evaluate WMMSE and FP+ZF baselines across N values.
Saves results/benchmarks_M{M}_K{K}.npz

Usage:
    python run_benchmarks.py                    # uses config defaults (M=K=64)
    python run_benchmarks.py --M 256 --K 256   # Part B
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))

from ris_miso import config as cfg
from ris_miso.channels import generate_channels
from ris_miso.benchmarks import run_wmmse, run_fp_zf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M",            type=int,   default=cfg.M)
    p.add_argument("--K",            type=int,   default=cfg.K)
    p.add_argument("--Pt_dB",        type=float, default=cfg.Pt_dB)
    p.add_argument("--sigma2",       type=float, default=cfg.sigma2)
    p.add_argument("--num_real",     type=int,   default=cfg.num_realizations)
    p.add_argument("--seed",         type=int,   default=cfg.seed)
    p.add_argument("--outer_iter",   type=int,   default=cfg.num_outer_alternating)
    p.add_argument("--wmmse_iter",   type=int,   default=cfg.wmmse_max_iter)
    p.add_argument("--phase_steps",  type=int,   default=cfg.phase_opt_steps)
    p.add_argument("--phase_lr",     type=float, default=cfg.phase_opt_lr)
    p.add_argument("--N_values",     type=int,   nargs="+", default=cfg.N_values)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    Pt = 10 ** (args.Pt_dB / 10)

    print(f"=== Benchmarks  M={args.M}  K={args.K}  Pt={args.Pt_dB} dB ===")
    print(f"    N sweep: {args.N_values}   realizations: {args.num_real}")

    wmmse_rates = []
    fpzf_rates  = []

    for N in args.N_values:
        r_wmmse, r_fpzf = [], []
        for _ in tqdm(range(args.num_real), desc=f"N={N:3d}", leave=False):
            H1, H2 = generate_channels(N, args.M, args.K)

            r_wmmse.append(run_wmmse(H1, H2, Pt, args.sigma2,
                                     num_outer=args.outer_iter,
                                     wmmse_iter=args.wmmse_iter,
                                     phase_steps=args.phase_steps,
                                     phase_lr=args.phase_lr))

            r_fpzf.append(run_fp_zf(H1, H2, Pt, args.sigma2,
                                     num_outer=args.outer_iter,
                                     phase_steps=args.phase_steps,
                                     phase_lr=args.phase_lr))

        mw, mf = np.mean(r_wmmse), np.mean(r_fpzf)
        wmmse_rates.append(mw)
        fpzf_rates.append(mf)
        print(f"  N={N:3d}  WMMSE={mw:.3f}  FP+ZF={mf:.3f}  bits/s/Hz")

    os.makedirs("results", exist_ok=True)
    out = f"results/benchmarks_M{args.M}_K{args.K}.npz"
    np.savez(out,
             N_values=np.array(args.N_values),
             wmmse=np.array(wmmse_rates),
             fpzf=np.array(fpzf_rates),
             M=args.M, K=args.K, Pt_dB=args.Pt_dB)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
