"""
Train DDPG for one fixed (M, K, N) setting and save the model.
Repeat for each N in N_values to build the DRL curve for Fig 5.

Usage (quick test, ~10 min on RTX 4050):
    python train_drl.py --N 64 --num_eps 200 --steps_ep 1000

Usage (paper scale, several hours per N):
    python train_drl.py --N 64

The trained model is saved to models/ddpg_M{M}_K{K}_N{N}.pt
Evaluation results saved to results/drl_M{M}_K{K}.npz (appended per N).
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))

from ris_miso import config as cfg
from ris_miso.environment  import RIS_MISO_Env
from ris_miso.ddpg         import DDPG
from ris_miso.replay_buffer import ReplayBuffer
from ris_miso.metrics      import build_phi, compute_effective_channel, compute_sum_rate
from ris_miso.channels     import generate_channels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--M",          type=int,   default=cfg.M)
    p.add_argument("--K",          type=int,   default=cfg.K)
    p.add_argument("--N",          type=int,   default=64,
                   help="Number of RIS elements for this training run")
    p.add_argument("--Pt_dB",      type=float, default=cfg.Pt_dB)
    p.add_argument("--sigma2",     type=float, default=cfg.sigma2)
    p.add_argument("--num_eps",    type=int,   default=cfg.num_episodes)
    p.add_argument("--steps_ep",   type=int,   default=cfg.steps_per_episode)
    p.add_argument("--batch",      type=int,   default=cfg.batch_size)
    p.add_argument("--buf",        type=int,   default=cfg.buffer_size)
    p.add_argument("--actor_lr",   type=float, default=cfg.actor_lr)
    p.add_argument("--critic_lr",  type=float, default=cfg.critic_lr)
    p.add_argument("--hidden_dim", type=int,   default=cfg.hidden_dim)
    p.add_argument("--seed",       type=int,   default=cfg.seed)
    p.add_argument("--eval_real",  type=int,   default=500,
                   help="Realizations for final evaluation")
    p.add_argument("--noise",      type=float, default=cfg.exploration_noise)
    return p.parse_args()


def evaluate(agent: DDPG, env: RIS_MISO_Env, num_realizations: int = 100) -> float:
    """Run trained policy on fresh channels; return average sum rate."""
    rates = []
    for _ in range(num_realizations):
        state = env.reset()
        # Run a few steps and take the last reward (policy has converged)
        for _ in range(20):
            action = agent.select_action(state)
            state, reward, _, _ = env.step(action)
        rates.append(reward)
    return float(np.mean(rates))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Pt     = 10 ** (args.Pt_dB / 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training DDPG  M={args.M}  K={args.K}  N={args.N}  Pt={args.Pt_dB} dB")
    print(f"Episodes={args.num_eps}  Steps/ep={args.steps_ep}")

    env = RIS_MISO_Env(args.M, args.N, args.K, Pt, args.sigma2)

    print(f"State dim: {env.state_dim}  Action dim: {env.action_dim}")

    agent = DDPG(
        state_dim   = env.state_dim,
        action_dim  = env.action_dim,
        M=args.M, K=args.K, N=args.N, Pt=Pt,
        actor_lr    = args.actor_lr,
        critic_lr   = args.critic_lr,
        actor_decay = cfg.actor_decay,
        critic_decay= cfg.critic_decay,
        device      = device,
        discount    = cfg.discount,
        tau         = cfg.tau,
        hidden_dim  = args.hidden_dim,
    )

    buffer = ReplayBuffer(env.state_dim, env.action_dim, max_size=args.buf)

    episode_rewards = []
    rng = np.random.default_rng(args.seed)

    for ep in range(args.num_eps):
        state = env.reset()
        ep_reward = 0.0

        for t in range(args.steps_ep):
            action = agent.select_action(state)
            # Add exploration noise
            noise  = rng.normal(0, args.noise, size=action.shape).astype(np.float32)
            action = np.clip(action + noise, -1.0, 1.0)

            next_state, reward, done, _ = env.step(action)
            buffer.add(state, action, next_state, reward, done)
            state      = next_state
            ep_reward += reward

            if buffer.size >= args.batch:
                agent.update(buffer, batch_size=args.batch)

            if done:
                state = env.reset()

        avg_r = ep_reward / args.steps_ep
        episode_rewards.append(avg_r)

        if (ep + 1) % max(1, args.num_eps // 20) == 0:
            print(f"  Ep {ep+1:4d}/{args.num_eps}  avg reward/step: {avg_r:.4f}")

    # ----- Save model -----
    os.makedirs("models", exist_ok=True)
    model_path = f"models/ddpg_M{args.M}_K{args.K}_N{args.N}.pt"
    agent.save(model_path)
    print(f"\nModel saved → {model_path}")

    # ----- Evaluate -----
    print(f"Evaluating on {args.eval_real} fresh realizations…")
    final_rate = evaluate(agent, env, num_realizations=args.eval_real)
    print(f"Final average sum rate: {final_rate:.4f} bits/s/Hz")

    # ----- Append result -----
    os.makedirs("results", exist_ok=True)
    result_path = f"results/drl_M{args.M}_K{args.K}.npz"

    if os.path.exists(result_path):
        old = dict(np.load(result_path, allow_pickle=True))
        Ns    = list(old["N_values"]) + [args.N]
        rates = list(old["drl"])      + [final_rate]
    else:
        Ns    = [args.N]
        rates = [final_rate]

    np.savez(result_path,
             N_values=np.array(Ns),
             drl=np.array(rates),
             M=args.M, K=args.K, Pt_dB=args.Pt_dB)
    print(f"Result appended → {result_path}")


if __name__ == "__main__":
    main()
