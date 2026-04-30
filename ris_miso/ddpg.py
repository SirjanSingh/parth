"""
DDPG for RIS-MISO.

Actor output post-processing:
  G    : scale 2*M*K real values so Frobenius norm = sqrt(Pt)
  Phi  : normalise (cos, sin) pairs to unit circle
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 M: int, K: int, N: int, Pt: float,
                 hidden_dim: int = 256):
        super().__init__()
        self.M, self.K, self.N = M, K, N
        self.Pt = float(Pt)
        MK = M * K

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),          # raw output in (-1, 1)
        )
        self._MK = MK

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state.float())   # (B, action_dim)
        MK  = self._MK

        # --- Normalize G to satisfy power constraint ---
        G_part = raw[:, :2 * MK]                          # (B, 2*M*K)
        frob   = torch.sqrt(torch.sum(G_part ** 2, dim=1, keepdim=True) + 1e-8)
        G_norm = G_part * (self.Pt ** 0.5) / frob         # ||G||_F = sqrt(Pt)

        # --- Normalize each Phi element to unit modulus ---
        ph_re  = raw[:, 2 * MK: 2 * MK + self.N]         # (B, N)
        ph_im  = raw[:, 2 * MK + self.N:]                 # (B, N)
        norm   = torch.sqrt(ph_re ** 2 + ph_im ** 2 + 1e-8)
        ph_re_n = ph_re / norm
        ph_im_n = ph_im / norm

        return torch.cat([G_norm, ph_re_n, ph_im_n], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(self.ln1(self.l1(state.float())))
        sa = torch.tanh(self.ln2(self.l2(torch.cat([s, action], dim=1))))
        return self.l3(sa)


# ---------------------------------------------------------------------------
# DDPG agent
# ---------------------------------------------------------------------------

class DDPG:
    def __init__(self, state_dim, action_dim, M, K, N, Pt,
                 actor_lr, critic_lr, actor_decay, critic_decay,
                 device, discount=0.99, tau=1e-3, hidden_dim=256):
        self.device   = device
        self.discount = discount
        self.tau      = tau
        self.M, self.K, self.N = M, K, N

        # Actor
        self.actor        = Actor(state_dim, action_dim, M, K, N, Pt, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(),
                                             lr=actor_lr, weight_decay=actor_decay)

        # Critic
        self.critic        = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt    = torch.optim.Adam(self.critic.parameters(),
                                              lr=critic_lr, weight_decay=critic_decay)

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            a = self.actor(s).cpu().numpy().flatten()
        self.actor.train()
        return a

    # ------------------------------------------------------------------
    def update(self, replay_buffer, batch_size: int = 16):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + not_done * self.discount * target_Q

        current_Q   = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    # ------------------------------------------------------------------
    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "actor_opt":    self.actor_opt.state_dict(),
            "critic_opt":   self.critic_opt.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
