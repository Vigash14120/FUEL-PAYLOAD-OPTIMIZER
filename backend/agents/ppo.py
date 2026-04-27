"""
PPO Agent (continuous action space — native)
=============================================
Proximal Policy Optimisation with:
  - Actor (policy network): outputs mean & log_std for each action dimension
  - Critic (value network): estimates V(s)
  - Clipped surrogate objective
  - Generalised Advantage Estimation (GAE)

State  : 4-D (from FuelPayloadEnv)
Action : 2-D continuous ∈ [0,1] (fuel_uplift_factor, cargo_fraction)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ── Networks ─────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Gaussian policy: outputs mean and log_std per action dimension."""
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
        )
        self.mean_head    = nn.Linear(256, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, x: torch.Tensor):
        h    = self.shared(x)
        mean = torch.sigmoid(self.mean_head(h))   # ∈ (0,1)
        std  = self.log_std_head.exp().clamp(1e-3, 1.0)
        return mean, std

    def get_dist(self, x: torch.Tensor) -> Normal:
        mean, std = self.forward(x)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),       nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values     = [], [], []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_advantages(self, last_value: float, gamma: float, lam: float):
        """GAE-λ advantage estimation."""
        rewards   = np.array(self.rewards, dtype=np.float32)
        dones     = np.array(self.dones,   dtype=np.float32)
        values    = np.array(self.values,  dtype=np.float32)
        T         = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else values[t + 1]
            delta    = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae      = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def get_tensors(self, advantages, returns, device):
        s   = torch.FloatTensor(np.stack(self.states)).to(device)
        a   = torch.FloatTensor(np.stack(self.actions)).to(device)
        lp  = torch.FloatTensor(self.log_probs).to(device)
        adv = torch.FloatTensor(advantages).to(device)
        ret = torch.FloatTensor(returns).to(device)
        return s, a, lp, adv, ret


# ── PPO Agent ────────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(
        self,
        state_dim:   int   = 4,
        action_dim:  int   = 2,
        lr:          float = 3e-4,
        gamma:       float = 0.99,
        lam:         float = 0.95,
        clip_eps:    float = 0.2,
        epochs:      int   = 10,
        batch_size:  int   = 64,
        vf_coef:     float = 0.5,
        ent_coef:    float = 0.01,
        device:      str   = "cpu",
    ):
        self.gamma      = gamma
        self.lam        = lam
        self.clip_eps   = clip_eps
        self.epochs     = epochs
        self.batch_size = batch_size
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.device     = torch.device(device)

        self.actor  = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer       = RolloutBuffer()
        self.loss_history: list[float] = []

    # ── Interact ──────────────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray):
        """Sample action from policy; return (action_np, log_prob_scalar, value_scalar)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist  = self.actor.get_dist(s)
        a     = dist.sample()
        a     = torch.clamp(a, 0.0, 1.0)
        lp    = dist.log_prob(a).sum(dim=-1)
        v     = self.critic(s)
        return (
            a.squeeze(0).cpu().numpy(),
            float(lp.item()),
            float(v.item()),
        )

    def store(self, state, action, log_prob, reward, done, value):
        self.buffer.store(state, action, log_prob, reward, done, value)

    # ── Update ────────────────────────────────────────────────────────────────
    def update(self, last_state: np.ndarray):
        with torch.no_grad():
            last_val = float(
                self.critic(
                    torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                ).item()
            )
        advantages, returns = self.buffer.compute_returns_advantages(
            last_val, self.gamma, self.lam
        )
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        s, a, old_lp, adv, ret = self.buffer.get_tensors(
            advantages, returns, self.device
        )
        total_loss = 0.0
        for _ in range(self.epochs):
            idx = torch.randperm(len(s))
            for start in range(0, len(s), self.batch_size):
                mb = idx[start: start + self.batch_size]
                sb, ab, old_lpb, advb, retb = s[mb], a[mb], old_lp[mb], adv[mb], ret[mb]

                dist  = self.actor.get_dist(sb)
                lp    = dist.log_prob(ab).sum(dim=-1)
                ratio = (lp - old_lpb).exp()
                s1    = ratio * advb
                s2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advb
                policy_loss = -torch.min(s1, s2).mean()

                val_pred = self.critic(sb)
                val_loss = nn.MSELoss()(val_pred, retb)

                ent_loss = -dist.entropy().mean()

                loss = policy_loss + self.vf_coef * val_loss + self.ent_coef * ent_loss
                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_actor.step()
                self.opt_critic.step()
                total_loss += float(loss.item())

        self.buffer.reset()
        avg_loss = total_loss / max(1, self.epochs)
        self.loss_history.append(avg_loss)
        return avg_loss

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
