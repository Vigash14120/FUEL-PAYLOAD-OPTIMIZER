"""
DQN Agent (discretised action space)
======================================
Because DQN requires a discrete action space, we discretise the continuous
action space into a grid:
  fuel_uplift_factor ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}  (8 levels)
  cargo_fraction     ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} (10 levels)
  ⟹  80 discrete actions total

Network: 4 → 256 → 256 → 80  (ReLU, batch-normalised hidden layers)
Training: Experience replay + ε-greedy exploration + target network
"""

import random
from collections import deque
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── Action discretisation ────────────────────────────────────────────────────
FUEL_LEVELS  = np.linspace(0.3, 1.0, 8)
CARGO_LEVELS = np.linspace(0.1, 1.0, 10)
ACTION_LIST  = [np.array([f, c], dtype=np.float32)
                for f, c in product(FUEL_LEVELS, CARGO_LEVELS)]
N_ACTIONS    = len(ACTION_LIST)   # 80


# ── Neural Network ───────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),       nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buf.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.stack(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.stack(ns)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(
        self,
        state_dim:     int   = 4,
        lr:            float = 1e-3,
        gamma:         float = 0.99,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     int   = 2_000,
        batch_size:    int   = 64,
        target_update: int   = 50,
        buffer_size:   int   = 50_000,
        device:        str   = "cpu",
    ):
        self.gamma         = gamma
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)
        self.steps         = 0

        self.policy_net = QNetwork(state_dim, N_ACTIONS).to(self.device)
        self.target_net = QNetwork(state_dim, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)
        self.loss_history: list[float] = []

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection; returns action index."""
        self.eps = self.eps_end + (self.eps - self.eps_end) * np.exp(
            -self.steps / self.eps_decay
        )
        self.steps += 1
        if random.random() < self.eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(s).argmax(dim=1).item())

    def action_to_continuous(self, action_idx: int) -> np.ndarray:
        return ACTION_LIST[action_idx]

    # ── Learning ──────────────────────────────────────────────────────────────
    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.push(state, action_idx, reward, next_state, float(done))

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = s.to(self.device)
        a  = a.to(self.device)
        r  = r.to(self.device)
        ns = ns.to(self.device)
        d  = d.to(self.device)

        q_vals  = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            target = r + self.gamma * next_q * (1 - d)

        loss = nn.SmoothL1Loss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        l = float(loss.item())
        self.loss_history.append(l)
        return l

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
