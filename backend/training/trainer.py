"""
Training Orchestrator
======================
Runs the full pipeline:
  1. Generate (or load) flight data
  2. Train regression model (SVM / Random Forest)
  3. Train RL agent (DQN or PPO) using the simulation environment
  4. Emit training progress via a callback
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.generator     import generate_flight, save_dataset
from models.regression  import train as train_regression, load_best_model
from environment.fuel_env import FuelPayloadEnv
from agents.dqn          import DQNAgent
from agents.ppo          import PPOAgent

CHECKPOINT_DIR = Path("agents/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ── Regression ────────────────────────────────────────────────────────────────

def run_regression_training(n_samples: int = 10_000) -> dict:
    """Generate data, train regression models, return metrics."""
    data_path = "data/flight_logs.csv"
    if not Path(data_path).exists():
        save_dataset(data_path, n=n_samples)
    df = pd.read_csv(data_path)
    return train_regression(df)


# ── DQN Training ─────────────────────────────────────────────────────────────

def run_dqn_training(
    n_episodes:    int  = 1_000,
    progress_cb=None,
    fuel_predictor=None,
) -> dict:
    env   = FuelPayloadEnv(fuel_burn_predictor=fuel_predictor)
    agent = DQNAgent()

    rewards_hist, profits_hist = [], []

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        action_idx = agent.select_action(state)
        action     = agent.action_to_continuous(action_idx)
        next_state, reward, done, _, info = env.step(action)

        agent.push(state, action_idx, reward, next_state, done)
        loss = agent.update()

        rewards_hist.append(reward)
        profits_hist.append(info["profit"])

        if ep % 50 == 0 or ep == n_episodes:
            avg_r = float(np.mean(rewards_hist[-50:]))
            avg_p = float(np.mean(profits_hist[-50:]))
            if progress_cb:
                progress_cb({
                    "agent":   "DQN",
                    "episode": ep,
                    "total":   n_episodes,
                    "avg_reward": round(avg_r, 4),
                    "avg_profit": round(avg_p, 2),
                    "epsilon":    round(agent.eps, 4),
                    "loss":       round(loss, 4) if loss is not None else None,
                })

    agent.save(str(CHECKPOINT_DIR / "dqn_final.pt"))
    return {
        "agent":       "DQN",
        "episodes":    n_episodes,
        "final_avg_reward": round(float(np.mean(rewards_hist[-100:])), 4),
        "final_avg_profit": round(float(np.mean(profits_hist[-100:])), 2),
        "rewards":     [round(r, 4) for r in rewards_hist],
        "profits":     [round(p, 2) for p in profits_hist],
    }


# ── PPO Training ─────────────────────────────────────────────────────────────

def run_ppo_training(
    n_episodes:    int  = 1_000,
    rollout_len:   int  = 32,
    progress_cb=None,
    fuel_predictor=None,
) -> dict:
    env   = FuelPayloadEnv(fuel_burn_predictor=fuel_predictor)
    agent = PPOAgent()

    rewards_hist, profits_hist = [], []
    step_count = 0

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _, info = env.step(action)

        agent.store(state, action, log_prob, reward, done, value)
        step_count += 1
        rewards_hist.append(reward)
        profits_hist.append(info["profit"])

        # Update every rollout_len steps
        if step_count % rollout_len == 0:
            agent.update(next_state)

        if ep % 50 == 0 or ep == n_episodes:
            avg_r = float(np.mean(rewards_hist[-50:]))
            avg_p = float(np.mean(profits_hist[-50:]))
            if progress_cb:
                progress_cb({
                    "agent":      "PPO",
                    "episode":    ep,
                    "total":      n_episodes,
                    "avg_reward": round(avg_r, 4),
                    "avg_profit": round(avg_p, 2),
                    "loss":       round(agent.loss_history[-1], 4)
                                  if agent.loss_history else None,
                })

    agent.save(str(CHECKPOINT_DIR / "ppo_final.pt"))
    return {
        "agent":            "PPO",
        "episodes":         n_episodes,
        "final_avg_reward": round(float(np.mean(rewards_hist[-100:])), 4),
        "final_avg_profit": round(float(np.mean(profits_hist[-100:])), 2),
        "rewards":          [round(r, 4) for r in rewards_hist],
        "profits":          [round(p, 2) for p in profits_hist],
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    agent_type: str,
    pax: float,
    weather: float,
    fuel_price: float,
    distance: float,
    fuel_predictor=None,
) -> dict:
    """Run a single-step inference with the trained agent."""
    env = FuelPayloadEnv(fuel_burn_predictor=fuel_predictor)
    # Manually set the environment state
    env._state_raw = {
        "pax": pax, "weather": weather,
        "fuel_price": fuel_price, "distance": distance,
    }
    state = env._normalise(env._state_raw)

    if agent_type.upper() == "DQN":
        agent = DQNAgent()
        ckpt  = CHECKPOINT_DIR / "dqn_final.pt"
        if not ckpt.exists():
            raise FileNotFoundError("DQN model not trained yet.")
        agent.load(str(ckpt))
        agent.eps = 0.0   # greedy
        idx    = agent.select_action(state)
        action = agent.action_to_continuous(idx)
    else:
        agent = PPOAgent()
        ckpt  = CHECKPOINT_DIR / "ppo_final.pt"
        if not ckpt.exists():
            raise FileNotFoundError("PPO model not trained yet.")
        agent.load(str(ckpt))
        action, _, _ = agent.select_action(state)

    _, reward, _, _, info = env.step(action)
    return info
