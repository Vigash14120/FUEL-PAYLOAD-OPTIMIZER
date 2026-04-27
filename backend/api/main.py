"""
FastAPI Application — Fuel Payload Optimizer
=============================================
Endpoints:
  POST /api/train/regression          → Train SVM + Random Forest
  POST /api/train/rl                  → Train DQN or PPO agent
  POST /api/infer                     → Single-flight inference
  GET  /api/status                    → Pipeline status
  GET  /api/data/sample               → Sample flight records
  GET  /api/metrics                   → Latest training metrics
  WebSocket /ws/training              → Live training progress stream
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.generator      import generate_flight, save_dataset
from models.regression   import train as train_regression, load_best_model
from training.trainer    import (
    run_regression_training,
    run_dqn_training,
    run_ppo_training,
    run_inference,
)

app = FastAPI(title="Fuel Payload Optimizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "regression_trained": False,
    "dqn_trained":        False,
    "ppo_trained":        False,
    "regression_metrics": None,
    "dqn_results":        None,
    "ppo_results":        None,
    "training_running":   False,
}
_ws_clients: list[WebSocket] = []
_fuel_predictor = None


async def _broadcast(msg: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _sync_broadcast(msg: dict):
    """Thread-safe broadcast from a background thread."""
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_broadcast(msg))
    loop.close()


# ── Request / Response models ─────────────────────────────────────────────────

class RegressionTrainRequest(BaseModel):
    n_samples: int = Field(10_000, ge=1_000, le=100_000)


class RLTrainRequest(BaseModel):
    agent:      str = Field("DQN", pattern="^(DQN|PPO)$")
    n_episodes: int = Field(500, ge=50, le=5_000)


class InferRequest(BaseModel):
    agent:      str   = Field("DQN", pattern="^(DQN|PPO)$")
    pax:        float = Field(..., ge=1,   le=220)
    weather:    float = Field(..., ge=0.0, le=1.0)
    fuel_price: float = Field(..., ge=0.1, le=3.0)
    distance:   float = Field(..., ge=50,  le=15_000)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    return {
        "regression_trained": _state["regression_trained"],
        "dqn_trained":        _state["dqn_trained"],
        "ppo_trained":        _state["ppo_trained"],
        "training_running":   _state["training_running"],
    }


@app.get("/api/data/sample")
def data_sample(n: int = 20):
    path = Path("data/flight_logs.csv")
    if path.exists():
        df = pd.read_csv(path).sample(min(n, 200))
    else:
        df = generate_flight(n)
    return df.round(2).to_dict(orient="records")


@app.post("/api/train/regression")
async def train_regression_endpoint(
    req: RegressionTrainRequest,
    background_tasks: BackgroundTasks,
):
    if _state["training_running"]:
        return {"status": "error", "detail": "Training already running"}

    def _do():
        global _fuel_predictor
        _state["training_running"] = True
        try:
            result = run_regression_training(n_samples=req.n_samples)
            _state["regression_metrics"] = result
            _state["regression_trained"] = True
            _fuel_predictor = load_best_model()
            _sync_broadcast({"type": "regression_done", "data": result})
        except Exception as e:
            _sync_broadcast({"type": "error", "detail": str(e)})
        finally:
            _state["training_running"] = False

    background_tasks.add_task(threading.Thread(target=_do).start)
    return {"status": "started", "agent": "regression"}


@app.post("/api/train/rl")
async def train_rl_endpoint(
    req: RLTrainRequest,
    background_tasks: BackgroundTasks,
):
    if _state["training_running"]:
        return {"status": "error", "detail": "Training already running"}

    def _do():
        _state["training_running"] = True
        try:
            cb = lambda msg: _sync_broadcast({"type": "progress", "data": msg})

            if req.agent == "DQN":
                result = run_dqn_training(
                    n_episodes=req.n_episodes,
                    progress_cb=cb,
                    fuel_predictor=_fuel_predictor,
                )
                _state["dqn_results"] = result
                _state["dqn_trained"] = True
            else:
                result = run_ppo_training(
                    n_episodes=req.n_episodes,
                    progress_cb=cb,
                    fuel_predictor=_fuel_predictor,
                )
                _state["ppo_results"] = result
                _state["ppo_trained"] = True

            _sync_broadcast({"type": "rl_done", "data": result})
        except Exception as e:
            _sync_broadcast({"type": "error", "detail": str(e)})
        finally:
            _state["training_running"] = False

    background_tasks.add_task(threading.Thread(target=_do).start)
    return {"status": "started", "agent": req.agent}


@app.post("/api/infer")
def infer(req: InferRequest):
    try:
        result = run_inference(
            agent_type  = req.agent,
            pax         = req.pax,
            weather     = req.weather,
            fuel_price  = req.fuel_price,
            distance    = req.distance,
            fuel_predictor = _fuel_predictor,
        )
        return {"status": "ok", "result": result}
    except FileNotFoundError as e:
        return {"status": "error", "detail": str(e)}


@app.get("/api/metrics")
def metrics():
    return {
        "regression": _state["regression_metrics"],
        "dqn":        _state["dqn_results"],
        "ppo":        _state["ppo_results"],
    }


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/training")
async def ws_training(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()   # keep-alive ping
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
