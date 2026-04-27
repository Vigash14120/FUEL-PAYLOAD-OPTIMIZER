# Fuel Payload Optimizer — Deep Reinforcement Learning

A full-stack Deep RL system that optimizes airline fuel uplift and cargo allocation decisions using SVM/Random Forest regression models combined with DQN/PPO reinforcement learning agents.

## Architecture

```
Flight Data Logs → Regression Model (SVM/RF) → Simulation Environment ↔ Deep RL Agent (DQN/PPO)
```

### Components
- **`backend/`** — Python FastAPI server + ML pipeline
  - `data/` — Synthetic flight data generation
  - `models/` — SVM / Random Forest regression
  - `environment/` — RL Gym-style simulation environment
  - `agents/` — DQN and PPO agents (PyTorch)
  - `api/` — REST endpoints for training & inference
- **`frontend/`** — React + Vite dashboard
  - Live training metrics
  - Policy visualization
  - Flight scenario simulator

## Setup

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```
