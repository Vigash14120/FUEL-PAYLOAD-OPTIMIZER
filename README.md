# ✈️ Fuel Payload Optimizer

[![GitHub stars](https://img.shields.io/github/stars/Vigash14120/FUEL-PAYLOAD-OPTIMIZER?style=for-the-badge)](https://github.com/Vigash14120/FUEL-PAYLOAD-OPTIMIZER/stargazers)
[![License](https://img.shields.io/github/license/Vigash14120/FUEL-PAYLOAD-OPTIMIZER?style=for-the-badge)](https://github.com/Vigash14120/FUEL-PAYLOAD-OPTIMIZER/blob/main/LICENSE)

A state-of-the-art Deep Reinforcement Learning (DRL) system designed to optimize airline fuel uplift and cargo allocation decisions. By combining **SVM/Random Forest regression** for fuel burn prediction with **DQN and PPO agents** for operational decision-making, the system maximizes flight profitability while maintaining strict safety margins.

---

## 🚀 Key Features

- **Hybrid AI Architecture**: Chained pipeline starting with high-accuracy regression models ($R^2 > 0.99$) feeding into DRL agents.
- **Deep RL Agents**: 
  - **DQN (Deep Q-Network)**: Discretized action grid with Experience Replay.
  - **PPO (Proximal Policy Optimization)**: Continuous action space with Actor-Critic architecture and GAE.
- **Gymnasium Simulation**: Custom flight simulation environment modeling revenue, fuel burn, and safety penalties.
- **Interactive Dashboard**: Real-time training progress visualization (WebSockets), metrics charting (Recharts), and per-flight AI inference.
- **Synthetic Data Pipeline**: Physics-grounded flight log generator with 10k+ records.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.13, FastAPI, PyTorch, scikit-learn, Gymnasium, Pandas, NumPy.
- **Frontend**: React 18, Vite, Recharts, Lucide Icons, CSS3 (Glassmorphism).
- **Protocol**: REST API + WebSockets for live telemetry.

---

## 📁 Project Structure

```text
├── backend/
│   ├── agents/         # DQN and PPO PyTorch implementations
│   ├── api/            # FastAPI REST + WebSocket endpoints
│   ├── data/           # Synthetic data generation logic
│   ├── environment/    # Custom Gymnasium flight environment
│   ├── models/         # SVM & Random Forest regression pipeline
│   └── training/       # Orchestrator for training tasks
├── frontend/
│   ├── src/
│   │   ├── pages/      # Training, Metrics, Inference, and Data views
│   │   ├── api.js      # API client definitions
│   │   └── App.jsx     # Main routing and layout
└── README.md
```

---

## 🚦 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Vigash14120/FUEL-PAYLOAD-OPTIMIZER.git
cd FUEL-PAYLOAD-OPTIMIZER
```

### 2. Setup Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

### 3. Setup Frontend
```bash
cd ../frontend
npm install
npm run dev
```

---

## 📊 Methodology

1. **Regression Stage**: Train SVM or Random Forest on flight logs to learn the relationship between payload, distance, weather, and fuel burn.
2. **Simulation Stage**: Wrap the best regression model in a `Gymnasium` environment to calculate rewards based on flight profit.
3. **RL Stage**: Train agents to find the optimal balance of fuel uplift (to avoid $50k shortage penalties) and cargo weight (to maximize revenue).

---

## 📈 Indicative Results

| Metric | Random Forest (Reg) | PPO Agent (RL) |
|---|---|---|
| **Accuracy / Profit** | $R^2 > 0.99$ | $~26,000 / flight |
| **Safety** | MAE < 60 kg | < 2% Shortage Rate |
| **Convergence** | < 1 min | ~250 Episodes |

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---

*Developed as part of the Fuel Payload Optimization project using Deep Reinforcement Learning.*
