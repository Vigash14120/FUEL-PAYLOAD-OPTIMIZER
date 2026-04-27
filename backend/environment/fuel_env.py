"""
Fuel Payload Simulation Environment
=====================================
A Gymnasium-compatible environment modelling an airline dispatch decision.

State  : [pax_norm, weather, fuel_price_norm, distance_norm]
Action : [fuel_uplift_factor, cargo_fraction]  ∈ [0,1]²  (continuous Box)

Reward : Revenue - Fuel_Cost - Penalty
  - Revenue    = pax * TICKET_PRICE + cargo_loaded * CARGO_REVENUE_KG
  - Fuel_Cost  = actual_fuel_burn * fuel_price
  - Penalty    = large negative if fuel_uplift < actual_burn (fuel shortage)
                + small negative proportional to excess fuel (dead weight)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Constants ────────────────────────────────────────────────────────────────
TICKET_PRICE        = 180.0   # USD / pax
CARGO_REVENUE_KG    = 0.8     # USD / kg
MAX_PAX             = 220
MAX_CARGO_KG        = 20_000
MAX_FUEL_CAPACITY   = 25_000  # kg
MAX_DISTANCE        = 8_000   # km
MAX_FUEL_PRICE      = 1.5     # USD / kg
SHORTAGE_PENALTY    = 50_000  # USD — grounding / diversion cost
EXCESS_FUEL_PENALTY = 0.05    # USD / kg over minimum (dead weight cost)

RNG = np.random.default_rng(None)   # re-seeded per episode


class FuelPayloadEnv(gym.Env):
    """
    Continuous action space environment for fuel & cargo optimisation.

    Observation (4-D, all in [0,1]):
        [pax_norm, weather, fuel_price_norm, distance_norm]

    Action (2-D, each in [0,1]):
        [fuel_uplift_factor, cargo_fraction]
        fuel_uplift_kg  = fuel_uplift_factor  * MAX_FUEL_CAPACITY
        cargo_loaded_kg = cargo_fraction       * MAX_CARGO_KG
    """

    metadata = {"render_modes": []}

    def __init__(self, fuel_burn_predictor=None):
        super().__init__()
        self.fuel_burn_predictor = fuel_burn_predictor  # sklearn pipeline or None

        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4,  dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2,  dtype=np.float32),
            dtype=np.float32,
        )
        self._state_raw = None   # unnormalised flight params

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self._state_raw = {
            "pax":        float(rng.integers(50, MAX_PAX + 1)),
            "weather":    float(rng.uniform(0, 1)),
            "fuel_price": float(rng.uniform(0.5, MAX_FUEL_PRICE)),
            "distance":   float(rng.uniform(200, MAX_DISTANCE)),
        }
        obs = self._normalise(self._state_raw)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, 0.0, 1.0)
        fuel_uplift_factor, cargo_fraction = float(action[0]), float(action[1])

        fuel_uplift_kg  = fuel_uplift_factor * MAX_FUEL_CAPACITY
        cargo_loaded_kg = cargo_fraction      * MAX_CARGO_KG

        s = self._state_raw
        actual_burn = self._predict_fuel_burn(
            pax        = s["pax"],
            weather    = s["weather"],
            fuel_price = s["fuel_price"],
            distance   = s["distance"],
            cargo_kg   = cargo_loaded_kg,
        )

        # ── Revenue ──────────────────────────────────────────────────────────
        revenue = s["pax"] * TICKET_PRICE + cargo_loaded_kg * CARGO_REVENUE_KG

        # ── Fuel cost ────────────────────────────────────────────────────────
        fuel_used  = min(fuel_uplift_kg, actual_burn)   # can't burn more than loaded
        fuel_cost  = fuel_used * s["fuel_price"]

        # ── Penalties ────────────────────────────────────────────────────────
        penalty = 0.0
        if fuel_uplift_kg < actual_burn:
            # Fuel shortage — diversion / grounding cost
            shortage   = actual_burn - fuel_uplift_kg
            penalty   += SHORTAGE_PENALTY + shortage * 10.0

        excess_fuel = max(0.0, fuel_uplift_kg - actual_burn)
        penalty    += excess_fuel * EXCESS_FUEL_PENALTY   # dead weight

        reward = (revenue - fuel_cost - penalty) / 1_000.0   # scale to ~[-50, 50]

        info = {
            "pax":            s["pax"],
            "weather":        s["weather"],
            "fuel_price":     s["fuel_price"],
            "distance":       s["distance"],
            "fuel_uplift_kg": round(fuel_uplift_kg, 1),
            "cargo_kg":       round(cargo_loaded_kg, 1),
            "actual_burn":    round(actual_burn, 1),
            "revenue":        round(revenue, 2),
            "fuel_cost":      round(fuel_cost, 2),
            "penalty":        round(penalty, 2),
            "profit":         round(reward * 1_000.0, 2),
        }
        # Single-step episode
        terminated = True
        truncated  = False
        obs        = self._normalise(s)
        return obs, float(reward), terminated, truncated, info

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _normalise(self, s: dict) -> np.ndarray:
        return np.array([
            s["pax"]        / MAX_PAX,
            s["weather"],
            (s["fuel_price"] - 0.5) / (MAX_FUEL_PRICE - 0.5),
            s["distance"]   / MAX_DISTANCE,
        ], dtype=np.float32)

    def _predict_fuel_burn(
        self, pax, weather, fuel_price, distance, cargo_kg
    ) -> float:
        if self.fuel_burn_predictor is not None:
            import numpy as _np
            X = _np.array([[pax, weather, fuel_price, distance, cargo_kg]])
            return max(100.0, float(self.fuel_burn_predictor.predict(X)[0]))
        # Fallback physics model
        from data.generator import (
            BASE_FUEL_PER_KM, PAX_FUEL_FACTOR,
            CARGO_FUEL_FACTOR, WEATHER_FUEL_SCALE,
        )
        burn = (
            BASE_FUEL_PER_KM * distance
            + PAX_FUEL_FACTOR   * pax     * distance
            + CARGO_FUEL_FACTOR * cargo_kg * distance
        ) * (1 + WEATHER_FUEL_SCALE * weather)
        return float(np.clip(burn, 100, MAX_FUEL_CAPACITY))
