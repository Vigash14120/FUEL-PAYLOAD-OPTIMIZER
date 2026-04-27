"""
Synthetic Flight Data Generator
================================
Generates realistic flight log data with:
  - Passengers (Pax): 50–220
  - Weather severity: 0.0 (clear) – 1.0 (storm)
  - Fuel price (USD/kg): 0.5 – 1.5
  - Distance (km): 200 – 8000
  - Actual fuel burn (kg): derived from physics + noise
  - Cargo weight (kg): 500 – 15000
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ── Aircraft reference parameters (narrow-body) ──────────────────────────────
BASE_FUEL_PER_KM   = 4.5    # kg/km at cruise (empty aircraft)
PAX_FUEL_FACTOR    = 0.012  # extra kg per km per passenger
CARGO_FUEL_FACTOR  = 0.0003 # extra kg per km per kg of cargo
WEATHER_FUEL_SCALE = 0.15   # max 15 % extra burn due to weather
TICKET_PRICE       = 180.0  # USD per passenger (avg)
CARGO_REVENUE_KG   = 0.8    # USD per kg of cargo
MAX_PAX            = 220
MAX_CARGO_KG       = 20_000
MAX_FUEL_CAPACITY  = 25_000 # kg


def generate_flight(n: int = 10_000) -> pd.DataFrame:
    """Generate n synthetic flight records."""
    pax          = RNG.integers(50, MAX_PAX + 1, size=n)
    weather      = RNG.uniform(0.0, 1.0, size=n)          # 0=clear, 1=storm
    fuel_price   = RNG.uniform(0.5, 1.5, size=n)          # USD / kg
    distance     = RNG.uniform(200, 8_000, size=n)         # km
    cargo_kg     = RNG.uniform(500, 15_000, size=n)

    # Physics-based fuel burn
    fuel_burn = (
        BASE_FUEL_PER_KM * distance
        + PAX_FUEL_FACTOR  * pax     * distance
        + CARGO_FUEL_FACTOR * cargo_kg * distance
    ) * (1 + WEATHER_FUEL_SCALE * weather)
    fuel_burn += RNG.normal(0, 50, size=n)                 # operational noise
    fuel_burn = np.clip(fuel_burn, 100, MAX_FUEL_CAPACITY)

    # Financials
    revenue    = pax * TICKET_PRICE + cargo_kg * CARGO_REVENUE_KG
    fuel_cost  = fuel_burn * fuel_price
    profit     = revenue - fuel_cost

    df = pd.DataFrame({
        "pax":         pax.astype(float),
        "weather":     weather,
        "fuel_price":  fuel_price,
        "distance":    distance,
        "cargo_kg":    cargo_kg,
        "fuel_burn":   fuel_burn,
        "revenue":     revenue,
        "fuel_cost":   fuel_cost,
        "profit":      profit,
    })
    return df


def save_dataset(path: str = "data/flight_logs.csv", n: int = 10_000) -> pd.DataFrame:
    df = generate_flight(n)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data] Saved {len(df)} records → {path}")
    return df


if __name__ == "__main__":
    save_dataset()
