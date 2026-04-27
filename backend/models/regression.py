"""
Regression Model Pipeline
==========================
Trains SVM and Random Forest regressors to predict fuel_burn given:
  features: [pax, weather, fuel_price, distance, cargo_kg]
  target:   fuel_burn (kg)

The best model is persisted via joblib and used by the Simulation Environment
to estimate expected fuel consumption for any given flight scenario.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["pax", "weather", "fuel_price", "distance", "cargo_kg"]
TARGET   = "fuel_burn"
MODEL_DIR = Path("models/saved")


def build_pipelines() -> dict:
    """Return untrained sklearn pipelines for SVM and Random Forest."""
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=100,
                epsilon=50,
                gamma="scale",
            )),
        ]),
    }


def train(df: pd.DataFrame) -> dict:
    """Train both regressors; return metrics dict and save best model."""
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipelines = build_pipelines()
    metrics = {}
    best_name, best_score = None, float("inf")

    for name, pipe in pipelines.items():
        print(f"[models] Training {name}…")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        cv   = cross_val_score(pipe, X_train, y_train,
                               cv=5, scoring="neg_mean_absolute_error",
                               n_jobs=-1)
        cv_mae = -cv.mean()

        metrics[name] = {
            "mae":    round(float(mae), 2),
            "r2":     round(float(r2), 4),
            "cv_mae": round(float(cv_mae), 2),
        }
        print(f"  MAE={mae:.1f} kg  R²={r2:.4f}  CV-MAE={cv_mae:.1f}")
        joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")

        if mae < best_score:
            best_score = mae
            best_name  = name

    # Mark best
    best_pipe = joblib.load(MODEL_DIR / f"{best_name}.pkl")
    joblib.dump(best_pipe, MODEL_DIR / "best_model.pkl")
    print(f"[models] Best model: {best_name}  (MAE={best_score:.1f} kg)")
    return {"best": best_name, "metrics": metrics}


def load_best_model():
    """Load the persisted best regression model."""
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "No trained model found. Run /api/train first."
        )
    return joblib.load(path)


def predict_fuel_burn(
    model,
    pax: float,
    weather: float,
    fuel_price: float,
    distance: float,
    cargo_kg: float,
) -> float:
    """Predict fuel burn (kg) for a single flight scenario."""
    X = np.array([[pax, weather, fuel_price, distance, cargo_kg]])
    return float(model.predict(X)[0])
