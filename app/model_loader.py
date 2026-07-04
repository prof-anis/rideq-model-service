"""
Loads XGBoost models from the models/ directory.

Supports both raw xgboost.Booster objects (saved via joblib or Booster.save_model)
and sklearn-wrapped XGBRegressor objects.

Drop real model files here:
  models/baseline.pkl     — trained WITHOUT event features
  models/with_events.pkl  — trained WITH event features (or same features, different training data)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import Booster, DMatrix

from app.features import validate_features, build_feature_row
from app.data_loader import load_rides, load_zone_stats

MODELS_DIR = Path(__file__).parent.parent / "models"

_baseline_model    = None
_events_model      = None
_baseline_features: list[str] = []
_events_features:   list[str] = []


class _BoosterWrapper:
    """Thin wrapper so both Booster and XGBRegressor expose the same .predict(df) interface."""

    def __init__(self, booster: Booster, feature_names: list[str]):
        self.booster       = booster
        self.feature_names = feature_names

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dmat = DMatrix(X[self.feature_names])
        return self.booster.predict(dmat)

    def get_feature_names(self) -> list[str]:
        return self.feature_names


def _extract_model_and_features(raw) -> tuple:
    """Return (wrapper, feature_names) regardless of model type."""
    # Raw Booster
    if isinstance(raw, Booster):
        names = raw.feature_names
        if not names:
            raise ValueError("Booster has no feature_names. Retrain with feature names set.")
        return _BoosterWrapper(raw, names), names

    # sklearn XGBRegressor
    booster = raw.get_booster()
    names   = booster.feature_names
    if not names:
        raise ValueError("XGBRegressor Booster has no feature_names.")
    return _BoosterWrapper(booster, names), names


def load_model(kind: str) -> tuple:
    """Returns (model_wrapper, feature_names). kind = 'baseline' | 'with_events'"""
    global _baseline_model, _events_model, _baseline_features, _events_features

    if kind == "baseline":
        if _baseline_model is not None:
            return _baseline_model, _baseline_features
        path = MODELS_DIR / "baseline.pkl"
        if path.exists():
            print("[model_loader] Loading baseline.pkl")
            raw = joblib.load(path)
            _baseline_model, _baseline_features = _extract_model_and_features(raw)
            validate_features(_baseline_features)
            print(f"[model_loader] baseline features ({len(_baseline_features)}): {_baseline_features}")
        else:
            raise FileNotFoundError("models/baseline.pkl not found.")
        return _baseline_model, _baseline_features

    if kind == "with_events":
        if _events_model is not None:
            return _events_model, _events_features
        path = MODELS_DIR / "with_events.pkl"
        if path.exists():
            print("[model_loader] Loading with_events.pkl")
            raw = joblib.load(path)
            _events_model, _events_features = _extract_model_and_features(raw)
            validate_features(_events_features)
            print(f"[model_loader] with_events features ({len(_events_features)}): {_events_features}")
        else:
            raise FileNotFoundError("models/with_events.pkl not found.")
        return _events_model, _events_features

    raise ValueError(f"Unknown model kind: {kind}")
