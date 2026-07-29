import sys
from unittest.mock import MagicMock, patch

# Stub xgboost before any project imports — it may fail to load locally on
# macOS without libomp. All actual xgboost calls are mocked in fixtures anyway.
_xgb = MagicMock()
sys.modules.setdefault("xgboost", _xgb)
sys.modules.setdefault("xgboost.core", _xgb)

import numpy as np
import pytest
from fastapi.testclient import TestClient

_FEATURES = [
    "pred_hour_of_day", "pred_dow", "pred_month", "pred_is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday", "day_before_holiday", "day_after_holiday",
    "zone_tier_encoded", "zone_mean_trips",
    "pred_temperature", "pred_precipitation", "pred_wind_speed",
    "pred_snowfall", "pred_rain", "pred_humidity",
    "pred_is_raining", "pred_is_snowing", "pred_heavy_rain", "pred_high_wind", "pred_freezing",
    "has_event", "event_magnitude", "event_direction", "event_count", "hours_to_event",
]

_MOCK_MODEL = MagicMock()
_MOCK_MODEL.predict.return_value = np.array([50.0] * 1000)

_HIST = {lag: 10.0 for lag in range(1, 337)}
_MOCK_ROW = {f: 0.0 for f in _FEATURES}


@pytest.fixture(scope="session")
def client():
    with (
        patch("main.load_rides", return_value={}),
        patch("main.load_zone_stats", return_value={}),
        patch("main._build_zone_peaks"),
        patch("main.load_model", return_value=(_MOCK_MODEL, _FEATURES)),
        patch("app.predictor.load_model", return_value=(_MOCK_MODEL, _FEATURES)),
        patch("app.predictor.load_rides", return_value={}),
        patch("app.predictor.get_zone_peak", return_value=200.0),
        patch("app.predictor.get_ride_history", return_value=_HIST),
        patch("app.predictor.get_ride_history_batch",
              return_value={z: _HIST for z in range(1, 300)}),
        patch("app.predictor.build_feature_row", return_value=_MOCK_ROW),
    ):
        import main
        with TestClient(main.app) as c:
            yield c
