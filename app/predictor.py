"""
Core prediction logic — four modes:

  naive()       — lag-168 seasonal naive: same hour last week (no ML)
  baseline()    — XGBoost, no event features
  with_events() — XGBoost, with event feature columns
  injected()    — rolling forecast from injection_hour forward;
                  each predicted hour feeds as lag into the next

Performance: baseline() and with_events() batch all zone×hour rows into
a single DMatrix prediction call instead of one call per row.

All return PredictionPoint with:
  rides        — raw model output rounded to whole rides
  demand_score — normalised 0–5 (rides / zone_peak * 5) for map colouring
"""

import numpy as np
import pandas as pd
from datetime import date

from app.schema import BaselineRequest, WithEventsRequest, InjectedRequest, NaiveRequest, PredictionPoint
from app.features import build_feature_row, FEATURE_REGISTRY
from app.data_loader import get_ride_history, get_ride_history_batch, get_zone_peak, load_rides
from app.model_loader import load_model
from xgboost import DMatrix

MAX_SCORE = 5.0


def _point(zone: int, hour: int, rides: float, peak: float) -> PredictionPoint:
    rides_int = max(0, int(round(rides)))
    score     = max(0.0, min(round((rides_int / peak) * MAX_SCORE, 2), MAX_SCORE)) if peak > 0 else 0.0
    return PredictionPoint(zone_id=zone, hour=hour, rides=rides_int, demand_score=score)


def naive(req: NaiveRequest) -> list[PredictionPoint]:
    """
    Seasonal naive forecast: prediction = actual rides at same hour, 7 days ago.
    No model, no features — just a direct lag-168 lookup from the rides index.
    Used as the benchmark baseline to show what XGBoost improves upon.
    """
    from datetime import timedelta
    idx   = load_rides()
    peaks = {z: get_zone_peak(z) for z in req.zones}
    out   = []

    dt_7d_ago = (date.fromisoformat(req.date) - timedelta(weeks=1)).isoformat()

    for zone in req.zones:
        peak = peaks[zone]
        for hour in range(req.hour_from, req.hour_to + 1):
            rides = idx.get((zone, dt_7d_ago, hour), 0.0)
            out.append(_point(zone, hour, rides, peak))

    return out


def _batch_predict(model_wrapper, feature_names, rows: list[dict]) -> np.ndarray:
    """Single DMatrix call for all rows — much faster than one call per row."""
    X = pd.DataFrame(rows, columns=feature_names)
    return model_wrapper.predict(X)


def baseline(req: BaselineRequest) -> list[PredictionPoint]:
    model, feature_names = load_model("baseline")
    dt    = date.fromisoformat(req.date)
    peaks = {z: get_zone_peak(z) for z in req.zones}
    rows, meta = [], []

    for hour in range(req.hour_from, req.hour_to + 1):
        # Batch lag history for all zones at this hour in one pass
        hist_batch = get_ride_history_batch(req.zones, req.date, hour)
        for zone in req.zones:
            rows.append(build_feature_row(zone, hour, dt, hist_batch[zone], [], feature_names))
            meta.append((zone, hour))

    preds = _batch_predict(model, feature_names, rows)
    return [_point(zone, hour, float(p), peaks[zone]) for (zone, hour), p in zip(meta, preds)]


def with_events(req: WithEventsRequest) -> list[PredictionPoint]:
    model, feature_names = load_model("with_events")
    dt    = date.fromisoformat(req.date)
    peaks = {z: get_zone_peak(z) for z in req.zones}
    rows, meta = [], []

    for hour in range(req.hour_from, req.hour_to + 1):
        hist_batch = get_ride_history_batch(req.zones, req.date, hour)
        for zone in req.zones:
            # Pass ALL zone events — feature builders filter to active at each hour.
            # This allows pre/post-event window features to see the full day schedule.
            zone_events = [e for e in req.events if e.zone_id == zone]
            rows.append(build_feature_row(zone, hour, dt, hist_batch[zone], zone_events, feature_names))
            meta.append((zone, hour))

    preds = _batch_predict(model, feature_names, rows)
    return [_point(zone, hour, float(p), peaks[zone]) for (zone, hour), p in zip(meta, preds)]


def injected(req: InjectedRequest) -> list[PredictionPoint]:
    """
    Rolling forecast — must stay sequential (each hour feeds the next),
    so we can't batch across hours. We batch across zones per hour instead.
    """
    base_model,   base_features   = load_model("baseline")
    events_model, events_features = load_model("with_events")
    dt  = date.fromisoformat(req.date)
    out = []

    # Seed rolling dict from actuals or parquet
    idx = load_rides()
    predicted_rides: dict[tuple, float] = {
        (a.zone_id, a.hour): a.rides for a in req.actuals_before_injection
    }
    for zone in req.zones:
        for hour in range(0, req.injection_hour):
            if (zone, hour) not in predicted_rides:
                predicted_rides[(zone, hour)] = idx.get((zone, req.date, hour), 0.0)

    peaks = {z: get_zone_peak(z) for z in req.zones}

    # Pre-injection: batch all zones for each hour
    for hour in range(0, req.injection_hour):
        rows, zones_this_hour = [], []
        for zone in req.zones:
            hist = get_ride_history(zone, req.date, hour)
            rows.append(build_feature_row(zone, hour, dt, hist, [], base_features))
            zones_this_hour.append(zone)

        preds = _batch_predict(base_model, base_features, rows)
        for zone, pred in zip(zones_this_hour, preds):
            predicted_rides[(zone, hour)] = float(pred)
            out.append(_point(zone, hour, float(pred), peaks[zone]))

    # From injection_hour: rolling forecast — batch zones per hour
    db_hist_caches = {
        zone: get_ride_history(zone, req.date, req.injection_hour)
        for zone in req.zones
    }

    for hour in range(req.injection_hour, 24):
        rows, zones_this_hour = [], []
        for zone in req.zones:
            cache = db_hist_caches[zone]
            hist  = {}
            for lag in range(1, 337):
                prior = hour - lag
                if prior >= 0:
                    hist[lag] = predicted_rides.get((zone, prior), cache.get(lag, 0.0))
                else:
                    hist[lag] = cache.get(lag, 0.0)

            zone_events = [e for e in req.events if e.zone_id == zone]
            rows.append(build_feature_row(zone, hour, dt, hist, zone_events, events_features))
            zones_this_hour.append(zone)

        preds = _batch_predict(events_model, events_features, rows)
        for zone, pred in zip(zones_this_hour, preds):
            predicted_rides[(zone, hour)] = float(pred)
            out.append(_point(zone, hour, float(pred), peaks[zone]))

    return out
