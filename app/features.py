"""
Feature registry — keyed by the exact column names stored in
model.feature_names (read from Booster at load time).

Adding a new feature: add one entry to FEATURE_REGISTRY with the
exact name the model expects. The builder receives:
  (zone_id: int, hour: int, dt: date, ride_history: dict, zone_events: list)
  zone_events is ALL events for the zone on that day (not pre-filtered to active).
  ride_history keys are integer lag offsets (1 = 1 hour ago, 24 = yesterday, etc.)
"""

import json
import math
import numpy as np
import pandas as pd
import holidays as hols
from datetime import date as DateType, timedelta
from pathlib import Path

# ── US Federal holiday set (cached) ──────────────────────────────────────────
_US_HOLIDAYS = hols.US(years=range(2023, 2030))


def _is_holiday(dt: DateType) -> int:
    return int(dt in _US_HOLIDAYS)


def _day_before_holiday(dt: DateType) -> int:
    return int((dt + timedelta(days=1)) in _US_HOLIDAYS)


def _day_after_holiday(dt: DateType) -> int:
    return int((dt - timedelta(days=1)) in _US_HOLIDAYS)


# ── NYC seasonal weather averages (monthly) ───────────────────────────────────
# Used as placeholders when live weather data isn't available.
# Temperature in °F, precipitation in inches/hr, wind in mph.
_WEATHER_AVG = {
    #  month: (temp, precip, wind, humidity)
    1:  (32,  0.15, 12, 60),
    2:  (34,  0.12, 12, 58),
    3:  (43,  0.14, 11, 57),
    4:  (53,  0.16, 10, 56),
    5:  (62,  0.17,  9, 58),
    6:  (71,  0.18,  8, 61),
    7:  (76,  0.18,  8, 64),
    8:  (75,  0.17,  8, 65),
    9:  (67,  0.15,  9, 63),
    10: (57,  0.13,  9, 60),
    11: (47,  0.15, 10, 61),
    12: (38,  0.15, 12, 62),
}


def _weather(dt: DateType, key: str):
    t, p, w, h = _WEATHER_AVG[dt.month]
    return {"pred_temperature": t, "pred_precipitation": p, "pred_wind_speed": w,
            "pred_humidity": h, "pred_snowfall": 0, "pred_rain": p,
            "pred_is_raining": int(p > 0.1),
            "pred_is_snowing": int(dt.month in (12, 1, 2) and t < 35),
            "pred_heavy_rain": int(p > 0.3),
            "pred_high_wind": int(w > 20),
            "pred_freezing": int(t < 32)}[key]


# ── Zone lookup (populated on first use via data_loader) ─────────────────────
def _zone_mean(zone_id):
    from app.data_loader import get_zone_mean_trips
    return get_zone_mean_trips(zone_id)


def _zone_tier(zone_id):
    from app.data_loader import get_zone_tier
    return get_zone_tier(zone_id)


# ── Event helpers ─────────────────────────────────────────────────────────────
CATEGORIES = [
    "sports", "concert", "festival", "parade", "transit_disruption",
    "weather_emergency", "conference", "political", "holiday", "other",
]

# Categories tracked by the with_events model (holiday excluded — rare, low-lift)
_EVENT_LIFT_CATS = [
    "sports", "concert", "festival", "parade", "conference",
    "transit_disruption", "other", "political",
]


def _best_event(events):
    return max(events, key=lambda e: e.demand_magnitude, default=None)


def _at(events, hour: int) -> list:
    """Events currently active at this hour."""
    return [ev for ev in events if ev.start_hour <= hour <= ev.end_hour]


# ── Historical event demand (zone × category averages from training) ──────────
# Loaded from data/hist_event_demand.json if present; falls back to
# zone_mean_trips so predictions degrade gracefully rather than crashing.
# Format: { "zone_id_category": avg_rides_float, ... }  e.g. "161_sports": 420.5
_hist_event_demand_cache: dict | None = None
_HIST_DEMAND_PATH = Path(__file__).parent.parent / "data" / "hist_event_demand.json"


def _load_hist_event_demand() -> dict:
    global _hist_event_demand_cache
    if _hist_event_demand_cache is not None:
        return _hist_event_demand_cache
    if _HIST_DEMAND_PATH.exists():
        with open(_HIST_DEMAND_PATH) as f:
            _hist_event_demand_cache = json.load(f)
        print(f"[features] Loaded {len(_hist_event_demand_cache)} hist_event_demand entries")
    else:
        print("[features] hist_event_demand.json not found — using zone_mean_trips fallback")
        _hist_event_demand_cache = {}
    return _hist_event_demand_cache


def _hist_demand(zone_id: int, category: str) -> float:
    """Historical avg rides for zone when category event is active. Falls back to zone mean."""
    hist = _load_hist_event_demand()
    key  = f"{zone_id}_{category}"
    return float(hist[key]) if key in hist else _zone_mean(zone_id)


# ── Feature registry ─────────────────────────────────────────────────────────
FEATURE_REGISTRY = {
    # ── Lag features ────────────────────────────────────────────────────────
    "lag_1":   lambda z, h, d, r, e: r.get(1,   0.0),
    "lag_2":   lambda z, h, d, r, e: r.get(2,   0.0),
    "lag_3":   lambda z, h, d, r, e: r.get(3,   0.0),
    "lag_24":  lambda z, h, d, r, e: r.get(24,  0.0),
    "lag_48":  lambda z, h, d, r, e: r.get(48,  0.0),
    "lag_72":  lambda z, h, d, r, e: r.get(72,  0.0),
    "lag_168": lambda z, h, d, r, e: r.get(168, 0.0),

    # ── Rolling statistics ───────────────────────────────────────────────────
    "rolling_mean_3h":   lambda z, h, d, r, e: np.mean([r.get(i, 0.0) for i in range(1, 4)]),
    "rolling_mean_6h":   lambda z, h, d, r, e: np.mean([r.get(i, 0.0) for i in range(1, 7)]),
    "rolling_mean_24h":  lambda z, h, d, r, e: np.mean([r.get(i, 0.0) for i in range(1, 25)]),
    "rolling_mean_168h": lambda z, h, d, r, e: np.mean([r.get(i, 0.0) for i in range(1, 169)]),
    "rolling_std_24h":   lambda z, h, d, r, e: np.std([r.get(i,  0.0) for i in range(1, 25)]),
    "rolling_max_24h":   lambda z, h, d, r, e: max([r.get(i,     0.0) for i in range(1, 25)]),

    # ── Time features ────────────────────────────────────────────────────────
    "pred_hour_of_day": lambda z, h, d, r, e: h,
    "pred_dow":         lambda z, h, d, r, e: d.weekday(),
    "pred_month":       lambda z, h, d, r, e: d.month,
    "pred_is_weekend":  lambda z, h, d, r, e: int(d.weekday() >= 5),

    # ── Cyclical encodings ───────────────────────────────────────────────────
    "hour_sin":  lambda z, h, d, r, e: math.sin(2 * math.pi * h / 24),
    "hour_cos":  lambda z, h, d, r, e: math.cos(2 * math.pi * h / 24),
    "dow_sin":   lambda z, h, d, r, e: math.sin(2 * math.pi * d.weekday() / 7),
    "dow_cos":   lambda z, h, d, r, e: math.cos(2 * math.pi * d.weekday() / 7),
    "month_sin": lambda z, h, d, r, e: math.sin(2 * math.pi * d.month / 12),
    "month_cos": lambda z, h, d, r, e: math.cos(2 * math.pi * d.month / 12),

    # ── Holiday features ─────────────────────────────────────────────────────
    "is_holiday":        lambda z, h, d, r, e: _is_holiday(d),
    "day_before_holiday":lambda z, h, d, r, e: _day_before_holiday(d),
    "day_after_holiday": lambda z, h, d, r, e: _day_after_holiday(d),

    # ── Zone features ────────────────────────────────────────────────────────
    "PULocationID":      lambda z, h, d, r, e: z,
    "zone_tier_encoded": lambda z, h, d, r, e: _zone_tier(z),
    "zone_mean_trips":   lambda z, h, d, r, e: _zone_mean(z),

    # ── Weather features (seasonal averages as placeholder) ──────────────────
    "pred_temperature":  lambda z, h, d, r, e: _weather(d, "pred_temperature"),
    "pred_precipitation":lambda z, h, d, r, e: _weather(d, "pred_precipitation"),
    "pred_wind_speed":   lambda z, h, d, r, e: _weather(d, "pred_wind_speed"),
    "pred_snowfall":     lambda z, h, d, r, e: _weather(d, "pred_snowfall"),
    "pred_rain":         lambda z, h, d, r, e: _weather(d, "pred_rain"),
    "pred_humidity":     lambda z, h, d, r, e: _weather(d, "pred_humidity"),
    "pred_is_raining":   lambda z, h, d, r, e: _weather(d, "pred_is_raining"),
    "pred_is_snowing":   lambda z, h, d, r, e: _weather(d, "pred_is_snowing"),
    "pred_heavy_rain":   lambda z, h, d, r, e: _weather(d, "pred_heavy_rain"),
    "pred_high_wind":    lambda z, h, d, r, e: _weather(d, "pred_high_wind"),
    "pred_freezing":     lambda z, h, d, r, e: _weather(d, "pred_freezing"),

    # ── Legacy single-event features (baseline-era; kept for registry completeness) ──
    "has_event":         lambda z, h, d, r, e: int(any(ev.start_hour <= h <= ev.end_hour for ev in e)),
    "event_magnitude":   lambda z, h, d, r, e: (_best_event(_at(e, h)).demand_magnitude if _at(e, h) else 0),
    "event_direction":   lambda z, h, d, r, e: (_best_event(_at(e, h)).demand_direction if _at(e, h) else 0),
    "event_count":       lambda z, h, d, r, e: len(_at(e, h)),
    "hours_to_event":    lambda z, h, d, r, e: (max(0, _best_event(_at(e, h)).start_hour - h) if _at(e, h) else 0),
    "event_duration":    lambda z, h, d, r, e: ((_best_event(_at(e, h)).end_hour - _best_event(_at(e, h)).start_hour) if _at(e, h) else 0),
    **{
        f"cat_{cat}": (lambda c: lambda z, h, d, r, e: int(bool(_at(e, h)) and _best_event(_at(e, h)).category == c))(cat)
        for cat in CATEGORIES
    },

    # ── Multi-event aggregations ──────────────────────────────────────────────
    "high_impact_event_count":   lambda z, h, d, r, e: sum(1 for ev in e if ev.demand_magnitude >= 4 and ev.start_hour <= h <= ev.end_hour),
    "has_sports":                lambda z, h, d, r, e: int(any(ev.category == "sports"              for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_concert":               lambda z, h, d, r, e: int(any(ev.category == "concert"             for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_transit_disruption":    lambda z, h, d, r, e: int(any(ev.category == "transit_disruption"  for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_weather_emergency":     lambda z, h, d, r, e: int(any(ev.category == "weather_emergency"   for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_festival":              lambda z, h, d, r, e: int(any(ev.category == "festival"            for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_parade":                lambda z, h, d, r, e: int(any(ev.category == "parade"              for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_conference":            lambda z, h, d, r, e: int(any(ev.category == "conference"          for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_other":                 lambda z, h, d, r, e: int(any(ev.category == "other"               for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_political":             lambda z, h, d, r, e: int(any(ev.category == "political"           for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "max_event_magnitude":       lambda z, h, d, r, e: max((ev.demand_magnitude for ev in e if ev.start_hour <= h <= ev.end_hour), default=0),
    "sum_event_magnitude":       lambda z, h, d, r, e: sum(ev.demand_magnitude  for ev in e if ev.start_hour <= h <= ev.end_hour),
    "net_event_direction":       lambda z, h, d, r, e: sum(ev.demand_direction  for ev in e if ev.start_hour <= h <= ev.end_hour),
    "signed_event_impact":       lambda z, h, d, r, e: sum(ev.demand_magnitude * ev.demand_direction for ev in e if ev.start_hour <= h <= ev.end_hour),
    "max_positive_event_impact": lambda z, h, d, r, e: max((ev.demand_magnitude for ev in e if ev.demand_direction  > 0 and ev.start_hour <= h <= ev.end_hour), default=0),
    "max_negative_event_impact": lambda z, h, d, r, e: max((ev.demand_magnitude for ev in e if ev.demand_direction  < 0 and ev.start_hour <= h <= ev.end_hour), default=0),

    # ── Scope features (require impact_scope on EventFeature) ─────────────────
    "has_citywide_event":  lambda z, h, d, r, e: int(any(getattr(ev, "impact_scope", "localized") == "citywide"  for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_localized_event": lambda z, h, d, r, e: int(any(getattr(ev, "impact_scope", "localized") == "localized" for ev in e if ev.start_hour <= h <= ev.end_hour)),

    # ── Temporal window features (use all zone events, including pre/post) ─────
    # e is ALL zone events for the day; active filtering done per feature below.
    "pre_event_window":      lambda z, h, d, r, e: int(any(0 < ev.start_hour - h <= 2 for ev in e)),
    "post_event_window":     lambda z, h, d, r, e: int(any(0 < h - ev.end_hour   <= 2 for ev in e)),
    "hours_into_event":      lambda z, h, d, r, e: max((h - ev.start_hour for ev in e if ev.start_hour <= h <= ev.end_hour), default=0),
    "hours_until_event_end": lambda z, h, d, r, e: max((ev.end_hour - h   for ev in e if ev.start_hour <= h <= ev.end_hour), default=0),
    "is_event_peak_window":  lambda z, h, d, r, e: int(any(0 <= h - ev.start_hour <= 2 for ev in e if ev.start_hour <= h <= ev.end_hour)),

    # ── Impact type features (require affected_demand_type on EventFeature) ────
    "has_commute_event":       lambda z, h, d, r, e: int(any(getattr(ev, "affected_demand_type", "") == "commute"       for ev in e if ev.start_hour <= h <= ev.end_hour)),
    "has_discretionary_event": lambda z, h, d, r, e: int(any(getattr(ev, "affected_demand_type", "") == "discretionary" for ev in e if ev.start_hour <= h <= ev.end_hour)),

    # ── Normalised impact ─────────────────────────────────────────────────────
    "signed_event_impact_norm": lambda z, h, d, r, e: sum(ev.demand_magnitude * ev.demand_direction for ev in e if ev.start_hour <= h <= ev.end_hour) / 5.0,
    "max_event_magnitude_norm": lambda z, h, d, r, e: max((ev.demand_magnitude for ev in e if ev.start_hour <= h <= ev.end_hour), default=0) / 5.0,

    # ── Historical demand by category (zone × category avg from training data) ─
    "hist_demand_sports":             lambda z, h, d, r, e: _hist_demand(z, "sports"),
    "hist_demand_concert":            lambda z, h, d, r, e: _hist_demand(z, "concert"),
    "hist_demand_festival":           lambda z, h, d, r, e: _hist_demand(z, "festival"),
    "hist_demand_parade":             lambda z, h, d, r, e: _hist_demand(z, "parade"),
    "hist_demand_conference":         lambda z, h, d, r, e: _hist_demand(z, "conference"),
    "hist_demand_transit_disruption": lambda z, h, d, r, e: _hist_demand(z, "transit_disruption"),
    "hist_demand_other":              lambda z, h, d, r, e: _hist_demand(z, "other"),
    "hist_demand_political":          lambda z, h, d, r, e: _hist_demand(z, "political"),

    # ── Event lift: active magnitude per category (interaction features) ───────
    **{
        f"event_lift_{cat}": (lambda c: lambda z, h, d, r, e: max(
            (ev.demand_magnitude for ev in e if ev.category == c and ev.start_hour <= h <= ev.end_hour),
            default=0,
        ))(cat)
        for cat in _EVENT_LIFT_CATS
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_features(feature_names: list[str]) -> None:
    missing = [f for f in feature_names if f not in FEATURE_REGISTRY]
    if missing:
        raise ValueError(
            f"Model requires features not in registry: {missing}\n"
            f"Add builders for these to app/features.py FEATURE_REGISTRY."
        )


def build_feature_row(
    zone_id: int,
    hour: int,
    dt: DateType,
    ride_history: dict,
    zone_events: list,    # ALL events for this zone on the day — features filter to active
    feature_names: list[str],
) -> dict:
    return {
        name: FEATURE_REGISTRY[name](zone_id, hour, dt, ride_history, zone_events)
        for name in feature_names
    }
