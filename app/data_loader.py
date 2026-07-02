"""
Loads the 2025 ride history and indexes it for O(1) lag lookups.

On startup the parquet is read once and pivoted into a flat dict:
  _rides_index[(zone_id, "YYYY-MM-DD", hour)] = rides_float

All subsequent lag lookups are instant dictionary gets — no DataFrame
scanning at prediction time.

Expected parquet schema:
  date     str        YYYY-MM-DD
  zone_id  int
  hour     int        0–23
  rides    float
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

DATA_PATH = Path(__file__).parent.parent / "data" / "rides_2025.parquet"

# (zone_id: int, date_str: str, hour: int) → rides: float
_rides_index: dict | None = None

_zone_stats_cache: pd.DataFrame | None = None


def load_rides() -> dict:
    """Returns the full rides index dict. Builds it on first call."""
    global _rides_index
    if _rides_index is not None:
        return _rides_index

    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
        required = {"date", "zone_id", "hour", "rides"}
        if not required.issubset(df.columns):
            raise ValueError(f"Parquet missing columns: {required - set(df.columns)}")
        print(f"[data_loader] Indexing {len(df):,} rows from rides_2025.parquet …")
    else:
        print("[data_loader] rides_2025.parquet not found — using synthetic data")
        df = _generate_synthetic()

    df["date"] = df["date"].astype(str).str[:10]

    # Build flat lookup dict — fast to construct, O(1) to read
    _rides_index = {
        (int(row.zone_id), row.date, int(row.hour)): float(row.rides)
        for row in df.itertuples(index=False)
    }
    print(f"[data_loader] Index ready: {len(_rides_index):,} entries")
    return _rides_index


def load_zone_stats() -> pd.DataFrame:
    """Returns DataFrame indexed by zone_id with mean_trips and zone_tier_encoded."""
    global _zone_stats_cache
    if _zone_stats_cache is not None:
        return _zone_stats_cache

    if DATA_PATH.exists():
        df = pd.read_parquet(DATA_PATH)
        df["date"] = df["date"].astype(str).str[:10]
    else:
        df = _generate_synthetic()

    stats = df.groupby("zone_id")["rides"].mean().reset_index()
    stats.columns = ["zone_id", "mean_trips"]
    stats["zone_tier_encoded"] = pd.qcut(
        stats["mean_trips"], 4, labels=[3, 2, 1, 0], duplicates="drop"
    ).astype(int)

    _zone_stats_cache = stats.set_index("zone_id")
    return _zone_stats_cache


def get_zone_mean_trips(zone_id: int) -> float:
    stats = load_zone_stats()
    return float(stats.loc[zone_id, "mean_trips"]) if zone_id in stats.index else 0.0


def get_zone_tier(zone_id: int) -> int:
    stats = load_zone_stats()
    return int(stats.loc[zone_id, "zone_tier_encoded"]) if zone_id in stats.index else 2


_zone_peak_cache: dict[int, float] = {}

def get_zone_peak(zone_id: int) -> float:
    """95th-percentile rides per zone — precomputed from the rides index."""
    if zone_id in _zone_peak_cache:
        return _zone_peak_cache[zone_id]

    # Build peaks for all zones at once on first call
    if not _zone_peak_cache:
        _build_zone_peaks()

    return _zone_peak_cache.get(zone_id, 1.0)


def _build_zone_peaks() -> None:
    """Compute 95th-percentile rides per zone from the in-memory index."""
    idx = load_rides()
    # Group rides by zone
    zone_rides: dict[int, list] = {}
    for (zone, date_str, hour), rides in idx.items():
        zone_rides.setdefault(zone, []).append(rides)

    for zone, rides_list in zone_rides.items():
        arr = np.array(rides_list)
        _zone_peak_cache[zone] = float(np.percentile(arr, 95))
    print(f"[data_loader] Zone peaks computed for {len(_zone_peak_cache)} zones")


def _build_lag_offsets(date_str: str, up_to_hour: int, max_lag: int = 168) -> list[tuple]:
    """
    Precompute (lag, date_str, hour) for every lag offset up to max_lag.
    Called once per prediction slot — shared across all zones.
    """
    dt      = date.fromisoformat(date_str)
    offsets = []
    for lag in range(1, max_lag + 1):
        lag_hour = up_to_hour - lag
        lag_date = dt
        while lag_hour < 0:
            lag_date = lag_date - timedelta(days=1)
            lag_hour += 24
        offsets.append((lag, str(lag_date), lag_hour))
    return offsets


# Cache of precomputed offsets — keyed by (date_str, hour)
_offset_cache: dict[tuple, list] = {}

def get_lag_offsets(date_str: str, up_to_hour: int) -> list[tuple]:
    key = (date_str, up_to_hour)
    if key not in _offset_cache:
        _offset_cache[key] = _build_lag_offsets(date_str, up_to_hour)
    return _offset_cache[key]


def get_ride_history(zone_id: int, date_str: str, up_to_hour: int) -> dict:
    """Returns {lag_offset: rides} for lags 1–168 hours back."""
    idx     = load_rides()
    offsets = get_lag_offsets(date_str, up_to_hour)
    return {lag: idx.get((zone_id, d, h), 0.0) for lag, d, h in offsets}


def get_ride_history_batch(zone_ids: list, date_str: str, up_to_hour: int) -> dict:
    """
    Batch version — returns {zone_id: {lag: rides}} for all zones.
    Precomputes lag offsets once, then runs all zone lookups in one pass.
    Much faster than calling get_ride_history() per zone.
    """
    idx     = load_rides()
    offsets = get_lag_offsets(date_str, up_to_hour)
    return {
        zone: {lag: idx.get((zone, d, h), 0.0) for lag, d, h in offsets}
        for zone in zone_ids
    }


# ── Synthetic fallback ─────────────────────────────────────────────────────────

def _generate_synthetic() -> pd.DataFrame:
    rng  = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    zones = list(range(1, 264))
    rows  = []

    hourly_shape = np.array([
        0.3, 0.2, 0.15, 0.12, 0.12, 0.18,
        0.4, 0.9, 1.4,  1.2,  1.0,  1.1,
        1.2, 1.1, 1.0,  1.0,  1.1,  1.5,
        1.8, 1.6, 1.4,  1.2,  0.9,  0.6,
    ])
    hourly_shape /= hourly_shape.mean()
    BUSY = {61, 79, 132, 138, 161, 162, 163, 186, 230, 231}

    for d in dates:
        dow_mult = 1.3 if d.dayofweek >= 4 else 1.0
        for zone in zones:
            base = (200 if zone in BUSY else 50) * dow_mult
            for hour in range(24):
                rides = base * hourly_shape[hour] * rng.lognormal(0, 0.15)
                rows.append({"date": d.strftime("%Y-%m-%d"), "zone_id": zone,
                             "hour": hour, "rides": rides})
    return pd.DataFrame(rows)
