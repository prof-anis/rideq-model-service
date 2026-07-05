from pydantic import BaseModel
from typing import Optional

class EventFeature(BaseModel):
    zone_id: int
    start_hour: int
    end_hour: int
    demand_magnitude: int        # 1–5
    demand_direction: int        # -1 / 0 / 1
    category: str
    impact_scope: Optional[str] = "localized"          # "citywide" | "localized"
    affected_demand_type: Optional[str] = ""           # "commute" | "discretionary" | ""

class HourlyRide(BaseModel):
    zone_id: int
    hour: int
    rides: float

class PredictionPoint(BaseModel):
    zone_id: int
    hour: int
    rides: int                   # raw model output rounded to whole rides
    demand_score: float          # normalised 0–5 for map colouring (rides / zone_peak * 5)

class BaselineRequest(BaseModel):
    date: str                    # YYYY-MM-DD
    zones: list[int]
    hour_from: int = 0
    hour_to: int = 23

class WithEventsRequest(BaseModel):
    date: str
    zones: list[int]
    events: list[EventFeature]
    hour_from: int = 0
    hour_to: int = 23

class InjectedRequest(BaseModel):
    date: str
    zones: list[int]
    injection_hour: int
    events: list[EventFeature]
    actuals_before_injection: list[HourlyRide]   # rides[zone][hour] for hours < injection_hour

class ActualsPoint(BaseModel):
    zone_id: int
    hour: int
    rides: float

class NaiveRequest(BaseModel):
    date: str
    zones: list[int]
    hour_from: int = 0
    hour_to: int = 23
