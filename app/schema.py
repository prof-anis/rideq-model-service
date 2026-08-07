from datetime import date as Date
from pydantic import BaseModel, field_validator
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

def _validate_date(v: str) -> str:
    try:
        Date.fromisoformat(v)
    except ValueError:
        raise ValueError(f"date must be YYYY-MM-DD, got '{v}'")
    return v

class BaselineRequest(BaseModel):
    date: str                    # YYYY-MM-DD
    zones: list[int]
    hour_from: int = 0
    hour_to: int = 23

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        return _validate_date(v)

class WithEventsRequest(BaseModel):
    date: str
    zones: list[int]
    events: list[EventFeature]
    hour_from: int = 0
    hour_to: int = 23

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        return _validate_date(v)

class InjectedRequest(BaseModel):
    date: str
    zones: list[int]
    injection_hour: int
    events: list[EventFeature]
    actuals_before_injection: list[HourlyRide]

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        return _validate_date(v)

class ActualsPoint(BaseModel):
    zone_id: int
    hour: int
    rides: float

class NaiveRequest(BaseModel):
    date: str
    zones: list[int]
    hour_from: int = 0
    hour_to: int = 23

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        return _validate_date(v)
