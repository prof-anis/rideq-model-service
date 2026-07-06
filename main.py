from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schema import BaselineRequest, WithEventsRequest, InjectedRequest, NaiveRequest, PredictionPoint, ActualsPoint
from app.data_loader import load_rides, load_zone_stats, _build_zone_peaks
from app.model_loader import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up on startup — index 2.3M rows + load both models
    load_rides()
    load_zone_stats()
    _build_zone_peaks()
    load_model("baseline")
    load_model("with_events")
    yield


app = FastAPI(title="RideQ Model Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/naive", response_model=list[PredictionPoint])
def predict_naive(req: NaiveRequest):
    """Seasonal naive: same hour, 7 days ago. No ML — benchmark only."""
    from app.predictor import naive
    return naive(req)


@app.post("/predict/baseline", response_model=list[PredictionPoint])
def predict_baseline(req: BaselineRequest):
    from app.predictor import baseline
    return baseline(req)


@app.post("/predict/with-events", response_model=list[PredictionPoint])
def predict_with_events(req: WithEventsRequest):
    from app.predictor import with_events
    return with_events(req)


@app.post("/predict/injected", response_model=list[PredictionPoint])
def predict_injected(req: InjectedRequest):
    from app.predictor import injected
    return injected(req)


@app.get("/actuals", response_model=list[ActualsPoint])
def get_actuals(date: str, zone_id: int):
    """Returns the real ride counts from the 2025 parquet for a zone across all 24 hours."""
    idx = load_rides()
    return [
        ActualsPoint(zone_id=zone_id, hour=hour, rides=idx.get((zone_id, date, hour), 0.0))
        for hour in range(24)
    ]
