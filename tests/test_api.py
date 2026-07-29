"""
Tests for the RideQ model service API.
Heavy dependencies (model files, parquet data) are mocked in conftest.py.
"""


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# ── /predict/naive ────────────────────────────────────────────────────────────

def test_naive_returns_one_point_per_zone_per_hour(client):
    res = client.post("/predict/naive", json={
        "date": "2025-01-15",
        "zones": [1, 162],
        "hour_from": 8,
        "hour_to": 10,
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 6  # 2 zones × 3 hours


def test_naive_point_shape(client):
    res = client.post("/predict/naive", json={
        "date": "2025-03-01",
        "zones": [1],
        "hour_from": 0,
        "hour_to": 0,
    })
    point = res.json()[0]
    assert set(point.keys()) == {"zone_id", "hour", "rides", "demand_score"}
    assert point["zone_id"] == 1
    assert point["hour"] == 0
    assert point["rides"] >= 0
    assert 0.0 <= point["demand_score"] <= 5.0


def test_naive_defaults_to_full_day(client):
    res = client.post("/predict/naive", json={"date": "2025-06-01", "zones": [1]})
    assert res.status_code == 200
    assert len(res.json()) == 24  # 1 zone × 24 hours (default hour_from=0, hour_to=23)


def test_naive_rejects_missing_date(client):
    res = client.post("/predict/naive", json={"zones": [1]})
    assert res.status_code == 422


def test_naive_rejects_missing_zones(client):
    res = client.post("/predict/naive", json={"date": "2025-06-01"})
    assert res.status_code == 422


# ── /predict/baseline ─────────────────────────────────────────────────────────

def test_baseline_returns_predictions(client):
    res = client.post("/predict/baseline", json={
        "date": "2025-06-10",
        "zones": [1],
        "hour_from": 0,
        "hour_to": 2,
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 3
    for point in data:
        assert point["rides"] >= 0
        assert 0.0 <= point["demand_score"] <= 5.0


def test_baseline_rejects_invalid_date_format(client):
    res = client.post("/predict/baseline", json={"date": "not-a-date", "zones": [1]})
    assert res.status_code == 422


# ── /predict/with-events ──────────────────────────────────────────────────────

def test_with_events_returns_predictions(client):
    res = client.post("/predict/with-events", json={
        "date": "2025-07-04",
        "zones": [1],
        "hour_from": 18,
        "hour_to": 20,
        "events": [{
            "zone_id": 1,
            "start_hour": 18,
            "end_hour": 23,
            "demand_magnitude": 4,
            "demand_direction": 1,
            "category": "festival",
        }],
    })
    assert res.status_code == 200
    assert len(res.json()) == 3


def test_with_events_requires_events_field(client):
    res = client.post("/predict/with-events", json={
        "date": "2025-06-10",
        "zones": [1],
    })
    assert res.status_code == 422


def test_with_events_rejects_invalid_event_schema(client):
    res = client.post("/predict/with-events", json={
        "date": "2025-06-10",
        "zones": [1],
        "events": [{"zone_id": "not-an-int"}],
    })
    assert res.status_code == 422


# ── /actuals ──────────────────────────────────────────────────────────────────

def test_actuals_returns_24_hours(client):
    res = client.get("/actuals", params={"date": "2025-01-15", "zone_id": 1})
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 24
    assert all(p["zone_id"] == 1 for p in data)
    assert [p["hour"] for p in data] == list(range(24))


def test_actuals_returns_zero_for_unknown_date(client):
    res = client.get("/actuals", params={"date": "1900-01-01", "zone_id": 1})
    assert res.status_code == 200
    assert all(p["rides"] == 0.0 for p in res.json())


def test_actuals_requires_both_params(client):
    res = client.get("/actuals", params={"date": "2025-01-15"})
    assert res.status_code == 422
