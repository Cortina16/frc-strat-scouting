"""Unit tests for 3D Match Digital Twin FastAPI Webserver.

Verifies:
1. Health check endpoint status and port configuration.
2. Discovery of available matches in parquet storage layer.
3. Field CAD topology verification (651.2 length, 317.7 width, 20.0 guardrail, 32 AprilTags, key field elements).
4. Match summary calculation (total frames, 60fps frame rate, duration, red/blue teams, action counts).
5. Action localization event reconciliation endpoint.
6. Frame scrubbing and telemetry streaming endpoint with robot and ball kinematics.
7. Static asset delivery and 404 error handling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from src.server.app import app
from src.server.data_provider import MatchDataProvider
from src.server.__main__ import parse_args


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provides a synchronous FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture(scope="module")
def match_id() -> str:
    """Default match identifier present in sample parquet storage."""
    return "2026cmptx_f1m3"


def test_healthcheck(client: TestClient) -> None:
    """Verifies GET /api/health returns status ok and port 3001."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("port") == 3001


def test_list_matches(client: TestClient, match_id: str) -> None:
    """Verifies GET /api/matches scans data/parquet and returns matches list."""
    response = client.get("/api/matches")
    assert response.status_code == 200
    matches = response.json()
    assert isinstance(matches, list)
    assert len(matches) >= 1

    match_ids = [m["match_id"] for m in matches]
    assert match_id in match_ids

    target = next(m for m in matches if m["match_id"] == match_id)
    assert target["frame_count"] > 0
    assert "trajectories" in target["available_layers"] or "trajectories" in target["layers"]


def test_field_cad_endpoint(client: TestClient, match_id: str) -> None:
    """Verifies GET /api/match/{match_id}/field returns 651.2 length, 317.7 width, 20.0 guardrail, and 32 AprilTags."""
    response = client.get(f"/api/match/{match_id}/field")
    assert response.status_code == 200
    data = response.json()

    # Verify dimensions
    length = data.get("length") or data.get("field", {}).get("length") or data.get("dimensions", {}).get("length")
    width = data.get("width") or data.get("field", {}).get("width") or data.get("dimensions", {}).get("width")
    guardrail_height = (
        data.get("guardrail_height")
        or data.get("field", {}).get("guardrail_height")
        or data.get("dimensions", {}).get("guardrail_height")
    )

    assert length == pytest.approx(651.2, abs=1e-2)
    assert width == pytest.approx(317.7, abs=1e-2)
    assert guardrail_height == pytest.approx(20.0, abs=1e-2)

    # Verify elements (Red Hub, Blue Hub, Towers, Depots, Trenches)
    elements = data.get("elements", {})
    assert "red_hub" in elements or "Red Hub" in elements
    assert "blue_hub" in elements or "Blue Hub" in elements
    assert "towers" in elements or "Towers" in elements
    assert "depots" in elements or "Depots" in elements
    assert "trenches" in elements or "Trenches" in elements

    # Verify all 32 AprilTags with (id, x, y, z, roll, pitch, yaw)
    tags = data.get("apriltags", [])
    assert len(tags) == 32
    for tag in tags:
        assert "id" in tag
        assert "x" in tag
        assert "y" in tag
        assert "z" in tag
        assert "roll" in tag
        assert "pitch" in tag
        assert "yaw" in tag
        assert 1 <= tag["id"] <= 32


def test_match_summary_endpoint(client: TestClient, match_id: str) -> None:
    """Verifies GET /api/match/{match_id}/summary returns frame totals, 60fps, duration, teams, and action counts."""
    response = client.get(f"/api/match/{match_id}/summary")
    assert response.status_code == 200
    data = response.json()

    assert data.get("match_id") == match_id
    assert data.get("total_frames") > 0
    assert data.get("frame_rate") == pytest.approx(60.0, abs=1e-2)
    assert data.get("duration_seconds") > 0.0

    # Alliances
    assert isinstance(data.get("red_teams"), list)
    assert isinstance(data.get("blue_teams"), list)
    assert len(data["red_teams"]) > 0
    assert len(data["blue_teams"]) > 0

    # Action summary counts
    action_counts = data.get("action_counts") or data.get("action_summary") or data.get("action_event_counts")
    assert isinstance(action_counts, dict)
    assert "SCORE_HUB" in action_counts
    assert action_counts["SCORE_HUB"] > 0


def test_match_events_endpoint(client: TestClient, match_id: str) -> None:
    """Verifies GET /api/match/{match_id}/events returns reconciled action events with required schema."""
    response = client.get(f"/api/match/{match_id}/events")
    assert response.status_code == 200
    events = response.json()

    assert isinstance(events, list)
    assert len(events) > 0

    first_event = events[0]
    required_keys = [
        "event_id",
        "track_id",
        "team_number",
        "action_type",
        "element",
        "start_frame",
        "end_frame",
        "start_time_ms",
        "end_time_ms",
        "duration_s",
        "confidence",
    ]
    for key in required_keys:
        assert key in first_event, f"Missing required event key: {key}"

    assert first_event["start_frame"] <= first_event["end_frame"]
    assert first_event["duration_s"] >= 0.0
    assert 0.0 <= first_event["confidence"] <= 1.0


def test_match_frames_endpoint(client: TestClient, match_id: str) -> None:
    """Verifies GET /api/match/{match_id}/frames returns robot trajectories and ball kinematics."""
    # Query a sub-slice with limit
    response = client.get(f"/api/match/{match_id}/frames?start=190&end=200&limit=5")
    assert response.status_code == 200
    data = response.json()

    assert data.get("match_id") == match_id
    frames = data.get("frames", [])
    assert 1 <= len(frames) <= 5

    # Check top-level / frame fields
    frame0 = frames[0]
    assert "frame_index" in frame0
    assert "timestamp_ms" in frame0
    assert "robots" in frame0
    assert "balls" in frame0

    # Verify robot kinematics schema
    robots = frame0["robots"]
    assert isinstance(robots, list)
    if robots:
        r0 = robots[0]
        for k in ["track_id", "team_number", "alliance", "field_x_inch", "field_y_inch", "heading_rad", "speed_ips", "in_defense"]:
            assert k in r0, f"Missing robot key: {k}"

    # Verify ball kinematics schema
    balls = frame0["balls"]
    assert isinstance(balls, list)
    if balls:
        b0 = balls[0]
        for k in ["ball_id", "field_x_inch", "field_y_inch", "velocity_x_ips", "velocity_y_ips", "ball_state", "is_cluster", "cluster_ball_count", "shooter_team_number"]:
            assert k in b0, f"Missing ball key: {k}"


def test_root_index_html(client: TestClient) -> None:
    """Verifies root endpoint GET / serves index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "FRC 2026 REBUILT" in response.text


def test_static_vendor_assets(client: TestClient) -> None:
    """Verifies static assets are properly mounted at /static."""
    response = client.get("/static/vendor/three.min.js")
    assert response.status_code == 200
    assert len(response.content) > 10000


def test_nonexistent_match_404(client: TestClient) -> None:
    """Verifies 404 response for non-existent match identifiers."""
    response = client.get("/api/match/nonexistent_match_9999/summary")
    assert response.status_code == 404

    response_events = client.get("/api/match/nonexistent_match_9999/events")
    assert response_events.status_code == 404


def test_data_provider_unit() -> None:
    """Unit test for MatchDataProvider caching and direct operations."""
    provider = MatchDataProvider()
    matches = provider.get_available_matches()
    assert len(matches) >= 1

    cad = provider.get_field_cad()
    assert len(cad["apriltags"]) == 32

    # Test cache retrieval
    frames1 = provider.get_match_frames("2026cmptx_f1m3", start_frame=200, end_frame=205)
    frames2 = provider.get_match_frames("2026cmptx_f1m3", start_frame=200, end_frame=205)
    assert len(frames1["frames"]) == len(frames2["frames"])

    # Test clearing cache
    provider.clear_cache()
    assert len(provider._frame_cache) == 0


def test_cli_arg_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies CLI argument parsing for webserver runner."""
    monkeypatch.setattr("sys.argv", ["__main__.py", "--host", "127.0.0.1", "--port", "8080", "--reload"])
    args = parse_args()
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.reload is True
