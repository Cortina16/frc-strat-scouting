"""FastAPI Application for 3D Match Digital Twin Webserver.

Exposes REST endpoints for match metadata, field CAD geometry, robot and ball telemetry frames,
and reconciled action events, with CORS support and static asset delivery.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.server.data_provider import MatchDataProvider

logger = logging.getLogger(__name__)

# Directory setup
SERVER_DIR = Path(__file__).resolve().parent
STATIC_DIR = SERVER_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Application instantiation
app = FastAPI(
    title="3D Match Digital Twin Webserver",
    description="Telemetry and CAD topology service for FRC 2026 REBUILT scouting and digital twin visualization",
    version="1.0.0",
)

# Global data provider instance
data_provider = MatchDataProvider()
app.state.data_provider = data_provider

# Enable CORS for cross-origin client consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset directory with symlink following enabled
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), follow_symlink=True), name="static")


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Translates FileNotFoundError into a 404 HTTP response."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.get("/")
@app.head("/")
def serve_index() -> FileResponse:
    """Serves the 3D visualizer web application at root."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static assets directory")
    return FileResponse(str(index_file))


@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Health check endpoint indicating service availability and default port."""
    return {"status": "ok", "port": 3001}


@app.get("/api/matches")
def get_matches() -> List[Dict[str, Any]]:
    """Returns available matches discovered in data/parquet."""
    provider: MatchDataProvider = getattr(app.state, "data_provider", data_provider)
    return provider.get_available_matches()


@app.get("/api/match/{match_id}/field")
def get_match_field_cad(match_id: str) -> Dict[str, Any]:
    """Returns field CAD geometry, elements, and AprilTags for the match."""
    provider: MatchDataProvider = getattr(app.state, "data_provider", data_provider)
    try:
        return provider.get_field_cad()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/field")
def get_global_field_cad() -> Dict[str, Any]:
    """Global endpoint alias for field CAD topology."""
    provider: MatchDataProvider = getattr(app.state, "data_provider", data_provider)
    try:
        return provider.get_field_cad()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/field/model")
@app.get("/model/field.glb")
@app.head("/model/field.glb")
def get_field_model_glb() -> FileResponse:
    """Returns the official 3D CAD model (field.glb) with GLTF binary MIME type."""
    model_path = Path("field.glb").resolve()
    if not model_path.exists():
        model_path = STATIC_DIR / "models" / "field.glb"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="field.glb CAD model not found")
    return FileResponse(
        str(model_path),
        media_type="model/gltf-binary",
        filename="field.glb",
    )


@app.get("/api/match/{match_id}/summary")
def get_match_summary(match_id: str) -> Dict[str, Any]:
    """Returns aggregated summary metrics, durations, teams, and action counts."""
    provider: MatchDataProvider = getattr(app.state, "data_provider", data_provider)
    try:
        return provider.get_match_summary(match_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/match/{match_id}/events")
def get_match_events(match_id: str) -> List[Dict[str, Any]]:
    """Returns reconciled action localization events for the match."""
    provider: MatchDataProvider = getattr(app.state, "data_provider", data_provider)
    try:
        return provider.get_match_events(match_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/match/{match_id}/frames")
def get_match_frames(
    match_id: str,
    start: Optional[int] = Query(None, description="Starting frame index (inclusive)"),
    end: Optional[int] = Query(None, description="Ending frame index (inclusive)"),
    start_frame: Optional[int] = Query(None, description="Alias for start frame index"),
    end_frame: Optional[int] = Query(None, description="Alias for end frame index"),
    limit: Optional[int] = Query(None, description="Maximum number of frames to return"),
) -> Dict[str, Any]:
    """Returns cached robot trajectories and ball kinematics for playback and scrubbing."""
    provider: MatchDataProvider = getattr(app.state, "data_provider", data_provider)
    actual_start = start if start is not None else start_frame
    actual_end = end if end is not None else end_frame

    try:
        return provider.get_match_frames(
            match_id=match_id,
            start_frame=actual_start,
            end_frame=actual_end,
            limit=limit,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
