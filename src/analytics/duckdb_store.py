"""DuckDB In-Process Analytical OLAP Database & Spatio-Temporal Queries.

Module 08: DuckDB Analytical Engine & Spatio-Temporal Queries
Provides high-throughput batch ingestion of robot trajectories and localized action events,
vectorized spatio-temporal ASOF JOIN linking scoring events to continuous trajectory dynamics,
Parquet file export via native DuckDB COPY, and specialized queries for cycle times,
velocity degradation under defensive pressure, and match aggregates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import duckdb

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

from src.common.interfaces import StorageActorBase, StorageActorProtocol
from src.common.types import (
    ActionSegment,
    ActionType,
    AllianceColor,
    BallTrackletState,
    BoundingBox,
    FieldElement,
    RobotTrackletState,
    TrackState,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtendedRobotTrackletState(RobotTrackletState):
    """Extension of RobotTrackletState supporting dynamic defense, timestamp, and match annotations."""
    in_defense: bool = False
    timestamp_ms: Optional[int] = None
    match_id: Optional[str] = None


@dataclass(slots=True)
class ExtendedActionSegment(ActionSegment):
    """Extension of ActionSegment supporting match_id annotation."""
    match_id: Optional[str] = None


# ============================================================================
# Field Geometry Coordinates & Zone Constants (Inches)
# ============================================================================

RED_HUB_POS: Tuple[float, float] = (162.8, 158.85)
BLUE_HUB_POS: Tuple[float, float] = (488.4, 158.85)
RED_TOWER_POS: Tuple[float, float] = (90.0, 158.85)
BLUE_TOWER_POS: Tuple[float, float] = (561.2, 158.85)
DEPOT_NORTH_POS: Tuple[float, float] = (325.6, 280.0)
DEPOT_SOUTH_POS: Tuple[float, float] = (325.6, 37.7)

# Spatial Zones
SOURCE_RED_X_MAX: float = 120.0
SOURCE_BLUE_X_MIN: float = 530.0
NEUTRAL_ZONE_X_MIN: float = 250.0
NEUTRAL_ZONE_X_MAX: float = 400.0

# Trench lanes: North (y >= 250) and South (y <= 65) across mid-field (x in 260..390)
TRENCH_X_MIN: float = 260.0
TRENCH_X_MAX: float = 390.0
TRENCH_NORTH_Y_MIN: float = 250.0
TRENCH_SOUTH_Y_MAX: float = 65.0

HUB_INTERACTION_RADIUS: float = 60.0

# ============================================================================
# Columnar Table Schemas (DDL)
# ============================================================================

DDL_ROBOT_TRAJECTORIES: str = """
CREATE TABLE IF NOT EXISTS robot_trajectories (
    match_id VARCHAR,
    frame_index BIGINT,
    timestamp_ms BIGINT,
    track_id INTEGER,
    team_number INTEGER,
    alliance VARCHAR,
    pixel_center_x FLOAT,
    pixel_center_y FLOAT,
    pixel_width FLOAT,
    pixel_height FLOAT,
    pixel_confidence FLOAT,
    field_x_inch FLOAT,
    field_y_inch FLOAT,
    field_z_inch FLOAT,
    center_rel_x_inch FLOAT,
    center_rel_y_inch FLOAT,
    center_rel_z_inch FLOAT,
    depth_inch FLOAT,
    robot_width_inch FLOAT,
    robot_length_inch FLOAT,
    robot_height_inch FLOAT,
    velocity_x_ips FLOAT,
    velocity_y_ips FLOAT,
    speed_ips FLOAT,
    heading_rad FLOAT,
    track_state VARCHAR,
    in_defense BOOLEAN
);
"""

DDL_ACTION_EVENTS: str = """
CREATE TABLE IF NOT EXISTS action_events (
    event_id VARCHAR,
    match_id VARCHAR,
    track_id INTEGER,
    team_number INTEGER,
    action_type VARCHAR,
    element VARCHAR,
    start_time_ms BIGINT,
    end_time_ms BIGINT,
    start_frame BIGINT,
    end_frame BIGINT,
    duration_s FLOAT,
    confidence FLOAT,
    diou FLOAT,
    pyramid_level INTEGER,
    field_x_inch FLOAT,
    field_y_inch FLOAT,
    is_reconciled BOOLEAN,
    reconciled_posterior FLOAT
);
"""

DDL_BALL_TRAJECTORIES: str = """
CREATE TABLE IF NOT EXISTS ball_trajectories (
    match_id VARCHAR,
    frame_index BIGINT,
    timestamp_ms BIGINT,
    ball_id INTEGER,
    pixel_center_x FLOAT,
    pixel_center_y FLOAT,
    pixel_radius FLOAT,
    field_x_inch FLOAT,
    field_y_inch FLOAT,
    field_z_inch FLOAT,
    center_rel_x_inch FLOAT,
    center_rel_y_inch FLOAT,
    center_rel_z_inch FLOAT,
    depth_inch FLOAT,
    velocity_x_ips FLOAT,
    velocity_y_ips FLOAT,
    ball_state VARCHAR,
    occluded_by_robot_id INTEGER,
    is_cluster BOOLEAN,
    cluster_ball_count INTEGER
);
"""

DDL_CAMERA_POSES: str = """
CREATE TABLE IF NOT EXISTS camera_poses (
    match_id VARCHAR,
    frame_index BIGINT,
    timestamp_ms BIGINT,
    camera_x_inch FLOAT,
    camera_y_inch FLOAT,
    camera_z_inch FLOAT,
    rvec_0 FLOAT,
    rvec_1 FLOAT,
    rvec_2 FLOAT,
    tvec_0 FLOAT,
    tvec_1 FLOAT,
    tvec_2 FLOAT,
    focal_length FLOAT,
    reproj_rmse FLOAT,
    pnp_converged BOOLEAN,
    visible_tags VARCHAR,
    tag_distances VARCHAR
);
"""

DDL_INDEXES: str = """
CREATE INDEX IF NOT EXISTS idx_trajectories_asof ON robot_trajectories (match_id, team_number, timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_actions_asof ON action_events (match_id, team_number, start_time_ms);
CREATE INDEX IF NOT EXISTS idx_balls_frame ON ball_trajectories (match_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_camera_poses_frame ON camera_poses (match_id, frame_index);
"""


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Creates the columnar tables and indexes for trajectories, balls, action events, and camera poses."""
    conn.execute(DDL_ROBOT_TRAJECTORIES)
    conn.execute(DDL_ACTION_EVENTS)
    conn.execute(DDL_BALL_TRAJECTORIES)
    conn.execute(DDL_CAMERA_POSES)
    try:
        conn.execute(DDL_INDEXES)
    except Exception as e:
        logger.debug("Indexes not applied or already present: %s", e)


# ============================================================================
# PyArrow Schemas (For High-Throughput SIMD Vectorized Ingestion)
# ============================================================================

if pa is not None:
    PYARROW_TRAJECTORY_SCHEMA = pa.schema([
        ("match_id", pa.string()),
        ("frame_index", pa.int64()),
        ("timestamp_ms", pa.int64()),
        ("track_id", pa.int32()),
        ("team_number", pa.int32()),
        ("alliance", pa.string()),
        ("pixel_center_x", pa.float32()),
        ("pixel_center_y", pa.float32()),
        ("pixel_width", pa.float32()),
        ("pixel_height", pa.float32()),
        ("pixel_confidence", pa.float32()),
        ("field_x_inch", pa.float32()),
        ("field_y_inch", pa.float32()),
        ("field_z_inch", pa.float32()),
        ("center_rel_x_inch", pa.float32()),
        ("center_rel_y_inch", pa.float32()),
        ("center_rel_z_inch", pa.float32()),
        ("depth_inch", pa.float32()),
        ("robot_width_inch", pa.float32()),
        ("robot_length_inch", pa.float32()),
        ("robot_height_inch", pa.float32()),
        ("velocity_x_ips", pa.float32()),
        ("velocity_y_ips", pa.float32()),
        ("speed_ips", pa.float32()),
        ("heading_rad", pa.float32()),
        ("track_state", pa.string()),
        ("in_defense", pa.bool_()),
    ])

    PYARROW_ACTION_SCHEMA = pa.schema([
        ("event_id", pa.string()),
        ("match_id", pa.string()),
        ("track_id", pa.int32()),
        ("team_number", pa.int32()),
        ("action_type", pa.string()),
        ("element", pa.string()),
        ("start_time_ms", pa.int64()),
        ("end_time_ms", pa.int64()),
        ("start_frame", pa.int64()),
        ("end_frame", pa.int64()),
        ("duration_s", pa.float32()),
        ("confidence", pa.float32()),
        ("diou", pa.float32()),
        ("pyramid_level", pa.int32()),
        ("field_x_inch", pa.float32()),
        ("field_y_inch", pa.float32()),
        ("is_reconciled", pa.bool_()),
        ("reconciled_posterior", pa.float32()),
    ])
else:
    PYARROW_TRAJECTORY_SCHEMA = None  # type: ignore[assignment]
    PYARROW_ACTION_SCHEMA = None  # type: ignore[assignment]


# ============================================================================
# Standalone Analytics Query Helpers
# ============================================================================

def query_asof_join(
    conn: duckdb.DuckDBPyConnection,
    match_id: str,
    team_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Vectorized ASOF JOIN merging action events with closest preceding trajectory sample.

    Matches: e.match_id = t.match_id AND e.team_number = t.team_number AND e.start_time_ms >= t.timestamp_ms
    """
    if team_number is not None and team_number > 0:
        query = """
        SELECT 
            e.event_id,
            e.match_id,
            e.track_id,
            e.team_number,
            e.action_type,
            e.element,
            e.start_time_ms,
            e.end_time_ms,
            e.duration_s,
            e.confidence,
            e.is_reconciled,
            e.reconciled_posterior,
            t.frame_index AS trajectory_frame_index,
            t.timestamp_ms AS trajectory_timestamp_ms,
            t.field_x_inch,
            t.field_y_inch,
            t.velocity_x_ips,
            t.velocity_y_ips,
            t.speed_ips,
            t.heading_rad,
            t.track_state,
            t.in_defense
        FROM action_events e
        ASOF JOIN robot_trajectories t
            ON e.match_id = t.match_id
           AND e.team_number = t.team_number
           AND e.start_time_ms >= t.timestamp_ms
        WHERE e.match_id = ? AND e.team_number = ?
        ORDER BY e.start_time_ms ASC
        """
        params = [match_id, team_number]
    else:
        query = """
        SELECT 
            e.event_id,
            e.match_id,
            e.track_id,
            e.team_number,
            e.action_type,
            e.element,
            e.start_time_ms,
            e.end_time_ms,
            e.duration_s,
            e.confidence,
            e.is_reconciled,
            e.reconciled_posterior,
            t.frame_index AS trajectory_frame_index,
            t.timestamp_ms AS trajectory_timestamp_ms,
            t.field_x_inch,
            t.field_y_inch,
            t.velocity_x_ips,
            t.velocity_y_ips,
            t.speed_ips,
            t.heading_rad,
            t.track_state,
            t.in_defense
        FROM action_events e
        ASOF JOIN robot_trajectories t
            ON e.match_id = t.match_id
           AND e.team_number = t.team_number
           AND e.start_time_ms >= t.timestamp_ms
        WHERE e.match_id = ?
        ORDER BY e.start_time_ms ASC
        """
        params = [match_id]

    cursor = conn.execute(query, params)
    desc = cursor.description
    if desc is None:
        return []
    columns = [d[0] for d in desc]
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def query_velocity_degradation(
    conn: duckdb.DuckDBPyConnection,
    match_id: str,
    team_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Calculates velocity degradation under defensive pressure (in_defense = TRUE vs in_defense = FALSE)."""
    filter_team = "AND team_number = ?" if (team_number is not None and team_number > 0) else ""
    params: List[Any] = [match_id]
    if team_number is not None and team_number > 0:
        params.append(team_number)

    sql = f"""
    SELECT 
        team_number,
        match_id,
        AVG(CASE WHEN in_defense = FALSE THEN speed_ips ELSE NULL END) AS avg_free_speed_ips,
        AVG(CASE WHEN in_defense = TRUE THEN speed_ips ELSE NULL END) AS avg_defense_speed_ips,
        MAX(CASE WHEN in_defense = FALSE THEN speed_ips ELSE NULL END) AS max_free_speed_ips,
        MAX(CASE WHEN in_defense = TRUE THEN speed_ips ELSE NULL END) AS max_defense_speed_ips,
        COUNT(CASE WHEN in_defense = FALSE THEN 1 ELSE NULL END) AS free_samples,
        COUNT(CASE WHEN in_defense = TRUE THEN 1 ELSE NULL END) AS defense_samples,
        COUNT(*) AS total_samples
    FROM robot_trajectories
    WHERE match_id = ? {filter_team}
    GROUP BY match_id, team_number
    ORDER BY team_number ASC
    """
    cursor = conn.execute(sql, params)
    desc = cursor.description
    if desc is None:
        return []
    columns = [d[0] for d in desc]
    rows = cursor.fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        item = dict(zip(columns, r))
        free_spd = float(item["avg_free_speed_ips"]) if item["avg_free_speed_ips"] is not None else 0.0
        def_spd = float(item["avg_defense_speed_ips"]) if item["avg_defense_speed_ips"] is not None else 0.0
        
        has_free = item["avg_free_speed_ips"] is not None
        has_def = item["avg_defense_speed_ips"] is not None

        if has_free and has_def and free_spd > 0.0:
            speed_drop = free_spd - def_spd
            deg_pct = ((free_spd - def_spd) / free_spd) * 100.0
            deg_ratio = def_spd / free_spd
        else:
            speed_drop = 0.0
            deg_pct = 0.0
            deg_ratio = 1.0 if not has_def else 0.0

        item["avg_free_speed_ips"] = free_spd
        item["avg_defense_speed_ips"] = def_spd
        item["speed_drop_ips"] = speed_drop
        item["degradation_pct"] = deg_pct
        item["degradation_ratio"] = deg_ratio
        results.append(item)

    return results


def query_cycle_times(
    conn: duckdb.DuckDBPyConnection,
    match_id: str,
    team_number: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Calculates scoring cycle times.

    Evaluates both:
    1. Action-level cycle transitions (INTAKE_FUEL / OUTPOST_CORRAL_CYCLE -> SCORE_HUB, or consecutive SCORE_HUB).
    2. Spatial spatio-temporal route detection from Source (Human Starting Line / Neutral Zone)
       through Trenches to Alliance Hub.
    """
    filter_team_act = "AND team_number = ?" if (team_number is not None and team_number > 0) else ""
    params_act: List[Any] = [match_id]
    if team_number is not None and team_number > 0:
        params_act.append(team_number)

    # 1. Action-based cycle query
    action_sql = f"""
    WITH ordered_actions AS (
        SELECT 
            match_id,
            team_number,
            action_type,
            element,
            start_time_ms,
            end_time_ms,
            LAG(action_type) OVER (PARTITION BY match_id, team_number ORDER BY start_time_ms) AS prev_type,
            LAG(element) OVER (PARTITION BY match_id, team_number ORDER BY start_time_ms) AS prev_element,
            LAG(start_time_ms) OVER (PARTITION BY match_id, team_number ORDER BY start_time_ms) AS prev_start_ms
        FROM action_events
        WHERE match_id = ? {filter_team_act}
          AND action_type IN ('INTAKE_FUEL', 'OUTPOST_CORRAL_CYCLE', 'TRANSIT_NEUTRAL_ZONE', 'SCORE_HUB')
    )
    SELECT 
        match_id,
        team_number,
        prev_type AS intake_action,
        prev_element AS intake_element,
        element AS score_element,
        prev_start_ms AS start_time_ms,
        start_time_ms AS end_time_ms,
        (start_time_ms - prev_start_ms) / 1000.0 AS duration_s
    FROM ordered_actions
    WHERE action_type = 'SCORE_HUB'
      AND prev_type IN ('INTAKE_FUEL', 'OUTPOST_CORRAL_CYCLE', 'TRANSIT_NEUTRAL_ZONE', 'SCORE_HUB')
      AND prev_start_ms IS NOT NULL
      AND start_time_ms > prev_start_ms
    ORDER BY team_number, start_time_ms
    """
    cursor = conn.execute(action_sql, params_act)
    action_cycle_rows = cursor.fetchall()
    desc = cursor.description
    cols = [d[0] for d in desc] if desc else []

    action_cycles_by_team: Dict[int, List[Dict[str, Any]]] = {}
    for r in action_cycle_rows:
        item = dict(zip(cols, r))
        t_num = item["team_number"]
        if t_num not in action_cycles_by_team:
            action_cycles_by_team[t_num] = []
        action_cycles_by_team[t_num].append(item)

    # 2. Trajectory-based spatial cycle analysis (detecting Source -> Trench -> Hub)
    filter_team_traj = "AND team_number = ?" if (team_number is not None and team_number > 0) else ""
    params_traj: List[Any] = [match_id]
    if team_number is not None and team_number > 0:
        params_traj.append(team_number)

    traj_sql = f"""
    SELECT 
        team_number,
        timestamp_ms,
        field_x_inch,
        field_y_inch
    FROM robot_trajectories
    WHERE match_id = ? {filter_team_traj}
    ORDER BY team_number, timestamp_ms ASC
    """
    cursor = conn.execute(traj_sql, params_traj)
    traj_rows = cursor.fetchall()

    spatial_cycles_by_team: Dict[int, List[Dict[str, Any]]] = {}
    current_team = None
    source_ts = None
    saw_trench = False

    for t_num, ts, x, y in traj_rows:
        if t_num != current_team:
            current_team = t_num
            source_ts = None
            saw_trench = False
            if t_num not in spatial_cycles_by_team:
                spatial_cycles_by_team[t_num] = []

        is_source = (
            (x <= SOURCE_RED_X_MAX)
            or (x >= SOURCE_BLUE_X_MIN)
            or (NEUTRAL_ZONE_X_MIN <= x <= NEUTRAL_ZONE_X_MAX and TRENCH_SOUTH_Y_MAX < y < TRENCH_NORTH_Y_MIN)
        )
        is_trench = (
            (TRENCH_X_MIN <= x <= TRENCH_X_MAX)
            and (y <= TRENCH_SOUTH_Y_MAX or y >= TRENCH_NORTH_Y_MIN)
        )
        dist_red = math.hypot(x - RED_HUB_POS[0], y - RED_HUB_POS[1])
        dist_blue = math.hypot(x - BLUE_HUB_POS[0], y - BLUE_HUB_POS[1])
        is_hub = dist_red <= HUB_INTERACTION_RADIUS or dist_blue <= HUB_INTERACTION_RADIUS

        if is_source:
            source_ts = ts
            saw_trench = False
        elif is_trench and source_ts is not None:
            saw_trench = True
        elif is_hub and source_ts is not None and saw_trench:
            dur_s = (ts - source_ts) / 1000.0
            if dur_s > 0.0:
                spatial_cycles_by_team[t_num].append({
                    "match_id": match_id,
                    "team_number": t_num,
                    "start_time_ms": source_ts,
                    "end_time_ms": ts,
                    "duration_s": dur_s,
                    "route": "SOURCE -> TRENCH -> ALLIANCE_HUB",
                })
            source_ts = None
            saw_trench = False

    # Consolidate results per team
    all_teams = set(action_cycles_by_team.keys()).union(spatial_cycles_by_team.keys())
    if team_number is not None and team_number > 0 and team_number not in all_teams:
        all_teams.add(team_number)

    results: List[Dict[str, Any]] = []
    for t_num in sorted(all_teams):
        # Prefer action cycles if available, otherwise spatial trajectory cycles
        act_cycles = action_cycles_by_team.get(t_num, [])
        spat_cycles = spatial_cycles_by_team.get(t_num, [])
        primary_cycles = act_cycles if len(act_cycles) > 0 else spat_cycles

        durations = [c["duration_s"] for c in primary_cycles]
        if durations:
            avg_cycle = float(sum(durations) / len(durations))
            med_cycle = float(sorted(durations)[len(durations) // 2])
            min_cycle = float(min(durations))
            max_cycle = float(max(durations))
        else:
            avg_cycle = 0.0
            med_cycle = 0.0
            min_cycle = 0.0
            max_cycle = 0.0

        results.append({
            "match_id": match_id,
            "team_number": t_num,
            "cycle_count": len(primary_cycles),
            "average_cycle_time_s": avg_cycle,
            "median_cycle_time_s": med_cycle,
            "min_cycle_time_s": min_cycle,
            "max_cycle_time_s": max_cycle,
            "action_cycles": act_cycles,
            "spatial_cycles": spat_cycles,
            "cycles": primary_cycles,
        })

    return results


def query_match_summary(
    conn: duckdb.DuckDBPyConnection,
    match_id: str,
) -> List[Dict[str, Any]]:
    """Calculates comprehensive full-match summary aggregates for each robot/team."""
    # 1. Trajectory aggregate metrics
    traj_agg_sql = """
    WITH ordered_pts AS (
        SELECT 
            team_number,
            alliance,
            timestamp_ms,
            speed_ips,
            in_defense,
            field_x_inch,
            field_y_inch,
            LAG(field_x_inch) OVER (PARTITION BY team_number ORDER BY timestamp_ms) AS prev_x,
            LAG(field_y_inch) OVER (PARTITION BY team_number ORDER BY timestamp_ms) AS prev_y
        FROM robot_trajectories
        WHERE match_id = ?
    )
    SELECT 
        team_number,
        MAX(alliance) AS alliance,
        COUNT(*) AS total_points,
        (MAX(timestamp_ms) - MIN(timestamp_ms)) / 1000.0 AS duration_s,
        AVG(speed_ips) AS avg_speed_ips,
        MAX(speed_ips) AS max_speed_ips,
        AVG(CASE WHEN in_defense = FALSE THEN speed_ips END) AS avg_free_speed_ips,
        AVG(CASE WHEN in_defense = TRUE THEN speed_ips END) AS avg_defense_speed_ips,
        COUNT(CASE WHEN in_defense = TRUE THEN 1 END) AS defense_points,
        SUM(CASE 
            WHEN prev_x IS NOT NULL THEN SQRT(POW(field_x_inch - prev_x, 2) + POW(field_y_inch - prev_y, 2))
            ELSE 0.0 
        END) AS total_distance_inches
    FROM ordered_pts
    GROUP BY team_number
    ORDER BY team_number ASC
    """
    cursor = conn.execute(traj_agg_sql, [match_id])
    desc = cursor.description
    cols = [d[0] for d in desc] if desc else []
    traj_rows = cursor.fetchall()
    traj_map = {r[0]: dict(zip(cols, r)) for r in traj_rows}

    # 2. Action aggregate metrics
    action_agg_sql = """
    SELECT 
        team_number,
        COUNT(CASE WHEN action_type = 'SCORE_HUB' THEN 1 END) AS score_hub_count,
        COUNT(CASE WHEN action_type IN ('INTAKE_FUEL', 'OUTPOST_CORRAL_CYCLE') THEN 1 END) AS intake_count,
        COUNT(CASE WHEN action_type = 'CLIMB_TOWER' THEN 1 END) AS climb_count,
        COUNT(CASE WHEN action_type = 'DEFENSIVE_ENGAGEMENT' THEN 1 END) AS defense_event_count,
        COUNT(CASE WHEN action_type = 'SCORE_HUB' AND is_reconciled THEN 1 END) AS reconciled_scores_count
    FROM action_events
    WHERE match_id = ?
    GROUP BY team_number
    """
    cursor = conn.execute(action_agg_sql, [match_id])
    desc = cursor.description
    act_cols = [d[0] for d in desc] if desc else []
    action_rows = cursor.fetchall()
    action_map = {r[0]: dict(zip(act_cols, r)) for r in action_rows}

    # 3. Cycle time metrics
    cycle_metrics = query_cycle_times(conn, match_id)
    cycle_map = {c["team_number"]: c for c in cycle_metrics}

    all_teams = sorted(set(traj_map.keys()).union(action_map.keys()))
    summaries: List[Dict[str, Any]] = []

    for t_num in all_teams:
        t_data = traj_map.get(t_num, {})
        a_data = action_map.get(t_num, {})
        c_data = cycle_map.get(t_num, {})

        total_pts = t_data.get("total_points", 0)
        def_pts = t_data.get("defense_points", 0)
        dur_s = float(t_data.get("duration_s", 0.0) or 0.0)
        dist_in = float(t_data.get("total_distance_inches", 0.0) or 0.0)
        dist_ft = dist_in / 12.0

        free_spd = float(t_data.get("avg_free_speed_ips", 0.0) or 0.0)
        def_spd = float(t_data.get("avg_defense_speed_ips", 0.0) or 0.0)
        if free_spd > 0.0 and t_data.get("avg_defense_speed_ips") is not None:
            deg_pct = ((free_spd - def_spd) / free_spd) * 100.0
        else:
            deg_pct = 0.0

        def_time_s = (float(def_pts) / float(total_pts) * dur_s) if total_pts > 0 else 0.0

        summary = {
            "match_id": match_id,
            "team_number": t_num,
            "alliance": t_data.get("alliance", "UNKNOWN"),
            "total_trajectory_points": total_pts,
            "active_duration_s": dur_s,
            "total_distance_inches": dist_in,
            "total_distance_feet": dist_ft,
            "avg_speed_ips": float(t_data.get("avg_speed_ips", 0.0) or 0.0),
            "max_speed_ips": float(t_data.get("max_speed_ips", 0.0) or 0.0),
            "avg_free_speed_ips": free_spd,
            "avg_defense_speed_ips": def_spd,
            "velocity_degradation_pct": deg_pct,
            "time_in_defense_s": def_time_s,
            "score_hub_count": a_data.get("score_hub_count", 0),
            "reconciled_scores_count": a_data.get("reconciled_scores_count", 0),
            "intake_count": a_data.get("intake_count", 0),
            "climb_count": a_data.get("climb_count", 0),
            "defense_event_count": a_data.get("defense_event_count", 0),
            "average_cycle_time_s": float(c_data.get("average_cycle_time_s", 0.0)),
            "cycle_count": int(c_data.get("cycle_count", 0)),
        }
        summaries.append(summary)

    return summaries


def export_table_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    file_path: str,
) -> str:
    """Exports a DuckDB table to a Parquet file using vectorized COPY command."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    sql = f"COPY {table_name} TO '{file_path}' (FORMAT PARQUET)"
    conn.execute(sql)
    return file_path


# ============================================================================
# DuckDBStore Class (Synchronous In-Process Analytical OLAP Engine)
# ============================================================================

class DuckDBStore:
    """Synchronous embedded DuckDB analytical OLAP engine."""

    def __init__(
        self,
        db_path: str = ":memory:",
        default_match_id: str = "match_001",
        fps: float = 30.0,
    ) -> None:
        self.db_path = db_path
        self.default_match_id = default_match_id
        self.fps = fps
        self._con: duckdb.DuckDBPyConnection = duckdb.connect(database=db_path)
        try:
            self._con.execute("SET preserve_insertion_order = false;")
        except Exception:
            pass
        create_tables(self._con)

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Underlying DuckDB database connection."""
        return self._con

    def insert_trajectory_batch(
        self,
        trajectories: List[RobotTrackletState],
        frame_index: int,
        match_id: Optional[str] = None,
    ) -> int:
        """Appends a batch of robot trajectory states to the `robot_trajectories` table.

        Uses PyArrow batch registration if available for SIMD vectorized columnar loading;
        otherwise falls back to parameterized batch INSERT.
        """
        if not trajectories:
            return 0

        m_id = match_id or self.default_match_id
        count = len(trajectories)

        match_ids = [getattr(t, "match_id", None) or m_id for t in trajectories]
        frame_indices = [frame_index] * count
        timestamps = [
            getattr(t, "timestamp_ms", None)
            if getattr(t, "timestamp_ms", None) is not None
            else int(frame_index * (1000.0 / self.fps))
            for t in trajectories
        ]
        track_ids = [int(t.track_id) for t in trajectories]
        team_numbers = [int(t.team_number) if t.team_number is not None else 0 for t in trajectories]
        alliances = [
            t.alliance.value if hasattr(t.alliance, "value") else str(t.alliance)
            for t in trajectories
        ]
        px_cx = [float(t.pixel_bbox.center_x) if t.pixel_bbox else 0.0 for t in trajectories]
        px_cy = [float(t.pixel_bbox.center_y) if t.pixel_bbox else 0.0 for t in trajectories]
        px_w = [float(t.pixel_bbox.width) if t.pixel_bbox else 0.0 for t in trajectories]
        px_h = [float(t.pixel_bbox.height) if t.pixel_bbox else 0.0 for t in trajectories]
        px_conf = [float(t.pixel_bbox.confidence) if t.pixel_bbox else 0.0 for t in trajectories]
        fx = [float(t.field_x_inch) for t in trajectories]
        fy = [float(t.field_y_inch) for t in trajectories]
        fz = [float(getattr(t, "field_z_inch", 0.0) or 0.0) for t in trajectories]
        cr_x = [
            float(getattr(t, "center_rel_x_inch", None) if getattr(t, "center_rel_x_inch", None) is not None else (t.field_x_inch - 325.6))
            for t in trajectories
        ]
        cr_y = [
            float(getattr(t, "center_rel_y_inch", None) if getattr(t, "center_rel_y_inch", None) is not None else (t.field_y_inch - 158.85))
            for t in trajectories
        ]
        cr_z = [
            float(getattr(t, "center_rel_z_inch", None) if getattr(t, "center_rel_z_inch", None) is not None else getattr(t, "field_z_inch", 0.0) or 0.0)
            for t in trajectories
        ]
        depths = [float(getattr(t, "depth_inch", 0.0) or 0.0) for t in trajectories]
        r_widths = [float(getattr(t, "robot_width_inch", 22.0) or 22.0) for t in trajectories]
        r_lengths = [float(getattr(t, "robot_length_inch", 28.0) or 28.0) for t in trajectories]
        r_heights = [float(getattr(t, "robot_height_inch", 28.0) or 28.0) for t in trajectories]
        vx = [float(t.velocity_x_ips) for t in trajectories]
        vy = [float(t.velocity_y_ips) for t in trajectories]
        speeds = [
            float(
                getattr(t, "speed_ips", None)
                if getattr(t, "speed_ips", None) is not None
                else math.hypot(t.velocity_x_ips, t.velocity_y_ips)
            )
            for t in trajectories
        ]
        headings = [float(t.heading_rad) for t in trajectories]
        track_states = [
            t.track_state.value if hasattr(t.track_state, "value") else str(t.track_state)
            for t in trajectories
        ]
        in_def: List[bool] = []
        for t in trajectories:
            val = getattr(t, "in_defense", None)
            if val is not None:
                in_def.append(bool(val))
            else:
                defended = False
                for other in trajectories:
                    if other.track_id != t.track_id and other.alliance != t.alliance:
                        dist = math.hypot(t.field_x_inch - other.field_x_inch, t.field_y_inch - other.field_y_inch)
                        if dist <= 72.0:
                            defended = True
                            break
                in_def.append(defended)

        if pa is not None and PYARROW_TRAJECTORY_SCHEMA is not None:
            cols = [
                pa.array(match_ids, type=pa.string()),
                pa.array(frame_indices, type=pa.int64()),
                pa.array(timestamps, type=pa.int64()),
                pa.array(track_ids, type=pa.int32()),
                pa.array(team_numbers, type=pa.int32()),
                pa.array(alliances, type=pa.string()),
                pa.array(px_cx, type=pa.float32()),
                pa.array(px_cy, type=pa.float32()),
                pa.array(px_w, type=pa.float32()),
                pa.array(px_h, type=pa.float32()),
                pa.array(px_conf, type=pa.float32()),
                pa.array(fx, type=pa.float32()),
                pa.array(fy, type=pa.float32()),
                pa.array(fz, type=pa.float32()),
                pa.array(cr_x, type=pa.float32()),
                pa.array(cr_y, type=pa.float32()),
                pa.array(cr_z, type=pa.float32()),
                pa.array(depths, type=pa.float32()),
                pa.array(r_widths, type=pa.float32()),
                pa.array(r_lengths, type=pa.float32()),
                pa.array(r_heights, type=pa.float32()),
                pa.array(vx, type=pa.float32()),
                pa.array(vy, type=pa.float32()),
                pa.array(speeds, type=pa.float32()),
                pa.array(headings, type=pa.float32()),
                pa.array(track_states, type=pa.string()),
                pa.array(in_def, type=pa.bool_()),
            ]
            tbl = pa.Table.from_arrays(cols, schema=PYARROW_TRAJECTORY_SCHEMA)
            token = f"_temp_traj_{uuid.uuid4().hex[:8]}"
            self._con.register(token, tbl)
            try:
                self._con.execute(f"INSERT INTO robot_trajectories SELECT * FROM {token}")
            finally:
                self._con.unregister(token)
        else:
            rows = list(
                zip(
                    match_ids,
                    frame_indices,
                    timestamps,
                    track_ids,
                    team_numbers,
                    alliances,
                    px_cx,
                    px_cy,
                    px_w,
                    px_h,
                    px_conf,
                    fx,
                    fy,
                    fz,
                    cr_x,
                    cr_y,
                    cr_z,
                    depths,
                    r_widths,
                    r_lengths,
                    r_heights,
                    vx,
                    vy,
                    speeds,
                    headings,
                    track_states,
                    in_def,
                )
            )
            sql = "INSERT INTO robot_trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            self._con.executemany(sql, rows)

        return count

    def insert_ball_trajectory_batch(
        self,
        balls: Sequence[BallTrackletState],
        frame_index: int,
        match_id: Optional[str] = None,
    ) -> int:
        """Appends a batch of yellow ball tracklet states to the `ball_trajectories` table."""
        if not balls:
            return 0

        m_id = match_id or self.default_match_id
        count = len(balls)
        ts = int(frame_index * (1000.0 / self.fps))

        rows = [
            (
                m_id,
                int(frame_index),
                int(ts),
                int(b.ball_id),
                float(b.pixel_bbox.center_x),
                float(b.pixel_bbox.center_y),
                float(b.radius_px),
                float(b.field_x_inch),
                float(b.field_y_inch),
                float(getattr(b, "field_z_inch", 2.955) if getattr(b, "field_z_inch", None) is not None else 2.955),
                float(getattr(b, "center_rel_x_inch", None) if getattr(b, "center_rel_x_inch", None) is not None else (b.field_x_inch - 325.6)),
                float(getattr(b, "center_rel_y_inch", None) if getattr(b, "center_rel_y_inch", None) is not None else (b.field_y_inch - 158.85)),
                float(getattr(b, "center_rel_z_inch", None) if getattr(b, "center_rel_z_inch", None) is not None else getattr(b, "field_z_inch", 2.955)),
                float(getattr(b, "depth_inch", 0.0) or 0.0),
                float(b.velocity_x_ips),
                float(b.velocity_y_ips),
                b.state.value if hasattr(b.state, "value") else str(b.state),
                int(b.occluded_by_robot_id) if b.occluded_by_robot_id is not None else None,
                bool(getattr(b, "is_cluster", False)),
                int(getattr(b, "cluster_ball_count", 1)),
            )
            for b in balls
        ]
        sql = "INSERT INTO ball_trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self._con.executemany(sql, rows)
        return count

    def insert_camera_poses(
        self,
        poses: Sequence[Any],
        match_id: Optional[str] = None,
    ) -> int:
        """Appends camera pose estimation records to the `camera_poses` table."""
        if not poses:
            return 0
        m_id = match_id or self.default_match_id
        count = len(poses)
        rows = []
        for p in poses:
            c = getattr(p, "camera_center", (0.0, 0.0, 0.0))
            if c is None:
                c = (0.0, 0.0, 0.0)
            rv = getattr(p, "rvec", (0.0, 0.0, 0.0))
            if rv is None:
                rv = (0.0, 0.0, 0.0)
            tv = getattr(p, "tvec", (0.0, 0.0, 0.0))
            if tv is None:
                tv = (0.0, 0.0, 0.0)
            vis_tags = getattr(p, "visible_tags", []) or []
            tag_dists = getattr(p, "tag_distances", {}) or {}
            rows.append((
                getattr(p, "match_id", m_id) or m_id,
                int(p.frame_index),
                int(p.timestamp_ms),
                float(c[0]), float(c[1]), float(c[2]),
                float(rv[0]), float(rv[1]), float(rv[2]),
                float(tv[0]), float(tv[1]), float(tv[2]),
                float(getattr(p, "focal_length", 1250.6) or 1250.6),
                float(getattr(p, "reproj_rmse", 0.0) or 0.0),
                bool(getattr(p, "converged", False)),
                json.dumps(vis_tags),
                json.dumps({str(k): float(v) for k, v in tag_dists.items()}),
            ))
        sql = "INSERT INTO camera_poses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        self._con.executemany(sql, rows)
        return count

    def insert_action_events(
        self,
        actions: List[ActionSegment],
        match_id: Optional[str] = None,
    ) -> int:
        """Inserts localized action events into the `action_events` table."""
        if not actions:
            return 0

        m_id = match_id or self.default_match_id
        count = len(actions)

        event_ids = [str(a.event_id) for a in actions]
        match_ids = [getattr(a, "match_id", None) or m_id for a in actions]
        track_ids = [int(a.track_id) for a in actions]
        team_numbers = [int(a.team_number) if a.team_number is not None else 0 for a in actions]
        action_types = [
            a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
            for a in actions
        ]
        elements = [
            a.element.value if hasattr(a.element, "value") else str(a.element)
            for a in actions
        ]
        start_ts = [int(a.start_time_ms) for a in actions]
        end_ts = [int(a.end_time_ms) for a in actions]
        start_frames = [int(a.start_frame) for a in actions]
        end_frames = [int(a.end_frame) for a in actions]
        durations = [
            float(
                getattr(a, "duration_s", None)
                if getattr(a, "duration_s", None) is not None
                else (a.end_time_ms - a.start_time_ms) / 1000.0
            )
            for a in actions
        ]
        confidences = [float(a.confidence) for a in actions]
        dious = [float(a.diou) for a in actions]
        pyramid_levels = [int(a.pyramid_level) for a in actions]
        fx = [float(a.field_coords[0]) if a.field_coords else 0.0 for a in actions]
        fy = [float(a.field_coords[1]) if a.field_coords else 0.0 for a in actions]
        is_reconciled = [bool(a.is_reconciled) for a in actions]
        posteriors = [float(a.reconciled_posterior) for a in actions]

        if pa is not None and PYARROW_ACTION_SCHEMA is not None:
            cols = [
                pa.array(event_ids, type=pa.string()),
                pa.array(match_ids, type=pa.string()),
                pa.array(track_ids, type=pa.int32()),
                pa.array(team_numbers, type=pa.int32()),
                pa.array(action_types, type=pa.string()),
                pa.array(elements, type=pa.string()),
                pa.array(start_ts, type=pa.int64()),
                pa.array(end_ts, type=pa.int64()),
                pa.array(start_frames, type=pa.int64()),
                pa.array(end_frames, type=pa.int64()),
                pa.array(durations, type=pa.float32()),
                pa.array(confidences, type=pa.float32()),
                pa.array(dious, type=pa.float32()),
                pa.array(pyramid_levels, type=pa.int32()),
                pa.array(fx, type=pa.float32()),
                pa.array(fy, type=pa.float32()),
                pa.array(is_reconciled, type=pa.bool_()),
                pa.array(posteriors, type=pa.float32()),
            ]
            tbl = pa.Table.from_arrays(cols, schema=PYARROW_ACTION_SCHEMA)
            token = f"_temp_actions_{uuid.uuid4().hex[:8]}"
            self._con.register(token, tbl)
            try:
                self._con.execute(f"INSERT INTO action_events SELECT * FROM {token}")
            finally:
                self._con.unregister(token)
        else:
            rows = list(
                zip(
                    event_ids,
                    match_ids,
                    track_ids,
                    team_numbers,
                    action_types,
                    elements,
                    start_ts,
                    end_ts,
                    start_frames,
                    end_frames,
                    durations,
                    confidences,
                    dious,
                    pyramid_levels,
                    fx,
                    fy,
                    is_reconciled,
                    posteriors,
                )
            )
            sql = "INSERT INTO action_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            self._con.executemany(sql, rows)

        return count

    def execute_asof_join(self, match_id: str, team_number: int) -> Dict[str, Any]:
        """Executes vectorized ASOF JOIN and returns matched events with query execution latency."""
        query = """
        SELECT 
            e.event_id,
            e.match_id,
            e.track_id,
            e.team_number,
            e.action_type,
            e.element,
            e.start_time_ms,
            e.end_time_ms,
            e.duration_s,
            e.confidence,
            e.is_reconciled,
            e.reconciled_posterior,
            t.frame_index AS trajectory_frame_index,
            t.timestamp_ms AS trajectory_timestamp_ms,
            t.field_x_inch,
            t.field_y_inch,
            t.velocity_x_ips,
            t.velocity_y_ips,
            t.speed_ips,
            t.heading_rad,
            t.track_state,
            t.in_defense
        FROM action_events e
        ASOF JOIN robot_trajectories t
            ON e.match_id = t.match_id
           AND e.team_number = t.team_number
           AND e.start_time_ms >= t.timestamp_ms
        WHERE e.match_id = ? AND e.team_number = ?
        ORDER BY e.start_time_ms ASC
        """
        params = [match_id, team_number]
        t0 = time.perf_counter()
        cursor = self._con.execute(query, params)
        rows = cursor.fetchall()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        desc = cursor.description
        columns = [d[0] for d in desc] if desc else []
        events = [dict(zip(columns, row)) for row in rows]

        return {
            "match_id": match_id,
            "team_number": team_number,
            "event_count": len(events),
            "query_latency_ms": latency_ms,
            "events": events,
        }

    def export_parquet(self, output_path: str, table_name: str = "robot_trajectories") -> str:
        """Exports DuckDB table(s) to Parquet file(s).

        If `output_path` ends with `.parquet`, writes `table_name` directly to that file.
        If `output_path` is a directory, writes `robot_trajectories.parquet` and `action_events.parquet`.
        """
        if output_path.endswith(".parquet"):
            return export_table_to_parquet(self._con, table_name, output_path)

        os.makedirs(output_path, exist_ok=True)
        traj_file = os.path.join(output_path, "robot_trajectories.parquet")
        actions_file = os.path.join(output_path, "action_events.parquet")
        balls_file = os.path.join(output_path, "ball_trajectories.parquet")
        camera_file = os.path.join(output_path, "camera_poses.parquet")
        export_table_to_parquet(self._con, "robot_trajectories", traj_file)
        export_table_to_parquet(self._con, "action_events", actions_file)
        try:
            cnt_row = self._con.execute("SELECT COUNT(*) FROM ball_trajectories").fetchone()
            if cnt_row and cnt_row[0] > 0:
                export_table_to_parquet(self._con, "ball_trajectories", balls_file)
        except Exception:
            pass
        try:
            cam_cnt = self._con.execute("SELECT COUNT(*) FROM camera_poses").fetchone()
            if cam_cnt and cam_cnt[0] > 0:
                export_table_to_parquet(self._con, "camera_poses", camera_file)
        except Exception:
            pass
        return traj_file

    def get_cycle_analytics(
        self,
        match_id: str,
        team_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns cycle time analytics for a specific team or all teams."""
        res = query_cycle_times(self._con, match_id=match_id, team_number=team_number)
        if team_number is not None and team_number > 0:
            for item in res:
                if item["team_number"] == team_number:
                    return item
            return {
                "match_id": match_id,
                "team_number": team_number,
                "cycle_count": 0,
                "average_cycle_time_s": 0.0,
                "median_cycle_time_s": 0.0,
                "min_cycle_time_s": 0.0,
                "max_cycle_time_s": 0.0,
                "action_cycles": [],
                "spatial_cycles": [],
                "cycles": [],
            }
        return {"match_id": match_id, "teams": res}

    def get_velocity_degradation(
        self,
        match_id: str,
        team_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns velocity degradation under defensive pressure."""
        res = query_velocity_degradation(self._con, match_id=match_id, team_number=team_number)
        if team_number is not None and team_number > 0:
            for item in res:
                if item["team_number"] == team_number:
                    return item
            return {
                "match_id": match_id,
                "team_number": team_number,
                "avg_free_speed_ips": 0.0,
                "avg_defense_speed_ips": 0.0,
                "speed_drop_ips": 0.0,
                "degradation_pct": 0.0,
                "degradation_ratio": 1.0,
                "free_samples": 0,
                "defense_samples": 0,
                "total_samples": 0,
            }
        return {"match_id": match_id, "teams": res}

    def get_match_summary_metrics(self, match_id: str) -> List[Dict[str, Any]]:
        """Returns full-match aggregate summary metrics for all robots in a match."""
        return query_match_summary(self._con, match_id=match_id)

    def get_team_summary(self, match_id: str, team_number: int) -> Optional[Dict[str, Any]]:
        """Returns summary metrics for a specific team in a match."""
        summaries = query_match_summary(self._con, match_id=match_id)
        for s in summaries:
            if s["team_number"] == team_number:
                return s
        return None

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Tuple[Any, ...]]:
        """Executes arbitrary SQL query and returns rows as tuples."""
        if params is not None:
            return self._con.execute(sql, params).fetchall()
        return self._con.execute(sql).fetchall()

    def query_dicts(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        """Executes arbitrary SQL query and returns rows as dictionaries."""
        cursor = self._con.execute(sql, params) if params is not None else self._con.execute(sql)
        desc = cursor.description
        if not desc:
            return []
        cols = [d[0] for d in desc]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        """Closes the underlying DuckDB connection."""
        try:
            self._con.close()
        except Exception as e:
            logger.debug("Error closing connection: %s", e)


# ============================================================================
# DuckDBStorageActor (Ray Pipeline Actor Implementing StorageActorBase)
# ============================================================================

class DuckDBStorageActor(StorageActorBase):
    """Ray Core actor for in-process DuckDB OLAP analytics & Parquet trajectory storage.

    Inherits from StorageActorBase and satisfies StorageActorProtocol.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        match_id: str = "default_match",
        fps: float = 30.0,
    ) -> None:
        self._store = DuckDBStore(db_path=db_path, default_match_id=match_id, fps=fps)
        self._lock = asyncio.Lock()
        self.match_id = match_id
        self.fps = fps

    @property
    def store(self) -> DuckDBStore:
        """Direct access to internal DuckDBStore."""
        return self._store

    async def insert_trajectory_batch(
        self,
        trajectories: List[RobotTrackletState],
        frame_index: int,
    ) -> None:
        """Appends trajectory state batch to DuckDB memory-mapped columnar table."""
        async with self._lock:
            self._store.insert_trajectory_batch(
                trajectories=trajectories,
                frame_index=frame_index,
                match_id=self.match_id,
            )

    async def insert_ball_trajectories(self, balls: Sequence[BallTrackletState], frame_index: int) -> None:
        """Appends ball trajectory state batch to DuckDB ball_trajectories table."""
        async with self._lock:
            self._store.insert_ball_trajectory_batch(
                balls=balls,
                frame_index=frame_index,
                match_id=self.match_id,
            )

    async def insert_action_events(self, actions: List[ActionSegment]) -> None:
        """Inserts reconciled action localization events into DuckDB action tables."""
        async with self._lock:
            self._store.insert_action_events(
                actions=actions,
                match_id=self.match_id,
            )

    async def insert_camera_poses(self, poses: Sequence[Any]) -> None:
        """Appends camera pose estimation records to DuckDB camera_poses table."""
        async with self._lock:
            self._store.insert_camera_poses(
                poses=poses,
                match_id=self.match_id,
            )

    async def execute_asof_join_query(
        self,
        match_id: str,
        team_number: int,
    ) -> Dict[str, Any]:
        """Executes vectorized ASOF JOIN merging trajectories with discrete scoring events."""
        async with self._lock:
            return self._store.execute_asof_join(
                match_id=match_id,
                team_number=team_number,
            )

    async def export_match_parquet(
        self,
        output_path: str,
        table_name: str = "robot_trajectories",
    ) -> str:
        """Flushes in-memory columnar data to disk for persistent archival."""
        async with self._lock:
            return self._store.export_parquet(
                output_path=output_path,
                table_name=table_name,
            )

    async def get_cycle_analytics(
        self,
        match_id: str,
        team_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculates average cycle time and route duration breakdown."""
        async with self._lock:
            return self._store.get_cycle_analytics(
                match_id=match_id,
                team_number=team_number,
            )

    async def get_velocity_degradation(
        self,
        match_id: str,
        team_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculates velocity degradation under defensive pressure."""
        async with self._lock:
            return self._store.get_velocity_degradation(
                match_id=match_id,
                team_number=team_number,
            )

    async def get_match_summary_metrics(self, match_id: str) -> List[Dict[str, Any]]:
        """Returns full-match summary metrics for all robots."""
        async with self._lock:
            return self._store.get_match_summary_metrics(match_id=match_id)

    async def get_team_summary(self, match_id: str, team_number: int) -> Optional[Dict[str, Any]]:
        """Returns match summary metrics for a specific team."""
        async with self._lock:
            return self._store.get_team_summary(match_id=match_id, team_number=team_number)

    async def close(self) -> None:
        """Closes the underlying database connection."""
        async with self._lock:
            self._store.close()
