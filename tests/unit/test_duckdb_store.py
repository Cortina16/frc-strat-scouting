"""Unit tests for Module 08: DuckDB Analytical Engine & Spatio-Temporal Queries.

Verifies:
1. Interface Contract Validation: DuckDBStorageActor inherits StorageActorBase and conforms to StorageActorProtocol.
2. Ingestion & High-Cardinality Performance: Ingest >= 10,000 synthetic trajectory rows across 6 competing robots;
   verify all rows are stored accurately with correct schema types.
3. Vectorized Spatio-Temporal ASOF JOIN: Merge trajectory coordinates with discrete scoring events for each robot;
   assert execution latency is < 15 ms leveraging DuckDB SIMD vectorization.
4. Parquet Export & Round-Trip Verification: Export table to Parquet file via export_match_parquet; read back with
   DuckDB SELECT * FROM read_parquet(...) and verify row count and column integrity.
5. Specialized Analytical Queries: Verify cycle time calculation from Starting Line / Neutral Zone through Trenches
   to Alliance Hub, velocity degradation metrics under defense, and full match summary aggregates.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import tempfile
import time
from typing import Dict, List, Tuple

import duckdb
import pytest

logger = logging.getLogger(__name__)

from src.analytics import (
    BLUE_HUB_POS,
    DuckDBStorageActor,
    DuckDBStore,
    ExtendedActionSegment,
    ExtendedRobotTrackletState,
    RED_HUB_POS,
    create_tables,
    export_table_to_parquet,
    query_asof_join,
    query_cycle_times,
    query_match_summary,
    query_velocity_degradation,
)
from src.common.interfaces import StorageActorBase, StorageActorProtocol
from src.common.types import (
    ActionSegment,
    ActionType,
    AllianceColor,
    BoundingBox,
    FieldElement,
    RobotTrackletState,
    TrackState,
)


# ============================================================================
# Fixtures & Synthetic Data Generators
# ============================================================================

SIX_TEAMS = [254, 1678, 1323, 971, 1114, 2056]


def _make_tracklet(
    track_id: int,
    team_number: int,
    alliance: AllianceColor,
    field_x: float,
    field_y: float,
    velocity_x: float = 10.0,
    velocity_y: float = 0.0,
    speed: float = 10.0,
    heading: float = 0.0,
    track_state: TrackState = TrackState.TRACKED,
    in_defense: bool = False,
    frame_index: int = 0,
    timestamp_ms: int = 0,
    match_id: str = "test_match_001",
) -> ExtendedRobotTrackletState:
    """Helper constructing a valid RobotTrackletState fixture."""
    if speed is not None:
        velocity_x = speed
        velocity_y = 0.0
    bbox = BoundingBox(
        center_x=320.0,
        center_y=240.0,
        width=80.0,
        height=60.0,
        confidence=0.92,
        class_id=0,
    )
    return ExtendedRobotTrackletState(
        track_id=track_id,
        team_number=team_number,
        alliance=alliance,
        pixel_bbox=bbox,
        field_x_inch=field_x,
        field_y_inch=field_y,
        velocity_x_ips=velocity_x,
        velocity_y_ips=velocity_y,
        heading_rad=heading,
        track_state=track_state,
        in_defense=in_defense,
        timestamp_ms=timestamp_ms,
        match_id=match_id,
    )


def _make_action(
    event_id: str,
    track_id: int,
    team_number: int,
    action_type: ActionType,
    element: FieldElement,
    start_time_ms: int,
    end_time_ms: int,
    confidence: float = 0.95,
    field_coords: Tuple[float, float] = (162.8, 158.85),
    is_reconciled: bool = True,
    reconciled_posterior: float = 0.98,
    match_id: str = "test_match_001",
) -> ExtendedActionSegment:
    """Helper constructing a valid ActionSegment fixture."""
    return ExtendedActionSegment(
        event_id=event_id,
        track_id=track_id,
        team_number=team_number,
        action_type=action_type,
        element=element,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        start_frame=int(start_time_ms * 30 / 1000),
        end_frame=int(end_time_ms * 30 / 1000),
        confidence=confidence,
        diou=0.88,
        pyramid_level=1,
        field_coords=field_coords,
        is_reconciled=is_reconciled,
        reconciled_posterior=reconciled_posterior,
        match_id=match_id,
    )



# ============================================================================
# 1. Interface Contract Validation Tests
# ============================================================================

def test_duckdb_storage_actor_inherits_base() -> None:
    """Assert DuckDBStorageActor inherits from StorageActorBase."""
    assert issubclass(DuckDBStorageActor, StorageActorBase)


def test_duckdb_storage_actor_satisfies_protocol() -> None:
    """Assert DuckDBStorageActor satisfies StorageActorProtocol."""
    actor = DuckDBStorageActor(match_id="test_contract")
    assert isinstance(actor, StorageActorBase)
    assert isinstance(actor, StorageActorProtocol)


def test_duckdb_storage_actor_methods_are_async() -> None:
    """Assert all protocol methods are coroutine functions with correct signatures."""
    actor = DuckDBStorageActor(match_id="test_contract")
    assert inspect.iscoroutinefunction(actor.insert_trajectory_batch)
    assert inspect.iscoroutinefunction(actor.insert_action_events)
    assert inspect.iscoroutinefunction(actor.execute_asof_join_query)
    assert inspect.iscoroutinefunction(actor.export_match_parquet)


# ============================================================================
# 2. Ingestion & High-Cardinality Performance (>= 10,000 Rows)
# ============================================================================

@pytest.mark.asyncio
async def test_high_cardinality_trajectory_ingestion() -> None:
    """Ingest >= 10,000 synthetic trajectory rows across 6 competing robots.

    Verify exact row counts, columnar schema types, and data integrity.
    """
    actor = DuckDBStorageActor(match_id="match_cardinality_001", fps=30.0)

    # 6 robots across 1,700 frames = 10,200 trajectory points
    num_frames = 1700
    rows_per_frame = 6
    total_expected_rows = num_frames * rows_per_frame  # 10,200 rows

    alliances = [
        AllianceColor.RED,
        AllianceColor.RED,
        AllianceColor.RED,
        AllianceColor.BLUE,
        AllianceColor.BLUE,
        AllianceColor.BLUE,
    ]

    t0 = time.perf_counter()
    # Batch frames in chunks of 50 frames (300 rows per batch)
    chunk_size = 50
    for chunk_start in range(0, num_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_frames)
        batch_trajectories: List[RobotTrackletState] = []
        for f_idx in range(chunk_start, chunk_end):
            ts_ms = int(f_idx * 1000.0 / 30.0)
            for r_idx in range(rows_per_frame):
                team = SIX_TEAMS[r_idx]
                alliance = alliances[r_idx]
                in_defense = (f_idx % 7 == 0) and (r_idx == 0)  # Team 254 periodic defense
                spd = 35.0 if in_defense else (90.0 + (f_idx % 20))

                t = _make_tracklet(
                    track_id=r_idx + 1,
                    team_number=team,
                    alliance=alliance,
                    field_x=100.0 + (f_idx % 400),
                    field_y=50.0 + (r_idx * 40.0),
                    velocity_x=spd * 0.8,
                    velocity_y=spd * 0.6,
                    speed=spd,
                    heading=0.05 * (f_idx % 60),
                    in_defense=in_defense,
                    frame_index=f_idx,
                    timestamp_ms=ts_ms,
                    match_id="match_cardinality_001",
                )
                batch_trajectories.append(t)

        await actor.insert_trajectory_batch(batch_trajectories, frame_index=chunk_start)

    ingest_time_s = time.perf_counter() - t0
    logger.info("Ingested %d rows in %.3f seconds (%.0f rows/sec)", total_expected_rows, ingest_time_s, total_expected_rows / ingest_time_s)

    # Verify row count in DuckDB
    res_count = actor.store.query("SELECT COUNT(*) FROM robot_trajectories WHERE match_id = 'match_cardinality_001'")
    assert res_count[0][0] == total_expected_rows
    assert total_expected_rows >= 10000

    # Verify columnar schema definition
    columns_desc = actor.store.query("DESCRIBE robot_trajectories")
    column_types = {col[0]: col[1] for col in columns_desc}

    expected_types = {
        "match_id": "VARCHAR",
        "frame_index": "BIGINT",
        "timestamp_ms": "BIGINT",
        "track_id": "INTEGER",
        "team_number": "INTEGER",
        "alliance": "VARCHAR",
        "pixel_center_x": "FLOAT",
        "pixel_center_y": "FLOAT",
        "pixel_width": "FLOAT",
        "pixel_height": "FLOAT",
        "pixel_confidence": "FLOAT",
        "field_x_inch": "FLOAT",
        "field_y_inch": "FLOAT",
        "velocity_x_ips": "FLOAT",
        "velocity_y_ips": "FLOAT",
        "speed_ips": "FLOAT",
        "heading_rad": "FLOAT",
        "track_state": "VARCHAR",
        "in_defense": "BOOLEAN",
    }
    for col_name, expected_type in expected_types.items():
        assert col_name in column_types, f"Missing column {col_name}"
        assert expected_type in column_types[col_name].upper(), f"Type mismatch for {col_name}: expected {expected_type}, got {column_types[col_name]}"

    # Verify per-robot row count parity (each robot must have exactly 1,700 rows)
    team_counts = actor.store.query("""
        SELECT team_number, COUNT(*) 
        FROM robot_trajectories 
        WHERE match_id = 'match_cardinality_001' 
        GROUP BY team_number 
        ORDER BY team_number
    """)
    assert len(team_counts) == 6
    for t_num, count in team_counts:
        assert count == num_frames, f"Robot {t_num} expected {num_frames} rows, got {count}"

    await actor.close()


# ============================================================================
# 3. Vectorized Spatio-Temporal ASOF JOIN & Sub-15ms Latency Benchmark
# ============================================================================

@pytest.mark.asyncio
async def test_vectorized_asof_join_latency_and_accuracy() -> None:
    """Execute vectorized ASOF JOIN linking scoring events to continuous trajectory dynamics.

    Assert query execution latency < 15 ms leveraging DuckDB SIMD vectorization,
    and verify exact temporal and spatial correspondence.
    """
    actor = DuckDBStorageActor(match_id="asof_perf_match", fps=30.0)

    # 1. Populate high-cardinality trajectory points (12,000 rows across 6 robots)
    total_frames = 2000
    all_trajectories: List[RobotTrackletState] = []
    for f in range(total_frames):
        ts = int(f * (1000.0 / 30.0))
        for r_idx, team in enumerate(SIX_TEAMS):
            alliance = AllianceColor.RED if r_idx < 3 else AllianceColor.BLUE
            t = _make_tracklet(
                track_id=r_idx + 1,
                team_number=team,
                alliance=alliance,
                field_x=160.0 + (f % 50),
                field_y=155.0 + (f % 40),
                velocity_x=15.0,
                velocity_y=5.0,
                speed=15.81,
                heading=0.32,
                frame_index=f,
                timestamp_ms=ts,
                match_id="asof_perf_match",
            )
            all_trajectories.append(t)

    await actor.insert_trajectory_batch(all_trajectories, frame_index=0)

    # 2. Populate 50 discrete scoring events for Team 254
    scoring_actions: List[ActionSegment] = []
    for i in range(50):
        # Action starts at frame 30 * i + 10 (approx t = i * 1000 + 333 ms)
        action_ts = int((i * 35 + 10) * 1000.0 / 30.0)
        action = _make_action(
            event_id=f"score_event_{i:03d}",
            track_id=1,
            team_number=254,
            action_type=ActionType.SCORE_HUB,
            element=FieldElement.RED_HUB,
            start_time_ms=action_ts,
            end_time_ms=action_ts + 1200,
            confidence=0.96,
            field_coords=RED_HUB_POS,
            is_reconciled=True,
            reconciled_posterior=0.99,
            match_id="asof_perf_match",
        )
        scoring_actions.append(action)

    await actor.insert_action_events(scoring_actions)

    # 3. Execute ASOF JOIN query and assert execution latency < 15 ms
    # Perform a single warmup query to populate DuckDB query plan cache
    await actor.execute_asof_join_query(match_id="asof_perf_match", team_number=254)
    res = await actor.execute_asof_join_query(match_id="asof_perf_match", team_number=254)

    latency_ms = res["query_latency_ms"]
    event_count = res["event_count"]
    events = res["events"]

    logger.info("ASOF JOIN latency: %.2f ms for %d events", latency_ms, event_count)
    assert event_count == 50
    assert latency_ms < 15.0, f"Expected ASOF JOIN latency < 15 ms, got {latency_ms:.2f} ms"

    # 4. Verify spatio-temporal accuracy:
    # Each joined trajectory timestamp must be <= action start_time_ms and within 1 frame interval (<= 34 ms)
    for ev in events:
        action_start = ev["start_time_ms"]
        traj_ts = ev["trajectory_timestamp_ms"]
        assert traj_ts is not None
        assert traj_ts <= action_start, f"ASOF invariant violated: traj_ts {traj_ts} > action_start {action_start}"
        delta_ms = action_start - traj_ts
        assert delta_ms <= 35, f"Trajectory gap too large: {delta_ms} ms"
        assert ev["team_number"] == 254
        assert ev["action_type"] == "SCORE_HUB"
        assert ev["element"] == "RED_HUB"
        assert abs(ev["speed_ips"] - 15.81) < 0.1

    await actor.close()


# ============================================================================
# 4. Parquet Export & Round-Trip Verification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_parquet_export_and_roundtrip_integrity() -> None:
    """Export table to Parquet file via export_match_parquet.

    Read exported Parquet back with DuckDB SELECT * FROM read_parquet(...)
    and verify row count and column integrity.
    """
    actor = DuckDBStorageActor(match_id="parquet_match", fps=30.0)

    # Populate 1,200 trajectory points
    sample_trajectories: List[RobotTrackletState] = []
    for f in range(200):
        for r_idx, team in enumerate(SIX_TEAMS):
            t = _make_tracklet(
                track_id=r_idx + 1,
                team_number=team,
                alliance=AllianceColor.RED if r_idx < 3 else AllianceColor.BLUE,
                field_x=100.0 + f,
                field_y=80.0 + r_idx * 10,
                speed=25.5,
                frame_index=f,
                timestamp_ms=int(f * 1000.0 / 30.0),
                match_id="parquet_match",
            )
            sample_trajectories.append(t)

    await actor.insert_trajectory_batch(sample_trajectories, frame_index=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case A: Export directly to .parquet file
        parquet_file = os.path.join(tmpdir, "trajectories_export.parquet")
        exported_path = await actor.export_match_parquet(parquet_file)
        assert exported_path == parquet_file
        assert os.path.exists(parquet_file)
        assert os.path.getsize(parquet_file) > 0

        # Read back with DuckDB read_parquet
        verify_con = duckdb.connect()
        read_res = verify_con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()
        assert read_res is not None
        assert read_res[0] == 1200

        # Verify column names and non-null values
        cols_res = verify_con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_file}')").fetchall()
        col_names = [c[0] for c in cols_res]
        assert "match_id" in col_names
        assert "field_x_inch" in col_names
        assert "velocity_x_ips" in col_names
        assert "speed_ips" in col_names
        assert "in_defense" in col_names

        # Case B: Export to directory
        dir_export = os.path.join(tmpdir, "parquet_dir")
        await actor.export_match_parquet(dir_export)
        assert os.path.exists(os.path.join(dir_export, "robot_trajectories.parquet"))
        assert os.path.exists(os.path.join(dir_export, "action_events.parquet"))

        verify_con.close()

    await actor.close()


# ============================================================================
# 5. Specialized Analytical Queries: Cycle Times & Velocity Degradation
# ============================================================================

@pytest.mark.asyncio
async def test_cycle_time_calculation_action_based() -> None:
    """Verify average cycle time calculation from sequential intake to score action events."""
    actor = DuckDBStorageActor(match_id="cycle_match", fps=30.0)

    # 3 scoring cycles for Team 254:
    # Cycle 1: Intake at 2000 ms, Score at 8000 ms -> Duration = 6.0s
    # Cycle 2: Intake at 11000 ms, Score at 17500 ms -> Duration = 6.5s
    # Cycle 3: Intake at 20000 ms, Score at 28000 ms -> Duration = 8.0s
    # Expected average = (6.0 + 6.5 + 8.0) / 3 = 6.833s
    actions = [
        _make_action("a1", 1, 254, ActionType.INTAKE_FUEL, FieldElement.DEPOT, 2000, 3000, field_coords=(325.6, 280.0), match_id="cycle_match"),
        _make_action("a2", 1, 254, ActionType.SCORE_HUB, FieldElement.RED_HUB, 8000, 9000, field_coords=RED_HUB_POS, match_id="cycle_match"),
        _make_action("a3", 1, 254, ActionType.OUTPOST_CORRAL_CYCLE, FieldElement.RED_OUTPOST, 11000, 12000, field_coords=(0.0, 158.85), match_id="cycle_match"),
        _make_action("a4", 1, 254, ActionType.SCORE_HUB, FieldElement.RED_HUB, 17500, 18500, field_coords=RED_HUB_POS, match_id="cycle_match"),
        _make_action("a5", 1, 254, ActionType.INTAKE_FUEL, FieldElement.DEPOT, 20000, 21000, field_coords=(325.6, 37.7), match_id="cycle_match"),
        _make_action("a6", 1, 254, ActionType.SCORE_HUB, FieldElement.RED_HUB, 28000, 29000, field_coords=RED_HUB_POS, match_id="cycle_match"),
    ]
    await actor.insert_action_events(actions)

    metrics = await actor.get_cycle_analytics(match_id="cycle_match", team_number=254)

    assert metrics["team_number"] == 254
    assert metrics["cycle_count"] == 3
    assert abs(metrics["average_cycle_time_s"] - 6.833) < 0.05
    assert metrics["min_cycle_time_s"] == 6.0
    assert metrics["max_cycle_time_s"] == 8.0
    assert metrics["median_cycle_time_s"] == 6.5

    await actor.close()


@pytest.mark.asyncio
async def test_cycle_time_calculation_spatial_route() -> None:
    """Verify cycle time detection from trajectory spatial waypoints:

    Human Starting Line / Neutral Zone -> Trenches -> Alliance Hub.
    """
    actor = DuckDBStorageActor(match_id="spatial_cycle_match", fps=30.0)

    # Robot moves from Neutral Zone (x=300, y=150) -> Trench (x=280, y=30) -> Red Hub (x=162.8, y=158.85)
    # Cycle 1: Source at 1000 ms, Trench at 3000 ms, Hub at 6000 ms (Duration = 5.0s)
    # Cycle 2: Source at 10000 ms, Trench at 13000 ms, Hub at 17000 ms (Duration = 7.0s)
    traj_points = [
        # Cycle 1
        (1000, 300.0, 150.0),  # Neutral Zone
        (3000, 280.0, 30.0),   # Trench South
        (6000, 165.0, 158.0),  # Red Hub
        # Reset / Transit
        (8000, 200.0, 150.0),
        # Cycle 2
        (10000, 310.0, 160.0), # Neutral Zone
        (13000, 290.0, 280.0), # Trench North
        (17000, 160.0, 160.0), # Red Hub
    ]

    trajectories: List[RobotTrackletState] = []
    for f_idx, (ts, x, y) in enumerate(traj_points):
        t = _make_tracklet(
            track_id=1,
            team_number=1678,
            alliance=AllianceColor.RED,
            field_x=x,
            field_y=y,
            speed=60.0,
            frame_index=f_idx,
            timestamp_ms=ts,
            match_id="spatial_cycle_match",
        )
        trajectories.append(t)

    await actor.insert_trajectory_batch(trajectories, frame_index=0)

    metrics = await actor.get_cycle_analytics(match_id="spatial_cycle_match", team_number=1678)
    assert metrics["team_number"] == 1678
    assert metrics["cycle_count"] == 2
    assert abs(metrics["average_cycle_time_s"] - 6.0) < 0.01
    assert metrics["min_cycle_time_s"] == 5.0
    assert metrics["max_cycle_time_s"] == 7.0

    await actor.close()


@pytest.mark.asyncio
async def test_velocity_degradation_under_defense() -> None:
    """Verify velocity degradation metrics comparing speed when in defense vs free transit."""
    actor = DuckDBStorageActor(match_id="defense_eval_match", fps=30.0)

    # 100 free samples @ 100.0 ips, 50 defense samples @ 40.0 ips
    # Expected:
    # avg_free_speed = 100.0
    # avg_defense_speed = 40.0
    # speed_drop = 60.0 ips
    # degradation_pct = 60.0%
    trajectories: List[RobotTrackletState] = []
    for f in range(150):
        is_def = f >= 100
        spd = 40.0 if is_def else 100.0
        t = _make_tracklet(
            track_id=1,
            team_number=971,
            alliance=AllianceColor.BLUE,
            field_x=200.0 + f,
            field_y=150.0,
            velocity_x=spd,
            velocity_y=0.0,
            speed=spd,
            in_defense=is_def,
            frame_index=f,
            timestamp_ms=int(f * 1000.0 / 30.0),
            match_id="defense_eval_match",
        )
        trajectories.append(t)

    await actor.insert_trajectory_batch(trajectories, frame_index=0)

    res = await actor.get_velocity_degradation(match_id="defense_eval_match", team_number=971)

    assert res["team_number"] == 971
    assert abs(res["avg_free_speed_ips"] - 100.0) < 0.01
    assert abs(res["avg_defense_speed_ips"] - 40.0) < 0.01
    assert abs(res["speed_drop_ips"] - 60.0) < 0.01
    assert abs(res["degradation_pct"] - 60.0) < 0.01
    assert abs(res["degradation_ratio"] - 0.4) < 0.01
    assert res["free_samples"] == 100
    assert res["defense_samples"] == 50

    await actor.close()


@pytest.mark.asyncio
async def test_full_match_summary_metrics() -> None:
    """Verify comprehensive match summary aggregates across all competing robots."""
    actor = DuckDBStorageActor(match_id="summary_match_001", fps=30.0)

    # Ingest trajectories for 6 robots
    trajectories: List[RobotTrackletState] = []
    for f in range(60):
        ts = int(f * 1000.0 / 30.0)
        for r_idx, team in enumerate(SIX_TEAMS):
            alliance = AllianceColor.RED if r_idx < 3 else AllianceColor.BLUE
            in_def = (team == 254 and f >= 30)
            spd = 20.0 if in_def else 50.0
            t = _make_tracklet(
                track_id=r_idx + 1,
                team_number=team,
                alliance=alliance,
                field_x=50.0 + f * 2.0,  # Moves 120 inches
                field_y=100.0,
                speed=spd,
                in_defense=in_def,
                frame_index=f,
                timestamp_ms=ts,
                match_id="summary_match_001",
            )
            trajectories.append(t)

    await actor.insert_trajectory_batch(trajectories, frame_index=0)

    # Ingest some actions
    actions = [
        _make_action("act_1", 1, 254, ActionType.SCORE_HUB, FieldElement.RED_HUB, 500, 1200, match_id="summary_match_001"),
        _make_action("act_2", 1, 254, ActionType.INTAKE_FUEL, FieldElement.DEPOT, 1500, 2000, match_id="summary_match_001"),
        _make_action("act_3", 2, 1678, ActionType.CLIMB_TOWER, FieldElement.RED_TOWER, 1500, 2000, match_id="summary_match_001"),
    ]
    await actor.insert_action_events(actions)

    summaries = await actor.get_match_summary_metrics(match_id="summary_match_001")
    assert len(summaries) == 6

    # Verify team 254 metrics
    team_254 = await actor.get_team_summary(match_id="summary_match_001", team_number=254)
    assert team_254 is not None
    assert team_254["team_number"] == 254
    assert team_254["alliance"] == "RED"
    assert team_254["total_trajectory_points"] == 60
    assert team_254["score_hub_count"] == 1
    assert team_254["intake_count"] == 1
    assert team_254["climb_count"] == 0
    assert team_254["time_in_defense_s"] > 0.0
    assert team_254["total_distance_inches"] > 100.0
    assert team_254["total_distance_feet"] > 8.0

    # Verify team 1678 climb count
    team_1678 = await actor.get_team_summary(match_id="summary_match_001", team_number=1678)
    assert team_1678 is not None
    assert team_1678["climb_count"] == 1

    await actor.close()


# ============================================================================
# 6. Edge Cases & Robustness
# ============================================================================

@pytest.mark.asyncio
async def test_empty_batches_and_graceful_handling() -> None:
    """Assert empty batch insertions and queries on non-existent data do not fail."""
    actor = DuckDBStorageActor(match_id="empty_match")

    # Empty batch inserts
    await actor.insert_trajectory_batch([], frame_index=0)
    await actor.insert_action_events([])

    # Empty ASOF join
    asof_res = await actor.execute_asof_join_query(match_id="empty_match", team_number=9999)
    assert asof_res["event_count"] == 0
    assert asof_res["events"] == []
    assert asof_res["query_latency_ms"] >= 0.0

    # Empty cycle analytics
    cycle_res = await actor.get_cycle_analytics(match_id="empty_match", team_number=9999)
    assert cycle_res["cycle_count"] == 0
    assert cycle_res["average_cycle_time_s"] == 0.0

    # Empty velocity degradation
    deg_res = await actor.get_velocity_degradation(match_id="empty_match", team_number=9999)
    assert deg_res["degradation_pct"] == 0.0

    # Non-existent team summary
    summary = await actor.get_team_summary(match_id="empty_match", team_number=9999)
    assert summary is None

    await actor.close()
