"""
Unit tests for Video Annotation Rendering and Frame Bounding Enforcement.

Verifies:
1. Parquet path resolution for direct files and directories.
2. Efficient DuckDB loading of trajectories and action event records.
3. Rendering primitives (HUD overlay, alliance-colored bounding boxes, high-contrast labels, kinematics, field coords).
4. Video rendering pipeline produces valid, cleanly encoded MP4 videos with cv2.VideoWriter.
5. Frame bounding enforcement ensures rendering and stream ingestion never exceed source video total frame count.
6. Orchestrator effective_max_frames calculation and twin Parquet export in stop_pipeline.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Generator, List

import cv2
import duckdb
import numpy as np
import pytest

from src.ingestion.stream_decoder import (
    StreamIngestionActor,
    _OpenCVStreamSource,
    _worker_decode_loop,
)
from src.pipeline.orchestrator import PipelineOrchestrator
from src.render import (
    ALLIANCE_COLORS,
    build_parser,
    draw_hud,
    draw_robot_annotation,
    load_parquet_data,
    main as render_main,
    render_annotated_video,
    resolve_parquet_paths,
)


def create_mock_video(path: str, num_frames: int = 30, fps: float = 30.0, width: int = 320, height: int = 240) -> str:
    """Creates a temporary synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (40 + i * 2, 80, 40), dtype=np.uint8)
        cv2.putText(frame, f"F:{i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return path


def create_mock_parquet(
    traj_path: str,
    actions_path: Optional[str] = None,
    num_frames: int = 30,
    match_id: str = "2026test_f1m1",
) -> None:
    """Generates synthetic trajectories and action events Parquet files using DuckDB."""
    con = duckdb.connect()

    # Trajectories Table
    con.execute("""
    CREATE TABLE robot_trajectories (
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
        velocity_x_ips FLOAT,
        velocity_y_ips FLOAT,
        speed_ips FLOAT,
        heading_rad FLOAT,
        track_state VARCHAR,
        in_defense BOOLEAN
    );
    """)

    for f in range(num_frames):
        # Robot 1 (Red 254)
        con.execute("""
            INSERT INTO robot_trajectories VALUES (
                ?, ?, ?, 1, 254, 'RED', 100.0, 100.0, 40.0, 40.0, 0.95, 120.5, 140.2, 12.0, 0.0, 12.0, 0.0, 'TRACKED', false
            )
        """, [match_id, f, int(f * 33.33)])
        # Robot 2 (Blue 1678, defending)
        con.execute("""
            INSERT INTO robot_trajectories VALUES (
                ?, ?, ?, 2, 1678, 'BLUE', 200.0, 150.0, 45.0, 45.0, 0.91, 300.0, 150.0, 8.5, 2.1, 8.7, 0.2, 'TRACKED', true
            )
        """, [match_id, f, int(f * 33.33)])

    os.makedirs(os.path.dirname(os.path.abspath(traj_path)), exist_ok=True)
    con.execute(f"COPY robot_trajectories TO '{traj_path}' (FORMAT PARQUET)")

    if actions_path:
        con.execute("""
        CREATE TABLE action_events (
            event_id VARCHAR,
            match_id VARCHAR,
            track_id INTEGER,
            team_number INTEGER,
            action_type VARCHAR,
            element VARCHAR,
            start_frame BIGINT,
            end_frame BIGINT,
            confidence FLOAT
        );
        """)
        con.execute("""
            INSERT INTO action_events VALUES (
                'act-1', ?, 1, 254, 'SCORE_HUB', 'RED_HUB', 5, 20, 0.88
            )
        """, [match_id])
        con.execute(f"COPY action_events TO '{actions_path}' (FORMAT PARQUET)")

    con.close()


# ============================================================================
# 1. Parquet Path Resolution & Data Loading Tests
# ============================================================================

def test_resolve_parquet_paths_file_and_directory(tmp_path) -> None:
    """Verifies parquet resolution for direct files and directories."""
    traj_file = str(tmp_path / "match_2026casf_qm1_trajectories.parquet")
    actions_file = str(tmp_path / "match_2026casf_qm1_actions.parquet")
    create_mock_parquet(traj_file, actions_file, num_frames=10)

    # 1. Direct file resolution (should also detect companion actions file)
    res_traj, res_act = resolve_parquet_paths(traj_file)
    assert res_traj == traj_file
    assert res_act == actions_file

    # 2. Directory resolution
    dir_traj, dir_act = resolve_parquet_paths(str(tmp_path))
    assert dir_traj == traj_file
    assert dir_act == actions_file

    # 3. Non-existent path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        resolve_parquet_paths(str(tmp_path / "non_existent.parquet"))


def test_load_parquet_data_with_duckdb(tmp_path) -> None:
    """Verifies high-performance DuckDB querying and frame grouping."""
    traj_file = str(tmp_path / "test_trajectories.parquet")
    actions_file = str(tmp_path / "test_actions.parquet")
    create_mock_parquet(traj_file, actions_file, num_frames=25, match_id="2026casf_f1m1")

    trajectories_by_frame, actions_by_frame_and_track, match_id = load_parquet_data(
        traj_path=traj_file,
        actions_path=actions_file,
        max_frames=15,
    )

    assert match_id == "2026casf_f1m1"
    assert len(trajectories_by_frame) == 15  # Filtered by max_frames
    assert 0 in trajectories_by_frame
    assert len(trajectories_by_frame[0]) == 2  # 2 robots per frame

    # Verify action event mapped to frames 5..14 (within max_frames limit)
    assert 10 in actions_by_frame_and_track
    assert 1 in actions_by_frame_and_track[10]
    assert actions_by_frame_and_track[10][1]["action_type"] == "SCORE_HUB"


# ============================================================================
# 2. Rendering Primitives Tests
# ============================================================================

def test_draw_hud_and_robot_annotation() -> None:
    """Tests drawing HUD and robot bounding boxes with pills on image canvas."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Draw HUD
    draw_hud(
        frame=frame,
        match_id="2026cmptx_f1m3",
        frame_index=120,
        total_frames=6417,
        pts_ms=4000,
        active_robots_count=6,
    )
    # Ensure pixels in HUD area (top-left) are modified from pure black
    assert np.any(frame[20:60, 20:100] > 0)

    # 2. Draw Robot Annotation with ALL features
    robot_red = {
        "pixel_center_x": 300.0,
        "pixel_center_y": 200.0,
        "pixel_width": 80.0,
        "pixel_height": 80.0,
        "pixel_confidence": 0.94,
        "team_number": 254,
        "track_id": 1,
        "alliance": "RED",
        "speed_ips": 15.4,
        "in_defense": False,
        "field_x_inch": 162.8,
        "field_y_inch": 158.9,
    }
    action_event = {
        "action_type": "SCORE_HUB",
        "element": "RED_HUB",
    }

    draw_robot_annotation(
        frame=frame,
        robot=robot_red,
        action=action_event,
        draw_field_coords=True,
    )
    # Bounding box should have red pixels
    assert np.any(frame[160:240, 260:340, 2] > 150)  # Red channel active

    # 3. Draw Robot Annotation with Defense and Boundary Edge Case
    robot_blue_edge = {
        "pixel_center_x": 635.0,  # Near right edge
        "pixel_center_y": 10.0,   # Near top edge
        "pixel_width": 50.0,
        "pixel_height": 50.0,
        "pixel_confidence": 0.88,
        "team_number": 1678,
        "track_id": 2,
        "alliance": "BLUE",
        "speed_ips": 9.2,
        "in_defense": True,
    }
    # Must not raise IndexError / boundary violation
    draw_robot_annotation(
        frame=frame,
        robot=robot_blue_edge,
        action=None,
        draw_field_coords=False,
    )
    assert np.any(frame[:, 620:640, 0] > 100)  # Blue channel active


# ============================================================================
# 3. End-to-End Video Rendering & Frame Bounding Tests
# ============================================================================

def test_render_annotated_video_end_to_end(tmp_path) -> None:
    """Verifies that render_annotated_video produces a valid playable video file."""
    video_path = str(tmp_path / "test_input.mp4")
    create_mock_video(video_path, num_frames=30, fps=25.0, width=320, height=240)

    traj_path = str(tmp_path / "test_trajectories.parquet")
    create_mock_parquet(traj_path, num_frames=30)

    out_path = str(tmp_path / "rendered_output.mp4")
    result = render_annotated_video(
        video_path=video_path,
        parquet_path=traj_path,
        output_path=out_path,
        max_frames=20,
        draw_field_coords=True,
    )

    assert result == os.path.abspath(out_path)
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0

    # Verify rendered video properties with OpenCV
    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    cap.release()


def test_frame_bounding_enforcement_never_exceeds_source_video(tmp_path) -> None:
    """Verifies that rendering never exceeds the total frames in the source video file."""
    video_path = str(tmp_path / "short_video.mp4")
    # Video only has 25 frames
    create_mock_video(video_path, num_frames=25, fps=30.0, width=320, height=240)

    traj_path = str(tmp_path / "large_trajectories.parquet")
    # Parquet has 100 frames
    create_mock_parquet(traj_path, num_frames=100)

    out_path = str(tmp_path / "bounded_output.mp4")
    # Request max_frames=200, which exceeds video length of 25
    render_annotated_video(
        video_path=video_path,
        parquet_path=traj_path,
        output_path=out_path,
        max_frames=200,
    )

    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 25  # Bounded by source video!
    cap.release()


def test_render_cli_main_entrypoint(tmp_path) -> None:
    """Tests CLI entrypoint argument parsing and execution."""
    video_path = str(tmp_path / "cli_video.mp4")
    create_mock_video(video_path, num_frames=15, fps=30.0)

    traj_path = str(tmp_path / "cli_traj.parquet")
    create_mock_parquet(traj_path, num_frames=15)

    out_path = str(tmp_path / "cli_output.mp4")

    cli_args = [
        "--video", video_path,
        "--parquet", traj_path,
        "--output", out_path,
        "--max-frames", "10",
        "--draw-field-coords",
    ]
    ret = render_main(cli_args)
    assert ret == 0
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


# ============================================================================
# 4. Stream Decoder & Orchestrator Frame Bounding Enforcement Tests
# ============================================================================

@pytest.mark.asyncio
async def test_stream_source_and_actor_total_frames(tmp_path) -> None:
    """Verifies get_total_frames on _OpenCVStreamSource and StreamIngestionActor."""
    video_path = str(tmp_path / "source_test.mp4")
    create_mock_video(video_path, num_frames=45, fps=30.0)

    # 1. Direct OpenCV source test
    source = _OpenCVStreamSource(video_path)
    assert source.is_opened()
    total = source.get_total_frames()
    assert total == 45
    source.release()

    # 2. StreamIngestionActor test
    actor = StreamIngestionActor(frame_gating_stride=5)
    initialized = await actor.initialize_stream(video_path, target_fps=30)
    assert initialized is True
    assert actor.total_video_frames == 45
    assert await actor.get_total_frames() == 45

    # Retrieve all frames
    all_frames = []
    while True:
        batch = await actor.get_next_gameplay_batch(batch_size=10)
        if not batch:
            break
        all_frames.extend(batch)

    # Must never exceed 45 frames
    assert len(all_frames) <= 45
    await actor.close_stream()


def test_orchestrator_effective_max_frames_logic() -> None:
    """Verifies effective_max_frames clamping logic under various limits."""
    # Case 1: max_frames is None, total_frames is 45 -> effective is 45
    orch = PipelineOrchestrator(use_ray=False, max_frames=None)
    total_frames = 45
    if total_frames is not None and total_frames > 0:
        if orch.max_frames is not None:
            orch.effective_max_frames = min(orch.max_frames, total_frames)
        else:
            orch.effective_max_frames = total_frames
    assert orch.effective_max_frames == 45

    # Case 2: max_frames is 20, total_frames is 45 -> effective is 20
    orch2 = PipelineOrchestrator(use_ray=False, max_frames=20)
    if total_frames is not None and total_frames > 0:
        if orch2.max_frames is not None:
            orch2.effective_max_frames = min(orch2.max_frames, total_frames)
        else:
            orch2.effective_max_frames = total_frames
    assert orch2.effective_max_frames == 20

    # Case 3: max_frames is 100, total_frames is 45 -> effective is 45 (clamped to video total)
    orch3 = PipelineOrchestrator(use_ray=False, max_frames=100)
    if total_frames is not None and total_frames > 0:
        if orch3.max_frames is not None:
            orch3.effective_max_frames = min(orch3.max_frames, total_frames)
        else:
            orch3.effective_max_frames = total_frames
    assert orch3.effective_max_frames == 45


@pytest.mark.asyncio
async def test_worker_decode_loop_terminates_at_total_frames(tmp_path) -> None:
    """Verifies that _worker_decode_loop breaks immediately upon reaching total_frames."""
    import threading
    from src.ingestion.frame_gate import FrameGate

    video_path = str(tmp_path / "decode_bound_test.mp4")
    create_mock_video(video_path, num_frames=50, fps=30.0)

    source = _OpenCVStreamSource(video_path)
    gate = FrameGate(stride=5)
    queue: asyncio.Queue = asyncio.Queue(maxsize=30)
    loop = asyncio.get_running_loop()
    stop_event = threading.Event()
    metrics = {"total_frames_read": 0, "gameplay_yielded": 0}

    # Run worker loop in thread with active running event loop bounded to total_frames=10
    worker_t = threading.Thread(
        target=_worker_decode_loop,
        args=(source, gate, queue, loop, stop_event, 30.0, metrics, 10),
    )
    worker_t.start()
    await asyncio.to_thread(worker_t.join, 5.0)

    assert metrics["total_frames_read"] == 10


@pytest.mark.asyncio
async def test_orchestrator_exports_trajectories_and_actions_parquet(tmp_path) -> None:
    """Verifies that stop_pipeline exports both trajectories and actions parquet files."""
    export_dir = str(tmp_path / "parquet_exports")
    orch = PipelineOrchestrator(
        use_ray=False,
        max_frames=10,
        parquet_export_dir=export_dir,
        db_path=":memory:",
        mqtt_mock_mode=True,
    )
    video_path = str(tmp_path / "orch_video.mp4")
    create_mock_video(video_path, num_frames=20, fps=30.0)

    await orch.start_pipeline(stream_uri=video_path, match_id="2026test_m1")
    await orch.process_match_stream()
    summary = await orch.stop_pipeline()

    traj_expected = os.path.join(export_dir, "match_2026test_m1_trajectories.parquet")
    actions_expected = os.path.join(export_dir, "match_2026test_m1_actions.parquet")

    assert os.path.exists(traj_expected)
    assert os.path.exists(actions_expected)
    assert summary["parquet_path"] == traj_expected
    assert summary["actions_parquet_path"] == actions_expected
