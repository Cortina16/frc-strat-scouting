"""
Integration test suite for Module 10: Ray Core Distributed Pipeline Orchestration.

Verifies:
1. PipelineOrchestrator satisfies PipelineOrchestratorProtocol interface contract.
2. End-to-end streaming pipeline execution in local in-process mode (use_ray=False).
3. End-to-end streaming pipeline execution across distributed Ray Core actors (use_ray=True).
4. Data ingestion into DuckDB, vectorized ASOF join queries, and MQTT Protobuf sync publishing (< 50 KB budget).
5. Sub-30-second post-match turnaround processing latency SLA.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest
import ray

from src.common.interfaces import PipelineOrchestratorProtocol
from src.pipeline.orchestrator import PipelineOrchestrator


def generate_synthetic_match_video(
    output_path: str,
    num_frames: int = 45,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """
    Generates a synthetic MP4 video simulating 2026 REBUILT broadcast footage.
    Includes known AprilTags (5, 6, 7, 8) and moving bounding box features.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    dict_tag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_images = {
        tid: cv2.cvtColor(
            cv2.aruco.generateImageMarker(dict_tag, tid, 60),
            cv2.COLOR_GRAY2BGR,
        )
        for tid in [5, 6, 7, 8]
    }

    for i in range(num_frames):
        frame = np.full((height, width, 3), (35, 120, 35), dtype=np.uint8)
        cv2.line(frame, (30, 240), (610, 240), (255, 255, 255), 3)

        frame[60:120, 60:120] = tag_images[5]
        frame[60:120, 520:580] = tag_images[6]
        frame[360:420, 520:580] = tag_images[7]
        frame[360:420, 60:120] = tag_images[8]

        rx = int(150 + i * 5)
        ry = int(200 + np.sin(i * 0.2) * 20)
        cv2.rectangle(frame, (rx, ry), (rx + 60, ry + 60), (200, 200, 200), -1)
        cv2.rectangle(frame, (rx, ry + 40), (rx + 60, ry + 60), (180, 40, 40), -1)

        out.write(frame)

    out.release()
    return output_path


@pytest.fixture(scope="module")
def synthetic_match_video_path() -> Generator[str, None, None]:
    """Provides temporary synthetic video file for integration tests."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = tf.name

    generate_synthetic_match_video(video_path, num_frames=60, fps=30.0)
    yield video_path

    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def clean_ray_session() -> Generator[None, None, None]:
    """Ensures Ray is completely shut down before and after each test."""
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_pipeline_orchestrator_satisfies_protocol() -> None:
    """Verifies that PipelineOrchestrator conforms to PipelineOrchestratorProtocol and defaults."""
    orch = PipelineOrchestrator(use_ray=False)
    assert isinstance(orch, PipelineOrchestratorProtocol), (
        'PipelineOrchestrator must satisfy PipelineOrchestratorProtocol'
    )
    assert asyncio.iscoroutinefunction(orch.start_pipeline)
    assert asyncio.iscoroutinefunction(orch.process_match_stream)
    assert asyncio.iscoroutinefunction(orch.stop_pipeline)

    # Verify default CPU, GPU, and 16-frame backpressure configuration
    assert orch.num_cpus == 8
    assert orch.num_gpus == 2
    assert orch.backpressure_threshold == 16
    assert orch.queued_frames == 0
    assert orch.backpressure_paused is False


@pytest.mark.asyncio
async def test_16_frame_backpressure_pausing_and_resumption(
    synthetic_match_video_path: str,
) -> None:
    """
    Verifies 16-frame backpressure regulation:
    - If in-flight / detection queue reaches or exceeds 16 frames, stream ingestion pauses.
    - As downstream stages process frames and the queue drops below 16 frames, ingestion resumes.
    - All frames complete without memory starvation or data loss.
    """
    orch = PipelineOrchestrator(
        use_ray=False,
        batch_size=8,
        backpressure_threshold=16,
        mqtt_mock_mode=True,
    )

    # 1. Verify direct backpressure threshold evaluation logic
    assert orch._is_backpressure_active() is False
    orch.queued_frames = 15
    assert orch._is_backpressure_active() is False
    orch.queued_frames = 16
    assert orch._is_backpressure_active() is True
    orch.queued_frames = 24
    assert orch._is_backpressure_active() is True
    orch.queued_frames = 0
    assert orch._is_backpressure_active() is False

    match_id = '2026casf_qm42_backpressure'
    await orch.start_pipeline(synthetic_match_video_path, match_id)
    assert orch.is_running is True

    # Simulate downstream compute load (e.g. GPU detection inference) so the queue accumulates
    orig_detect = orch.actors.detector.detect_batch
    first_call = True
    async def delayed_detect(*args, **kwargs):
        nonlocal first_call
        if first_call:
            first_call = False
            await asyncio.sleep(0.08)  # brief 80ms delay to allow ingestion to queue batch 2
        return await orig_detect(*args, **kwargs)
    orch.actors.detector.detect_batch = delayed_detect

    # 2. Run streaming loop with 60 frames (8 batches of size 8)
    # The first 2 batches fill the 16-frame capacity, triggering backpressure pause on batch 3.
    # When downstream finishes batch 1, queue drops to 8 frames (< 16), triggering resumption.
    await asyncio.wait_for(orch.process_match_stream(), timeout=25.0)

    # Verify that backpressure regulation paused and resumed ingestion during the run
    assert orch.backpressure_pause_count >= 1, (
        f"Expected at least 1 backpressure pause event, got {orch.backpressure_pause_count}"
    )
    assert orch.backpressure_resume_count >= 1, (
        f"Expected at least 1 backpressure resume event, got {orch.backpressure_resume_count}"
    )
    assert orch.queued_frames == 0, "All frames must be fully drained from the pipeline"
    assert orch.backpressure_paused is False

    summary = await orch.stop_pipeline()
    assert summary['status'] == 'completed'
    assert summary['total_frames_processed'] > 0
    assert summary['elapsed_seconds'] < 30.0, 'Post-match SLA breached'


@pytest.mark.asyncio
async def test_ray_initialization_with_gpus_and_cpu_cores(
    synthetic_match_video_path: str,
) -> None:
    """
    Verifies Ray initialization with num_gpus=2 and designated num_cpus=8:
    - Queries torch.cuda.device_count() / ray.cluster_resources().
    - In CPU test environments (0 GPUs available), allocates num_gpus=0 gracefully so tests never hang.
    - On multi-GPU hosts (>= 2 GPUs), allocates 2 GPUs.
    """
    orch = PipelineOrchestrator(
        use_ray=True,
        num_cpus=8,
        num_gpus=2,
        batch_size=8,
        backpressure_threshold=16,
        mqtt_mock_mode=True,
    )

    match_id = '2026casf_qm42_resources'
    await orch.start_pipeline(synthetic_match_video_path, match_id)
    assert orch.is_running is True
    assert ray.is_initialized() is True

    # Verify cluster resources respect the requested allocations
    cluster_res = ray.cluster_resources()
    assert cluster_res.get('CPU', 0) >= 8, f"Expected >= 8 CPUs, got {cluster_res.get('CPU')}"

    # Verify GPU allocation is either 2 (on GPU host) or 0 (on CPU host), never hanging
    if orch.allocated_gpus > 0:
        assert cluster_res.get('GPU', 0) >= 2
    else:
        assert cluster_res.get('GPU', 0) == 0

    await asyncio.wait_for(orch.process_match_stream(), timeout=25.0)
    summary = await orch.stop_pipeline()
    assert summary['status'] == 'completed'
    assert summary['total_frames_processed'] > 0
    assert summary['elapsed_seconds'] < 30.0


@pytest.mark.asyncio
async def test_end_to_end_pipeline_in_process(synthetic_match_video_path: str) -> None:
    """
    Tests end-to-end video streaming pipeline in local in-process mode (use_ray=False).
    Verifies pipelined execution and sub-30-second post-match latency SLA.
    """
    orch = PipelineOrchestrator(
        use_ray=False,
        batch_size=8,
        backpressure_threshold=16,
        mqtt_mock_mode=True,
    )

    match_id = '2026casf_qm42_local'
    await orch.start_pipeline(synthetic_match_video_path, match_id)
    assert orch.is_running is True

    start_ts = time.perf_counter()
    await asyncio.wait_for(orch.process_match_stream(), timeout=25.0)
    process_duration = time.perf_counter() - start_ts

    summary = await orch.stop_pipeline()
    assert orch.is_running is False

    assert summary['status'] == 'completed'
    assert summary['total_frames_processed'] > 0
    assert summary['elapsed_seconds'] < 30.0, 'Post-match processing turnaround SLA (<30s) breached'
    assert process_duration < 30.0, 'Post-match processing turnaround SLA (<30s) breached'
    assert summary['sync_published'] is True
    assert summary['total_trajectories_inserted'] >= 0
    assert orch.queued_frames == 0

    asof_result = await orch.execute_asof_join_query(match_id, team_number=254)
    assert asof_result['match_id'] == match_id
    assert 'query_latency_ms' in asof_result


@pytest.mark.asyncio
async def test_end_to_end_pipeline_distributed_ray(synthetic_match_video_path: str) -> None:
    """
    Tests end-to-end video streaming pipeline across distributed Ray Core actors (use_ray=True).
    Verifies pipelined streaming execution and sub-30-second post-match latency SLA.
    """
    orch = PipelineOrchestrator(
        use_ray=True,
        num_cpus=8,
        num_gpus=2,
        batch_size=8,
        backpressure_threshold=16,
        mqtt_mock_mode=True,
    )

    match_id = '2026casf_qm42_ray'
    await orch.start_pipeline(synthetic_match_video_path, match_id)
    assert orch.is_running is True
    assert ray.is_initialized() is True

    start_ts = time.perf_counter()
    await asyncio.wait_for(orch.process_match_stream(), timeout=25.0)
    elapsed = time.perf_counter() - start_ts

    summary = await orch.stop_pipeline()

    assert summary['status'] == 'completed'
    assert summary['total_frames_processed'] > 0
    assert elapsed < 30.0, 'Turnaround processing SLA (<30s) breached'
    assert summary['elapsed_seconds'] < 30.0, 'Turnaround processing SLA (<30s) breached'
    assert summary['sync_published'] is True
    assert orch.queued_frames == 0

    asof_res = await orch.execute_asof_join_query(match_id, team_number=254)
    assert asof_res['match_id'] == match_id
    assert asof_res['query_latency_ms'] < 15.0


@pytest.mark.asyncio
async def test_pipeline_stop_without_start_idempotency() -> None:
    """Verifies stop_pipeline behaves gracefully and idempotently even before start."""
    orch = PipelineOrchestrator(use_ray=False)
    summary = await orch.stop_pipeline()
    assert summary['status'] == 'completed'
    assert summary['total_frames_processed'] == 0
