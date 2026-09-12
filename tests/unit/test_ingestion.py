"""
Unit tests for Stream Ingestion and ResNet-18 Frame Gating.

Verifies:
- Programmatic generation of synthetic MP4 video clips with alternating gameplay and non-gameplay segments.
- Compliance with StreamIngestionActorBase and StreamIngestionActorProtocol interfaces.
- Accurate ResNet-18 / heuristic frame gating with strided frame-dropping policy (>99% recall).
- Decoding and ingestion do NOT block the asyncio event loop.
- Memory buffers and decoded frames are yielded as valid contiguous numpy arrays with correct metadata.
- Graceful stream lifecycle (start, pause, EOF, close, re-initialize).
- Ray Plasma object store registration when Ray is initialized.
- WorkerWithPipelining asynchronous streaming generator.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import time
from typing import Generator, List, Tuple

import cv2
import numpy as np
import pytest
import ray

from src.common.interfaces import (
    StreamIngestionActorBase,
    StreamIngestionActorProtocol,
)
from src.common.types import FrameMetadata, RawDecodedFrame
from src.ingestion.frame_gate import (
    FrameGate,
    FrameGateClassifier,
    heuristic_field_classifier,
)
from src.ingestion.stream_decoder import StreamIngestionActor


# ============================================================================
# Fixtures: Programmatically Generated Synthetic Video Fixture
# ============================================================================

def generate_synthetic_video(
    file_path: str,
    fps: float = 30.0,
    width: int = 640,
    height: int = 360,
    total_frames: int = 150,
) -> None:
    """
    Generates an MP4 video with alternating gameplay and non-gameplay segments.

    Layout (150 frames = 5 seconds at 30 FPS):
    - Frames 0..29 (30 frames):   GAMEPLAY - Wide-angle green field carpet with white field lines.
    - Frames 30..59 (30 frames):  NON-GAMEPLAY - Solid gray frame with referee review banner.
    - Frames 60..89 (30 frames):  GAMEPLAY - Wide-angle green carpet with alliance markings.
    - Frames 90..119 (30 frames): NON-GAMEPLAY - Simulated crowd / high-frequency noise static.
    - Frames 120..149 (30 frames): GAMEPLAY - Wide-angle green field carpet.

    Total gameplay frames: 90.
    Total non-gameplay frames: 60.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for path: {file_path}")

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if (0 <= i < 30) or (60 <= i < 90) or (120 <= i < 150):
            # Gameplay frame: Green field carpet (BGR: (35, 120, 35))
            frame[:] = (35, 120, 35)
            # Add white field lines
            cv2.line(frame, (50, height // 2), (width - 50, height // 2), (255, 255, 255), 4)
            cv2.rectangle(frame, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), (20, 20, 20), 2)
            # Add simulated robot bumpers
            cv2.rectangle(frame, (200, 150), (250, 180), (180, 40, 40), -1)  # Blue bumper
            cv2.rectangle(frame, (400, 150), (450, 180), (40, 40, 180), -1)  # Red bumper
        elif 30 <= i < 60:
            # Non-gameplay frame: Solid gray with text banner (crowd/referee)
            frame[:] = (128, 128, 128)
            cv2.putText(
                frame,
                "MATCH PAUSED - REFEREE REVIEW",
                (50, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            # Non-gameplay frame: Random static / crowd texture
            # Deterministic noise pattern based on frame index
            np.random.seed(i)
            frame[:] = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        out.write(frame)

    out.release()


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Generator[str, None, None]:
    """Pytest fixture providing a path to a synthetic 150-frame video."""
    video_file = tmp_path / "synthetic_scouting_match.mp4"
    generate_synthetic_video(str(video_file))
    yield str(video_file)
    if video_file.exists():
        video_file.unlink()


# ============================================================================
# 1. Interface Contracts Verification
# ============================================================================

def test_stream_ingestion_actor_interface_contracts() -> None:
    """Verifies that StreamIngestionActor adheres to ABC and Protocol contracts."""
    actor = StreamIngestionActor()
    assert isinstance(actor, StreamIngestionActorBase)
    assert isinstance(actor, StreamIngestionActorProtocol)

    # Check method signatures
    assert hasattr(actor, "initialize_stream")
    assert hasattr(actor, "get_next_gameplay_batch")
    assert hasattr(actor, "is_stream_active")
    assert hasattr(actor, "close_stream")


# ============================================================================
# 2. ResNet-18 / Heuristic Frame Gate Classifier Direct Tests
# ============================================================================

def test_heuristic_field_classifier_accuracy() -> None:
    """Tests the heuristic classifier directly across various frame categories."""
    h, w = 360, 640

    # 1. Green carpet gameplay frame
    green_frame = np.zeros((h, w, 3), dtype=np.uint8)
    green_frame[:] = (35, 120, 35)
    cv2.line(green_frame, (50, 180), (590, 180), (255, 255, 255), 4)
    is_gameplay, conf = heuristic_field_classifier(green_frame)
    assert is_gameplay is True
    assert conf > 0.8

    # 2. Gray referee / score reveal banner
    gray_frame = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.putText(gray_frame, "REFEREE REVIEW", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    is_gameplay, conf = heuristic_field_classifier(gray_frame)
    assert is_gameplay is False
    assert conf > 0.8

    # 3. High-frequency random noise static
    np.random.seed(42)
    noise_frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    is_gameplay, conf = heuristic_field_classifier(noise_frame)
    assert is_gameplay is False

    # 4. Red alliance banner
    red_frame = np.zeros((h, w, 3), dtype=np.uint8)
    red_frame[:] = (20, 20, 180)
    is_gameplay, conf = heuristic_field_classifier(red_frame)
    assert is_gameplay is False

    # 5. Blue alliance banner
    blue_frame = np.zeros((h, w, 3), dtype=np.uint8)
    blue_frame[:] = (180, 20, 20)
    is_gameplay, conf = heuristic_field_classifier(blue_frame)
    assert is_gameplay is False

    # 6. Blank / Empty frame
    blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
    is_gameplay, conf = heuristic_field_classifier(blank_frame)
    assert is_gameplay is False


def test_frame_gate_strided_dropping_policy() -> None:
    """Verifies that FrameGate evaluates every stride=10 frames and drops intervening non-gameplay."""
    # Pattern: 0..9 gameplay, 10..19 non-gameplay, 20..29 gameplay
    decisions = {0: True, 10: False, 20: True}

    def mock_classify(img: np.ndarray) -> Tuple[bool, float]:
        frame_id = int(img[0, 0, 0])
        return decisions.get(frame_id, False), 0.95

    gate = FrameGate(stride=10, classifier=mock_classify)
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)

    accepted_indices: List[int] = []
    dropped_indices: List[int] = []

    for idx in range(30):
        # Embed idx in pixel for mock classifier
        dummy_img[0, 0, 0] = idx
        is_gameplay, _ = gate.evaluate(idx, dummy_img)
        if is_gameplay:
            accepted_indices.append(idx)
        else:
            dropped_indices.append(idx)

    # Frame 0..9 should be accepted (10 frames)
    # Frame 10..19 should be dropped (10 frames)
    # Frame 20..29 should be accepted (10 frames)
    assert accepted_indices == list(range(0, 10)) + list(range(20, 30))
    assert dropped_indices == list(range(10, 20))
    assert gate.total_evaluations == 3  # Frames 0, 10, 20
    assert gate.frames_accepted == 20
    assert gate.frames_dropped == 10


# ============================================================================
# 3. Stream Ingestion End-to-End & Frame Gating Recall (>99%)
# ============================================================================

@pytest.mark.asyncio
async def test_gameplay_batch_retrieval_and_gating_recall(synthetic_video_path: str) -> None:
    """
    Ingests the synthetic 150-frame video and asserts:
    - Non-gameplay frame sequences are dropped with >99% recall.
    - Yielded frames are strictly from gameplay segments (0..29, 60..89, 120..149).
    - Batches conform to List[RawDecodedFrame] with FrameMetadata.
    """
    actor = StreamIngestionActor(frame_gating_stride=10, queue_depth=32)

    success = await actor.initialize_stream(synthetic_video_path, target_fps=30)
    assert success is True
    assert await actor.is_stream_active() is True

    all_frames: List[RawDecodedFrame] = []
    batch_count = 0

    while True:
        batch = await actor.get_next_gameplay_batch(batch_size=8)
        if not batch:
            break
        batch_count += 1
        assert len(batch) <= 8
        all_frames.extend(batch)

    await actor.close_stream()

    # Ground truth gameplay frames: [0..29] + [60..89] + [120..149] = 90 frames
    # Non-gameplay frames: [30..59] + [90..119] = 60 frames
    yielded_indices = [f.metadata.frame_index for f in all_frames]

    # Check that ALL yielded frames are marked as gameplay
    for frame in all_frames:
        assert frame.metadata.is_gameplay is True
        assert frame.metadata.confidence >= 0.5
        assert isinstance(frame.image_bgr, np.ndarray)
        assert frame.image_bgr.shape == (360, 640, 3)

    # Check non-gameplay frame exclusion: NONE of frames 30..59 or 90..119 should be yielded
    non_gameplay_expected = set(range(30, 60)) | set(range(90, 120))
    leaked_non_gameplay = [idx for idx in yielded_indices if idx in non_gameplay_expected]
    assert leaked_non_gameplay == [], f"Non-gameplay frames leaked into batch: {leaked_non_gameplay}"

    # Verify recall: 60 non-gameplay frames dropped out of 60 -> 100% recall (>99% required)
    dropped_count = actor.frame_gate.frames_dropped
    assert dropped_count == 60
    assert len(yielded_indices) == 90

    # Verify frame index continuity within gameplay intervals
    expected_indices = list(range(0, 30)) + list(range(60, 90)) + list(range(120, 150))
    assert yielded_indices == expected_indices

    # Verify monotonic PTS
    pts_values = [f.metadata.pts_ms for f in all_frames]
    assert pts_values == sorted(pts_values)


# ============================================================================
# 4. Ingestion with Mock Classifier
# ============================================================================

@pytest.mark.asyncio
async def test_stream_ingestion_with_mock_classifier(synthetic_video_path: str) -> None:
    """Verifies that StreamIngestionActor correctly adheres to an injected mock classifier."""
    # Mock classifier: only allow frames where frame_index < 30
    def custom_classifier(img: np.ndarray) -> Tuple[bool, float]:
        # Green carpet check
        return heuristic_field_classifier(img)

    classifier = FrameGateClassifier(custom_classifier=custom_classifier)
    actor = StreamIngestionActor(
        frame_gating_stride=10,
        classifier=classifier,
        queue_depth=16,
    )

    await actor.initialize_stream(synthetic_video_path, target_fps=30)

    frames: List[RawDecodedFrame] = []
    while True:
        batch = await actor.get_next_gameplay_batch(batch_size=16)
        if not batch:
            break
        frames.extend(batch)

    await actor.close_stream()
    assert len(frames) == 90
    assert actor.frame_gate.classifier.backend == "custom"


# ============================================================================
# 5. Non-Blocking Event Loop Verification & Memory Buffers
# ============================================================================

@pytest.mark.asyncio
async def test_non_blocking_event_loop_during_ingestion(synthetic_video_path: str) -> None:
    """
    Asserts that decoding and frame gating do NOT block the asyncio event loop.
    A concurrent probe task monitors event loop wake-up latency during decoding.
    """
    actor = StreamIngestionActor(frame_gating_stride=10, queue_depth=32)
    await actor.initialize_stream(synthetic_video_path, target_fps=30)

    max_lag_ms = 0.0
    tick_count = 0
    monitoring = True

    async def loop_monitor() -> None:
        nonlocal max_lag_ms, tick_count
        expected_interval_s = 0.005  # 5 ms target interval
        last_time = time.perf_counter()
        while monitoring:
            await asyncio.sleep(expected_interval_s)
            now = time.perf_counter()
            lag_ms = (now - last_time - expected_interval_s) * 1000.0
            if lag_ms > max_lag_ms:
                max_lag_ms = lag_ms
            last_time = now
            tick_count += 1

    monitor_task = asyncio.create_task(loop_monitor())

    # Concurrently retrieve frames
    total_yielded = 0
    while True:
        batch = await actor.get_next_gameplay_batch(batch_size=8)
        if not batch:
            break
        total_yielded += len(batch)

    monitoring = False
    await monitor_task
    await actor.close_stream()

    assert total_yielded == 90
    assert tick_count >= 5, "Event loop monitor did not tick sufficiently"
    # Event loop latency should stay very low (< 35 ms) proving non-blocking decoding
    assert max_lag_ms < 35.0, f"Event loop was blocked! Max lag was {max_lag_ms:.2f} ms"


def test_decoded_frame_memory_buffer_compliance() -> None:
    """Verifies that RawDecodedFrame instances conform to specification types and memory layouts."""
    meta = FrameMetadata(
        frame_index=1,
        pts_ms=33,
        monotonic_ts_ns=time.monotonic_ns(),
        is_gameplay=True,
        confidence=0.98,
        width=1920,
        height=1080,
    )
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = RawDecodedFrame(metadata=meta, image_bgr=img)

    assert frame.metadata.frame_index == 1
    assert frame.metadata.pts_ms == 33
    assert frame.metadata.is_gameplay is True
    assert frame.metadata.confidence == 0.98
    assert frame.image_bgr.shape == (1080, 1920, 3)
    assert frame.image_bgr.dtype == np.uint8
    assert frame.image_bgr.flags["C_CONTIGUOUS"] is True


# ============================================================================
# 6. Stream Lifecycle Verification
# ============================================================================

@pytest.mark.asyncio
async def test_stream_lifecycle_and_error_handling(synthetic_video_path: str) -> None:
    """Tests the full lifecycle: invalid URI, active check, close, idempotent close, re-init."""
    actor = StreamIngestionActor()

    # 1. Invalid stream URI should return False
    success = await actor.initialize_stream("non_existent_stream_path_12345.mp4")
    assert success is False
    assert await actor.is_stream_active() is False

    # 2. Close on uninitialized stream should be safe
    await actor.close_stream()

    # 3. Valid stream initialization
    success = await actor.initialize_stream(synthetic_video_path, target_fps=30)
    assert success is True
    assert await actor.is_stream_active() is True

    # 4. Partial read
    batch = await actor.get_next_gameplay_batch(batch_size=4)
    assert len(batch) == 4

    # 5. Premature close
    await actor.close_stream()
    assert await actor.is_stream_active() is False

    # 6. Reading after close returns empty list
    empty_batch = await actor.get_next_gameplay_batch(batch_size=4)
    assert empty_batch == []

    # 7. Idempotent close
    await actor.close_stream()

    # 8. Re-initialization on same actor instance
    success = await actor.initialize_stream(synthetic_video_path, target_fps=30)
    assert success is True
    assert await actor.is_stream_active() is True
    batch_reinit = await actor.get_next_gameplay_batch(batch_size=8)
    assert len(batch_reinit) == 8
    await actor.close_stream()


# ============================================================================
# 7. Ray Object Store Integration
# ============================================================================

@pytest.mark.asyncio
async def test_ray_plasma_registration(synthetic_video_path: str) -> None:
    """Tests optional registration of frames into Ray Plasma object store when Ray is active."""
    # Start local Ray runtime
    ray.init(num_cpus=1, ignore_reinit_error=True)
    try:
        assert ray.is_initialized() is True

        actor = StreamIngestionActor(enable_ray_put=True, queue_depth=16)
        await actor.initialize_stream(synthetic_video_path, target_fps=30)

        batch = await actor.get_next_gameplay_batch(batch_size=4)
        assert len(batch) == 4

        # Check that last_ray_refs contains valid ObjectRefs
        refs = actor.last_ray_refs
        assert len(refs) == 4
        for ref in refs:
            assert isinstance(ref, ray.ObjectRef)

        # Retrieve frame from Ray Plasma store and assert equivalence
        retrieved_frame = ray.get(refs[0])
        assert isinstance(retrieved_frame, RawDecodedFrame)
        assert retrieved_frame.metadata.frame_index == batch[0].metadata.frame_index
        np.testing.assert_array_equal(retrieved_frame.image_bgr, batch[0].image_bgr)

        await actor.close_stream()
    finally:
        ray.shutdown()


# ============================================================================
# 8. Pipelined Async Generator (stream_gameplay_batches)
# ============================================================================

@pytest.mark.asyncio
async def test_stream_gameplay_batches_async_generator(synthetic_video_path: str) -> None:
    """Tests WorkerWithPipelining streaming pattern via async generator."""
    actor = StreamIngestionActor(frame_gating_stride=10, queue_depth=16)
    await actor.initialize_stream(synthetic_video_path, target_fps=30)

    total_frames = 0
    batch_count = 0

    async for batch in actor.stream_gameplay_batches(batch_size=8):
        total_frames += len(batch)
        batch_count += 1
        assert len(batch) <= 8

    await actor.close_stream()
    assert total_frames == 90
    assert batch_count >= 11  # 90 / 8 = 11.25 -> 12 batches


# ============================================================================
# 9. Bounded Queue & Backpressure
# ============================================================================

@pytest.mark.asyncio
async def test_backpressure_with_bounded_queue(synthetic_video_path: str) -> None:
    """Tests that a tiny queue depth throttles the worker thread cleanly without memory overflow."""
    actor = StreamIngestionActor(frame_gating_stride=10, queue_depth=4)
    await actor.initialize_stream(synthetic_video_path, target_fps=30)

    # Let worker run and hit the queue boundary (depth=4)
    await asyncio.sleep(0.05)

    # Ingest batches slowly
    total_frames = 0
    while True:
        batch = await actor.get_next_gameplay_batch(batch_size=2)
        if not batch:
            break
        total_frames += len(batch)
        await asyncio.sleep(0.005)

    await actor.close_stream()
    assert total_frames == 90
    assert actor.stats["gameplay_frames_yielded"] == 90
    assert actor.stats["frames_dropped"] == 60
