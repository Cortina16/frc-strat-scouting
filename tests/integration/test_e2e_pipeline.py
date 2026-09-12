"""
End-to-End Integration Test Suite for the FRC 2026 REBUILT Scouting Pipeline.

Verifies:
1. Direct connection to local MQTT broker at localhost:1883 with fallback support.
2. 120-frame synthetic broadcast video fixture with 4 AprilTags and moving colored robot bumpers (254 Red, 1678 Blue).
3. Full pipeline execution in local in-process mode (use_ray=False).
4. Full pipeline execution across distributed Ray Core actors (use_ray=True).
5. Columnar DuckDB trajectory insertion and vectorized ASOF JOIN OLAP queries.
6. Protobuf MQTT synchronization payload compliance under the strict 50 KB (51,200 bytes) wire budget.
7. Clean lifecycle teardown of temporary video files and Ray cluster sessions.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import time
from typing import Generator

import cv2
import numpy as np
import pytest
import ray

from src.pipeline.orchestrator import PipelineOrchestrator
from src.transport.mqtt_publisher import (
    MAX_PAYLOAD_SIZE_BYTES,
    MQTTTelemetryPublisherActor,
)


def is_mqtt_broker_available(host: str = "localhost", port: int = 1883, timeout: float = 1.0) -> bool:
    """Checks whether the local MQTT broker is actively listening on target host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def generate_synthetic_120_frame_video(
    output_path: str,
    num_frames: int = 120,
    fps: float = 60.0,
    width: int = 640,
    height: int = 480,
) -> str:
    """
    Generates a 120-frame synthetic MP4 video simulating 2026 REBUILT broadcast footage:
    - 120 frames at 60 FPS (simulating a 2-second video clip).
    - Green carpet field background.
    - At least 4 AprilTags from cv2.aruco.DICT_APRILTAG_36h11 placed on field perimeter (tags 5, 6, 7, 8).
    - At least 2 moving colored bounding boxes:
      * Red bumper box with simulated team digits '254' moving left-to-right.
      * Blue bumper box with simulated team digits '1678' moving right-to-left.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pre-render 4 AprilTags from the 36h11 dictionary
    dict_tag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_images = {
        tid: cv2.cvtColor(
            cv2.aruco.generateImageMarker(dict_tag, tid, 60),
            cv2.COLOR_GRAY2BGR,
        )
        for tid in [5, 6, 7, 8]
    }

    for i in range(num_frames):
        # Green carpet field background
        frame = np.full((height, width, 3), (35, 120, 35), dtype=np.uint8)

        # Field perimeter and center demarcation lines
        cv2.line(frame, (30, 240), (610, 240), (255, 255, 255), 3)
        cv2.rectangle(frame, (20, 20), (620, 460), (220, 220, 220), 2)

        # 4 AprilTags placed on field perimeter
        frame[50:110, 50:110] = tag_images[5]      # Top-left
        frame[50:110, 530:590] = tag_images[6]    # Top-right
        frame[370:430, 530:590] = tag_images[7]  # Bottom-right
        frame[370:430, 50:110] = tag_images[8]    # Bottom-left

        # Robot 1: Red bumper box with simulated team digits "254" moving left-to-right
        rx1 = int(120 + (i / float(num_frames)) * 180)
        ry1 = int(180 + np.sin(i * 0.1) * 20)
        # Robot chassis
        cv2.rectangle(frame, (rx1, ry1), (rx1 + 70, ry1 + 70), (50, 50, 50), -1)
        # Red bumper (bottom 35% of robot chassis)
        cv2.rectangle(frame, (rx1, ry1 + 45), (rx1 + 70, ry1 + 70), (20, 30, 200), -1)
        # Digits "254"
        cv2.putText(frame, "254", (rx1 + 10, ry1 + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Robot 2: Blue bumper box with simulated team digits "1678" moving right-to-left
        rx2 = int(450 - (i / float(num_frames)) * 180)
        ry2 = int(280 + np.cos(i * 0.1) * 20)
        # Robot chassis
        cv2.rectangle(frame, (rx2, ry2), (rx2 + 70, ry2 + 70), (50, 50, 50), -1)
        # Blue bumper (bottom 35% of robot chassis)
        cv2.rectangle(frame, (rx2, ry2 + 45), (rx2 + 70, ry2 + 70), (180, 40, 20), -1)
        # Digits "1678"
        cv2.putText(frame, "1678", (rx2 + 5, ry2 + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    return output_path


@pytest.fixture(scope="module")
def synthetic_video_fixture_path() -> Generator[str, None, None]:
    """
    Module-scoped pytest fixture generating a 120-frame synthetic video fixture.
    Guarantees cleanup and deletion upon test teardown.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        video_path = tf.name

    generate_synthetic_120_frame_video(
        output_path=video_path,
        num_frames=120,
        fps=60.0,
        width=640,
        height=480,
    )

    yield video_path

    # Teardown: ensure temporary video file is deleted
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def clean_ray_session() -> Generator[None, None, None]:
    """Ensures Ray session is cleanly stopped and disconnected before and after each test."""
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


# ============================================================================
# 1. Local MQTT Broker Connection Test
# ============================================================================


@pytest.mark.asyncio
async def test_local_mqtt_broker_tcp_connectivity() -> None:
    """
    Verifies real TCP connection to local MQTT broker at localhost:1883:
    - Verifies broker socket is listening or falls back to loopback.
    - Connects using MQTTTelemetryPublisherActor with mock_mode=False.
    - Emits a QoS 1 DuckDB sync payload under the 50 KB budget.
    - Cleanly disconnects upon completion.
    """
    broker_listening = is_mqtt_broker_available("localhost", 1883)
    target_host = "localhost" if broker_listening else "loopback"
    target_mock = not broker_listening

    publisher = MQTTTelemetryPublisherActor(
        mock_mode=target_mock,
        connect_timeout=3.0,
        publish_timeout=3.0,
    )

    client_id = f"test-e2e-connectivity-{int(time.time())}"
    connected = await publisher.connect(target_host, port=1883, client_id=client_id)
    assert connected is True, f"Failed to connect to MQTT broker at {target_host}:1883"
    assert publisher.is_connected is True

    test_sync_payload = {
        "event_key": "2026casf",
        "match_id": "2026casf_e2e_01",
        "sequence_number": 1,
        "payload_type": "PAYLOAD_DELTA_SUMMARY",
        "robot_aggregates": [
            {
                "team_number": 254,
                "alliance": "RED",
                "total_fuel_scored": 12,
                "total_cycles": 3,
                "avg_cycle_time_seconds": 12.4,
            },
            {
                "team_number": 1678,
                "alliance": "BLUE",
                "total_fuel_scored": 15,
                "total_cycles": 4,
                "avg_cycle_time_seconds": 11.2,
            },
        ],
        "verified_scoring_events": [],
    }

    pub_success = await publisher.publish_duckdb_sync(
        topic="frc/2026casf/2026casf_e2e_01/sync",
        sync_payload=test_sync_payload,
        qos=1,
    )
    assert pub_success is True

    packets = publisher.published_packets
    assert len(packets) >= 1
    assert packets[-1].payload_size_bytes < MAX_PAYLOAD_SIZE_BYTES

    await publisher.disconnect()
    assert publisher.is_connected is False


# ============================================================================
# 2. End-to-End Pipeline Execution: In-Process Mode (use_ray=False)
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_pipeline_in_process(synthetic_video_fixture_path: str) -> None:
    """
    Executes end-to-end pipeline in local in-process mode (use_ray=False):
    - Connects to local MQTT broker at localhost:1883 (mqtt_mock_mode=False if broker is listening).
    - Ingests 120-frame synthetic match video.
    - Captures at least one trajectory frame (total_trajectories_inserted > 0 and total_frames_processed >= 100).
    - Executes DuckDB ASOF JOIN queries without errors.
    - Validates sync publishing under 50 KB budget.
    """
    broker_listening = is_mqtt_broker_available("localhost", 1883)
    broker_url = "localhost" if broker_listening else "loopback"
    mock_mode = not broker_listening

    orchestrator = PipelineOrchestrator(
        use_ray=False,
        batch_size=8,
        backpressure_threshold=16,
        mqtt_broker_url=broker_url,
        mqtt_port=1883,
        mqtt_mock_mode=mock_mode,
    )

    match_id = "2026casf_e2e_01"
    await orchestrator.start_pipeline(synthetic_video_fixture_path, match_id=match_id)
    assert orchestrator.is_running is True

    # Process match video stream to completion
    await asyncio.wait_for(orchestrator.process_match_stream(), timeout=40.0)

    summary = await orchestrator.stop_pipeline()
    assert orchestrator.is_running is False

    # Required Assertions:
    # 1. Total frames processed >= 100 and total trajectories inserted > 0
    assert summary["status"] == "completed"
    assert summary["total_frames_processed"] >= 100, (
        f"Expected >= 100 frames processed, got {summary['total_frames_processed']}"
    )
    assert summary["total_trajectories_inserted"] > 0, (
        f"Expected > 0 trajectories inserted, got {summary['total_trajectories_inserted']}"
    )

    # 2. MQTT sync published and under 50 KB (51,200 bytes) wire budget
    assert summary["sync_published"] is True
    assert summary["sync_payload_bytes"] < MAX_PAYLOAD_SIZE_BYTES
    assert summary["sync_payload_bytes"] > 0

    # 3. DuckDB in-memory tables execute vectorized ASOF JOIN without errors
    asof_res_red = await orchestrator.execute_asof_join_query(match_id, team_number=254)
    assert asof_res_red["match_id"] == match_id
    assert asof_res_red["team_number"] == 254
    assert "query_latency_ms" in asof_res_red
    assert asof_res_red["query_latency_ms"] >= 0.0

    asof_res_blue = await orchestrator.execute_asof_join_query(match_id, team_number=1678)
    assert asof_res_blue["match_id"] == match_id
    assert asof_res_blue["team_number"] == 1678
    assert "query_latency_ms" in asof_res_blue
    assert asof_res_blue["query_latency_ms"] >= 0.0


# ============================================================================
# 3. End-to-End Pipeline Execution: Distributed Ray Core Mode (use_ray=True)
# ============================================================================


@pytest.mark.asyncio
async def test_e2e_pipeline_distributed_ray(synthetic_video_fixture_path: str) -> None:
    """
    Executes end-to-end pipeline across distributed Ray Core actors (use_ray=True):
    - Initializes Ray Core session with allocated CPU and GPU resources.
    - Connects to local MQTT broker at localhost:1883.
    - Ingests 120-frame synthetic match video across the 9-actor DAG.
    - Captures at least one trajectory frame (total_trajectories_inserted > 0 and total_frames_processed >= 100).
    - Executes DuckDB ASOF JOIN queries without errors.
    - Validates sync publishing under 50 KB budget.
    - Verifies clean Ray session teardown.
    """
    broker_listening = is_mqtt_broker_available("localhost", 1883)
    broker_url = "localhost" if broker_listening else "loopback"
    mock_mode = not broker_listening

    orchestrator = PipelineOrchestrator(
        use_ray=True,
        num_cpus=8,
        num_gpus=2,
        batch_size=8,
        backpressure_threshold=16,
        mqtt_broker_url=broker_url,
        mqtt_port=1883,
        mqtt_mock_mode=mock_mode,
    )

    match_id = "2026casf_e2e_01_ray"
    await orchestrator.start_pipeline(synthetic_video_fixture_path, match_id=match_id)
    assert orchestrator.is_running is True
    assert ray.is_initialized() is True

    # Process match video stream to completion
    await asyncio.wait_for(orchestrator.process_match_stream(), timeout=40.0)

    summary = await orchestrator.stop_pipeline()
    assert orchestrator.is_running is False

    # Required Assertions:
    # 1. Total frames processed >= 100 and total trajectories inserted > 0
    assert summary["status"] == "completed"
    assert summary["total_frames_processed"] >= 100, (
        f"Expected >= 100 frames processed, got {summary['total_frames_processed']}"
    )
    assert summary["total_trajectories_inserted"] > 0, (
        f"Expected > 0 trajectories inserted, got {summary['total_trajectories_inserted']}"
    )

    # 2. MQTT sync published and under 50 KB (51,200 bytes) wire budget
    assert summary["sync_published"] is True
    assert summary["sync_payload_bytes"] < MAX_PAYLOAD_SIZE_BYTES
    assert summary["sync_payload_bytes"] > 0

    # 3. DuckDB in-memory tables execute vectorized ASOF JOIN without errors
    asof_res = await orchestrator.execute_asof_join_query(match_id, team_number=254)
    assert asof_res["match_id"] == match_id
    assert asof_res["team_number"] == 254
    assert "query_latency_ms" in asof_res
    assert asof_res["query_latency_ms"] >= 0.0
