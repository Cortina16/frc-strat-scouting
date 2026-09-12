"""
Unit tests for FRC Robot Detection Accuracy and RobotBumperVerificationGate.

Verifies:
- Bumper fabric patch evaluation for authentic FRC Red and Blue bumpers.
- Non-bumper rejection (referee shirts, gray carpet, stadium noise, lighting glares).
- Upper-arena / scoreboard masking (cy < 0.12 * H).
- Geometric aspect ratio and dimensional bounds sanity filtering.
- Competition cardinality enforcement (max 3 Red, max 3 Blue, max 6 total).
- PyTorch YOLO model batched inference and verification pipeline with bumper gate.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import pytest
import torch

from src.common.types import (
    BoundingBox,
    DetectionResult,
    FrameMetadata,
    RawDecodedFrame,
)
from src.vision.detector_trt import (
    RobotBumperVerificationGate,
    TensorRTDetectorActor,
)


def create_test_frame(
    frame_index: int = 0,
    width: int = 854,
    height: int = 480,
    bg_color: Tuple[int, int, int] = (60, 60, 60),
) -> RawDecodedFrame:
    """Helper to generate a clean synthetic video frame."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    meta = FrameMetadata(
        frame_index=frame_index,
        pts_ms=frame_index * 33,
        monotonic_ts_ns=time.monotonic_ns(),
        is_gameplay=True,
        confidence=1.0,
        width=width,
        height=height,
    )
    return RawDecodedFrame(metadata=meta, image_bgr=img)


def draw_synthetic_robot(
    frame: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    alliance: str,  # "RED", "BLUE", or "NONE"
    draw_chassis: bool = True,
) -> BoundingBox:
    """
    Draws a synthetic robot with chassis and authentic FRC bumpers onto frame.
    Returns the corresponding BoundingBox.
    """
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))

    if draw_chassis:
        # Metallic gray robot chassis
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)

    # Bumpers are in the bottom 35% of the robot height
    bumper_h = max(4, int(round(h * 0.35)))
    by1 = y2 - bumper_h
    by2 = y2

    if alliance.upper() == "RED":
        # Saturated FRC Red bumper fabric (BGR: ~20, 20, 210)
        cv2.rectangle(frame, (x1, by1), (x2, by2), (20, 20, 210), -1)
    elif alliance.upper() == "BLUE":
        # Saturated FRC Blue bumper fabric (BGR: ~210, 50, 20)
        cv2.rectangle(frame, (x1, by1), (x2, by2), (210, 50, 20), -1)
    elif alliance.upper() == "REFEREE":
        # Black and white vertical stripes simulating referee jersey
        stripe_w = 4
        for sx in range(x1, x2, stripe_w * 2):
            cv2.rectangle(frame, (sx, y1), (min(x2, sx + stripe_w), y2), (240, 240, 240), -1)
            cv2.rectangle(frame, (min(x2, sx + stripe_w), y1), (min(x2, sx + stripe_w * 2), y2), (15, 15, 15), -1)

    return BoundingBox(
        center_x=float(cx),
        center_y=float(cy),
        width=float(w),
        height=float(h),
        confidence=0.85,
        class_id=0,
    )


# ============================================================================
# 1. Bumper Patch Multi-Spectral Evaluation Tests
# ============================================================================


def test_bumper_gate_patch_evaluation_red() -> None:
    """Tests that red bumper fabric yields high bumper score and 'RED' alliance."""
    gate = RobotBumperVerificationGate(min_bumper_score=0.10)

    # Pure FRC Red fabric patch (BGR: B=20, G=20, R=210)
    red_patch = np.full((20, 60, 3), (20, 20, 210), dtype=np.uint8)
    score, alliance = gate.evaluate_bumper_patch(red_patch)

    assert score > 0.50
    assert alliance == "RED"


def test_bumper_gate_patch_evaluation_blue() -> None:
    """Tests that blue bumper fabric yields high bumper score and 'BLUE' alliance."""
    gate = RobotBumperVerificationGate(min_bumper_score=0.10)

    # Pure FRC Blue fabric patch (BGR: B=210, G=40, R=20)
    blue_patch = np.full((20, 60, 3), (210, 40, 20), dtype=np.uint8)
    score, alliance = gate.evaluate_bumper_patch(blue_patch)

    assert score > 0.50
    assert alliance == "BLUE"


def test_bumper_gate_patch_evaluation_neutral_noise() -> None:
    """Tests that gray carpet, dark chassis, or empty patches yield near 0 score."""
    gate = RobotBumperVerificationGate(min_bumper_score=0.10)

    # Gray carpet patch
    carpet_patch = np.full((20, 60, 3), (100, 100, 100), dtype=np.uint8)
    score, alliance = gate.evaluate_bumper_patch(carpet_patch)
    assert score < 0.05

    # Black chassis patch
    black_patch = np.zeros((20, 60, 3), dtype=np.uint8)
    score, alliance = gate.evaluate_bumper_patch(black_patch)
    assert score == 0.0

    # Empty patch
    empty_patch = np.empty((0, 0, 3), dtype=np.uint8)
    score, alliance = gate.evaluate_bumper_patch(empty_patch)
    assert score == 0.0
    assert alliance == "UNKNOWN"


# ============================================================================
# 2. Geometric & Spatial Masking Tests
# ============================================================================


def test_bumper_gate_upper_arena_masking() -> None:
    """Tests that detections in the upper 12% (scoreboard, crowd rafters) are masked."""
    gate = RobotBumperVerificationGate(upper_mask_fraction=0.12)
    h, w = 480, 854
    frame = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)

    # Bounding box in upper scoreboard area (cy = 30 < 0.12 * 480 = 57.6)
    upper_box = draw_synthetic_robot(frame, cx=400, cy=30, w=80, h=50, alliance="RED")

    # Bounding box on competition floor (cy = 250)
    field_box = draw_synthetic_robot(frame, cx=400, cy=250, w=80, h=50, alliance="RED")

    verified = gate.verify_candidates(frame, [upper_box, field_box])

    assert len(verified) == 1
    assert verified[0].center_y == pytest.approx(250, abs=1.0)


def test_bumper_gate_aspect_ratio_filtering() -> None:
    """Tests that abnormally narrow or wide non-robot objects are rejected."""
    gate = RobotBumperVerificationGate(min_aspect_ratio=0.40, max_aspect_ratio=2.60)
    h, w = 480, 854
    frame = np.full((h, w, 3), (50, 50, 50), dtype=np.uint8)

    # Vertical light pole / truss (w=15, h=100 -> aspect = 0.15)
    narrow_box = draw_synthetic_robot(frame, cx=200, cy=250, w=15, h=100, alliance="RED")

    # Ultra-wide barrier banner (w=300, h=30 -> aspect = 10.0)
    wide_box = draw_synthetic_robot(frame, cx=500, cy=250, w=300, h=30, alliance="RED")

    # Normal robot (w=80, h=60 -> aspect = 1.33)
    normal_box = draw_synthetic_robot(frame, cx=350, cy=250, w=80, h=60, alliance="RED")

    verified = gate.verify_candidates(frame, [narrow_box, wide_box, normal_box])

    assert len(verified) == 1
    assert verified[0].center_x == pytest.approx(350, abs=1.0)


def test_bumper_gate_referee_jersey_rejection() -> None:
    """Tests that black/white striped referee jerseys without bumper fabric are rejected."""
    gate = RobotBumperVerificationGate(min_bumper_score=0.10)
    h, w = 480, 854
    frame = np.full((h, w, 3), (70, 70, 70), dtype=np.uint8)

    referee_box = draw_synthetic_robot(frame, cx=200, cy=200, w=60, h=90, alliance="REFEREE")
    robot_box = draw_synthetic_robot(frame, cx=500, cy=200, w=70, h=60, alliance="BLUE")

    verified = gate.verify_candidates(frame, [referee_box, robot_box])

    assert len(verified) == 1
    assert verified[0].center_x == pytest.approx(500, abs=1.0)


def test_bumper_gate_normalized_bounds() -> None:
    """Tests that normalized screen coordinates [0.02, 0.98, 0.13, 0.62] scale dynamically in 1080p."""
    gate = RobotBumperVerificationGate(
        field_bounds=(0.02, 0.98, 0.13, 0.62),
        min_bumper_score=0.10,
    )
    h, w = 1080, 1920
    frame = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)

    # Valid robot in wide broadcast camera (cx=960, cy=400) -> x_norm=0.50, y_norm=0.37
    valid_box = draw_synthetic_robot(frame, cx=960, cy=400, w=120, h=90, alliance="RED")
    # Invalid robot in upper scoreboard banner (cx=960, cy=80) -> y_norm=0.074 < 0.13
    upper_box = draw_synthetic_robot(frame, cx=960, cy=80, w=120, h=90, alliance="RED")
    # Invalid robot in lower floor inset camera (cx=960, cy=850) -> y_norm=0.787 > 0.62
    lower_box = draw_synthetic_robot(frame, cx=960, cy=850, w=120, h=90, alliance="RED")

    verified = gate.verify_candidates(frame, [valid_box, upper_box, lower_box])
    assert len(verified) == 1
    assert verified[0].center_x == pytest.approx(960, abs=1.0)
    assert verified[0].center_y == pytest.approx(400, abs=1.0)


def test_bumper_gate_yellow_ball_occlusion_acceptance() -> None:
    """Tests that a robot whose bumper is occluded by yellow fuel balls is accepted."""
    gate = RobotBumperVerificationGate(min_bumper_score=0.10)
    h, w = 480, 854
    frame = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)

    # Robot chassis at cx=300, cy=250, w=70, h=60
    x1, y1 = 265, 220
    x2, y2 = 335, 280
    # Metallic chassis
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)
    # Bottom bumper area is submerged in yellow balls (BGR: 20, 220, 220)
    cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), (20, 220, 220), -1)

    ball_occluded_box = BoundingBox(
        center_x=300.0,
        center_y=250.0,
        width=70.0,
        height=60.0,
        confidence=0.75,
        class_id=0,
    )

    verified = gate.verify_candidates(frame, [ball_occluded_box])
    assert len(verified) == 1
    assert verified[0].center_x == pytest.approx(300.0, abs=1.0)


# ============================================================================
# 3. Competition Cardinality Enforcement Tests
# ============================================================================


def test_bumper_gate_cardinality_enforcement() -> None:
    """
    Tests that RobotBumperVerificationGate strictly enforces FRC match limits:
    max 3 Red robots, max 3 Blue robots, and max 6 total.
    """
    gate = RobotBumperVerificationGate(
        max_robots_total=6,
        max_robots_per_alliance=3,
    )
    h, w = 480, 854
    frame = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)

    candidates: List[BoundingBox] = []

    # Generate 6 Red candidates
    for i in range(6):
        box = draw_synthetic_robot(frame, cx=100 + i * 50, cy=200, w=40, h=35, alliance="RED")
        box.confidence = 0.50 + i * 0.05
        candidates.append(box)

    # Generate 6 Blue candidates
    for i in range(6):
        box = draw_synthetic_robot(frame, cx=100 + i * 50, cy=300, w=40, h=35, alliance="BLUE")
        box.confidence = 0.50 + i * 0.05
        candidates.append(box)

    assert len(candidates) == 12

    verified = gate.verify_candidates(frame, candidates)

    # Must be capped at exactly 6 total (3 Red + 3 Blue)
    assert len(verified) == 6

    red_count = 0
    blue_count = 0
    for b in verified:
        crop = frame[int(b.center_y - b.height / 2) : int(b.center_y + b.height / 2), int(b.center_x - b.width / 2) : int(b.center_x + b.width / 2)]
        bh = max(2, int(crop.shape[0] * 0.35))
        _, alliance = gate.evaluate_bumper_patch(crop[-bh:, :])
        if alliance == "RED":
            red_count += 1
        elif alliance == "BLUE":
            blue_count += 1

    assert red_count == 3
    assert blue_count == 3


# ============================================================================
# 4. Detector Integration with PyTorch and Bumper Gate Tests
# ============================================================================


@pytest.mark.asyncio
async def test_detector_with_bumper_gate_mock_mode() -> None:
    """Verifies TensorRTDetectorActor with bumper verification gate in mock mode."""
    gate = RobotBumperVerificationGate(min_bumper_score=0.10)
    actor = TensorRTDetectorActor(mock_mode=True, bumper_gate=gate)
    await actor.load_engine("mock", precision="fp16")

    # Generate frame with 1 authentic Red robot and 1 neutral block
    frame_raw = create_test_frame(frame_index=10)
    draw_synthetic_robot(frame_raw.image_bgr, cx=300, cy=250, w=70, h=60, alliance="RED")
    cv2.rectangle(frame_raw.image_bgr, (500, 200), (580, 270), (80, 80, 80), -1)

    res = await actor.detect_batch([frame_raw])
    assert len(res) == 1
    assert len(res[0].boxes) <= 6


@pytest.mark.asyncio
async def test_detector_with_real_weights_if_present() -> None:
    """
    If data/models/medium_robo_v3.pt is present on disk, tests loading and inference.
    """
    weights_path = "data/models/medium_robo_v3.pt"
    if not os.path.isfile(weights_path):
        pytest.skip(f"Model weights not found at {weights_path}")

    gate = RobotBumperVerificationGate(min_bumper_score=0.10)
    actor = TensorRTDetectorActor(bumper_gate=gate)
    await actor.load_engine(weights_path, precision="fp16")

    assert actor.is_loaded
    assert actor.execution_mode in ("pytorch_cuda", "tensorrt", "opencv_dnn")

    frame_raw = create_test_frame(frame_index=1, width=854, height=480)
    draw_synthetic_robot(frame_raw.image_bgr, cx=400, cy=250, w=80, h=70, alliance="RED")

    results = await actor.detect_batch([frame_raw])
    assert len(results) == 1
    assert len(results[0].boxes) <= 6
