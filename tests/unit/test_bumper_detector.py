"""
Unit tests for FRC Bumper Detector, Extrapolation, and Proposal Fusion.
"""

import numpy as np
import pytest

from src.common.types import BoundingBox
from src.vision.bumper_detector import (
    BumperDetection,
    BumperDetector,
    find_best_bumper_model_path,
)
from src.vision.deterministic_robot_tracker import DeterministicRobotTracker
from src.vision.spatial_pipeline import SpatialAnalyticsPipeline


def test_find_best_bumper_model_path():
    """Verify bumper model weights locator discovers existing weights or handles None."""
    path = find_best_bumper_model_path()
    # If weights exist on disk, path should be returned
    if path is not None:
        assert "best.pt" in path or "bumper" in path

    # Non-existent path falls back to default candidates
    fake_path = find_best_bumper_model_path("non_existent_path_xyz.pt")
    assert fake_path == path


def test_bumper_detector_initialization():
    """Verify BumperDetector initializes cleanly."""
    detector = BumperDetector(conf_threshold=0.30)
    # Detector should not throw even if model is not present or present
    assert detector.conf_threshold == 0.30
    assert detector.imgsz == 1280


def test_extrapolate_robot_box():
    """Verify robot proposal extrapolation from a bumper base detection."""
    detector = BumperDetector()

    red_bumper = BumperDetection(
        box=BoundingBox(center_x=400.0, center_y=600.0, width=120.0, height=35.0, confidence=0.88, class_id=1),
        confidence=0.88,
        alliance="RED",
        color_purity=0.75,
        field_coords_inch=(150.0, 100.0),
    )

    blue_bumper = BumperDetection(
        box=BoundingBox(center_x=800.0, center_y=700.0, width=100.0, height=30.0, confidence=0.82, class_id=2),
        confidence=0.82,
        alliance="BLUE",
        color_purity=0.70,
        field_coords_inch=(450.0, 120.0),
    )

    red_robot = detector.extrapolate_robot_box(red_bumper)
    blue_robot = detector.extrapolate_robot_box(blue_bumper)

    # Robot width should match bumper width
    assert red_robot.width == 120.0
    assert blue_robot.width == 100.0

    # Ground contact (bottom of box) must match bumper ground contact
    red_ground_bumper = red_bumper.box.center_y + red_bumper.box.height / 2.0
    red_ground_robot = red_robot.center_y + red_robot.height / 2.0
    assert abs(red_ground_bumper - red_ground_robot) < 1e-4

    # Robot height must be significantly taller than bumper
    assert red_robot.height > red_bumper.box.height * 2.0
    assert red_robot.class_id == 1  # RED
    assert blue_robot.class_id == 2  # BLUE


def test_fuse_proposals():
    """Verify fusion of robot superstructure proposals with bumper detections."""
    detector = BumperDetector()

    # Robot proposal: general box [340, 500, 460, 620] -> cx=400, cy=560, w=120, h=120
    robot_proposals = [
        BoundingBox(center_x=400.0, center_y=560.0, width=120.0, height=120.0, confidence=0.60, class_id=0),
        BoundingBox(center_x=1200.0, center_y=500.0, width=110.0, height=110.0, confidence=0.55, class_id=0),
    ]

    # Bumper detections:
    # 1. Overlaps with robot 1 (RED)
    # 2. Unmatched high-confidence bumper (BLUE) -> should be recovered as robot 3
    bumper_detections = [
        BumperDetection(
            box=BoundingBox(center_x=402.0, center_y=605.0, width=118.0, height=30.0, confidence=0.85, class_id=1),
            confidence=0.85,
            alliance="RED",
            color_purity=0.80,
        ),
        BumperDetection(
            box=BoundingBox(center_x=800.0, center_y=700.0, width=100.0, height=32.0, confidence=0.82, class_id=2),
            confidence=0.82,
            alliance="BLUE",
            color_purity=0.75,
        ),
    ]

    fused = detector.fuse_proposals(robot_proposals, bumper_detections)

    # Expect: 1 fused RED robot, 1 unmatched generic robot, 1 recovered BLUE robot = 3 total
    assert len(fused) == 3

    # Find the fused robot near X=400
    fused_r1 = next(b for b in fused if abs(b.center_x - 400.0) < 10.0)
    assert fused_r1.class_id == 1  # Successfully tagged as RED
    assert fused_r1.confidence > 0.60  # Confidence boosted

    # Find the recovered robot near X=800
    recovered_r3 = next(b for b in fused if abs(b.center_x - 800.0) < 10.0)
    assert recovered_r3.class_id == 2  # Tagged as BLUE
    assert recovered_r3.confidence == 0.82


def test_deterministic_tracker_with_bumper_class_id():
    """Verify DeterministicRobotTracker directly leverages bumper-verified class_id."""
    tracker = DeterministicRobotTracker()

    # Provide proposals pre-tagged by bumper model (class_id 1 = RED, class_id 2 = BLUE)
    proposals = [
        BoundingBox(center_x=300.0, center_y=350.0, width=80.0, height=70.0, confidence=0.90, class_id=1),
        BoundingBox(center_x=300.0, center_y=450.0, width=80.0, height=70.0, confidence=0.90, class_id=1),
        BoundingBox(center_x=300.0, center_y=550.0, width=80.0, height=70.0, confidence=0.90, class_id=1),
        BoundingBox(center_x=1200.0, center_y=350.0, width=80.0, height=70.0, confidence=0.90, class_id=2),
        BoundingBox(center_x=1200.0, center_y=450.0, width=80.0, height=70.0, confidence=0.90, class_id=2),
        BoundingBox(center_x=1200.0, center_y=550.0, width=80.0, height=70.0, confidence=0.90, class_id=2),
    ]

    # Create dummy frame and identity-like homography
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    H = np.array([
        [0.3, 0.0, 50.0],
        [0.0, 0.3, 50.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    res = tracker.track_frame(
        frame_bgr=frame,
        H_image_to_field=H,
        external_proposals=proposals,
    )

    assert len(res.red_units) == 3
    assert len(res.blue_units) == 3
    assert all(u.alliance == "RED" for u in res.red_units)
    assert all(u.alliance == "BLUE" for u in res.blue_units)


def test_spatial_pipeline_with_bumper_detector():
    """Verify SpatialAnalyticsPipeline initializes and runs with BumperDetector."""
    pipeline = SpatialAnalyticsPipeline(
        scale_px_per_inch=2.0,
        enable_bumper_model=True,
    )

    assert pipeline.bumper_detector is not None

    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    res = pipeline.process_frame(frame, frame_index=1, timestamp_ms=33)

    assert len(res.robots.all_units) == 6
    assert len(res.robots.red_units) == 3
    assert len(res.robots.blue_units) == 3
