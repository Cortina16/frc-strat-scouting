"""
Unit tests for RobotEKF, Proximity-Emulated IoU, and Multi-Detection Fusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.types import BoundingBox
from src.vision.deterministic_robot_tracker import DeterministicRobotTracker
from src.vision.robot_ekf import (
    RobotEKF,
    emulate_iou,
    fuse_bounding_boxes,
    pairwise_emulate_iou,
)
from src.vision.spatial_calibrator import SpatialCalibrator
from src.vision.spatial_types import RobotLocalizationResult


def test_emulate_iou_overlapping_boxes() -> None:
    """Verifies that intersecting boxes yield standard IoU values."""
    b1 = BoundingBox(center_x=100.0, center_y=100.0, width=50.0, height=50.0, confidence=0.9, class_id=0)
    # Identical box -> IoU = 1.0
    iou_exact = emulate_iou(b1, b1)
    assert pytest.approx(iou_exact, 1e-4) == 1.0

    # Partially overlapping box (shifted by 25 px in X: half overlap)
    b2 = BoundingBox(center_x=125.0, center_y=100.0, width=50.0, height=50.0, confidence=0.9, class_id=0)
    iou_half = emulate_iou(b1, b2)
    # Intersection = 25 * 50 = 1250, Union = 2500 + 2500 - 1250 = 3750 -> IoU = 1/3 ~ 0.333
    assert 0.30 <= iou_half <= 0.40


def test_emulate_iou_non_overlapping_proximity() -> None:
    """
    Verifies that non-overlapping boxes separated by a gap < D_prox yield
    a non-zero emulated IoU (1.0 - d / D_prox), whereas far apart boxes yield 0.0.
    """
    b1 = BoundingBox(center_x=100.0, center_y=100.0, width=50.0, height=50.0, confidence=0.9, class_id=0)

    # Box 2: Separated by a gap of 20 px along X (b1 right edge = 125, b2 left edge = 145)
    # center_x = 170 -> left edge = 145, right edge = 195. Gap = 145 - 125 = 20 px
    b2 = BoundingBox(center_x=170.0, center_y=100.0, width=50.0, height=50.0, confidence=0.9, class_id=0)

    prox_thresh = 100.0
    iou_emul = emulate_iou(b1, b2, proximity_thresh_px=prox_thresh)

    # Standard box IoU is strictly 0.0
    # Emulated IoU should be approx 1.0 - (20 / 100) = 0.80
    assert iou_emul > 0.70
    assert iou_emul <= 1.0

    # Box 3: Distant box (center_x = 500, gap = 350 px >> 100 px)
    b3 = BoundingBox(center_x=500.0, center_y=100.0, width=50.0, height=50.0, confidence=0.9, class_id=0)
    iou_distant = emulate_iou(b1, b3, proximity_thresh_px=prox_thresh)
    assert iou_distant == 0.0


def test_emulate_iou_field_proximity() -> None:
    """Verifies proximity emulation using physical CAD field coordinates."""
    b1 = BoundingBox(center_x=100.0, center_y=100.0, width=50.0, height=50.0, confidence=0.9, class_id=0)
    b2 = BoundingBox(center_x=400.0, center_y=400.0, width=50.0, height=50.0, confidence=0.9, class_id=0)

    # In pixel space, gap is large (>> 120 px)
    # In field space: coords are within 10 inches of each other (thresh = 40 inches)
    f1 = (150.0, 100.0)
    f2 = (156.0, 108.0) # dist = sqrt(36 + 64) = 10 inches

    iou_field = emulate_iou(
        b1,
        b2,
        proximity_thresh_px=50.0,
        field_coords1=f1,
        field_coords2=f2,
        proximity_thresh_inch=40.0,
    )
    # Expected emulated IoU = 1.0 - (10.0 / 40.0) = 0.75
    assert iou_field >= 0.70


def test_pairwise_emulate_iou_matrix() -> None:
    """Verifies vectorized pairwise emulated IoU matrix computation."""
    boxes1 = [
        BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=200.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
    ]
    boxes2 = [
        BoundingBox(center_x=110.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=800.0, center_y=800.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
    ]

    mat = pairwise_emulate_iou(boxes1, boxes2, proximity_thresh_px=100.0)
    assert mat.shape == (2, 2)
    assert mat[0, 0] > 0.50  # Overlapping / close
    assert mat[0, 1] == 0.0  # Far away
    assert mat[1, 1] == 0.0  # Far away


def test_fuse_bounding_boxes() -> None:
    """Verifies consolidating multiple bounding boxes into an enclosing union box."""
    b1 = BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.8, class_id=0)
    b2 = BoundingBox(center_x=110.0, center_y=105.0, width=30.0, height=35.0, confidence=0.7, class_id=0)

    fused = fuse_bounding_boxes([b1, b2])
    # b1 bounds: x in [80, 120], y in [80, 120]
    # b2 bounds: x in [95, 125], y in [87.5, 122.5]
    # Union bounds: x in [80, 125] -> w = 45, cx = 102.5
    # y in [80, 122.5] -> h = 42.5, cy = 101.25
    assert pytest.approx(fused.width, 1e-2) == 45.0
    assert pytest.approx(fused.height, 1e-2) == 42.5
    assert pytest.approx(fused.center_x, 1e-2) == 102.5
    assert pytest.approx(fused.center_y, 1e-2) == 101.25
    assert fused.confidence > 0.80


def test_robot_ekf_prediction_and_kinematics() -> None:
    """Verifies EKF state prediction, velocity damping, and covariance propagation."""
    ekf = RobotEKF(init_field_x=100.0, init_field_y=150.0, init_vx=30.0, init_vy=0.0)
    p_init_trace = np.trace(ekf.covariance)

    # Predict forward by 1 second (30 frames at dt = 1/30s)
    for _ in range(30):
        ekf.predict(dt=1.0 / 30.0)

    # Robot moved along X
    assert ekf.field_x > 120.0
    # Velocity was damped
    assert ekf.vx < 30.0
    # Covariance grew due to process noise
    assert np.trace(ekf.covariance) > p_init_trace


def test_robot_ekf_multi_update_covariance_shrinkage() -> None:
    """
    Formally verifies that updating the EKF with an additional detection within
    proximity strictly contracts the estimation covariance (trace(P_multi) < trace(P_single)).
    """
    # 1. Single detection update
    ekf_single = RobotEKF(init_field_x=100.0, init_field_y=150.0)
    ekf_single.predict(dt=1.0 / 30.0)

    box_primary = BoundingBox(center_x=300.0, center_y=400.0, width=80.0, height=70.0, confidence=0.85, class_id=0)
    ekf_single.update(box_primary, field_coords=(105.0, 152.0))
    p_single_trace = np.trace(ekf_single.covariance)

    # 2. Multi-detection update (primary + 1 additional detection within proximity)
    ekf_multi = RobotEKF(init_field_x=100.0, init_field_y=150.0)
    ekf_multi.predict(dt=1.0 / 30.0)

    box_additional = BoundingBox(center_x=306.0, center_y=404.0, width=78.0, height=68.0, confidence=0.80, class_id=0)
    ekf_multi.update_multi(
        primary_box=box_primary,
        additional_boxes=[box_additional],
        primary_field=(105.0, 152.0),
        additional_fields=[(106.0, 153.0)],
    )
    p_multi_trace = np.trace(ekf_multi.covariance)

    # Multi-observation fusion MUST strictly contract estimation uncertainty
    assert p_multi_trace < p_single_trace
    assert ekf_multi.last_update_count == 2


def test_deterministic_tracker_with_proximity_additional_detections() -> None:
    """
    Verifies that in DeterministicRobotTracker:
    1. An extra detection within proximity to a robot acts as an additional detection.
    2. Strict 3v3 cardinality is preserved (3 Red, 3 Blue).
    3. The matched unit receives the additional detection in unit.additional_detections.
    4. The unit's bounding box is fused to cover both proposals.
    """
    tracker = DeterministicRobotTracker()
    calibrator = SpatialCalibrator()

    # Synthetic homography matrix (identity-like mapping)
    H_img_to_field = np.array([
        [0.35, 0.0, 0.0],
        [0.0, 0.35, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # 3 Red targets matching bootstrap starting line, with an extra detection 6 px away from target 1
    red_proposals = [
        # Target 1 (Primary)
        BoundingBox(center_x=350.0, center_y=350.0, width=90.0, height=70.0, confidence=0.85, class_id=0),
        # Target 1 Extra detection (6 px offset, within proximity -> acts like additional detection)
        BoundingBox(center_x=356.0, center_y=354.0, width=88.0, height=68.0, confidence=0.80, class_id=0),
        # Target 2
        BoundingBox(center_x=350.0, center_y=430.0, width=90.0, height=70.0, confidence=0.88, class_id=0),
        # Target 3
        BoundingBox(center_x=350.0, center_y=510.0, width=90.0, height=70.0, confidence=0.84, class_id=0),
    ]

    # Blue proposals (3 targets)
    blue_proposals = [
        BoundingBox(center_x=1400.0, center_y=350.0, width=90.0, height=70.0, confidence=0.86, class_id=0),
        BoundingBox(center_x=1400.0, center_y=430.0, width=90.0, height=70.0, confidence=0.85, class_id=0),
        BoundingBox(center_x=1400.0, center_y=510.0, width=90.0, height=70.0, confidence=0.87, class_id=0),
    ]

    all_proposals = red_proposals + blue_proposals

    # Create dummy frame with colored bumper pixels
    frame = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    for p in red_proposals:
        x1 = int(p.center_x - p.width * 0.5)
        y1 = int(p.center_y - p.height * 0.5)
        frame[y1:y1 + int(p.height), x1:x1 + int(p.width)] = (30, 30, 220)  # Red
    for p in blue_proposals:
        x1 = int(p.center_x - p.width * 0.5)
        y1 = int(p.center_y - p.height * 0.5)
        frame[y1:y1 + int(p.height), x1:x1 + int(p.width)] = (220, 30, 30)  # Blue

    res: RobotLocalizationResult = tracker.track_frame(
        frame_bgr=frame,
        H_image_to_field=H_img_to_field,
        external_proposals=all_proposals,
    )

    # 1. Assert Strict 3v3 cardinality
    assert len(res.red_units) == 3
    assert len(res.blue_units) == 3
    assert len(res.all_units) == 6

    # 2. Check that the extra detection was captured as an additional detection on one of the red units
    units_with_additional = [u for u in res.red_units if len(u.additional_detections) > 0]
    assert len(units_with_additional) == 1

    unit_fused = units_with_additional[0]
    assert len(unit_fused.additional_detections) == 1
    # Check that fused bounding box width is larger than either individual box
    assert unit_fused.pixel_bbox.width >= 90.0


def test_strict_candidate_confidence_rejection() -> None:
    """Verifies that proposals with confidence < min_track_confidence (0.50) are rejected."""
    tracker = DeterministicRobotTracker(min_track_confidence=0.50)
    frame = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    H_eye = np.eye(3, dtype=np.float64)

    weak_proposal = BoundingBox(
        center_x=350.0, center_y=350.0, width=90.0, height=70.0, confidence=0.35, class_id=0
    )
    res = tracker.track_frame(frame_bgr=frame, H_image_to_field=H_eye, external_proposals=[weak_proposal])

    assert len(res.candidate_proposals) == 1
    assert "CONFIDENCE_TOO_LOW" in str(res.candidate_proposals[0].filter_reason)
    # Track units should remain projected because weak proposal was rejected
    assert all(u.is_projected for u in res.red_units)


def test_strict_candidate_geometry_filtering() -> None:
    """Verifies that proposals violating strict area (< 1200 px) or AR bounds are rejected."""
    tracker = DeterministicRobotTracker(min_pixel_area=1200.0, min_aspect_ratio=0.52, max_aspect_ratio=2.10)
    frame = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    H_eye = np.eye(3, dtype=np.float64)

    proposals = [
        # Small area (30 * 30 = 900 < 1200)
        BoundingBox(center_x=300.0, center_y=150.0, width=30.0, height=30.0, confidence=0.85, class_id=0),
        # Slender aspect ratio (width 30, height 80 -> AR = 0.375 < 0.52)
        BoundingBox(center_x=400.0, center_y=150.0, width=30.0, height=80.0, confidence=0.85, class_id=0),
        # Wide aspect ratio (width 120, height 40 -> AR = 3.0 > 2.10)
        BoundingBox(center_x=500.0, center_y=150.0, width=120.0, height=40.0, confidence=0.85, class_id=0),
    ]
    res = tracker.track_frame(frame_bgr=frame, H_image_to_field=H_eye, external_proposals=proposals)

    reasons = [p.filter_reason for p in res.candidate_proposals]
    assert any("AREA_TOO_SMALL" in str(r) for r in reasons)
    assert any("ASPECT_RATIO_TOO_SLENDER" in str(r) for r in reasons)
    assert any("ASPECT_RATIO_TOO_WIDE" in str(r) for r in reasons)


def test_strict_hungarian_gating_rejects_outliers() -> None:
    """
    Verifies that a proposal located very far away (> max_gating_dist_inch)
    is rejected by Hungarian gating, preventing the EKF track from teleporting across the field.
    """
    tracker = DeterministicRobotTracker(max_gating_dist_inch=200.0, max_matching_cost=0.70)
    frame = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    # Identity mapping where (x, y) px -> (x, y) inches
    H_eye = np.eye(3, dtype=np.float64)

    # Red bootstrap tracks are at X=120, Y in [70, 160, 250].
    # Inject an outlier Red candidate at X=600, Y=160 (distance ~ 480 inches >> 200 inches)
    outlier = BoundingBox(
        center_x=600.0, center_y=160.0, width=90.0, height=70.0, confidence=0.90, class_id=0
    )
    # Paint red bumper pixels at outlier location so color classification passes
    x1, y1 = int(600 - 45), int(160 - 35)
    frame[y1:y1 + 70, x1:x1 + 90] = (20, 20, 220)

    res = tracker.track_frame(frame_bgr=frame, H_image_to_field=H_eye, external_proposals=[outlier])

    # The outlier proposal should NOT have matched any Red unit due to gating distance
    # All red units should remain projected at their bootstrap / coasting positions
    assert all(u.is_projected for u in res.red_units)
    # Red 2 should not have jumped to X=600
    assert all(u.field_x_inch < 300.0 for u in res.red_units)


def test_relaxed_proximity_grouping_wide_margin() -> None:
    """
    Verifies that a corroborating detection 130 px away (relaxed proximity threshold = 160 px)
    is successfully grouped as an additional detection and fused.
    """
    tracker = DeterministicRobotTracker(
        grouping_proximity_px=160.0,
        grouping_proximity_inch=65.0,
        grouping_emul_threshold=0.02,
    )
    frame = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    H_eye = np.eye(3, dtype=np.float64)

    # 3 Primary Red detections matching RED_1, RED_2, RED_3 bootstrap positions
    red_1_prim = BoundingBox(center_x=120.0, center_y=70.0, width=90.0, height=70.0, confidence=0.88, class_id=0)
    red_2_prim = BoundingBox(center_x=120.0, center_y=160.0, width=90.0, height=70.0, confidence=0.88, class_id=0)
    red_3_prim = BoundingBox(center_x=120.0, center_y=250.0, width=90.0, height=70.0, confidence=0.88, class_id=0)
    # Secondary detection 130 px away from RED_1 (extended intake/mechanism, gap = 130 - 90 = 40 px < 160 px)
    secondary = BoundingBox(center_x=250.0, center_y=70.0, width=80.0, height=70.0, confidence=0.82, class_id=0)

    all_props = [red_1_prim, red_2_prim, red_3_prim, secondary]

    # Colorize red bumpers
    for b in all_props:
        bx1, by1 = int(b.center_x - b.width / 2), int(b.center_y - b.height / 2)
        frame[by1:by1 + int(b.height), bx1:bx1 + int(b.width)] = (20, 20, 220)

    res = tracker.track_frame(frame_bgr=frame, H_image_to_field=H_eye, external_proposals=all_props)

    red_1 = next(u for u in res.red_units if u.unit_id == "RED_1")
    assert not red_1.is_projected
    assert len(red_1.additional_detections) == 1
    assert red_1.pixel_bbox.width > 150.0  # Fused width covers both primary and secondary


def test_relaxed_multi_fragment_grouping() -> None:
    """
    Verifies that multiple unassigned detections (e.g., bumper, chassis, intake)
    are iteratively grouped into a single tracked unit with a unified enclosing bounding box.
    """
    tracker = DeterministicRobotTracker(
        grouping_proximity_px=160.0,
        grouping_proximity_inch=65.0,
        grouping_emul_threshold=0.02,
    )
    frame = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    H_eye = np.eye(3, dtype=np.float64)

    # 3 fragmented detections belonging to RED_1
    frag1 = BoundingBox(center_x=120.0, center_y=70.0, width=70.0, height=60.0, confidence=0.85, class_id=0)
    frag2 = BoundingBox(center_x=160.0, center_y=75.0, width=60.0, height=55.0, confidence=0.80, class_id=0)
    frag3 = BoundingBox(center_x=200.0, center_y=70.0, width=65.0, height=60.0, confidence=0.78, class_id=0)

    # Detections for RED_2 and RED_3
    red_2_prim = BoundingBox(center_x=120.0, center_y=160.0, width=90.0, height=70.0, confidence=0.88, class_id=0)
    red_3_prim = BoundingBox(center_x=120.0, center_y=250.0, width=90.0, height=70.0, confidence=0.88, class_id=0)

    all_props = [frag1, red_2_prim, red_3_prim, frag2, frag3]

    for b in all_props:
        bx1, by1 = int(b.center_x - b.width / 2), int(b.center_y - b.height / 2)
        frame[by1:by1 + int(b.height), bx1:bx1 + int(b.width)] = (20, 20, 220)

    res = tracker.track_frame(frame_bgr=frame, H_image_to_field=H_eye, external_proposals=all_props)

    red_1 = next(u for u in res.red_units if u.unit_id == "RED_1")
    assert not red_1.is_projected
    # Both secondary fragments should have been clustered into RED_1
    assert len(red_1.additional_detections) == 2
    # Fused width should enclose all 3 fragments: from (120 - 35) = 85 to (200 + 32.5) = 232.5 -> width ~ 147.5
    assert red_1.pixel_bbox.width >= 140.0


def test_robot_ekf_velocity_clamping() -> None:
    """Verifies that RobotEKF physically caps linear velocity magnitude to max_velocity (240 in/s)."""
    ekf = RobotEKF(init_field_x=100.0, init_field_y=100.0, init_vx=300.0, init_vy=400.0, max_velocity_ips=240.0)
    # Initial velocity hypot(300, 400) = 500 in/s
    ekf.predict(dt=1.0 / 30.0)
    v_mag = np.hypot(ekf.vx, ekf.vy)
    assert pytest.approx(v_mag, abs=1e-2) == 240.0
