"""
Unit tests for Robot Superstructure Verification, Hub Physical Exclusion Gating,
Center-Field Ball Recovery, and 3D Z-Axis Ball Elevation.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.common.types import (
    BallState,
    BoundingBox,
    HomographyEstimate,
)
from src.tracking.ball_tracker import (
    CARPET_CENTER_ELEVATION_INCH,
    FUEL_RADIUS_INCH,
    BallTracker,
)
from src.tracking.bot_sort import STrack
from src.vision.ball_detector import YellowBallDetector
from src.vision.detector_trt import RobotBumperVerificationGate


def test_superstructure_evaluation_robot_vs_hub_chute() -> None:
    """
    Tests that evaluate_superstructure_patch discriminates between a real mechanical
    robot superstructure (edges, tubing, variance) and a solid plastic Hub chute/banner.
    """
    gate = RobotBumperVerificationGate()

    # 1. Real Robot Superstructure: metallic gray with vertical tubing and wiring
    robot_super = np.full((60, 80, 3), (50, 50, 50), dtype=np.uint8)
    # Add vertical aluminum uprights / elevator rails
    cv2.line(robot_super, (15, 0), (15, 60), (200, 200, 200), 2)
    cv2.line(robot_super, (65, 0), (65, 60), (200, 200, 200), 2)
    # Add horizontal crossbar and intake roller
    cv2.line(robot_super, (15, 20), (65, 20), (160, 160, 160), 2)
    cv2.circle(robot_super, (40, 40), 10, (20, 180, 220), -1)  # Fuel ball in hopper

    score_robot, metrics_robot = gate.evaluate_superstructure_patch(robot_super)
    assert score_robot > 0.15, f"Expected robot superstructure score > 0.15, got {score_robot}"
    assert metrics_robot["edge_variance"] > 20.0

    # 2. Hub Chute / Colored Plastic Banner: uniform saturated red with zero internal mechanisms
    hub_chute = np.full((60, 80, 3), (25, 25, 220), dtype=np.uint8)
    score_hub, metrics_hub = gate.evaluate_superstructure_patch(hub_chute)
    assert score_hub < 0.05, f"Expected Hub chute superstructure score < 0.05, got {score_hub}"
    assert metrics_hub["excessive_fabric_color"] > 0.50


def test_hub_physical_exclusion_gate() -> None:
    """
    Tests that RobotBumperVerificationGate rejects candidate boxes whose projected
    contact patch penetrates inside the solid Hub cylinders (Red Hub: 162.8, 158.85).
    """
    gate = RobotBumperVerificationGate(hub_exclusion_radius_inch=30.0)

    # Construct synthetic frame
    h, w = 1080, 1920
    frame = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)

    # The homography matrix maps field (inches) to image (pixels).
    # Its inverse maps image (pixels) to field (inches).
    # For pixel u to map to field x = u * scale:
    scale = 0.35
    H = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    homography = HomographyEstimate(
        frame_index=1,
        timestamp_ms=0,
        h_matrix=H,
        dlt_condition=1.0,
        lm_rmse=0.1,
        lm_iterations=1,
        converged=True,
        camera_cut=False,
        affine_cmc=np.eye(2, 3),
    )

    box_w, box_h = 60.0, 60.0
    # Box 1: Contact patch (bottom center) inside Red Hub (field: 163.0, 159.0)
    # contact_v = cy + 0.5 * h -> cy = (field_y / scale) - 0.5 * h
    cx_hub = 163.0 / scale
    cy_hub = (159.0 / scale) - (box_h / 2.0)
    # Draw bumper on frame
    cv2.rectangle(frame, (int(cx_hub - 30), int(cy_hub - 30)), (int(cx_hub + 30), int(cy_hub + 30)), (40, 40, 40), -1)
    cv2.rectangle(frame, (int(cx_hub - 30), int(cy_hub + 10)), (int(cx_hub + 30), int(cy_hub + 30)), (20, 20, 210), -1)
    box_inside_hub = BoundingBox(center_x=cx_hub, center_y=cy_hub, width=box_w, height=box_h, confidence=0.85, class_id=0)

    # Box 2: Safely outside Hub in open field (field: 280.0, 158.85)
    cx_safe = 280.0 / scale
    cy_safe = (158.85 / scale) - (box_h / 2.0)
    cv2.rectangle(frame, (int(cx_safe - 30), int(cy_safe - 30)), (int(cx_safe + 30), int(cy_safe + 30)), (40, 40, 40), -1)
    cv2.rectangle(frame, (int(cx_safe - 30), int(cy_safe + 10)), (int(cx_safe + 30), int(cy_safe + 30)), (20, 20, 210), -1)
    # Add vertical structure to safe robot
    cv2.line(frame, (int(cx_safe - 15), int(cy_safe - 30)), (int(cx_safe - 15), int(cy_safe + 10)), (220, 220, 220), 2)
    cv2.line(frame, (int(cx_safe + 15), int(cy_safe - 30)), (int(cx_safe + 15), int(cy_safe + 10)), (220, 220, 220), 2)
    box_outside_hub = BoundingBox(center_x=cx_safe, center_y=cy_safe, width=box_w, height=box_h, confidence=0.85, class_id=0)

    verified = gate.verify_candidates(frame, [box_inside_hub, box_outside_hub], homography=homography)

    # The box inside the Hub must be rejected, only the box outside the Hub should survive
    assert len(verified) == 1
    assert abs(verified[0].center_x - cx_safe) < 2.0


def test_center_field_ball_detection_expanded_bounds() -> None:
    """
    Tests that YellowBallDetector detects balls in the foreground and center field
    (y > 0.65) that were previously discarded by restrictive screen boundaries.
    """
    detector = YellowBallDetector(
        field_roi_y=(0.05, 0.94),
        field_bounds_screen=(0.01, 0.99, 0.05, 0.92),
    )

    h, w = 1080, 1920
    frame = np.full((h, w, 3), (35, 110, 35), dtype=np.uint8)

    # Place 10 yellow balls in the center and lower foreground: y in [700, 950] (y_norm: 0.65 to 0.88)
    for i in range(10):
        bx = int(200 + i * 160)
        by = int(720 + (i % 3) * 60)
        cv2.circle(frame, (bx, by), 9, (20, 220, 240), -1)

    boxes = detector.detect(frame)
    assert len(boxes) >= 8, f"Expected at least 8 center/foreground balls, got {len(boxes)}"
    for b in boxes:
        assert b.center_y > 0.60 * h, "Detected balls must be in the center/foreground"


def test_ball_3d_z_axis_elevation_calculation() -> None:
    """
    Tests that BallTracker computes true 3D Z elevation for in-flight balls from
    known 5.91\" diameter and camera extrinsics, and sets state = IN_FLIGHT.
    """
    tracker = BallTracker(target_fps=60.0)

    # Construct camera extrinsics (camera elevated at X=325.6, Y=-180, Z=220 inches)
    f_cam = 1250.6
    K_cam = np.array([[f_cam, 0.0, 960.0], [0.0, f_cam, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # Looking down towards field center: pitch ~129 degrees around X (camera frame Y is down)
    rvec = np.array([2.25, 0.0, 0.0], dtype=np.float64)
    R_cam, _ = cv2.Rodrigues(rvec)
    C_cam = np.array([325.6, -180.0, 220.0], dtype=np.float64)
    tvec = -R_cam @ C_cam

    H_floor = K_cam @ np.column_stack([R_cam[:, 0], R_cam[:, 1], tvec])
    H_floor /= H_floor[2, 2]

    homography = HomographyEstimate(
        frame_index=1,
        timestamp_ms=0,
        h_matrix=H_floor,
        dlt_condition=1.0,
        lm_rmse=0.1,
        lm_iterations=1,
        converged=True,
        camera_cut=False,
        affine_cmc=np.eye(2, 3),
        focal_length=f_cam,
        camera_center=C_cam,
        camera_pose_rvec=rvec,
        camera_pose_tvec=tvec,
    )

    # 1. Carpet ball: observed at floor position (u=960, v=750, radius=8.0)
    carpet_box = BoundingBox(center_x=960.0, center_y=750.0, width=16.0, height=16.0, confidence=0.90, class_id=1)

    # 2. In-flight ball: flying high up near Upper Hub (u=960, v=350, larger apparent radius r=14.0 px)
    # Depth = f * 2.955 / 14.0 ~ 264 inches -> elevated Z ~ 65-75 inches!
    inflight_box = BoundingBox(center_x=960.0, center_y=350.0, width=28.0, height=28.0, confidence=0.95, class_id=1)

    states = tracker.update(
        detections=[carpet_box, inflight_box],
        frame_index=1,
        homography=homography,
    )

    assert len(states) == 2
    carpet_state = next(s for s in states if s.pixel_bbox.center_y > 500.0)
    inflight_state = next(s for s in states if s.pixel_bbox.center_y < 500.0)

    # Carpet ball must be on carpet
    assert carpet_state.state == BallState.ON_CARPET
    assert carpet_state.field_z_inch == pytest.approx(CARPET_CENTER_ELEVATION_INCH, abs=0.1)

    # In-flight ball must have elevated Z
    assert inflight_state.state == BallState.IN_FLIGHT
    assert inflight_state.field_z_inch > 20.0, f"Expected elevated ball Z > 20.0, got {inflight_state.field_z_inch}"
    assert inflight_state.center_rel_z_inch > 20.0


def test_robot_field_coordinates_hub_collision_clamping() -> None:
    """
    Tests that BotTrack.update_field_coords clamps robot coordinates attempting to
    penetrate inside the Hub base cylinder to the physical bumper contact perimeter (r=33.5\").
    """
    # Create bounding box whose contact patch (xc=163.0, yc_contact=125.0 + 35.0 = 160.0)
    # maps directly inside the Red Hub (center 162.8, 158.85), distance = 1.17" < 33.5"
    bbox = BoundingBox(center_x=163.0, center_y=125.0, width=70.0, height=70.0, confidence=0.85, class_id=0)
    track = STrack(track_id=1, bbox=bbox)

    H = np.eye(3, dtype=np.float64)
    est = HomographyEstimate(
        frame_index=1,
        timestamp_ms=0,
        h_matrix=H,
        dlt_condition=1.0,
        lm_rmse=0.1,
        lm_iterations=1,
        converged=True,
        camera_cut=False,
        affine_cmc=np.eye(2, 3),
        camera_center=np.array([325.6, -180.0, 220.0]),
    )

    track.update_field_coords(est, frame_index=1)

    # Distance to Red Hub must be at least 33.5"
    dist_to_red_hub = np.hypot(track.field_x_inch - 162.8, track.field_y_inch - 158.85)
    assert dist_to_red_hub >= 33.4, f"Expected robot clearance >= 33.5, got {dist_to_red_hub}"
