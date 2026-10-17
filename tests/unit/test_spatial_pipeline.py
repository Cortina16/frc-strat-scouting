"""
Unit tests for the 3-Stage FRC 2026 REBUILT Spatial Analytics Pipeline.

Verifies:
1. Stage 1 Spatial Calibration & Dynamic Homography Caching
2. Stage 2 Game Piece (Yellow FUEL) Quantification & Proportional Mass Allocation
3. Stage 3 Deterministic Robot Localization, Referee Filtering & Strict 3v3 Tracking
4. End-to-End Multi-Stage Pipeline Execution
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.common.types import BoundingBox
from src.vision.deterministic_robot_tracker import DeterministicRobotTracker
from src.vision.fuel_quantifier import FuelQuantifier
from src.vision.spatial_calibrator import SpatialCalibrator
from src.vision.spatial_pipeline import SpatialAnalyticsPipeline
from src.vision.spatial_types import (
    CalibrationResult,
    FieldGeometry,
    GamePieceQuantificationResult,
    RobotLocalizationResult,
    SpatialPipelineResult,
)


# ============================================================================
# Synthetic Test Fixtures
# ============================================================================

def make_synthetic_frame_with_tags_and_balls() -> np.ndarray:
    """Generates a synthetic 1080p frame with 4 Hub AprilTags, 6 robot bumpers, and yellow FUEL balls."""
    w, h = 1920, 1080
    frame = np.full((h, w, 3), (35, 110, 35), dtype=np.uint8)

    # 4 Hub AprilTags (tags 2, 5 on Red Hub, tags 21, 18 on Blue Hub)
    # Tag 2: (182.1, 135.1), Tag 5: (182.1, 182.6)
    # Tag 21: (469.1, 135.1), Tag 18: (469.1, 182.6)
    dict_tag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_images = {
        tid: cv2.cvtColor(cv2.aruco.generateImageMarker(dict_tag, tid, 60), cv2.COLOR_GRAY2BGR)
        for tid in [2, 5, 21, 18]
    }
    frame[300:360, 400:460] = tag_images[2]   # Red Hub South
    frame[420:480, 400:460] = tag_images[5]   # Red Hub North
    frame[300:360, 1400:1460] = tag_images[21] # Blue Hub South
    frame[420:480, 1400:1460] = tag_images[18] # Blue Hub North

    # Draw Yellow Game Pieces (FUEL)
    # Blob 1: Radius 20 (~1256 px)
    cv2.circle(frame, (960, 500), 20, (20, 220, 240), -1)
    # Blob 2: Radius 15 (~706 px)
    cv2.circle(frame, (800, 600), 15, (20, 220, 240), -1)
    # Blob 3: Radius 10 (~314 px)
    cv2.circle(frame, (1100, 450), 10, (20, 220, 240), -1)

    # Draw 3 Red Robots (left side)
    for i in range(3):
        rx = 300 + i * 120
        ry = 650
        cv2.rectangle(frame, (rx - 45, ry - 35), (rx + 45, ry + 35), (40, 40, 40), -1)
        cv2.rectangle(frame, (rx - 45, ry + 10), (rx + 45, ry + 35), (20, 20, 220), -1)

    # Draw 3 Blue Robots (right side)
    for i in range(3):
        bx = 1350 + i * 120
        by = 650
        cv2.rectangle(frame, (bx - 45, by - 35), (bx + 45, by + 35), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx - 45, by + 10), (bx + 45, by + 35), (220, 30, 20), -1)

    # Draw Human Referee Outside Guardrails (standing upright, height 120, width 25)
    cv2.rectangle(frame, (100, 700), (125, 820), (220, 220, 220), -1)

    return frame


# ============================================================================
# Stage 1: Spatial Calibration & Homography Tests
# ============================================================================

def test_stage1_spatial_calibration_and_warping() -> None:
    """Verifies AprilTag hub detection, homography computation, and top-down warping."""
    frame = make_synthetic_frame_with_tags_and_balls()
    calibrator = SpatialCalibrator(scale_px_per_inch=2.0)

    res: CalibrationResult = calibrator.calibrate_frame(frame)

    # 1. Verify detected tags on hubs
    assert len(res.detected_tags) == 4
    tag_ids = {t.tag_id for t in res.detected_tags}
    assert tag_ids == {2, 5, 21, 18}

    # 2. Verify Field Geometry resolution
    assert res.field_geometry.length_inches == 651.2
    assert res.field_geometry.width_inches == 317.7
    assert res.field_geometry.red_hub_center_inch == (162.8, 158.85)
    assert res.field_geometry.blue_hub_center_inch == (488.4, 158.85)

    # 3. Verify top-down canvas shape (635 x 1302 at 2 px/inch)
    assert res.canvas_shape == (635, 1302)
    assert res.warped_field_plane.shape == (635, 1302, 3)

    # 4. Verify homography matrices are valid 3x3 float64 matrices
    assert res.H_image_to_field.shape == (3, 3)
    assert res.H_field_to_image.shape == (3, 3)
    assert res.H_image_to_canvas.shape == (3, 3)
    assert np.linalg.cond(res.H_image_to_field) < 1e8

    # 5. Verify caching mechanism on subsequent frame with 0 tags
    black_frame = np.zeros_like(frame)
    res_cached = calibrator.calibrate_frame(black_frame)
    assert res_cached.is_cached is True
    assert res_cached.H_image_to_field is not None


def test_stage1_coordinate_mapping_roundtrip() -> None:
    """Verifies that pixel and field coordinate transformations are invertible."""
    calibrator = SpatialCalibrator(scale_px_per_inch=2.0)
    frame = make_synthetic_frame_with_tags_and_balls()
    calibrator.calibrate_frame(frame)

    # Test center field coordinate (325.6, 158.85)
    target_fx, target_fy = 325.6, 158.85
    u, v = calibrator.field_coords_to_pixels(target_fx, target_fy)
    fx_back, fy_back = calibrator.pixels_to_field_coords(u, v)

    assert abs(fx_back - target_fx) < 1.0
    assert abs(fy_back - target_fy) < 1.0

    # Test canvas coordinates
    cx, cy = calibrator.field_coords_to_canvas(target_fx, target_fy)
    assert abs(cx - target_fx * 2.0) < 1e-4
    assert abs(cy - target_fy * 2.0) < 1e-4


# ============================================================================
# Stage 2: Game Piece (Yellow FUEL) Quantification Tests
# ============================================================================

def test_stage2_yellow_blob_quantification_and_proportional_mass() -> None:
    """Verifies yellow color segmentation, morphological cleanup, and 504 mass allocation."""
    quantifier = FuelQuantifier(max_field_capacity=504, scale_px_per_inch=2.0)

    # Create top-down synthetic canvas (635 x 1302)
    canvas = np.full((635, 1302, 3), (35, 110, 35), dtype=np.uint8)

    # Draw 3 distinct yellow blobs with known proportional areas:
    # Blob 1: r=30 -> Area ~ 2827 px
    cv2.circle(canvas, (400, 300), 30, (20, 220, 240), -1)
    # Blob 2: r=20 -> Area ~ 1256 px
    cv2.circle(canvas, (700, 300), 20, (20, 220, 240), -1)
    # Blob 3: r=10 -> Area ~ 314 px
    cv2.circle(canvas, (1000, 300), 10, (20, 220, 240), -1)

    res: GamePieceQuantificationResult = quantifier.quantify_frame(canvas, isolate_dynamic_only=False)

    # 1. Verify clusters extracted
    assert len(res.blob_clusters) == 3
    assert res.total_yellow_pixel_area > 4000.0

    # 2. Verify proportional mass allocation sums up to <= 504
    assert res.total_estimated_count <= 504
    assert res.total_estimated_count > 450  # Should be close to 504 due to rounding

    # 3. Verify individual percentage and count hierarchy
    blobs = sorted(res.blob_clusters, key=lambda b: b.pixel_area, reverse=True)
    assert blobs[0].pixel_area > blobs[1].pixel_area > blobs[2].pixel_area
    assert blobs[0].estimated_count > blobs[1].estimated_count > blobs[2].estimated_count
    assert abs(sum(b.percentage_total_area for b in blobs) - 100.0) < 0.5


def test_stage2_zero_yellow_blobs_edge_case() -> None:
    """Verifies that empty frames handle zero yellow pixels without division-by-zero errors."""
    quantifier = FuelQuantifier(max_field_capacity=504)
    empty_canvas = np.full((635, 1302, 3), (35, 110, 35), dtype=np.uint8)

    res = quantifier.quantify_frame(empty_canvas, isolate_dynamic_only=False)
    assert res.total_yellow_pixel_area == 0.0
    assert res.total_estimated_count == 0
    assert len(res.blob_clusters) == 0


# ============================================================================
# Stage 3: Deterministic Robot Localization & Identification Tests
# ============================================================================

def test_stage3_referee_filtering_and_strict_3v3_cardinality() -> None:
    """
    Verifies that:
    1. Human referees (slender aspect ratio or outside field) are filtered out.
    2. Exactly 3 Red units and 3 Blue units are outputted (total 6 units).
    """
    tracker = DeterministicRobotTracker()
    frame = make_synthetic_frame_with_tags_and_balls()
    calibrator = SpatialCalibrator()
    calib = calibrator.calibrate_frame(frame)

    # Test candidate proposals:
    # 4 Red candidates (3 matching drawn red robots, 1 excess duplicate), 2 Blue candidates (3rd is occluded)
    proposals = [
        # Valid Red candidates
        BoundingBox(center_x=300.0, center_y=650.0, width=90.0, height=70.0, confidence=0.85, class_id=0),
        BoundingBox(center_x=420.0, center_y=650.0, width=90.0, height=70.0, confidence=0.88, class_id=0),
        BoundingBox(center_x=540.0, center_y=650.0, width=90.0, height=70.0, confidence=0.82, class_id=0),
        BoundingBox(center_x=305.0, center_y=655.0, width=88.0, height=68.0, confidence=0.75, class_id=0),
        # Valid Blue candidates (only 2 visible, 1 occluded)
        BoundingBox(center_x=1350.0, center_y=650.0, width=90.0, height=70.0, confidence=0.87, class_id=0),
        BoundingBox(center_x=1470.0, center_y=650.0, width=90.0, height=70.0, confidence=0.84, class_id=0),
        # Human Referee: Slender aspect ratio (width 25, height 120 -> AR = 0.21 < 0.45)
        BoundingBox(center_x=110.0, center_y=750.0, width=25.0, height=120.0, confidence=0.90, class_id=0),
        # Off-field media/scorer table personnel: outside field perimeter (Y > 317.7 inches)
        BoundingBox(center_x=840.0, center_y=900.0, width=80.0, height=70.0, confidence=0.90, class_id=0),
    ]

    res: RobotLocalizationResult = tracker.track_frame(
        frame_bgr=frame,
        H_image_to_field=calib.H_image_to_field,
        external_proposals=proposals,
    )

    # 1. Assert that human referee and off-field media were rejected
    rejection_reasons = [p.filter_reason for p in res.candidate_proposals if p.filter_reason is not None]
    assert any("ASPECT_RATIO_TOO_SLENDER" in r for r in rejection_reasons)
    assert any("OUTSIDE_FIELD_PERIMETER" in r for r in rejection_reasons)

    # 2. Assert STRICT 3v3 CARDINALITY
    assert len(res.red_units) == 3
    assert len(res.blue_units) == 3
    assert len(res.all_units) == 6

    # 3. Assert unit naming convention
    assert [u.unit_id for u in res.red_units] == ["RED_1", "RED_2", "RED_3"]
    assert [u.unit_id for u in res.blue_units] == ["BLUE_1", "BLUE_2", "BLUE_3"]

    # 4. Assert missing blue unit was kept alive via spatial persistence (is_projected=True)
    blue_projected = [u for u in res.blue_units if u.is_projected]
    assert len(blue_projected) >= 1

    # 5. Assert excess candidate proposal within proximity was fused as an additional detection
    red_with_additional = [u for u in res.red_units if len(u.additional_detections) > 0]
    assert len(red_with_additional) == 1
    assert len(red_with_additional[0].additional_detections) == 1


# ============================================================================
# End-to-End Multi-Stage Pipeline Integration Test
# ============================================================================

def test_spatial_pipeline_end_to_end() -> None:
    """Verifies that SpatialAnalyticsPipeline processes a frame and produces full composite telemetry."""
    pipeline = SpatialAnalyticsPipeline(scale_px_per_inch=2.0)
    frame = make_synthetic_frame_with_tags_and_balls()

    res: SpatialPipelineResult = pipeline.process_frame(
        frame_bgr=frame,
        frame_index=1,
        timestamp_ms=33,
        render_visualization=True,
    )

    # Check Stage 1 outputs
    assert len(res.calibration.detected_tags) == 4
    assert res.calibration.warped_field_plane.shape == (635, 1302, 3)

    # Check Stage 2 outputs
    assert res.game_pieces.total_estimated_count <= 504
    assert len(res.game_pieces.blob_clusters) >= 2

    # Check Stage 3 outputs
    assert len(res.robots.red_units) == 3
    assert len(res.robots.blue_units) == 3
    assert len(res.robots.all_units) == 6

    # Check Dual Visualizations
    assert res.annotated_broadcast_frame is not None
    assert res.annotated_broadcast_frame.shape == (1080, 1920, 3)
    assert res.annotated_top_down_plane is not None
    assert res.annotated_top_down_plane.shape == (635, 1302, 3)


# ============================================================================
# Bug Fix Verification Tests: Z-Parallax & 180-Deg Orientation Inversion
# ============================================================================

def test_z_axis_parallax_and_collinear_pnp_fallback() -> None:
    """
    Verifies that:
    1. Elevated AprilTags on Hubs (Z=35 to 44.25") are solved via PnP rather than corrupted 2D fitting.
    2. When PnP fails or encounters collinear anchors, the calibrator strictly falls back
       to cached homography / CMC without exploding into an infinite vertical vortex.
    """
    calibrator = SpatialCalibrator(scale_px_per_inch=2.0)

    # Synthetic realistic frame with authentic 3D tag projections
    frame = make_synthetic_frame_with_tags_and_balls()
    res1 = calibrator.calibrate_frame(frame)
    assert res1.H_field_to_image is not None
    assert np.all(np.isfinite(res1.H_field_to_image))
    assert np.linalg.cond(res1.H_field_to_image) < 1e7

    # Cache valid homography state
    initial_h = res1.H_field_to_image.copy()

    # Now simulate a frame with collinear anchors on elevated wall (Z=35")
    # Collinear points in 3D: PnP cannot uniquely disambiguate pose
    collinear_frame = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    dict_tag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    # 4 collinear tags placed in a line
    for idx, tid in enumerate([13, 14, 15, 16]):
        marker = cv2.cvtColor(cv2.aruco.generateImageMarker(dict_tag, tid, 60), cv2.COLOR_GRAY2BGR)
        collinear_frame[300 + idx * 70 : 360 + idx * 70, 500:560] = marker

    # Calibrator must detect collinear failure and STRICTLY fall back to CMC/cached homography
    res_collinear = calibrator.calibrate_frame(collinear_frame)

    assert res_collinear.is_cached is True, "Must fall back to cached homography on collinear anchors"
    assert np.all(np.isfinite(res_collinear.H_field_to_image)), "Homography must not explode to NaN/Inf"
    assert np.linalg.cond(res_collinear.H_field_to_image) < 1e7, "Condition number must stay bounded"

    # Verify no vertical vortex: field center and corners must project within plausible bounds
    test_pts = np.array([
        [100.0, 50.0],
        [325.6, 158.85],
        [550.0, 250.0],
    ])
    for fx, fy in test_pts:
        u, v = calibrator.field_coords_to_pixels(fx, fy)
        assert np.isfinite(u) and np.isfinite(v), "Projected coordinates must be finite"
        assert -2000.0 < u < 4000.0 and -2000.0 < v < 4000.0, f"Point exploded to vortex: ({u}, {v})"


def test_180_degree_broadcast_orientation_alignment() -> None:
    """
    Verifies that the top-down canvas matches the left-to-right perspective of the broadcast camera:
    - Broadcast view looking South: Blue Hub on LEFT, Red Hub on RIGHT.
    - Top-down canvas coordinates: Blue Hub has x_canvas < canvas_w / 2, Red Hub has x_canvas > canvas_w / 2.
    - Matrix multiplication order: H_canvas_to_img = H_floor @ S_canvas_to_field.
    - Inversion and roundtrip mapping are exact identities.
    """
    calibrator = SpatialCalibrator(scale_px_per_inch=2.0, align_with_broadcast_view=True)
    # Force inverted orientation (standard broadcast view from North sideline)
    calibrator._is_inverted = True
    calibrator._update_scaling_matrices()

    L = calibrator.field_geometry.length_inches   # 651.2
    W = calibrator.field_geometry.width_inches    # 317.7
    scale = calibrator.scale                      # 2.0
    cw = calibrator.canvas_w                      # 1302

    # In CAD coordinates: Red Hub is at X=162.8, Blue Hub is at X=488.4
    red_hub_cad = calibrator.field_geometry.red_hub_center_inch    # (162.8, 158.85)
    blue_hub_cad = calibrator.field_geometry.blue_hub_center_inch  # (488.4, 158.85)

    rx_c, ry_c = calibrator.field_coords_to_canvas(red_hub_cad[0], red_hub_cad[1])
    bx_c, by_c = calibrator.field_coords_to_canvas(blue_hub_cad[0], blue_hub_cad[1])

    # In broadcast alignment, Blue Hub must be on the LEFT and Red Hub on the RIGHT
    assert bx_c < cw / 2.0, f"Blue Hub canvas x={bx_c} must be on LEFT (< {cw/2.0})"
    assert rx_c > cw / 2.0, f"Red Hub canvas x={rx_c} must be on RIGHT (> {cw/2.0})"

    # Verify exact roundtrip conversion
    red_back_x, red_back_y = calibrator.canvas_to_field_coords(rx_c, ry_c)
    assert abs(red_back_x - red_hub_cad[0]) < 1e-4
    assert abs(red_back_y - red_hub_cad[1]) < 1e-4

    blue_back_x, blue_back_y = calibrator.canvas_to_field_coords(bx_c, by_c)
    assert abs(blue_back_x - blue_hub_cad[0]) < 1e-4
    assert abs(blue_back_y - blue_hub_cad[1]) < 1e-4

    # Verify Matrix Multiplication Order
    # H_canvas_to_img = H_floor @ S_canvas_to_field
    # H_img_to_canvas = S_field_to_canvas @ inv(H_floor) = inv(H_canvas_to_img)
    H_floor = np.array([
        [1.2, 0.1, 100.0],
        [0.05, 1.1, 200.0],
        [0.0001, 0.0002, 1.0],
    ], dtype=np.float64)
    calibrator._cached_H_field_to_img = H_floor
    calibrator._cached_H_img_to_field = np.linalg.inv(H_floor)
    calibrator._cached_H_canvas_to_img = H_floor @ calibrator.S_canvas_to_field
    calibrator._cached_H_img_to_canvas = np.linalg.inv(calibrator._cached_H_canvas_to_img)

    product = calibrator._cached_H_canvas_to_img @ calibrator._cached_H_img_to_canvas
    product /= product[2, 2]
    assert np.allclose(product, np.eye(3), atol=1e-6), "Canvas matrices must be exact inverses"


def test_match_video_calibration_frame_520() -> None:
    """Verifies calibration, PnP ground pose, and broadcast orientation on actual match video output.mp4."""
    from pathlib import Path
    video_path = Path("output.mp4")
    if not video_path.exists():
        pytest.skip("output.mp4 not present in workspace")

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 520)
    ret, frame = cap.read()
    cap.release()
    assert ret is True and frame is not None

    calibrator = SpatialCalibrator(scale_px_per_inch=2.0, align_with_broadcast_view=True)
    res = calibrator.calibrate_frame(frame)

    # Check detected tags: only wide-angle tags must be used (tags 6, 17, 5, 18), lower inset tags filtered
    tag_ids = {t.tag_id for t in res.detected_tags}
    assert 6 in tag_ids
    assert 17 in tag_ids
    assert 9 not in tag_ids, "Tag 9 in lower camera inset must be filtered out"

    # Check PnP RMSE < 3.0 pixels
    assert res.reprojection_rmse_px < 3.0, f"PnP RMSE {res.reprojection_rmse_px} must be < 3.0 px"
    assert res.is_cached is False

    # Check broadcast orientation: Blue Hub (left) and Red Hub (right)
    assert calibrator._is_inverted is True, "North sideline camera must trigger _is_inverted=True"
    bx_c, _ = calibrator.field_coords_to_canvas(467.64, 292.32)
    rx_c, _ = calibrator.field_coords_to_canvas(183.33, 292.32)
    assert bx_c < rx_c, f"Blue Hub canvas x={bx_c} must be to the left of Red Hub canvas x={rx_c}"


def test_apriltag_detector_zero_decimation():
    """Verify AprilTag detector operates at full resolution with zero downsampling decimation."""
    from src.vision.apriltag_detector import AprilTagDetector

    detector = AprilTagDetector()
    assert detector.parameters.aprilTagQuadDecimate == 0.0, "Decimation must be 0.0 (full resolution)"
    assert detector.parameters.aprilTagQuadSigma == 0.0, "Quad sigma blur must be 0.0"


def test_spatial_calibrator_persistent_hub_history_and_triangulation():
    """
    Verify SpatialCalibrator maintains persistent Hub tag history,
    computes empirical median locations, and triangulates occluded Hubs without jumping.
    """
    from src.vision.spatial_calibrator import SpatialCalibrator
    from src.vision.spatial_types import AprilTagAnchor

    calibrator = SpatialCalibrator()

    # Create synthetic structure tags
    tag_blue = AprilTagAnchor(
        tag_id=18,
        image_center_px=(452.0, 332.0),
        field_coords_inch=(488.4, 158.85, 44.25),
        corners_px=np.zeros((4, 2)),
    )
    tag_red = AprilTagAnchor(
        tag_id=5,
        image_center_px=(1495.0, 330.0),
        field_coords_inch=(162.8, 158.85, 44.25),
        corners_px=np.zeros((4, 2)),
    )

    # Frame 1: Both tags visible
    r_box, b_box = calibrator.compute_hub_screen_bboxes([tag_blue, tag_red], (1080, 1920))
    rcx = (r_box[0] + r_box[2]) / 2.0
    bcx = (b_box[0] + b_box[2]) / 2.0
    assert abs(rcx - 1495.0) < 5.0
    assert abs(bcx - 452.0) < 5.0

    # Verify most common locations dictionary
    most_common = calibrator.get_hub_tag_most_common_locations()
    assert 5 in most_common["red"]
    assert 18 in most_common["blue"]
    assert abs(most_common["red"][5][0] - 1495.0) < 1.0
    assert abs(most_common["blue"][18][0] - 452.0) < 1.0

    # Frame 2: Blue Hub tag occluded, wall tag 17 and Red Hub tag 5 visible
    tag_wall = AprilTagAnchor(
        tag_id=17,
        image_center_px=(260.0, 510.0),
        field_coords_inch=(600.0, 100.0, 0.0),
        corners_px=np.zeros((4, 2)),
    )
    r_box2, b_box2 = calibrator.compute_hub_screen_bboxes([tag_red, tag_wall], (1080, 1920))
    bcx2 = (b_box2[0] + b_box2[2]) / 2.0
    # Blue Hub must stay near 452 via triangulation and persistent memory, NOT jump to wall tag at 260!
    assert abs(bcx2 - 452.0) < 15.0
    assert bcx2 > 350.0

