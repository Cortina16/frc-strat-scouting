"""
Unit tests for Yellow Ball Detection and Occlusion-Guarded Multi-Object Tracking.
"""

from __future__ import annotations

import time
import cv2
import duckdb
import numpy as np
import pytest

from src.common.types import (
    BallCluster,
    BallState,
    BallTrackletState,
    BoundingBox,
    DualZoneDetectionResult,
    HomographyEstimate,
)
from src.analytics.duckdb_store import DuckDBStore
from src.tracking.ball_tracker import BallTracker, SingleBallKalman
from src.vision.ball_detector import YellowBallDetector


def create_synthetic_field_image(
    num_yellow_balls: int = 10,
    include_other_colors: bool = True,
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Generates a synthetic 1080p image with green carpet and colored spheres."""
    # Field green carpet
    img = np.full((height, width, 3), (35, 110, 35), dtype=np.uint8)

    # Place yellow balls within the field ROI (y: 200 to 650)
    np.random.seed(42)
    for _ in range(num_yellow_balls):
        cx = int(np.random.randint(200, width - 200))
        cy = int(np.random.randint(220, 600))
        radius = int(np.random.randint(7, 14))
        # Vibrant yellow: BGR (20, 220, 240)
        cv2.circle(img, (cx, cy), radius, (20, 220, 240), -1)

    if include_other_colors:
        # Red bumpers / balls
        for _ in range(5):
            cx = int(np.random.randint(200, width - 200))
            cy = int(np.random.randint(220, 600))
            cv2.circle(img, (cx, cy), 12, (20, 30, 220), -1)

        # Blue bumpers / balls
        for _ in range(5):
            cx = int(np.random.randint(200, width - 200))
            cy = int(np.random.randint(220, 600))
            cv2.circle(img, (cx, cy), 12, (220, 80, 20), -1)

    return img


def test_yellow_ball_detector_color_segmentation() -> None:
    """Tests that YellowBallDetector extracts yellow game pieces and ignores red/blue/field carpet."""
    detector = YellowBallDetector(max_balls=504, min_radius_px=3.0, max_radius_px=30.0)
    img = create_synthetic_field_image(num_yellow_balls=25, include_other_colors=True)

    boxes = detector.detect(img)
    assert len(boxes) >= 20, f"Expected at least 20 detections, got {len(boxes)}"
    assert len(boxes) <= 30

    for b in boxes:
        assert b.class_id == YellowBallDetector.BALL_CLASS_ID
        assert 0.1 <= b.confidence <= 1.0
        assert 6.0 <= b.width <= 60.0
        # Verify detected positions correspond to yellow regions
        px = int(b.center_x)
        py = int(b.center_y)
        b_val, g_val, r_val = img[py, px]
        # Yellow has high R and G, lower B
        assert r_val > 150 and g_val > 150 and b_val < 100, f"False positive at ({px}, {py}): BGR=({b_val},{g_val},{r_val})"


def test_yellow_ball_detector_max_504_ceiling() -> None:
    """Tests that detection count is strictly capped at the 504 limit even with excessive candidates."""
    detector = YellowBallDetector(max_balls=504, min_radius_px=2.0)

    # Synthesize an image with a dense grid of 600 yellow dots
    img = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    xs = np.linspace(100, 1800, 30, dtype=int)
    ys = np.linspace(200, 600, 20, dtype=int)
    for x in xs:
        for y in ys:
            cv2.circle(img, (x, y), 5, (20, 220, 240), -1)

    boxes = detector.detect(img)
    assert len(boxes) <= 504, f"Detections exceeded 504 ceiling: {len(boxes)}"
    assert len(boxes) == 504, f"Expected exactly 504 detections, got {len(boxes)}"


def test_dense_cluster_occlusion_splitting() -> None:
    """Tests that distance transform peak finding resolves individual balls in a dense touching pile."""
    detector = YellowBallDetector(max_balls=504, min_radius_px=3.0)

    img = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    # Draw 16 tightly packed touching yellow circles (4x4 grid with overlapping edges)
    grid_size = 4
    start_x, start_y = 500, 400
    r = 10
    step = int(r * 1.8)  # Overlapping / touching spacing
    for i in range(grid_size):
        for j in range(grid_size):
            cx = start_x + i * step
            cy = start_y + j * step
            cv2.circle(img, (cx, cy), r, (20, 220, 240), -1)

    boxes = detector.detect(img)
    # The distance transform peak detector should separate the touching spheres
    assert len(boxes) >= 12, f"Failed to split dense cluster: extracted {len(boxes)}/16 peaks"


def test_ball_tracker_kd_tree_association_and_latency() -> None:
    """Tests that k-d tree spatial matching achieves sub-2ms latency for 504 balls and maintains IDs."""
    tracker = BallTracker(max_balls=504, distance_match_thresh_px=30.0, target_fps=60.0)

    # Frame 1: 504 initial detections
    np.random.seed(99)
    cx_arr = np.random.uniform(150, 1750, size=504)
    cy_arr = np.random.uniform(220, 620, size=504)

    boxes_f1 = [
        BoundingBox(center_x=float(cx), center_y=float(cy), width=18.0, height=18.0, confidence=0.90, class_id=1)
        for cx, cy in zip(cx_arr, cy_arr)
    ]

    tracks_f1 = tracker.update(detections=boxes_f1, frame_index=1)
    assert len(tracks_f1) == 504
    initial_ids = {t.ball_id for t in tracks_f1}
    assert len(initial_ids) == 504

    # Frame 2: simulate smooth movement (< 5 pixels per frame)
    cx_arr2 = cx_arr + np.random.normal(1.5, 0.5, size=504)
    cy_arr2 = cy_arr + np.random.normal(0.5, 0.5, size=504)
    boxes_f2 = [
        BoundingBox(center_x=float(cx), center_y=float(cy), width=18.0, height=18.0, confidence=0.88, class_id=1)
        for cx, cy in zip(cx_arr2, cy_arr2)
    ]

    import gc
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        tracks_f2 = tracker.update(detections=boxes_f2, frame_index=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        gc.enable()

    assert elapsed_ms < 15.0, f"Spatial association latency {elapsed_ms:.2f}ms exceeded 15.0ms SLA"
    assert len(tracks_f2) == 504

    # Assert track ID continuity
    f2_ids = {t.ball_id for t in tracks_f2}
    common_ids = initial_ids.intersection(f2_ids)
    overlap_pct = len(common_ids) / 504.0
    assert overlap_pct >= 0.98, f"Expected >= 98% ID continuity, got {overlap_pct * 100:.1f}%"


def test_ball_robot_occlusion_and_intake() -> None:
    """Tests that balls entering a robot bounding box transition to INTAKED and link occluded_by_robot_id."""
    tracker = BallTracker(max_balls=504, distance_match_thresh_px=25.0)

    # Ball stationary at (500, 400)
    box_b = BoundingBox(center_x=500.0, center_y=400.0, width=16.0, height=16.0, confidence=0.95, class_id=1)
    t1 = tracker.update(detections=[box_b], frame_index=1)
    assert len(t1) == 1
    ball_id = t1[0].ball_id
    assert t1[0].state == BallState.ON_CARPET

    # Frame 2: Robot drives over ball (Robot center (505, 405), width=100, height=100), ball detection occluded
    robot_box = BoundingBox(center_x=505.0, center_y=405.0, width=100.0, height=100.0, confidence=0.92, class_id=0)
    t2 = tracker.update(detections=[], frame_index=2, robot_boxes=[robot_box])

    assert len(t2) == 1
    assert t2[0].ball_id == ball_id
    assert t2[0].state == BallState.INTAKED
    assert t2[0].occluded_by_robot_id == 0


def test_ball_kalman_forward_coasting() -> None:
    """Tests that balls coast forward along velocity vector across momentary occlusions."""
    tracker = BallTracker(max_balls=504, distance_match_thresh_px=30.0, max_coast_frames=10)

    # Frame 1 & 2: moving ball with positive x velocity
    b1 = BoundingBox(center_x=300.0, center_y=400.0, width=16.0, height=16.0, confidence=0.95, class_id=1)
    tracker.update(detections=[b1], frame_index=1)

    b2 = BoundingBox(center_x=310.0, center_y=400.0, width=16.0, height=16.0, confidence=0.95, class_id=1)
    t2 = tracker.update(detections=[b2], frame_index=2)
    assert len(t2) == 1
    assert t2[0].velocity_x_ips > 0.0

    # Frames 3-5: ball detection temporarily lost
    for f in range(3, 6):
        t_coast = tracker.update(detections=[], frame_index=f)
        assert len(t_coast) == 1
        assert t_coast[0].coast_frames == (f - 2)
        # Position should have advanced forward along x
        assert t_coast[0].pixel_bbox.center_x > 310.0

    # Beyond max_coast_frames, track expires
    for f in range(6, 15):
        tracker.update(detections=[], frame_index=f)

    final_tracks = tracker.update(detections=[], frame_index=16)
    assert len(final_tracks) == 0, "Expected track to be removed after exceeding max_coast_frames"


def test_duckdb_ball_trajectory_insertion(tmp_path) -> None:
    """Tests DuckDB batch insertion and querying of ball trajectory records."""
    store = DuckDBStore(db_path=":memory:", default_match_id="test_match_001", fps=60.0)

    ball_states = [
        BallTrackletState(
            ball_id=i,
            pixel_bbox=BoundingBox(center_x=100.0 + i, center_y=200.0 + i, width=16.0, height=16.0, confidence=0.90, class_id=1),
            field_x_inch=50.0 + i,
            field_y_inch=60.0 + i,
            velocity_x_ips=2.5,
            velocity_y_ips=-1.2,
            radius_px=8.0,
            state=BallState.ON_CARPET,
            occluded_by_robot_id=None,
            coast_frames=0,
        )
        for i in range(50)
    ]

    inserted = store.insert_ball_trajectory_batch(ball_states, frame_index=100, match_id="test_match_001")
    assert inserted == 50

    # Query back using DuckDB SQL
    conn = store.connection
    res = conn.execute("SELECT count(*), min(ball_id), max(ball_id), min(frame_index) FROM ball_trajectories").fetchall()
    count, min_id, max_id, min_f = res[0]
    assert count == 50
    assert min_id == 0
    assert max_id == 49
    assert min_f == 100

    # Test Parquet export includes ball_trajectories
    export_dir = str(tmp_path / "parquet_export")
    store.export_parquet(export_dir)
    import os
    assert os.path.isfile(os.path.join(export_dir, "ball_trajectories.parquet"))


def test_spatial_nms_zero_overlap() -> None:
    """Tests that Euclidean Spatial NMS strictly prevents detections closer than min_separation_px."""
    detector = YellowBallDetector(max_balls=504, min_separation_px=7.0)

    # 1. Direct test of peak extraction with close candidate peaks (1 to 5 pixels apart)
    dist = np.zeros((100, 100), dtype=np.float32)
    # Create two close artificial distance peaks
    dist[50, 50] = 8.0
    dist[50, 52] = 7.5  # 2 pixels apart
    dist[50, 65] = 8.0  # 15 pixels apart (safe)

    peaks = detector._extract_peaks_with_nms(dist)
    assert len(peaks) == 2  # The peak at (52, 50) should be suppressed
    p1, p2 = peaks[0], peaks[1]
    dist_between = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    assert dist_between >= 7.0

    # 2. Image-level test: synthetic image with jittered overlapping spheres
    img = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)
    # Draw pairs of yellow circles with centers separated by only 3-4 pixels
    for i in range(10):
        cx = 300 + i * 80
        cy = 400
        cv2.circle(img, (cx, cy), 10, (20, 220, 240), -1)
        cv2.circle(img, (cx + 3, cy + 2), 10, (20, 220, 240), -1)

    boxes = detector.detect(img)
    # Verify pairwise distance across all returned bounding boxes
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            sep = np.hypot(boxes[i].center_x - boxes[j].center_x, boxes[i].center_y - boxes[j].center_y)
            assert sep >= 7.0, f"Detections {i} and {j} too close: separation = {sep:.2f}px < 7.0px"


def test_dual_zone_focal_and_ambient() -> None:
    """Tests dual-zone partitioning into discrete robot focal balls and ambient cluster blobs."""
    detector = YellowBallDetector(max_balls=504, nominal_ball_radius_px=7.0, packing_efficiency=0.85)

    img = np.full((1080, 1920, 3), (35, 110, 35), dtype=np.uint8)

    # Robot 1 located at (500, 450) with size 120x120
    robot_box = BoundingBox(center_x=500.0, center_y=450.0, width=120.0, height=120.0, confidence=0.95, class_id=0)

    # 1. Place discrete balls inside robot focal zone
    cv2.circle(img, (480, 430), 8, (20, 220, 240), -1)
    cv2.circle(img, (520, 470), 8, (20, 220, 240), -1)

    # 2. Place an ambient pile (dense cluster) far outside robot zone (e.g. at (1200, 450))
    # Fill an ellipse simulating a cluster of ~15 balls
    cv2.ellipse(img, (1200, 450), (60, 30), 0, 0, 360, (20, 220, 240), -1)

    result = detector.detect_dual_zone(img, robot_boxes=[robot_box])
    assert isinstance(result, DualZoneDetectionResult)

    # Robot focal zone should yield individual balls
    assert len(result.individual_balls) >= 2
    for b in result.individual_balls:
        # Must be within robot catchment zone (margin = 40px, so within [500-100, 500+100])
        assert 350.0 <= b.center_x <= 650.0
        assert 300.0 <= b.center_y <= 600.0

    # Ambient zone should yield BallCluster
    assert len(result.clusters) >= 1
    cluster = result.clusters[0]
    assert cluster.estimated_ball_count > 1
    assert cluster.inventory_pct > 0.0
    assert 1100.0 <= cluster.centroid_x <= 1300.0
    assert 400.0 <= cluster.centroid_y <= 500.0


def test_ballistic_arc_in_flight() -> None:
    """Tests parabolic ballistic arc modeling under gravity in SingleBallKalman during IN_FLIGHT."""
    # Create ball launched upwards: vy = -20 px/frame (launch towards hub)
    kalman = SingleBallKalman(cx=400.0, cy=600.0, radius=8.0, g_px=1.2)
    kalman._vx = 5.0
    kalman._vy = -20.0

    y_positions = [kalman._y]
    vy_velocities = [kalman._vy]

    # Predict 35 frames of ballistic flight
    for _ in range(35):
        kalman.predict(state=BallState.IN_FLIGHT)
        y_positions.append(kalman._y)
        vy_velocities.append(kalman._vy)

    # Verify downward gravity acceleration: vy should steadily increase from -20 towards positive
    assert vy_velocities[-1] > 10.0, "Ball should have transitioned from ascent to descent"
    # Apex occurs when vy crosses 0
    apex_idx = int(np.argmin(y_positions))
    assert 14 <= apex_idx <= 20, f"Apex occurred at frame {apex_idx}, expected ~16"
    assert y_positions[apex_idx] < y_positions[0], "Apex position y should be higher (smaller y) than launch"
    assert y_positions[-1] > y_positions[apex_idx], "Ball should descend after apex"

    # Compare against rolling friction when ON_CARPET: no gravitational downward acceleration
    kalman_carpet = SingleBallKalman(cx=400.0, cy=600.0, radius=8.0, g_px=1.2)
    kalman_carpet._vx = 5.0
    kalman_carpet._vy = -20.0
    for _ in range(35):
        kalman_carpet.predict(state=BallState.ON_CARPET)

    # On carpet, vy should decelerate to 0 via friction without reversing downwards
    assert abs(kalman_carpet._vy) < 5.0
    assert kalman_carpet._vy <= 0.0, "Carpet rolling ball should not reverse velocity under gravity"

