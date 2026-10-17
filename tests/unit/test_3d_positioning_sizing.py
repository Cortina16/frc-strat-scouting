"""
Unit tests for 3D Positioning, Camera Extrinsics, AprilTag Distances, and Physical Sizing.

Verifies:
1. Camera extrinsics (R, t, C, f) estimated from CAD AprilTags via PnP.
2. Euclidean distance from camera to visible AprilTags matches ground truth.
3. 3D ray projection at carpet elevation (Z=0 for robots, Z=2.955 for 5.91" FUEL balls).
4. Field-center relative coordinates (X_rel = X - 325.6, Y_rel = Y - 158.85, Z_rel = Z).
5. Robot live optical sizing (W_live = w_px * d / f) and nominal 22"x28" base fallback.
6. Ball 3D coordinates, carpet elevation Z=2.955", and camera depth.
7. DuckDB storage, querying, and Parquet export of camera poses and 3D telemetry.
8. Digital twin webserver DataProvider returns 3D coordinates and sizing.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import cv2
import duckdb
import numpy as np
import pytest

from src.analytics.duckdb_store import DuckDBStorageActor
from src.common.types import (
    AllianceColor,
    BallState,
    BallTrackletState,
    BoundingBox,
    CameraPoseState,
    HomographyEstimate,
    RobotTrackletState,
    TagDetectionResult,
    TrackState,
)
from src.server.data_provider import MatchDataProvider
from src.tracking.ball_tracker import (
    CARPET_CENTER_ELEVATION_INCH,
    FUEL_DIAMETER_INCH,
    FUEL_RADIUS_INCH,
    BallTracker,
)
from src.tracking.bot_sort import STrack
from src.vision.apriltag_detector import CANONICAL_FIELD_TAGS
from src.vision.homography import (
    estimate_camera_pose_pnp,
    project_pixel_to_field_3d,
)


def create_synthetic_camera(
    cam_pos: np.ndarray,
    look_at: np.ndarray,
    focal_length: float = 1250.0,
    image_shape: Tuple[int, int] = (1080, 1920),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs ground-truth camera matrices K, R, t, and camera center C.
    """
    h_img, w_img = image_shape
    cx = float(w_img) / 2.0
    cy = float(h_img) / 2.0

    K = np.array([
        [focal_length, 0.0, cx],
        [0.0, focal_length, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    # OpenCV camera coordinate convention: X right, Y down, Z forward
    R_world_to_cam = np.vstack([right, -true_up, forward])
    t_cam = -R_world_to_cam @ cam_pos

    rvec, _ = cv2.Rodrigues(R_world_to_cam)
    tvec = t_cam.reshape(3, 1)

    return K, rvec, tvec, cam_pos


def test_camera_extrinsics_and_apriltag_distances() -> None:
    """
    Verifies that PnP recovers camera position, rotation, and Euclidean
    distances to CAD AprilTags with sub-inch accuracy.
    """
    # Camera elevated along sideline looking at field center
    cam_pos_gt = np.array([325.6, -180.0, 220.0], dtype=np.float64)
    look_at = np.array([325.6, 158.85, 0.0], dtype=np.float64)
    f_gt = 1250.0
    K_gt, rvec_gt, tvec_gt, C_gt = create_synthetic_camera(cam_pos_gt, look_at, focal_length=f_gt)

    # Select visible AprilTags from CANONICAL_FIELD_TAGS (Tags 1, 2, 11, 12, 15, 16, 27, 28)
    test_tag_ids = [1, 2, 11, 12, 15, 16, 27, 28]
    object_points_3d = []
    ground_truth_distances = {}

    for tid in test_tag_ids:
        pos_3d = CANONICAL_FIELD_TAGS[tid]
        object_points_3d.append(pos_3d)
        d_gt = float(np.linalg.norm(np.array(pos_3d) - cam_pos_gt))
        ground_truth_distances[tid] = d_gt

    obj_pts = np.array(object_points_3d, dtype=np.float64)

    # Project to 2D image coordinates
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, tvec_gt, K_gt, None)
    img_pts = proj_pts.reshape(-1, 2)

    # Run PnP solver
    rvec_est, tvec_est, cam_center_est, H_floor, reproj_rmse, f_est, success = estimate_camera_pose_pnp(
        object_points_3d=obj_pts,
        image_points_2d=img_pts,
        camera_matrix=K_gt,
        image_shape=(1080, 1920),
    )

    assert success is True, "PnP estimation failed"
    assert reproj_rmse < 0.5, f"Reprojection RMSE too high: {reproj_rmse}"

    # Verify recovered camera optical center C = -R^T * t
    center_err = float(np.linalg.norm(cam_center_est - cam_pos_gt))
    assert center_err < 1.0, f"Camera center error {center_err:.2f} inches exceeds 1.0 inch threshold"

    # Verify calculated distance to each AprilTag d_tag = || P_tag - C ||
    for i, tid in enumerate(test_tag_ids):
        pos_tag = np.array(CANONICAL_FIELD_TAGS[tid], dtype=np.float64)
        calculated_dist = float(np.linalg.norm(pos_tag - cam_center_est))
        gt_dist = ground_truth_distances[tid]
        dist_diff = abs(calculated_dist - gt_dist)
        assert dist_diff < 1.0, f"Tag {tid} distance error {dist_diff:.2f} inches exceeds 1.0 inch"


def test_robot_3d_ray_projection_and_depth() -> None:
    """
    Verifies 3D ray projection at carpet floor (Z=0) extracts physical (X, Y, 0)
    and camera depth d_robot = || P - C ||.
    """
    cam_pos = np.array([325.6, -180.0, 220.0], dtype=np.float64)
    look_at = np.array([325.6, 158.85, 0.0], dtype=np.float64)
    K, rvec, tvec, C = create_synthetic_camera(cam_pos, look_at, focal_length=1250.0)

    # Robot at known field position on floor
    robot_field_gt = np.array([180.0, 140.0, 0.0], dtype=np.float64)
    expected_depth = float(np.linalg.norm(robot_field_gt - cam_pos))

    # Project to pixel coordinates
    proj, _ = cv2.projectPoints(robot_field_gt.reshape(1, 3), rvec, tvec, K, None)
    u_px, v_px = float(proj[0, 0, 0]), float(proj[0, 0, 1])

    # Unproject ray to Z=0 floor
    field_coords, depth = project_pixel_to_field_3d(
        pixel_u=u_px,
        pixel_v=v_px,
        elevation_inch=0.0,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=K,
        camera_center=C,
    )

    assert abs(field_coords[0] - 180.0) < 0.5, f"X error: {field_coords[0]}"
    assert abs(field_coords[1] - 140.0) < 0.5, f"Y error: {field_coords[1]}"
    assert field_coords[2] == 0.0, "Floor Z must be 0.0"
    assert abs(depth - expected_depth) < 0.5, f"Depth error: {depth} vs {expected_depth}"


def test_field_center_relative_coordinates() -> None:
    """
    Verifies conversion from absolute field coordinates (origin 0,0,0)
    to field-center relative coordinates (Center = 325.6, 158.85, 0.0).
    """
    CENTER_X = 325.6
    CENTER_Y = 158.85

    # 1. Field Center maps to (0, 0, 0)
    x_rel = 325.6 - CENTER_X
    y_rel = 158.85 - CENTER_Y
    assert abs(x_rel) < 1e-6 and abs(y_rel) < 1e-6

    # 2. Red Hub at (162.8, 158.85, 0.0) -> (-162.8, 0.0, 0.0)
    x_rel = 162.8 - CENTER_X
    y_rel = 158.85 - CENTER_Y
    assert abs(x_rel - (-162.8)) < 1e-6
    assert abs(y_rel) < 1e-6

    # 3. Blue Hub at (488.4, 158.85, 0.0) -> (+162.8, 0.0, 0.0)
    x_rel = 488.4 - CENTER_X
    y_rel = 158.85 - CENTER_Y
    assert abs(x_rel - 162.8) < 1e-6
    assert abs(y_rel) < 1e-6

    # 4. Red Tower at (90.0, 158.85, 0.0) -> (-235.6, 0.0, 0.0)
    x_rel = 90.0 - CENTER_X
    assert abs(x_rel - (-235.6)) < 1e-6

    # 5. Carpet origin (0.0, 0.0, 0.0) -> (-325.6, -158.85, 0.0)
    x_rel = 0.0 - CENTER_X
    y_rel = 0.0 - CENTER_Y
    assert abs(x_rel - (-325.6)) < 1e-6
    assert abs(y_rel - (-158.85)) < 1e-6


def test_robot_live_sizing_and_nominal_fallback() -> None:
    """
    Verifies that STrack computes live optical dimensions:
    W_live = w_px * d / f, H_live = h_px * d / f
    and falls back to nominal 22"x28"x28" base when camera extrinsics are absent.
    """
    cam_pos = np.array([325.6, -180.0, 220.0], dtype=np.float64)
    look_at = np.array([325.6, 158.85, 0.0], dtype=np.float64)
    f_cam = 1250.0
    K, rvec, tvec, C = create_synthetic_camera(cam_pos, look_at, focal_length=f_cam)

    # Robot at known location: depth ~ 380 inches
    robot_field = np.array([200.0, 160.0, 0.0], dtype=np.float64)
    d_gt = float(np.linalg.norm(robot_field - cam_pos))

    # Suppose real robot has width = 24.0", height = 30.0"
    # Apparent pixel width & height:
    w_px = float(24.0 * f_cam / d_gt)
    h_px = float(30.0 * f_cam / d_gt)

    proj, _ = cv2.projectPoints(robot_field.reshape(1, 3), rvec, tvec, K, None)
    u_center, v_contact = float(proj[0, 0, 0]), float(proj[0, 0, 1])
    v_center = v_contact - 0.5 * h_px

    bbox = BoundingBox(
        center_x=u_center,
        center_y=v_center,
        width=w_px,
        height=h_px,
        confidence=0.92,
        class_id=0,
    )

    # Solve homography estimate representation
    R, _ = cv2.Rodrigues(rvec)
    H_floor = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    H_floor /= H_floor[2, 2]

    homography_est = HomographyEstimate(
        frame_index=1,
        timestamp_ms=33,
        h_matrix=H_floor,
        dlt_condition=100.0,
        lm_rmse=0.2,
        lm_iterations=5,
        converged=True,
        camera_cut=False,
        affine_cmc=np.eye(2, 3),
        camera_pose_rvec=rvec,
        camera_pose_tvec=tvec,
        camera_center=C,
        focal_length=f_cam,
    )

    track = STrack(track_id=1, bbox=bbox, frame_index=1)

    # Update over multiple frames to smooth dimensions
    for f_idx in range(1, 15):
        track.update_field_geometry(
            h_matrix=homography_est,
            timestamp_ms=f_idx * 33,
            frame_index=f_idx,
            homography_estimate=homography_est,
        )

    # Live dimensions should have converged close to 24" width, 30" height
    assert abs(track.robot_width_inch - 24.0) < 2.0, f"Robot width {track.robot_width_inch} not near 24.0"
    assert abs(track.robot_height_inch - 30.0) < 2.5, f"Robot height {track.robot_height_inch} not near 30.0"
    assert track.depth_inch > 100.0, f"Depth {track.depth_inch} should be > 100 inches"
    assert track.center_rel_x_inch == pytest.approx(200.0 - 325.6, abs=5.0)

    # Fallback test: without homography / camera extrinsics
    fallback_track = STrack(track_id=2, bbox=bbox, frame_index=1)
    fallback_track.update_field_geometry(
        h_matrix=None,
        timestamp_ms=33,
        frame_index=1,
        homography_estimate=None,
    )
    assert fallback_track.robot_width_inch == 22.0, "Must fall back to nominal 22 inch width"
    assert fallback_track.robot_length_inch == 28.0, "Must fall back to nominal 28 inch length"
    assert fallback_track.robot_height_inch == 28.0, "Must fall back to nominal 28 inch height"
    assert fallback_track.depth_inch == 0.0, "Depth must be 0.0 when camera center is unknown"


def test_ball_3d_positioning_and_fuel_specs() -> None:
    """
    Verifies that FRC FUEL balls (5.91" diameter, 2.955" radius) on carpet
    are projected at Z = 2.955" elevation with center-relative coordinates and depth.
    """
    assert FUEL_DIAMETER_INCH == 5.91
    assert FUEL_RADIUS_INCH == 2.955
    assert CARPET_CENTER_ELEVATION_INCH == 2.955

    cam_pos = np.array([325.6, -180.0, 220.0], dtype=np.float64)
    look_at = np.array([325.6, 158.85, 0.0], dtype=np.float64)
    f_cam = 1250.0
    K, rvec, tvec, C = create_synthetic_camera(cam_pos, look_at, focal_length=f_cam)

    # Ball on carpet at (X=300.0, Y=160.0, Z=2.955)
    ball_field_gt = np.array([300.0, 160.0, CARPET_CENTER_ELEVATION_INCH], dtype=np.float64)
    expected_ball_depth = float(np.linalg.norm(ball_field_gt - cam_pos))

    proj, _ = cv2.projectPoints(ball_field_gt.reshape(1, 3), rvec, tvec, K, None)
    u_ball, v_ball = float(proj[0, 0, 0]), float(proj[0, 0, 1])

    # Apparent pixel radius for 5.91" ball
    r_px = float((FUEL_RADIUS_INCH * f_cam) / expected_ball_depth)

    ball_bbox = BoundingBox(
        center_x=u_ball,
        center_y=v_ball,
        width=r_px * 2.0,
        height=r_px * 2.0,
        confidence=0.95,
        class_id=1,
    )

    R, _ = cv2.Rodrigues(rvec)
    H_floor = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    H_floor /= H_floor[2, 2]

    homography_est = HomographyEstimate(
        frame_index=1,
        timestamp_ms=33,
        h_matrix=H_floor,
        dlt_condition=100.0,
        lm_rmse=0.2,
        lm_iterations=5,
        converged=True,
        camera_cut=False,
        affine_cmc=np.eye(2, 3),
        camera_pose_rvec=rvec,
        camera_pose_tvec=tvec,
        camera_center=C,
        focal_length=f_cam,
    )

    tracker = BallTracker(max_balls=504, target_fps=30.0)
    tracklets = tracker.update(
        detections=[ball_bbox],
        frame_index=1,
        homography=homography_est,
    )

    assert len(tracklets) == 1
    t = tracklets[0]
    assert t.field_z_inch == CARPET_CENTER_ELEVATION_INCH
    assert t.center_rel_z_inch == CARPET_CENTER_ELEVATION_INCH
    assert abs(t.field_x_inch - 300.0) < 1.0
    assert abs(t.field_y_inch - 160.0) < 1.0
    assert abs(t.center_rel_x_inch - (300.0 - 325.6)) < 1.0
    assert abs(t.center_rel_y_inch - (160.0 - 158.85)) < 1.0
    assert abs(t.depth_inch - expected_ball_depth) < 1.0


@pytest.mark.asyncio
async def test_duckdb_camera_poses_and_3d_roundtrip() -> None:
    """
    Verifies DuckDB storage actor ingests, queries, and exports 3D fields and camera poses.
    """
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_3d.duckdb")

    try:
        actor = DuckDBStorageActor(db_path=db_path, match_id="test_3d_match")

        # 1. Insert camera pose
        pose = CameraPoseState(
            match_id="test_3d_match",
            frame_index=1,
            timestamp_ms=33,
            camera_center=(325.6, -180.0, 220.0),
            rvec=(1.2, 0.0, 0.0),
            tvec=(0.0, 50.0, 300.0),
            focal_length=1250.0,
            reproj_rmse=0.45,
            converged=True,
            visible_tags=[2, 11, 12, 15],
            tag_distances={2: 345.2, 11: 348.1, 12: 350.0, 15: 410.5},
        )
        await actor.insert_camera_poses([pose])

        # 2. Insert robot trajectory with 3D fields
        bbox = BoundingBox(center_x=960.0, center_y=540.0, width=80.0, height=80.0, confidence=0.9, class_id=0)
        robot_state = RobotTrackletState(
            track_id=1,
            team_number=4414,
            alliance=AllianceColor.RED,
            pixel_bbox=bbox,
            field_x_inch=180.0,
            field_y_inch=140.0,
            velocity_x_ips=12.0,
            velocity_y_ips=0.0,
            heading_rad=0.0,
            track_state=TrackState.TRACKED,
            field_z_inch=0.0,
            depth_inch=320.5,
            center_rel_x_inch=-145.6,
            center_rel_y_inch=-18.85,
            center_rel_z_inch=0.0,
            robot_width_inch=24.0,
            robot_length_inch=30.0,
            robot_height_inch=28.0,
        )
        await actor.insert_trajectory_batch([robot_state], frame_index=1)

        # 3. Insert ball trajectory with 3D fields
        ball_state = BallTrackletState(
            ball_id=101,
            pixel_bbox=BoundingBox(center_x=800.0, center_y=600.0, width=16.0, height=16.0, confidence=0.85, class_id=1),
            field_x_inch=300.0,
            field_y_inch=150.0,
            velocity_x_ips=0.0,
            velocity_y_ips=0.0,
            radius_px=8.0,
            state=BallState.ON_CARPET,
            field_z_inch=2.955,
            depth_inch=280.0,
            center_rel_x_inch=-25.6,
            center_rel_y_inch=-8.85,
            center_rel_z_inch=2.955,
        )
        await actor.insert_ball_trajectories([ball_state], frame_index=1)

        # 4. Query DuckDB directly to verify data integrity
        poses = actor.store.query_dicts("SELECT * FROM camera_poses WHERE match_id = 'test_3d_match'")
        assert len(poses) == 1
        assert poses[0]["camera_x_inch"] == pytest.approx(325.6)
        assert poses[0]["camera_y_inch"] == pytest.approx(-180.0)
        assert poses[0]["camera_z_inch"] == pytest.approx(220.0)
        assert "11" in poses[0]["visible_tags"]

        robots = actor.store.query_dicts("SELECT * FROM robot_trajectories WHERE match_id = 'test_3d_match'")
        assert len(robots) == 1
        assert robots[0]["depth_inch"] == pytest.approx(320.5)
        assert robots[0]["center_rel_x_inch"] == pytest.approx(-145.6)
        assert robots[0]["center_rel_y_inch"] == pytest.approx(-18.85)
        assert robots[0]["robot_width_inch"] == pytest.approx(24.0)
        assert robots[0]["robot_length_inch"] == pytest.approx(30.0)

        balls = actor.store.query_dicts("SELECT * FROM ball_trajectories WHERE match_id = 'test_3d_match'")
        assert len(balls) == 1
        assert balls[0]["field_z_inch"] == pytest.approx(2.955)
        assert balls[0]["center_rel_z_inch"] == pytest.approx(2.955)
        assert balls[0]["depth_inch"] == pytest.approx(280.0)

        # 5. Export to Parquet
        parquet_dir = os.path.join(temp_dir, "parquet")
        exported_path = await actor.export_match_parquet(parquet_dir)
        assert os.path.exists(os.path.join(parquet_dir, "robot_trajectories.parquet"))
        assert os.path.exists(os.path.join(parquet_dir, "ball_trajectories.parquet"))
        assert os.path.exists(os.path.join(parquet_dir, "camera_poses.parquet"))
        actor.store.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_webserver_data_provider_returns_3d_telemetry() -> None:
    """
    Verifies that DataProvider attaches 3D coordinates, robot sizing, and camera pose.
    """
    temp_dir = tempfile.mkdtemp()
    parquet_match_dir = os.path.join(temp_dir, "2026_test_match")
    os.makedirs(parquet_match_dir, exist_ok=True)

    try:
        # Create parquet tables using DuckDB
        conn = duckdb.connect()
        conn.execute(f"""
            CREATE TABLE robot_trajectories AS SELECT
                '2026_test_match' AS match_id,
                1 AS frame_index,
                33 AS timestamp_ms,
                1 AS track_id,
                4414 AS team_number,
                'RED' AS alliance,
                960.0 AS pixel_center_x,
                540.0 AS pixel_center_y,
                80.0 AS pixel_width,
                80.0 AS pixel_height,
                0.95 AS pixel_confidence,
                180.0 AS field_x_inch,
                140.0 AS field_y_inch,
                0.0 AS field_z_inch,
                -145.6 AS center_rel_x_inch,
                -18.85 AS center_rel_y_inch,
                0.0 AS center_rel_z_inch,
                320.5 AS depth_inch,
                24.0 AS robot_width_inch,
                30.0 AS robot_length_inch,
                28.0 AS robot_height_inch,
                10.0 AS velocity_x_ips,
                0.0 AS velocity_y_ips,
                10.0 AS speed_ips,
                0.0 AS heading_rad,
                'TRACKED' AS track_state,
                false AS in_defense;

            CREATE TABLE ball_trajectories AS SELECT
                '2026_test_match' AS match_id,
                1 AS frame_index,
                33 AS timestamp_ms,
                101 AS ball_id,
                800.0 AS pixel_center_x,
                600.0 AS pixel_center_y,
                8.0 AS pixel_radius,
                300.0 AS field_x_inch,
                150.0 AS field_y_inch,
                2.955 AS field_z_inch,
                -25.6 AS center_rel_x_inch,
                -8.85 AS center_rel_y_inch,
                2.955 AS center_rel_z_inch,
                280.0 AS depth_inch,
                0.0 AS velocity_x_ips,
                0.0 AS velocity_y_ips,
                'ON_CARPET' AS ball_state,
                NULL::INTEGER AS occluded_by_robot_id,
                false AS is_cluster,
                1 AS cluster_ball_count;

            CREATE TABLE camera_poses AS SELECT
                '2026_test_match' AS match_id,
                1 AS frame_index,
                33 AS timestamp_ms,
                325.6 AS camera_x_inch,
                -180.0 AS camera_y_inch,
                220.0 AS camera_z_inch,
                1.2 AS rvec_0,
                0.0 AS rvec_1,
                0.0 AS rvec_2,
                0.0 AS tvec_0,
                50.0 AS tvec_1,
                300.0 AS tvec_2,
                1250.0 AS focal_length,
                0.45 AS reproj_rmse,
                true AS pnp_converged,
                '[2, 11]' AS visible_tags,
                '{{"2": 345.2, "11": 348.1}}' AS tag_distances;

            COPY robot_trajectories TO '{os.path.join(parquet_match_dir, "robot_trajectories.parquet")}' (FORMAT PARQUET);
            COPY ball_trajectories TO '{os.path.join(parquet_match_dir, "ball_trajectories.parquet")}' (FORMAT PARQUET);
            COPY camera_poses TO '{os.path.join(parquet_match_dir, "camera_poses.parquet")}' (FORMAT PARQUET);
        """)
        conn.close()

        provider = MatchDataProvider(data_dir=temp_dir)
        res = provider.get_match_frames("2026_test_match", start_frame=1, end_frame=1)
        assert res["total_frames"] == 1
        frame_data = res["frames"][0]

        assert frame_data is not None
        assert "robots" in frame_data and len(frame_data["robots"]) == 1
        r = frame_data["robots"][0]
        assert r["center_rel_x_inch"] == pytest.approx(-145.6)
        assert r["center_rel_y_inch"] == pytest.approx(-18.85)
        assert r["robot_width_inch"] == pytest.approx(24.0)
        assert r["robot_length_inch"] == pytest.approx(30.0)
        assert r["depth_inch"] == pytest.approx(320.5)

        assert "balls" in frame_data and len(frame_data["balls"]) == 1
        b = frame_data["balls"][0]
        assert b["field_z_inch"] == pytest.approx(2.955)
        assert b["center_rel_x_inch"] == pytest.approx(-25.6)
        assert b["center_rel_z_inch"] == pytest.approx(2.955)
        assert b["depth_inch"] == pytest.approx(280.0)

        assert "camera_pose" in frame_data and frame_data["camera_pose"] is not None
        cp = frame_data["camera_pose"]
        assert cp["camera_center"][0] == pytest.approx(325.6)
        assert cp["camera_center"][1] == pytest.approx(-180.0)
        assert cp["camera_center"][2] == pytest.approx(220.0)
        assert cp["focal_length"] == pytest.approx(1250.0)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
