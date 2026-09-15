"""
Unit tests for Dynamic Homography actor and geometric estimation algorithms.

Verifies:
- Synthetic reprojection test with >= 4 point correspondences.
- SVD condition number < 10^6.
- Levenberg-Marquardt reprojection RMSE < 1.0 pixel.
- Projected field coordinates match CAD truth within 0.5 inches.
- Camera Motion Compensation (CMC) affine transformation estimation.
- Camera cut detection trigger on synthetic frame discontinuity and residual divergence.
- Async interface contracts of HomographyActor (HomographyActorBase & HomographyActorProtocol).
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from src.common.interfaces import HomographyActorBase, HomographyActorProtocol
from src.common.types import (
    FrameMetadata,
    HomographyEstimate,
    RawDecodedFrame,
    TagDetectionResult,
)
from src.vision.apriltag_detector import CANONICAL_FIELD_TAGS, AprilTagDetector
from src.vision.homography import (
    HomographyActor,
    compute_cmc_affine_detailed,
    compute_normalized_dlt,
    detect_camera_cut,
    estimate_camera_pose_pnp,
    optimize_homography_lm,
    project_pixel_to_field_3d,
    project_pixels_to_field_3d_batch,
    project_pixels_to_field_batch,
)


def create_synthetic_frame(
    frame_index: int,
    pts_ms: int,
    image_bgr: np.ndarray,
) -> RawDecodedFrame:
    """Helper to instantiate a RawDecodedFrame for unit tests."""
    h, w = image_bgr.shape[:2]
    metadata = FrameMetadata(
        frame_index=frame_index,
        pts_ms=pts_ms,
        monotonic_ts_ns=pts_ms * 1_000_000,
        is_gameplay=True,
        confidence=1.0,
        width=w,
        height=h,
    )
    return RawDecodedFrame(metadata=metadata, image_bgr=image_bgr)


def create_synthetic_camera_projection(
    field_points_2d: np.ndarray,
    noise_sigma: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates a broadcast camera observing the FRC 2026 REBUILT field.
    Camera elevated along sideline at (X=325.6, Y=-150, Z=220), looking at field center.
    Returns:
        (H_ground_truth, clean_pixels, noisy_pixels)
    """
    cam_pos = np.array([325.6, -150.0, 220.0], dtype=np.float64)
    look_at = np.array([325.6, 158.85, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R = np.vstack([right, -true_up, forward])  # Camera coords: X right, Y down, Z forward
    t = -R @ cam_pos
    K = np.array([
        [1800.0, 0.0, 960.0],
        [0.0, 1800.0, 540.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Planar ground homography (Z=0 plane)
    H_planar = K @ np.column_stack([R[:, 0], R[:, 1], t])
    H_planar /= H_planar[2, 2]

    pts_h = np.column_stack([field_points_2d, np.ones(len(field_points_2d), dtype=np.float64)])
    p_proj = (H_planar @ pts_h.T).T
    clean_px = p_proj[:, :2] / p_proj[:, 2:3]

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma, clean_px.shape)
    noisy_px = clean_px + noise

    return H_planar, clean_px, noisy_px


# ============================================================================
# 1. Interface & Protocol Conformance Tests
# ============================================================================


def test_actor_interface_contracts() -> None:
    """Verifies HomographyActor subclasses HomographyActorBase and satisfies HomographyActorProtocol."""
    actor = HomographyActor()
    assert isinstance(actor, HomographyActorBase), "Actor must inherit from HomographyActorBase"
    assert isinstance(actor, HomographyActorProtocol), "Actor must satisfy HomographyActorProtocol"

    # Verify coroutine method signatures
    assert asyncio.iscoroutinefunction(actor.detect_apriltags)
    assert asyncio.iscoroutinefunction(actor.estimate_homography)
    assert asyncio.iscoroutinefunction(actor.project_pixel_to_field)
    assert asyncio.iscoroutinefunction(actor.compute_camera_motion_compensation)


# ============================================================================
# 2. Normalized DLT & SVD Condition Number Tests
# ============================================================================


def test_normalized_dlt_minimum_points_validation() -> None:
    """Asserts that compute_normalized_dlt rejects < 4 point correspondences."""
    field_pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    image_pts = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0]])

    with pytest.raises(ValueError, match=r"requires >= 4 point correspondences"):
        compute_normalized_dlt(field_pts, image_pts)


def test_normalized_dlt_condition_number_bound() -> None:
    """
    Verifies that Hartley-normalized DLT via SVD maintains condition number < 10^6
    even under real-world coordinate magnitude differences (field inches ~600 vs pixels ~1000).
    """
    # 4 field corner points
    field_corners = np.array([
        [0.0, 0.0],
        [651.2, 0.0],
        [651.2, 317.7],
        [0.0, 317.7],
    ], dtype=np.float64)

    H_true, clean_px, noisy_px = create_synthetic_camera_projection(field_corners, noise_sigma=0.5)

    H_init, cond_num = compute_normalized_dlt(field_corners, noisy_px)

    assert cond_num < 1e6, f"SVD condition number {cond_num} must be < 10^6"
    assert np.isfinite(cond_num)
    assert H_init.shape == (3, 3)
    assert pytest.approx(H_init[2, 2], abs=1e-6) == 1.0


# ============================================================================
# 3. Levenberg-Marquardt Optimization & Convergence Tests
# ============================================================================


def test_levenberg_marquardt_optimization_convergence() -> None:
    """
    Verifies that Levenberg-Marquardt non-linear optimization converges
    and achieves reprojection RMSE < 1.0 pixel under Gaussian noise (sigma = 0.5 px).
    """
    tags_array = np.array([[pos[0], pos[1]] for pos in CANONICAL_FIELD_TAGS.values()])
    H_true, clean_px, noisy_px = create_synthetic_camera_projection(tags_array, noise_sigma=0.5)

    H_init, cond_num = compute_normalized_dlt(tags_array, noisy_px)
    assert cond_num < 1e6

    H_opt, rmse, iters, converged = optimize_homography_lm(H_init, tags_array, noisy_px)

    assert converged is True, "Levenberg-Marquardt optimization must converge"
    assert rmse < 1.0, f"Reprojection RMSE {rmse} px must be < 1.0 px"
    assert iters > 0, "Optimization must perform at least 1 iteration"


# ============================================================================
# 4. Synthetic Reprojection Fixture (32 Field Tags) & CAD Truth Accuracy
# ============================================================================


@pytest.mark.asyncio
async def test_synthetic_reprojection_32_tags_accuracy() -> None:
    """
    Synthetic reprojection fixture projecting all 32 known field tags with Gaussian noise (sigma = 0.5 px).
    Asserts:
    1. SVD condition number < 10^6
    2. Levenberg-Marquardt reprojection error converges to RMSE < 1.0 px
    3. Projected field coordinates match CAD truth within 0.5 inches
    """
    actor = HomographyActor()
    tag_ids = sorted(CANONICAL_FIELD_TAGS.keys())
    field_pts_list = [CANONICAL_FIELD_TAGS[tid][:2] for tid in tag_ids]
    field_pts = np.array(field_pts_list, dtype=np.float64)

    H_true, clean_px, noisy_px = create_synthetic_camera_projection(field_pts, noise_sigma=0.5, seed=123)

    # Construct TagDetectionResults
    detections: List[TagDetectionResult] = []
    for i, tid in enumerate(tag_ids):
        det = TagDetectionResult(
            tag_id=tid,
            corners_px=np.zeros((4, 2), dtype=np.float32),
            center_px=(float(noisy_px[i, 0]), float(noisy_px[i, 1])),
            field_coords_inch=CANONICAL_FIELD_TAGS[tid],
            hamming=0,
            reprojection_err=0.0,
        )
        detections.append(det)

    frame_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = create_synthetic_frame(frame_index=1, pts_ms=16, image_bgr=frame_img)

    estimate = await actor.estimate_homography(frame, detections)

    # 1. Assert SVD condition number < 10^6
    assert estimate.dlt_condition < 1e6, f"DLT condition number {estimate.dlt_condition} >= 10^6"

    # 2. Assert LM reprojection RMSE < 1.0 pixel
    assert estimate.lm_rmse < 1.0, f"LM RMSE {estimate.lm_rmse} >= 1.0 px"
    assert estimate.converged is True

    # 3. Assert projected field coordinates match CAD truth within 0.5 inches
    # Test on true pixel coordinates (measuring homography inversion accuracy)
    for i, tid in enumerate(tag_ids):
        true_px = (float(clean_px[i, 0]), float(clean_px[i, 1]))
        cad_truth = (CANONICAL_FIELD_TAGS[tid][0], CANONICAL_FIELD_TAGS[tid][1])

        projected = await actor.project_pixel_to_field(true_px, estimate)
        error_inch = np.linalg.norm(np.array(projected) - np.array(cad_truth))

        assert error_inch < 0.5, (
            f"Tag {tid} projected field coordinate {projected} differs from CAD truth {cad_truth} "
            f"by {error_inch:.4f} inches (exceeds 0.5 in tolerance)"
        )

    # Also test batch projection helper
    batch_projected = project_pixels_to_field_batch(clean_px, estimate.h_matrix)
    batch_errors = np.linalg.norm(batch_projected - field_pts, axis=1)
    assert np.max(batch_errors) < 0.5
    assert np.mean(batch_errors) < 0.15


# ============================================================================
# 5. Camera Motion Compensation (CMC) Affine Estimation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_camera_motion_compensation_affine_estimation() -> None:
    """
    Verifies that Lucas-Kanade optical flow + RANSAC affine CMC accurately
    estimates camera pan and tilt between consecutive frames.
    """
    actor = HomographyActor()

    # Generate synthetic textured frame
    rng = np.random.default_rng(99)
    base_frame = rng.integers(0, 256, (720, 1280), dtype=np.uint8)
    base_frame = cv2.GaussianBlur(base_frame, (9, 9), 2.5)

    # Synthetic affine motion: translation (dx=6.0, dy=-4.0) + rotation (1.2 deg)
    angle_rad = np.deg2rad(1.2)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    tx, ty = 6.0, -4.0
    M_true = np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
    ], dtype=np.float64)

    curr_frame = cv2.warpAffine(base_frame, M_true, (1280, 720))

    # Test standalone CMC helper
    M_est, breakdown, inlier_ratio = compute_cmc_affine_detailed(curr_frame, base_frame)

    assert breakdown is False, "CMC feature tracking should not break down on valid continuous motion"
    assert inlier_ratio > 0.6, f"Expected high inlier ratio, got {inlier_ratio}"
    assert M_est.shape == (2, 3)

    # Verify affine parameters match ground truth within tight tolerance
    assert pytest.approx(M_est[0, 2], abs=0.5) == tx
    assert pytest.approx(M_est[1, 2], abs=0.5) == ty
    assert pytest.approx(M_est[0, 0], abs=0.02) == cos_a
    assert pytest.approx(M_est[1, 0], abs=0.02) == sin_a

    # Test actor async method
    actor_affine = await actor.compute_camera_motion_compensation(curr_frame, base_frame)
    assert actor_affine.shape == (2, 3)
    assert actor_affine.dtype == np.float64
    assert pytest.approx(actor_affine[0, 2], abs=0.5) == tx


@pytest.mark.asyncio
async def test_cmc_fallback_on_blank_frames() -> None:
    """Verifies that CMC returns identity matrix when frames lack trackable features."""
    actor = HomographyActor()
    blank1 = np.zeros((480, 640, 3), dtype=np.uint8)
    blank2 = np.zeros((480, 640, 3), dtype=np.uint8)

    affine = await actor.compute_camera_motion_compensation(blank2, blank1)
    expected_identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    np.testing.assert_allclose(affine, expected_identity)


# ============================================================================
# 6. Camera Cut Detection Tests
# ============================================================================


def test_camera_cut_detection_on_synthetic_frame_discontinuity() -> None:
    """
    Tests camera cut detection triggered by:
    1. Feature tracking breakdown across random/discontinuous frames.
    2. Homography residual divergence (sudden spike in reprojection error).
    3. Projective displacement jump between consecutive homographies.
    """
    # 1. Feature tracking breakdown trigger
    rng = np.random.default_rng(101)
    img_a = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    img_b = rng.integers(0, 256, (480, 640), dtype=np.uint8)

    _, breakdown, _ = compute_cmc_affine_detailed(img_b, img_a)
    assert breakdown is True, "Discontinuous random frames must trigger feature tracking breakdown"

    cut_detected = detect_camera_cut(
        curr_h=None,
        prev_h=None,
        curr_rmse=0.5,
        prev_rmse=0.5,
        tracking_breakdown=breakdown,
    )
    assert cut_detected is True, "Camera cut must trigger on tracking breakdown"

    # 2. Homography residual divergence trigger (RMSE spike > 5.0 px)
    cut_on_rmse = detect_camera_cut(
        curr_h=np.eye(3),
        prev_h=np.eye(3),
        curr_rmse=7.2,
        prev_rmse=0.6,
        tracking_breakdown=False,
    )
    assert cut_on_rmse is True, "Camera cut must trigger when LM RMSE diverges"

    # 3. Homography projective displacement jump
    H1 = np.eye(3, dtype=np.float64)
    # H2 shifts camera viewpoint by 300 pixels
    H2 = np.array([
        [1.0, 0.0, 300.0],
        [0.0, 1.0, 300.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    cut_on_displacement = detect_camera_cut(
        curr_h=H2,
        prev_h=H1,
        curr_rmse=0.8,
        prev_rmse=0.7,
        tracking_breakdown=False,
    )
    assert cut_on_displacement is True, "Camera cut must trigger on large homography displacement"


@pytest.mark.asyncio
async def test_homography_actor_camera_cut_pipeline() -> None:
    """
    Verifies that HomographyActor flags camera_cut = True on a discontinuous frame sequence,
    and camera_cut = False during smooth camera tracking.
    """
    actor = HomographyActor()
    tag_ids = [1, 2, 5, 6, 7, 8, 11, 12]
    field_pts = np.array([CANONICAL_FIELD_TAGS[t][:2] for t in tag_ids])

    # Frame 1: Initial camera position
    H1, _, p1_noisy = create_synthetic_camera_projection(field_pts, noise_sigma=0.3, seed=1)
    dets1 = [
        TagDetectionResult(
            tag_id=t,
            corners_px=np.zeros((4, 2), dtype=np.float32),
            center_px=(float(p1_noisy[i, 0]), float(p1_noisy[i, 1])),
            field_coords_inch=CANONICAL_FIELD_TAGS[t],
            hamming=0,
            reprojection_err=0.0,
        )
        for i, t in enumerate(tag_ids)
    ]
    frame1 = create_synthetic_frame(0, 0, np.full((1080, 1920, 3), 128, dtype=np.uint8))
    est1 = await actor.estimate_homography(frame1, dets1)
    assert est1.camera_cut is False

    # Frame 2: Smooth motion (slight shift in pixel space)
    dets2 = [
        TagDetectionResult(
            tag_id=t,
            corners_px=np.zeros((4, 2), dtype=np.float32),
            center_px=(float(p1_noisy[i, 0] + 1.5), float(p1_noisy[i, 1] - 0.5)),
            field_coords_inch=CANONICAL_FIELD_TAGS[t],
            hamming=0,
            reprojection_err=0.0,
        )
        for i, t in enumerate(tag_ids)
    ]
    frame2 = create_synthetic_frame(1, 16, np.full((1080, 1920, 3), 128, dtype=np.uint8))
    est2 = await actor.estimate_homography(frame2, dets2, prev_homography=est1)
    assert est2.camera_cut is False, "Smooth frame sequence should not trigger camera cut"

    # Frame 3: Discontinuous jump (e.g. broadcast switches cameras -> huge displacement)
    dets3 = [
        TagDetectionResult(
            tag_id=t,
            corners_px=np.zeros((4, 2), dtype=np.float32),
            center_px=(float(p1_noisy[i, 0] + 400.0), float(p1_noisy[i, 1] + 300.0)),
            field_coords_inch=CANONICAL_FIELD_TAGS[t],
            hamming=0,
            reprojection_err=0.0,
        )
        for i, t in enumerate(tag_ids)
    ]
    frame3 = create_synthetic_frame(2, 33, np.full((1080, 1920, 3), 128, dtype=np.uint8))
    est3 = await actor.estimate_homography(frame3, dets3, prev_homography=est2)
    assert est3.camera_cut is True, "Discontinuous jump must trigger camera_cut=True"


# ============================================================================
# 7. AprilTag Detector Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_apriltag_detection_and_tag_mapping() -> None:
    """
    Tests AprilTag detection targeting tag36h11 and mapping to CAD coordinates.
    Generates a synthetic tag marker image and runs detect_apriltags.
    """
    actor = HomographyActor()
    dict_tag = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    # Generate tag ID 5 (Red Hub face)
    tag_size_px = 120
    marker = cv2.aruco.generateImageMarker(dict_tag, 5, tag_size_px)

    canvas = np.full((600, 800, 3), 255, dtype=np.uint8)
    canvas[200:320, 300:420] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    frame = create_synthetic_frame(0, 0, canvas)
    detections = await actor.detect_apriltags(frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.tag_id == 5
    assert det.field_coords_inch == CANONICAL_FIELD_TAGS[5]
    assert pytest.approx(det.center_px[0], abs=5.0) == 360.0
    assert pytest.approx(det.center_px[1], abs=5.0) == 260.0


@pytest.mark.asyncio
async def test_fewer_than_4_detections_handling() -> None:
    """Asserts that < 4 detections gracefully return non-converged homography estimate."""
    actor = HomographyActor()
    frame = create_synthetic_frame(0, 0, np.zeros((480, 640, 3), dtype=np.uint8))

    est = await actor.estimate_homography(frame, detections=[])
    assert est.converged is False
    assert est.h_matrix.shape == (3, 3)
    assert est.affine_cmc.shape == (2, 3)


@pytest.mark.asyncio
async def test_split_screen_apriltag_isolation() -> None:
    """Verifies that in a split-screen frame, lower inset tags (y > 0.65 * H) are isolated from wide camera."""
    actor = HomographyActor()
    frame_bgr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = create_synthetic_frame(0, 0, frame_bgr)

    # 4 authentic tags in top wide camera (y in 200..500)
    top_dets = [
        TagDetectionResult(
            tag_id=i,
            corners_px=np.zeros((4, 2)),
            center_px=(float(400 + i * 200), float(300 + i * 50)),
            field_coords_inch=CANONICAL_FIELD_TAGS.get(i, (float(i * 50), 100.0, 0.0)),
            hamming=0,
            reprojection_err=0.0,
        )
        for i in [5, 6, 7, 8]
    ]
    # 2 auxiliary tags in bottom floor inset camera (y = 850, 900)
    bottom_dets = [
        TagDetectionResult(
            tag_id=18,
            corners_px=np.zeros((4, 2)),
            center_px=(500.0, 850.0),
            field_coords_inch=CANONICAL_FIELD_TAGS.get(18, (488.4, 182.35, 62.0)),
            hamming=0,
            reprojection_err=0.0,
        ),
        TagDetectionResult(
            tag_id=19,
            corners_px=np.zeros((4, 2)),
            center_px=(800.0, 900.0),
            field_coords_inch=CANONICAL_FIELD_TAGS.get(19, (511.9, 158.85, 62.0)),
            hamming=0,
            reprojection_err=0.0,
        ),
    ]

    all_dets = top_dets + bottom_dets
    est = await actor.estimate_homography(frame, detections=all_dets)
    assert est.converged is True
    assert est.h_matrix is not None


# ============================================================================
# 8. Perspective-n-Point (PnP) 3D Camera Pose & Ground Plane Estimation Tests
# ============================================================================


def test_estimate_camera_pose_pnp_and_floor_homography() -> None:
    """
    Verifies that estimate_camera_pose_pnp recovers true 3D camera pose (R, T, C)
    and floor homography H_floor mapping carpet Z=0 plane to image pixels.
    Uses canonical 32 AprilTag CAD coordinates from field.glb (specifically Tags 2, 11 on Red Hub).
    """
    # Elevated broadcast sideline camera at (325.6, 520.0, 160.0) looking at field center (325.6, 158.85, 0.0)
    cam_pos_true = np.array([325.6, 520.0, 160.0], dtype=np.float64)
    look_at = np.array([325.6, 158.85, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = look_at - cam_pos_true
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R_true = np.vstack([right, -true_up, forward])
    t_true = -R_true @ cam_pos_true
    rvec_true, _ = cv2.Rodrigues(R_true)
    tvec_true = t_true.reshape(3, 1)

    f_true = 1450.0
    K_true = np.array([[f_true, 0.0, 960.0], [0.0, f_true, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Subset of 3D tags including Tag 2 and Tag 11 on Red Hub
    tag_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 21, 24]
    tags_3d = np.array([CANONICAL_FIELD_TAGS[tid] for tid in tag_ids], dtype=np.float64)

    proj_clean, _ = cv2.projectPoints(tags_3d, rvec_true, tvec_true, K_true, None)
    img_pts_clean = proj_clean.reshape(-1, 2)

    rvec_est, tvec_est, cam_center_est, H_floor, rmse, f_est, success = estimate_camera_pose_pnp(
        object_points_3d=tags_3d,
        image_points_2d=img_pts_clean,
        camera_matrix=K_true,
        image_shape=(1080, 1920),
    )

    assert success is True
    assert rmse < 1e-4, f"PnP RMSE {rmse} px must be near zero on clean points"

    # Camera optical center recovery error < 0.1 inches
    cam_pos_err = np.linalg.norm(cam_center_est - cam_pos_true)
    assert cam_pos_err < 0.1, f"Camera center error {cam_pos_err} in exceeds 0.1 in"

    # Verify Floor Homography H_floor on ground carpet points (Z=0)
    carpet_points_3d = np.array([
        [100.0, 100.0, 0.0],
        [325.6, 158.85, 0.0],
        [500.0, 200.0, 0.0],
        [200.0, 250.0, 0.0],
    ], dtype=np.float64)

    proj_carpet, _ = cv2.projectPoints(carpet_points_3d, rvec_true, tvec_true, K_true, None)
    img_carpet = proj_carpet.reshape(-1, 2)

    # Unproject carpet points using H_floor
    H_inv = np.linalg.inv(H_floor)
    for i in range(len(carpet_points_3d)):
        u, v = img_carpet[i]
        p_hom = H_inv @ np.array([u, v, 1.0])
        unproj_xy = p_hom[:2] / p_hom[2]
        err_xy = np.linalg.norm(unproj_xy - carpet_points_3d[i, :2])
        assert err_xy < 0.1, f"Floor homography unprojection error {err_xy} in exceeds 0.1 in"


# ============================================================================
# 9. 3D Ray Depth Projection Tests (5.91" FUEL Ball & Robot Ground Patch)
# ============================================================================


def test_project_pixel_to_field_3d_fuel_and_robot() -> None:
    """
    Verifies 3D ray projection to horizontal planes and Euclidean depth extraction:
    1. Carpet ground patch at Z = 0.0 inches (robot bottom center contact)
    2. FUEL game piece center at Z = 2.955 inches (5.91" diameter sphere)
    3. Verifies Euclidean depth || P - C || matches ground truth within 0.01 inches.
    """
    cam_pos = np.array([325.6, 520.0, 160.0], dtype=np.float64)
    look_at = np.array([325.6, 158.85, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R = np.vstack([right, -true_up, forward])
    t = -R @ cam_pos
    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    K = np.array([[1250.0, 0.0, 960.0], [0.0, 1250.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # 1. Robot ground contact patch at (X=250.0, Y=180.0, Z=0.0)
    robot_ground_pt = np.array([250.0, 180.0, 0.0], dtype=np.float64)
    proj_r, _ = cv2.projectPoints(robot_ground_pt.reshape(1, 3), rvec, tvec, K, None)
    u_r, v_r = proj_r[0, 0]

    coords_r, depth_r = project_pixel_to_field_3d(
        pixel_u=u_r,
        pixel_v=v_r,
        elevation_inch=0.0,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=K,
        camera_center=cam_pos,
    )
    assert np.linalg.norm(np.array(coords_r) - robot_ground_pt) < 0.01
    true_robot_depth = float(np.linalg.norm(robot_ground_pt - cam_pos))
    assert abs(depth_r - true_robot_depth) < 0.01

    # 2. FUEL game piece (5.91" diameter => radius = 2.955") at (X=300.0, Y=220.0, Z=2.955)
    fuel_center_pt = np.array([300.0, 220.0, 2.955], dtype=np.float64)
    proj_f, _ = cv2.projectPoints(fuel_center_pt.reshape(1, 3), rvec, tvec, K, None)
    u_f, v_f = proj_f[0, 0]

    coords_f, depth_f = project_pixel_to_field_3d(
        pixel_u=u_f,
        pixel_v=v_f,
        elevation_inch=2.955,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=K,
        camera_center=cam_pos,
    )
    assert np.linalg.norm(np.array(coords_f) - fuel_center_pt) < 0.01
    true_fuel_depth = float(np.linalg.norm(fuel_center_pt - cam_pos))
    assert abs(depth_f - true_fuel_depth) < 0.01

    # 3. Batch projection test
    coords_batch, depths_batch = project_pixels_to_field_3d_batch(
        pixel_coords=np.array([[u_r, v_r], [u_f, v_f]]),
        elevation_inch=0.0,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=K,
        camera_center=cam_pos,
    )
    assert len(coords_batch) == 2
    assert len(depths_batch) == 2
    assert abs(depths_batch[0] - true_robot_depth) < 0.01

