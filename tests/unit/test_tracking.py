"""
Unit tests for BoT-SORT Multi-Object Tracking, CMC, ReID, and Tracklet Bridging.

Verifies:
- TrackingActor interface contract (TrackingActorBase and TrackingActorProtocol).
- KalmanFilter 8-state representation [xc, yc, w, h, vx, vy, vw, vh]^T.
- Camera Motion Compensation (CMC) affine transformation on state & covariance.
- Robot crossing without identity switch (ReID appearance discrimination).
- 4-second broadcast camera cut (240 frames) and tracklet bridging upon re-emergence.
- Multi-robot tracking with simulated Hub occlusion.
- OCR running consensus and weighted voting.
- Field projection to physical inches and velocity estimation.
- Expiration and pruning of lost tracklets past max_bridge_gap_frames.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.common.interfaces import TrackingActorBase, TrackingActorProtocol
from src.common.types import (
    AllianceColor,
    BoundingBox,
    HomographyEstimate,
    OCRResult,
    RobotTrackletState,
    TrackState,
)
from src.tracking.bot_sort import (
    BoTSORT,
    STrack,
    TrackingActor,
    box_iou,
    fuse_iou_reid_cost,
    iou_distance,
    linear_assignment,
)
from src.tracking.cmc import (
    apply_cmc_transform,
    build_cmc_matrix,
    compute_camera_motion_compensation,
    transform_box_cmc,
)
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.reid import (
    ReIDExtractor,
    cosine_distance,
    generate_synthetic_embedding,
    normalize_embedding,
)


def make_dummy_homography(
    frame_index: int,
    camera_cut: bool = False,
    affine_cmc: Optional[np.ndarray] = None,
    h_matrix: Optional[np.ndarray] = None,
    timestamp_ms: Optional[int] = None,
) -> HomographyEstimate:
    """Creates a synthetic HomographyEstimate for testing."""
    if affine_cmc is None:
        affine_cmc = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

    if h_matrix is None:
        # Identity 3x3 homography: 1 pixel = 1 inch
        h_matrix = np.eye(3, dtype=np.float64)

    if timestamp_ms is None:
        timestamp_ms = int(frame_index * (1000.0 / 60.0))

    return HomographyEstimate(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        h_matrix=h_matrix,
        dlt_condition=1.0,
        lm_rmse=0.0,
        lm_iterations=10,
        converged=True,
        camera_cut=camera_cut,
        affine_cmc=affine_cmc,
    )


# ============================================================================
# 1. Interface Contract Test
# ============================================================================

def test_actor_interface_contracts():
    """Verifies TrackingActor adheres to TrackingActorBase and TrackingActorProtocol."""
    actor = TrackingActor()
    assert isinstance(actor, TrackingActorBase), "TrackingActor must inherit TrackingActorBase"
    assert isinstance(actor, TrackingActorProtocol), "TrackingActor must conform to TrackingActorProtocol"


# ============================================================================
# 2. Test 1: Kalman Filter State & CMC Validation
# ============================================================================

def test_kalman_filter_state_and_cmc():
    """
    Validates:
    - State vector is explicitly 8-dimensional: [xc, yc, w, h, vx, vy, vw, vh]^T
    - Direct width and height tracking (no aspect ratio w/h)
    - CMC affine translation shifts predicted center coordinates accordingly
    """
    initial_box = np.array([200.0, 150.0, 60.0, 80.0], dtype=np.float64)
    kf = KalmanFilter(initial_box)

    # 1. Verify 8-dimensional state vector
    state = kf.state_vector
    assert state.shape == (8,), f"Expected 8-D state vector, got {state.shape}"
    assert state[0] == pytest.approx(200.0), "xc must match initial center_x"
    assert state[1] == pytest.approx(150.0), "yc must match initial center_y"
    assert state[2] == pytest.approx(60.0), "w must directly track width"
    assert state[3] == pytest.approx(80.0), "h must directly track height"
    assert state[4] == pytest.approx(0.0), "Initial vx must be 0"
    assert state[5] == pytest.approx(0.0), "Initial vy must be 0"
    assert state[6] == pytest.approx(0.0), "Initial vw must be 0"
    assert state[7] == pytest.approx(0.0), "Initial vh must be 0"

    # 2. Kalman prediction
    pred_mean, pred_cov = kf.predict()
    assert pred_mean.shape == (8,)
    assert pred_cov.shape == (8, 8)

    # 3. Apply Camera Motion Compensation (CMC) translation: tx = 50, ty = -30
    affine_translation = np.array([
        [1.0, 0.0, 50.0],
        [0.0, 1.0, -30.0],
    ], dtype=np.float64)

    cmc_mean, cmc_cov = kf.apply_cmc(affine_translation)

    # Assert predicted centers shifted by exactly [50, -30]
    assert cmc_mean[0] == pytest.approx(250.0), "xc must shift by tx=+50"
    assert cmc_mean[1] == pytest.approx(120.0), "yc must shift by ty=-30"
    # Width and height remain unaffected by pure translation
    assert cmc_mean[2] == pytest.approx(60.0), "Width must remain unaffected by pure translation"
    assert cmc_mean[3] == pytest.approx(80.0), "Height must remain unaffected by pure translation"

    # Covariance shape and symmetry
    assert cmc_cov.shape == (8, 8)
    assert np.allclose(cmc_cov, cmc_cov.T, atol=1e-5), "Covariance must remain symmetric after CMC"


def test_kalman_filter_scaling_growth():
    """Verifies Kalman filter tracks width and height expansion when approaching camera."""
    kf = KalmanFilter(np.array([100.0, 100.0, 40.0, 40.0]))
    # Robot approaches camera: measurements expand in size
    measurements = [
        np.array([100.0, 100.0, 45.0, 45.0]),
        np.array([100.0, 100.0, 50.0, 50.0]),
        np.array([100.0, 100.0, 56.0, 56.0]),
    ]
    for meas in measurements:
        kf.predict()
        kf.update(meas)

    # Width and height velocity vw, vh should be positive
    assert kf.mean[2] > 50.0
    assert kf.mean[3] > 50.0
    assert kf.mean[6] > 0.0, "vw must be positive when expanding"
    assert kf.mean[7] > 0.0, "vh must be positive when expanding"


# ============================================================================
# 3. ReID Distance & Dynamic Cost Fusion Tests
# ============================================================================

def test_reid_cosine_distance():
    """Verifies 512-D normalized cosine distance properties."""
    e1 = generate_synthetic_embedding(512, basis_index=0)
    e2 = generate_synthetic_embedding(512, basis_index=1)
    e1_dup = e1.copy()

    # Self-distance is 0
    dist_self = cosine_distance(e1, e1_dup)
    assert dist_self == pytest.approx(0.0, abs=1e-5)

    # Orthogonal vectors have distance 1.0
    dist_ortho = cosine_distance(e1, e2)
    assert dist_ortho == pytest.approx(1.0, abs=1e-5)

    # Opposite vectors have distance 2.0
    dist_opp = cosine_distance(e1, -e1)
    assert dist_opp == pytest.approx(2.0, abs=1e-5)


def test_dynamic_cost_fusion_gating():
    """Verifies strict distance threshold masking in IoU-ReID cost fusion."""
    # Two overlapping boxes (IoU > 0 => iou_dist < 1.0)
    boxes1 = np.array([[100.0, 100.0, 50.0, 50.0]])
    boxes2 = np.array([[110.0, 100.0, 50.0, 50.0]])
    iou_dist = iou_distance(boxes1, boxes2)
    assert iou_dist[0, 0] < 0.5, "Boxes must overlap"

    # Case A: Same appearance (ReID dist = 0.0 <= 0.4)
    reid_dist_match = np.array([[0.05]])
    fused_match = fuse_iou_reid_cost(iou_dist, reid_dist_match, reid_threshold=0.4)
    assert fused_match[0, 0] < 1.0, "Matching appearance should have low cost"

    # Case B: Dissimilar appearance (ReID dist = 0.9 > 0.4)
    reid_dist_mismatch = np.array([[0.9]])
    fused_mismatch = fuse_iou_reid_cost(iou_dist, reid_dist_mismatch, reid_threshold=0.4)
    assert fused_mismatch[0, 0] >= 1e4, "Mismatching appearance must be gated to high cost"

    # Case C: Zero IoU (disjoint boxes)
    disjoint_box = np.array([[500.0, 500.0, 50.0, 50.0]])
    disjoint_iou_dist = iou_distance(boxes1, disjoint_box)
    fused_disjoint = fuse_iou_reid_cost(disjoint_iou_dist, reid_dist_match, reid_threshold=0.4)
    assert fused_disjoint[0, 0] >= 1e4, "Zero IoU must be gated to high cost in Stage 1"


# ============================================================================
# 4. Test 2: Robot Crossing Without Identity Switch
# ============================================================================

@pytest.mark.asyncio
async def test_robot_crossing_without_identity_switch():
    """
    Simulates two identically sized robots crossing paths with overlapping bounding boxes.
    Verifies:
    - ReID appearance prevents identity swap during full box overlap.
    - Exactly 0 identity switches across entire trajectory.
    - Each robot retains its original track_id and team_number post-crossing.
    """
    actor = TrackingActor(target_fps=60.0)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Robot 1: moves from x=100 to x=300, team 254 (BLUE), ReID e1
    # Robot 2: moves from x=300 to x=100, team 1678 (RED), ReID e2
    emb_robot1 = generate_synthetic_embedding(512, basis_index=0)
    emb_robot2 = generate_synthetic_embedding(512, basis_index=1)

    ocr_robot1 = OCRResult(team_number=254, alliance=AllianceColor.BLUE, confidence=0.95, super_resolved=True)
    ocr_robot2 = OCRResult(team_number=1678, alliance=AllianceColor.RED, confidence=0.95, super_resolved=True)

    num_frames = 21  # t=0 to t=20, cross at t=10 (x=200)
    robot1_track_id: Optional[int] = None
    robot2_track_id: Optional[int] = None

    for frame_idx in range(num_frames):
        # Position interpolation
        x1 = 100.0 + frame_idx * 10.0  # 100 -> 300
        x2 = 300.0 - frame_idx * 10.0  # 300 -> 100
        y = 200.0

        box1 = BoundingBox(center_x=x1, center_y=y, width=50.0, height=50.0, confidence=0.9, class_id=0)
        box2 = BoundingBox(center_x=x2, center_y=y, width=50.0, height=50.0, confidence=0.9, class_id=0)

        homography = make_dummy_homography(frame_idx)

        active_tracks = await actor.update_tracks(
            frame_index=frame_idx,
            detections=[box1, box2],
            ocr_results=[ocr_robot1, ocr_robot2],
            homography=homography,
            frame_bgr=dummy_frame,
            embeddings=[emb_robot1, emb_robot2],
        )

        assert len(active_tracks) == 2, f"Expected 2 active tracks at frame {frame_idx}, got {len(active_tracks)}"

        # Find track corresponding to robot 1 (team 254) and robot 2 (team 1678)
        tr1 = next((t for t in active_tracks if t.team_number == 254), None)
        tr2 = next((t for t in active_tracks if t.team_number == 1678), None)

        assert tr1 is not None, f"Robot 1 (team 254) missing at frame {frame_idx}"
        assert tr2 is not None, f"Robot 2 (team 1678) missing at frame {frame_idx}"

        if frame_idx == 0:
            robot1_track_id = tr1.track_id
            robot2_track_id = tr2.track_id
            assert robot1_track_id != robot2_track_id, "Track IDs must be unique"
        else:
            # Check for 0 identity switches
            assert tr1.track_id == robot1_track_id, f"Identity switch on Robot 1 at frame {frame_idx}"
            assert tr2.track_id == robot2_track_id, f"Identity switch on Robot 2 at frame {frame_idx}"

        # Position tracking verification
        assert tr1.pixel_bbox.center_x == pytest.approx(x1, abs=5.0)
        assert tr2.pixel_bbox.center_x == pytest.approx(x2, abs=5.0)

    # Post-crossing verification at final frame
    assert tr1.pixel_bbox.center_x == pytest.approx(300.0, abs=5.0)
    assert tr2.pixel_bbox.center_x == pytest.approx(100.0, abs=5.0)
    assert tr1.track_id == robot1_track_id
    assert tr2.track_id == robot2_track_id
    assert tr1.team_number == 254
    assert tr2.team_number == 1678


# ============================================================================
# 5. Test 3: 4-Second Camera Cut & Tracklet Bridging
# ============================================================================

@pytest.mark.asyncio
async def test_camera_cut_and_tracklet_bridging():
    """
    Simulates a 4-second camera cut (240 frames without detections).
    Verifies:
    - Tracks transition to LOST during the broadcast discontinuity.
    - Re-emergence at a different field location reconnects to original pre-cut track_id.
    - State is updated to TrackState.BRIDGED.
    - Team number and alliance identity are preserved.
    """
    actor = TrackingActor(target_fps=60.0, max_lost_frames=300)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    emb_robot = generate_synthetic_embedding(512, seed=42)
    ocr_robot = OCRResult(team_number=971, alliance=AllianceColor.BLUE, confidence=0.9, super_resolved=True)

    # 1. Establish track before camera cut (frames 0..5)
    pre_cut_box = BoundingBox(center_x=150.0, center_y=150.0, width=60.0, height=60.0, confidence=0.9, class_id=0)
    for f in range(6):
        homography = make_dummy_homography(f, camera_cut=False)
        tracks = await actor.update_tracks(
            frame_index=f,
            detections=[pre_cut_box],
            ocr_results=[ocr_robot],
            homography=homography,
            frame_bgr=dummy_frame,
            embeddings=[emb_robot],
        )

    assert len(tracks) == 1
    pre_cut_track_id = tracks[0].track_id
    assert tracks[0].team_number == 971
    assert tracks[0].track_state == TrackState.TRACKED

    # 2. 4-second camera cut (240 frames: frame 6 to 245)
    for f in range(6, 246):
        is_cut_edge = (f == 6)
        homography_cut = make_dummy_homography(f, camera_cut=is_cut_edge)
        empty_tracks = await actor.update_tracks(
            frame_index=f,
            detections=[],
            ocr_results=[],
            homography=homography_cut,
            frame_bgr=dummy_frame,
            embeddings=[],
        )
        assert len(empty_tracks) == 0, "No active tracks should be returned during camera cut"

    # Verify track is currently in LOST state in memory
    assert len(actor.tracker.lost_tracks) == 1
    lost_track = actor.tracker.lost_tracks[0]
    assert lost_track.track_id == pre_cut_track_id
    assert lost_track.state == TrackState.LOST
    assert lost_track.time_since_update == 240

    # 3. Re-emergence at frame 246 at a completely different position (x=500, y=400)
    post_cut_box = BoundingBox(center_x=500.0, center_y=400.0, width=60.0, height=60.0, confidence=0.9, class_id=0)
    homography_resume = make_dummy_homography(246, camera_cut=False)

    reconnected_tracks = await actor.update_tracks(
        frame_index=246,
        detections=[post_cut_box],
        ocr_results=[],  # No OCR in this frame; tests identity preservation
        homography=homography_resume,
        frame_bgr=dummy_frame,
        embeddings=[emb_robot],
    )

    # 4. Assert tracklet bridging success
    assert len(reconnected_tracks) == 1, "Expected 1 active bridged track"
    bridged_track = reconnected_tracks[0]

    assert bridged_track.track_id == pre_cut_track_id, (
        f"Track ID mismatch: expected {pre_cut_track_id}, got {bridged_track.track_id}"
    )
    assert bridged_track.team_number == 971, "Team number must be retained across camera cut"
    assert bridged_track.alliance == AllianceColor.BLUE, "Alliance must be retained across camera cut"
    assert bridged_track.track_state == TrackState.BRIDGED, (
        f"Track state must be BRIDGED upon re-emergence, got {bridged_track.track_state}"
    )
    assert bridged_track.pixel_bbox.center_x == pytest.approx(500.0)
    assert bridged_track.pixel_bbox.center_y == pytest.approx(400.0)


# ============================================================================
# 6. Track Expiration & Pruning Test
# ============================================================================

@pytest.mark.asyncio
async def test_lost_tracklet_expiration_pruning():
    """Verifies that lost tracks older than max_bridge_gap_frames (300 frames) are removed."""
    actor = TrackingActor(target_fps=60.0, max_lost_frames=300)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    emb = generate_synthetic_embedding(512, seed=1)
    box = BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)

    # Initialize track
    await actor.update_tracks(0, [box], [], make_dummy_homography(0), dummy_frame, embeddings=[emb])
    assert len(actor.tracker.tracked_tracks) == 1

    # Drop detections for 301 frames
    for f in range(1, 302):
        await actor.update_tracks(f, [], [], make_dummy_homography(f), dummy_frame, embeddings=[])

    # After 301 frames, the tracklet must be removed from lost_tracks
    assert len(actor.tracker.lost_tracks) == 0
    assert len(actor.tracker.removed_tracks) >= 1
    assert actor.tracker.removed_tracks[0].state == TrackState.REMOVED


# ============================================================================
# 7. OCR Consensus & Alliance Identification Test
# ============================================================================

@pytest.mark.asyncio
async def test_ocr_running_consensus_majority_vote():
    """Verifies that running weighted consensus correctly determines team number and alliance."""
    actor = TrackingActor(target_fps=60.0)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = BoundingBox(center_x=200.0, center_y=200.0, width=50.0, height=50.0, confidence=0.95, class_id=0)
    emb = generate_synthetic_embedding(512, seed=7)

    # Frame 0: noisy OCR reading (team 254 with 0.4 confidence, BLUE)
    ocr_noise = OCRResult(team_number=254, alliance=AllianceColor.BLUE, confidence=0.4, super_resolved=False)
    tracks0 = await actor.update_tracks(0, [box], [ocr_noise], make_dummy_homography(0), dummy_frame, embeddings=[emb])
    assert tracks0[0].team_number == 254

    # Frame 1: strong correct OCR reading (team 1678 with 0.9 confidence, RED)
    ocr_strong = OCRResult(team_number=1678, alliance=AllianceColor.RED, confidence=0.9, super_resolved=True)
    tracks1 = await actor.update_tracks(1, [box], [ocr_strong], make_dummy_homography(1), dummy_frame, embeddings=[emb])
    # 1678 has total confidence 0.9 vs 254 with 0.4 => 1678 wins!
    assert tracks1[0].team_number == 1678
    assert tracks1[0].alliance == AllianceColor.RED

    # Frame 2: another confirming reading for 1678 (confidence 0.8)
    ocr_confirm = OCRResult(team_number=1678, alliance=AllianceColor.RED, confidence=0.8, super_resolved=True)
    tracks2 = await actor.update_tracks(2, [box], [ocr_confirm], make_dummy_homography(2), dummy_frame, embeddings=[emb])
    assert tracks2[0].team_number == 1678
    assert tracks2[0].alliance == AllianceColor.RED


# ============================================================================
# 8. Field Coordinates & Velocity IPS Estimation Test
# ============================================================================

@pytest.mark.asyncio
async def test_field_coordinates_and_velocity_estimation():
    """Verifies homography projection to field CAD coordinates and velocity calculation."""
    actor = TrackingActor(target_fps=60.0)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = generate_synthetic_embedding(512, seed=99)

    # Homography scale: 10 pixels = 1 field inch (H_inv multiplies pixel coords by 0.1)
    # H maps field -> image, so H = diag(10, 10, 1)
    h_matrix = np.diag([10.0, 10.0, 1.0]).astype(np.float64)

    # Robot moves at 10 px/frame = 1.0 inch/frame = 60 inches/sec (ips)
    track_res = None
    for f in range(10):
        px_x = 100.0 + f * 10.0
        px_y = 200.0
        ts = int(f * (1000.0 / 60.0))
        box = BoundingBox(center_x=px_x, center_y=px_y, width=40.0, height=40.0, confidence=0.9, class_id=0)
        homo = make_dummy_homography(f, h_matrix=h_matrix, timestamp_ms=ts)
        tracks = await actor.update_tracks(f, [box], [], homo, dummy_frame, embeddings=[emb])
        assert len(tracks) == 1
        track_res = tracks[0]

    assert track_res is not None
    # At frame 9: px_x = 190.0 => field_x = 19.0 inches
    assert track_res.field_x_inch == pytest.approx(19.0, abs=0.1)
    assert track_res.field_y_inch == pytest.approx(20.0, abs=0.1)
    # Velocity: 60 ips expected
    assert track_res.velocity_x_ips > 40.0, f"Expected vx_ips > 40, got {track_res.velocity_x_ips}"
    assert abs(track_res.velocity_y_ips) < 1.0
    # Heading should be along +x (approx 0 rad)
    assert track_res.heading_rad == pytest.approx(0.0, abs=0.1)


# ============================================================================
# 9. Two-Stage Association (Stage 2 Low-Confidence IoU Matching)
# ============================================================================

@pytest.mark.asyncio
async def test_two_stage_low_confidence_association():
    """Verifies low-confidence detections are matched in Stage 2 via IoU."""
    actor = TrackingActor(target_fps=60.0, track_high_thresh=0.6, track_low_thresh=0.1)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = generate_synthetic_embedding(512, seed=123)

    # Frame 0: High-confidence detection (0.9) initializes track
    box_high = BoundingBox(center_x=100.0, center_y=100.0, width=50.0, height=50.0, confidence=0.9, class_id=0)
    t0 = await actor.update_tracks(0, [box_high], [], make_dummy_homography(0), dummy_frame, embeddings=[emb])
    assert len(t0) == 1
    orig_id = t0[0].track_id

    # Frame 1: Low-confidence detection (0.25) overlapping with predicted box
    box_low = BoundingBox(center_x=102.0, center_y=101.0, width=50.0, height=50.0, confidence=0.25, class_id=0)
    t1 = await actor.update_tracks(1, [box_low], [], make_dummy_homography(1), dummy_frame, embeddings=[emb])

    assert len(t1) == 1, "Track must be maintained via Stage 2 second-chance matching"
    assert t1[0].track_id == orig_id
    assert t1[0].track_state == TrackState.TRACKED


# ============================================================================
# 10. Multi-Robot Trajectory Fixture (6 Robots with Hub Occlusion)
# ============================================================================

@pytest.mark.asyncio
async def test_six_robots_with_hub_occlusion():
    """
    Simulates a full 6-robot FRC match (3 RED, 3 BLUE) where one robot gets occluded
    behind the central Alliance Hub for 30 frames and re-emerges.
    """
    actor = TrackingActor(target_fps=60.0)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    team_numbers = [254, 1678, 118, 148, 1323, 2056]
    alliances = [
        AllianceColor.BLUE, AllianceColor.BLUE, AllianceColor.BLUE,
        AllianceColor.RED, AllianceColor.RED, AllianceColor.RED,
    ]
    embeddings = [generate_synthetic_embedding(512, basis_index=i) for i in range(6)]

    # Initial positions
    x_positions = [100.0, 200.0, 300.0, 700.0, 800.0, 900.0]
    y_positions = [150.0, 150.0, 150.0, 500.0, 500.0, 500.0]

    # Frame 0: Initialize all 6 robots
    dets_f0 = [
        BoundingBox(center_x=x, center_y=y, width=45.0, height=45.0, confidence=0.9, class_id=0)
        for x, y in zip(x_positions, y_positions)
    ]
    ocr_f0 = [
        OCRResult(team_number=tm, alliance=al, confidence=0.9, super_resolved=True)
        for tm, al in zip(team_numbers, alliances)
    ]

    tracks_f0 = await actor.update_tracks(
        frame_index=0,
        detections=dets_f0,
        ocr_results=ocr_f0,
        homography=make_dummy_homography(0),
        frame_bgr=dummy_frame,
        embeddings=embeddings,
    )
    assert len(tracks_f0) == 6
    track_ids = {t.team_number: t.track_id for t in tracks_f0}

    # Robot 0 (team 254) passes behind central Hub (occluded for frames 10..40)
    for f in range(1, 50):
        dets_curr = []
        ocrs_curr = []
        embs_curr = []

        for r_idx in range(6):
            if r_idx == 0 and 10 <= f <= 40:
                # Occluded by Hub
                continue

            # Moving slowly along x
            x = x_positions[r_idx] + f * 2.0
            y = y_positions[r_idx]
            dets_curr.append(BoundingBox(center_x=x, center_y=y, width=45.0, height=45.0, confidence=0.9, class_id=0))
            ocrs_curr.append(ocr_f0[r_idx])
            embs_curr.append(embeddings[r_idx])

        tracks_curr = await actor.update_tracks(
            frame_index=f,
            detections=dets_curr,
            ocr_results=ocrs_curr,
            homography=make_dummy_homography(f),
            frame_bgr=dummy_frame,
            embeddings=embs_curr,
        )

        if 10 <= f <= 40:
            assert len(tracks_curr) == 5, f"Expected 5 tracks during occlusion at frame {f}"
        elif f > 40:
            assert len(tracks_curr) == 6, f"Expected 6 tracks post-occlusion at frame {f}"

    # Verify team 254 re-associated with original track_id
    tr254 = next(t for t in tracks_curr if t.team_number == 254)
    assert tr254.track_id == track_ids[254], "Team 254 must retain original track ID after Hub occlusion"


# ============================================================================
# 11. Alliance Roster Bipartite Matching Tests
# ============================================================================

@pytest.mark.asyncio
async def test_alliance_roster_bipartite_matching_unique_assignment():
    """Verifies that 1-to-1 bipartite matching assigns unique roster teams to all alliance robots."""
    actor = TrackingActor(
        target_fps=60.0,
        red_teams=[1323, 4065, 4414],
        blue_teams=[868, 2046, 2910],
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    boxes = [
        BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=200.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=300.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
    ]
    embs = [
        generate_synthetic_embedding(512, seed=10),
        generate_synthetic_embedding(512, seed=20),
        generate_synthetic_embedding(512, seed=30),
    ]
    ocrs = [
        OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.95, super_resolved=True),
        OCRResult(team_number=1323, alliance=AllianceColor.RED, confidence=0.90, super_resolved=True),
        OCRResult(team_number=4065, alliance=AllianceColor.RED, confidence=0.85, super_resolved=True),
    ]

    tracks = await actor.update_tracks(
        frame_index=0,
        detections=boxes,
        ocr_results=ocrs,
        homography=make_dummy_homography(0),
        frame_bgr=dummy_frame,
        embeddings=embs,
    )

    assert len(tracks) == 3
    assigned_teams = {t.team_number for t in tracks}
    assert assigned_teams == {1323, 4065, 4414}, f"Expected all 3 roster teams assigned, got {assigned_teams}"


@pytest.mark.asyncio
async def test_alliance_roster_bipartite_matching_conflict_resolution():
    """Verifies that Hungarian algorithm resolves competing votes without duplicate team assignments."""
    actor = TrackingActor(
        target_fps=60.0,
        red_teams=[1323, 4065, 4414],
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Robot 1 at x=100, Robot 2 at x=200
    box1 = BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)
    box2 = BoundingBox(center_x=200.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)
    emb1 = generate_synthetic_embedding(512, seed=11)
    emb2 = generate_synthetic_embedding(512, seed=22)

    # Frame 0: Robot 1 sees 4414 (conf 0.9); Robot 2 sees 4414 falsely (conf 0.3)
    tracks0 = await actor.update_tracks(
        0,
        [box1, box2],
        [
            OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.9, super_resolved=True),
            OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.3, super_resolved=True),
        ],
        make_dummy_homography(0),
        dummy_frame,
        embeddings=[emb1, emb2],
    )
    # Robot 1 has stronger claim to 4414; Robot 2 must NOT duplicate 4414
    tr1 = next(t for t in tracks0 if t.pixel_bbox.center_x == pytest.approx(100.0, abs=1.0))
    tr2 = next(t for t in tracks0 if t.pixel_bbox.center_x == pytest.approx(200.0, abs=1.0))
    assert tr1.team_number == 4414
    assert tr2.team_number is None, "Robot 2 must not duplicate 4414 when outvoted"

    # Frame 1: Robot 2 gets true reading for 1323 (conf 0.8)
    tracks1 = await actor.update_tracks(
        1,
        [box1, box2],
        [
            OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.9, super_resolved=True),
            OCRResult(team_number=1323, alliance=AllianceColor.RED, confidence=0.8, super_resolved=True),
        ],
        make_dummy_homography(1),
        dummy_frame,
        embeddings=[emb1, emb2],
    )
    tr1 = next(t for t in tracks1 if t.pixel_bbox.center_x == pytest.approx(100.0, abs=1.0))
    tr2 = next(t for t in tracks1 if t.pixel_bbox.center_x == pytest.approx(200.0, abs=1.0))
    assert tr1.team_number == 4414
    assert tr2.team_number == 1323


@pytest.mark.asyncio
async def test_kalman_track_coasting_persistence():
    """Verifies that with max_coast_frames=15, a robot missing for 5 frames continues to coast with Kalman velocity and retains track ID."""
    actor = TrackingActor(target_fps=60.0, max_coast_frames=15)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = generate_synthetic_embedding(512, seed=42)

    # Frame 0: Initialize robot moving at vx = 10.0
    det0 = [BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)]
    tracks0 = await actor.update_tracks(
        frame_index=0,
        detections=det0,
        ocr_results=[OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.9, super_resolved=True)],
        homography=make_dummy_homography(0),
        frame_bgr=dummy_frame,
        embeddings=[emb],
    )
    assert len(tracks0) == 1
    init_id = tracks0[0].track_id
    assert tracks0[0].team_number == 4414

    # Frame 1: Update robot moving right
    det1 = [BoundingBox(center_x=105.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)]
    tracks1 = await actor.update_tracks(
        frame_index=1,
        detections=det1,
        ocr_results=[],
        homography=make_dummy_homography(1),
        frame_bgr=dummy_frame,
        embeddings=[emb],
    )
    assert len(tracks1) == 1
    assert tracks1[0].track_id == init_id

    # Frames 2..6: Robot is occluded by yellow balls (zero detections for 5 frames)
    tracks_coast = []
    for f in range(2, 7):
        tracks_coast = await actor.update_tracks(
            frame_index=f,
            detections=[],
            ocr_results=[],
            homography=make_dummy_homography(f),
            frame_bgr=dummy_frame,
            embeddings=[],
        )
        assert len(tracks_coast) == 1, f"Track must coast during occlusion at frame {f}"
        assert tracks_coast[0].track_id == init_id
        assert tracks_coast[0].team_number == 4414
        assert tracks_coast[0].pixel_bbox.center_x >= 105.0

    # Frame 7: Robot re-emerges near its predicted position
    det7 = [BoundingBox(center_x=tracks_coast[0].pixel_bbox.center_x, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)]
    tracks7 = await actor.update_tracks(
        frame_index=7,
        detections=det7,
        ocr_results=[],
        homography=make_dummy_homography(7),
        frame_bgr=dummy_frame,
        embeddings=[emb],
    )
    assert len(tracks7) == 1
    assert tracks7[0].track_id == init_id
    assert tracks7[0].team_number == 4414


@pytest.mark.asyncio
async def test_alliance_locking_immutability():
    """Verifies that a track assigned to Red alliance cannot be flipped to Blue by noisy OCR."""
    actor = TrackingActor(target_fps=60.0, red_teams=[4414], blue_teams=[2910])
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = generate_synthetic_embedding(512, seed=4414)

    # Frame 0: Initialize with high confidence Red reading
    det = [BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)]
    ocr_red = [OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.95, super_resolved=True)]
    tracks = await actor.update_tracks(0, det, ocr_red, make_dummy_homography(0), dummy_frame, embeddings=[emb])
    assert tracks[0].team_number == 4414
    assert tracks[0].alliance == AllianceColor.RED

    # Frame 1: Push match with opponent causes noisy Blue OCR reading
    ocr_blue_noise = [OCRResult(team_number=2910, alliance=AllianceColor.BLUE, confidence=0.3, super_resolved=True)]
    tracks1 = await actor.update_tracks(1, det, ocr_blue_noise, make_dummy_homography(1), dummy_frame, embeddings=[emb])
    assert tracks1[0].team_number == 4414
    assert tracks1[0].alliance == AllianceColor.RED


@pytest.mark.asyncio
async def test_deductive_elimination_inference():
    """Verifies that when 2 teams on an alliance are confirmed, a 3rd robot with unreadable text is deduced by elimination."""
    actor = TrackingActor(
        target_fps=60.0,
        red_teams=[1323, 4065, 4414],
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb1 = generate_synthetic_embedding(512, seed=1)
    emb2 = generate_synthetic_embedding(512, seed=2)
    emb3 = generate_synthetic_embedding(512, seed=3)

    b1 = BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)
    b2 = BoundingBox(center_x=200.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)
    b3 = BoundingBox(center_x=300.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)

    # Frame 0: Robot 1 is 4414 (conf 0.95), Robot 2 is 4065 (conf 0.90), Robot 3 has occluded text (no team number, only RED alliance)
    tracks0 = await actor.update_tracks(
        0,
        [b1, b2, b3],
        [
            OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.95, super_resolved=True),
            OCRResult(team_number=4065, alliance=AllianceColor.RED, confidence=0.90, super_resolved=True),
            OCRResult(team_number=None, alliance=AllianceColor.RED, confidence=0.70, super_resolved=True),
        ],
        make_dummy_homography(0),
        dummy_frame,
        embeddings=[emb1, emb2, emb3],
    )

    t1 = next(t for t in tracks0 if t.pixel_bbox.center_x == pytest.approx(100.0, abs=2.0))
    t2 = next(t for t in tracks0 if t.pixel_bbox.center_x == pytest.approx(200.0, abs=2.0))
    t3 = next(t for t in tracks0 if t.pixel_bbox.center_x == pytest.approx(300.0, abs=2.0))

    assert t1.team_number == 4414
    assert t2.team_number == 4065
    # Robot 3 must be deduced as 1323 by elimination!
    assert t3.team_number == 1323
    assert t3.alliance == AllianceColor.RED

    # Frame 1: Text continues to be completely unreadable (None) for all robots
    tracks1 = await actor.update_tracks(
        1,
        [b1, b2, b3],
        [],
        make_dummy_homography(1),
        dummy_frame,
        embeddings=[emb1, emb2, emb3],
    )
    t1_1 = next(t for t in tracks1 if t.pixel_bbox.center_x == pytest.approx(100.0, abs=2.0))
    t2_1 = next(t for t in tracks1 if t.pixel_bbox.center_x == pytest.approx(200.0, abs=2.0))
    t3_1 = next(t for t in tracks1 if t.pixel_bbox.center_x == pytest.approx(300.0, abs=2.0))
    assert t1_1.team_number == 4414
    assert t2_1.team_number == 4065
    assert t3_1.team_number == 1323
