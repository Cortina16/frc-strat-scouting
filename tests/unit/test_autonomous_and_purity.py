"""
Unit tests for Autonomous Rules, Physical Overlap Loosening, Strict Alliance Purity,
and Mandatory Team Attribution.

Verifies:
1. Autonomous Centerline Crossing Gating:
   Under FRC rules, robots cannot cross 100% over the center line (X = 325.6") during AUTO.
   Red robot at X > 335.0" or Blue robot at X < 315.0" is rejected/clamped during AUTO.
2. Physical Overlap Deduplication & Adaptive Loosening:
   Two robots cannot occupy the same physical space. Duplicate boxes with IoU >= 0.40 are
   suppressed, and if fewer than 3 robots are found, thresholds are loosened to find missing robots.
3. Strict Alliance Purity:
   A Red robot detection CANNOT have a Blue team number, and a Blue robot detection CANNOT
   have a Red team number.
4. Mandatory Team Attribution:
   When an alliance roster is provided, every active track on that alliance is guaranteed
   a unique, non-None team number from that alliance roster.
5. HSV (V - S) Differential OCR & Roster Template Correlation:
   Differential text mapping zeroes out red/blue fabric and achieves high correlation.
"""

from __future__ import annotations

import asyncio
import numpy as np
import pytest

from src.common.types import (
    AllianceColor,
    BoundingBox,
    HomographyEstimate,
    OCRResult,
    RawDecodedFrame,
    FrameMetadata,
)
from src.tracking.bot_sort import STrack, BoTSORT, TrackingActor
from src.vision.detector_trt import RobotBumperVerificationGate
from src.vision.ocr_enhancer import (
    direct_roster_template_match,
    extract_digit_candidates,
    recognize_team_number,
    OCREnhancerActor,
)


def make_dummy_homography(frame_index: int = 0, pts_ms: int = 0) -> HomographyEstimate:
    """Creates an identity-scaled homography mapping pixel (x, y) directly to field inches."""
    return HomographyEstimate(
        frame_index=frame_index,
        timestamp_ms=pts_ms,
        h_matrix=np.eye(3, dtype=np.float64),
        dlt_condition=1.0,
        lm_rmse=0.15,
        lm_iterations=10,
        converged=True,
        camera_cut=False,
        affine_cmc=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        focal_length=1250.0,
        camera_center=np.array([325.6, 158.85, 200.0], dtype=np.float64),
    )


# ============================================================================
# 1. Autonomous Centerline Crossing Gating Tests
# ============================================================================

def test_autonomous_centerline_crossing_gate_in_vision():
    """Verifies that candidates violating the autonomous centerline boundary are rejected."""
    gate = RobotBumperVerificationGate()

    # Synthetic frame (H=600, W=800)
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # 1. Red bumper box placed on Red side (X = 200)
    # Ground contact at (200, 400) -> X = 200.0 in field coordinates
    box_red_valid = BoundingBox(center_x=200.0, center_y=380.0, width=50.0, height=40.0, confidence=0.85, class_id=0)
    # Paint Red bumper (B=20, G=30, R=200)
    frame[380:400, 175:225] = [20, 30, 200]
    frame[360:380, 175:225] = [60, 60, 60]  # superstructure

    # 2. Red bumper box placed on Blue side (X = 400) during AUTO -> ILLEGAL!
    box_red_illegal = BoundingBox(center_x=400.0, center_y=380.0, width=50.0, height=40.0, confidence=0.85, class_id=0)
    frame[380:400, 375:425] = [20, 30, 200]
    frame[360:380, 375:425] = [60, 60, 60]

    # Homography: 1 pixel = 1 inch
    h_mat = np.eye(3, dtype=np.float64)

    # During Autonomous:
    verified_auto = gate.verify_candidates(
        frame_bgr=frame,
        candidates=[box_red_valid, box_red_illegal],
        homography=h_mat,
        is_autonomous=True,
    )
    # Only the Red robot on the Red side (X=200) must survive!
    assert len(verified_auto) == 1
    assert verified_auto[0].center_x == pytest.approx(200.0)

    # During Teleop:
    verified_teleop = gate.verify_candidates(
        frame_bgr=frame,
        candidates=[box_red_valid, box_red_illegal],
        homography=h_mat,
        is_autonomous=False,
    )
    # Both Red robots are permitted across the field during teleop!
    assert len(verified_teleop) == 2


def test_autonomous_centerline_clamping_in_strack():
    """Verifies that STrack clamps field coordinates to the centerline boundary during AUTO."""
    box = BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)
    track_red = STrack(track_id=1, bbox=box)
    track_red.alliance = AllianceColor.RED
    track_red.is_alliance_locked = True

    # Homography maps contact to X = 450.0 (Blue side)
    h_mat = np.array([
        [1.0, 0.0, -350.0],  # xc=100 -> new_x = 100 - (-350) = 450
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # In AUTO: Red robot cannot be at X=450, clamped to 325.6
    track_red.update_field_geometry(h_matrix=h_mat, is_autonomous=True)
    assert track_red.field_x_inch <= 335.0
    assert track_red.field_x_inch == pytest.approx(325.6)

    # In Teleop: Red robot can be at X=450
    track_red.update_field_geometry(h_matrix=h_mat, is_autonomous=False)
    assert track_red.field_x_inch == pytest.approx(450.0)


# ============================================================================
# 2. Physical Overlap Deduplication & Adaptive Loosening Tests
# ============================================================================

def test_physical_overlap_deduplication_and_adaptive_loosening():
    """Verifies that overlapping duplicate boxes are suppressed and unverified pool is loosened."""
    gate = RobotBumperVerificationGate()

    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Box 1: Primary Red robot detection (conf=0.90)
    box1 = BoundingBox(center_x=200.0, center_y=300.0, width=60.0, height=60.0, confidence=0.90, class_id=0)
    frame[300:330, 170:230] = [20, 30, 200]  # Red bumper
    frame[270:300, 170:230] = [50, 50, 50]

    # Box 2: Duplicate detection overlapping 95% with Box 1 (conf=0.85)
    box2 = BoundingBox(center_x=202.0, center_y=301.0, width=60.0, height=60.0, confidence=0.85, class_id=0)

    # Box 3: Distant Red robot with low confidence (conf=0.18, low bumper score)
    box3 = BoundingBox(center_x=500.0, center_y=300.0, width=50.0, height=50.0, confidence=0.18, class_id=0)
    frame[305:325, 475:525] = [20, 30, 170]  # Faint Red bumper
    frame[285:305, 475:525] = [40, 40, 40]

    verified = gate.verify_candidates(
        frame_bgr=frame,
        candidates=[box1, box2, box3],
        is_autonomous=False,
    )

    # Box 2 must be suppressed as a physical duplicate of Box 1.
    # Adaptive loosening must recover Box 3 so 2 distinct robots are output!
    assert len(verified) == 2
    centers = [round(b.center_x) for b in verified]
    assert 200 in centers
    assert 500 in centers


# ============================================================================
# 3. Strict Alliance Purity Tests
# ============================================================================

@pytest.mark.asyncio
async def test_strict_alliance_purity_ocr_rejection():
    """Verifies that an OCR result attributing a Blue team to a Red robot is strictly rejected."""
    actor = TrackingActor(
        target_fps=60.0,
        red_teams=[1323, 4065, 4414],
        blue_teams=[868, 2046, 2910],
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    emb = np.random.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    # Frame 0: Robot detected as RED with team 4414
    box = BoundingBox(center_x=100.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0)
    ocr_red = OCRResult(team_number=4414, alliance=AllianceColor.RED, confidence=0.95, super_resolved=True)
    tracks0 = await actor.update_tracks(0, [box], [ocr_red], make_dummy_homography(0), dummy_frame, embeddings=[emb])
    assert tracks0[0].team_number == 4414
    assert tracks0[0].alliance == AllianceColor.RED

    # Frame 1: Malformed / noisy OCR proposes Blue team 2910 on this Red robot
    ocr_polluted = OCRResult(team_number=2910, alliance=AllianceColor.BLUE, confidence=0.99, super_resolved=True)
    tracks1 = await actor.update_tracks(1, [box], [ocr_polluted], make_dummy_homography(1), dummy_frame, embeddings=[emb])

    # Strict Alliance Purity: Blue team 2910 must NOT pollute Red robot!
    assert tracks1[0].team_number == 4414
    assert tracks1[0].alliance == AllianceColor.RED
    assert 2910 not in tracks1[0].team_number if isinstance(tracks1[0].team_number, list) else tracks1[0].team_number != 2910


# ============================================================================
# 4. Mandatory Team Attribution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_mandatory_team_attribution_from_roster():
    """Verifies that every active robot track is guaranteed a unique team number from the roster."""
    red_roster = [1323, 4065, 4414]
    blue_roster = [868, 2046, 2910]

    actor = TrackingActor(
        target_fps=60.0,
        red_teams=red_roster,
        blue_teams=blue_roster,
        mandatory_attribution=True,
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # 3 Red robot detections and 3 Blue robot detections without any OCR numbers
    boxes = [
        # Red alliance side (x < 325.6)
        BoundingBox(center_x=50.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=120.0, center_y=150.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=200.0, center_y=200.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        # Blue alliance side (x > 325.6)
        BoundingBox(center_x=450.0, center_y=100.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=520.0, center_y=150.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
        BoundingBox(center_x=600.0, center_y=200.0, width=40.0, height=40.0, confidence=0.9, class_id=0),
    ]
    # No OCR readings available (bumpers dirty or occluded)
    ocr_empty = [
        OCRResult(team_number=None, alliance=AllianceColor.RED, confidence=0.5),
        OCRResult(team_number=None, alliance=AllianceColor.RED, confidence=0.5),
        OCRResult(team_number=None, alliance=AllianceColor.RED, confidence=0.5),
        OCRResult(team_number=None, alliance=AllianceColor.BLUE, confidence=0.5),
        OCRResult(team_number=None, alliance=AllianceColor.BLUE, confidence=0.5),
        OCRResult(team_number=None, alliance=AllianceColor.BLUE, confidence=0.5),
    ]

    tracks = await actor.update_tracks(0, boxes, ocr_empty, make_dummy_homography(0), dummy_frame)
    assert len(tracks) == 6

    red_tracks = [t for t in tracks if t.alliance == AllianceColor.RED]
    blue_tracks = [t for t in tracks if t.alliance == AllianceColor.BLUE]

    assert len(red_tracks) == 3
    assert len(blue_tracks) == 3

    # Mandatory attribution assertions:
    # 1. 0 tracks have team_number is None
    for t in tracks:
        assert t.team_number is not None, f"Track {t.track_id} must have attributed team number"

    # 2. All Red teams are uniquely attributed
    assigned_red = {t.team_number for t in red_tracks}
    assert assigned_red == set(red_roster), f"Expected {set(red_roster)}, got {assigned_red}"

    # 3. All Blue teams are uniquely attributed
    assigned_blue = {t.team_number for t in blue_tracks}
    assert assigned_blue == set(blue_roster), f"Expected {set(blue_roster)}, got {assigned_blue}"


# ============================================================================
# 5. HSV (V - S) Differential OCR & Direct Template Matching Tests
# ============================================================================

def test_hsv_differential_ocr_and_template_correlation():
    """Verifies that HSV (V - S) differential mapping suppresses fabric and matches roster templates."""
    # Synthesize bumper crop with red fabric and white text '4414'
    crop = np.full((40, 100, 3), (20, 30, 200), dtype=np.uint8)
    import cv2
    cv2.putText(crop, "4414", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Direct roster template match against candidate teams
    best_team, score = direct_roster_template_match(crop, [1323, 4065, 4414], AllianceColor.RED)
    assert best_team == 4414
    assert score >= 0.60
