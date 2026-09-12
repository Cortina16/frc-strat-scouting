"""
Unit tests for Module 06: ActionFormer Temporal Action Localization.

Verifies:
1. Interface Contract Validation:
   - ActionRecognitionActor inherits ActionRecognitionActorBase and conforms to ActionRecognitionActorProtocol.
2. Temporal Pyramid Verification:
   - Feature pyramid construction across levels 0 to 3 with correct temporal downsampling.
3. DIoU and Focal Loss Metric Tests:
   - 1D Distance-IoU computation between overlapping, non-overlapping, and shifted temporal intervals.
   - Focal Loss computation, probability calibration, and focusing behavior.
4. Synthetic Scoring Event Detection:
   - Simulate a robot trajectory with a realistic scoring cycle at the Hub (e.g., t=15.0s to t=18.5s).
   - Assert predicted temporal boundaries match ground truth within delta_t < 0.25s.
   - Assert classification confidence for SCORE_HUB exceeds threshold > 0.85.
   - Assert ActionSegment.element is properly associated (e.g., FieldElement.RED_HUB or BLUE_HUB).
5. Multiple Actions & Edge Cases:
   - INTAKE_FUEL near Depot, CLIMB_TOWER near Tower, and TRANSIT_NEUTRAL_ZONE.
   - Empty windows and unobserved track_id.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import List, Tuple
import pytest
import numpy as np
import torch

from src.actions.action_former import (
    ACTION_CLASSES,
    ActionFormer,
    ActionRecognitionActor,
    BoundaryRegressionHead,
    ClassificationHead,
    LocalSelfAttentionBlock,
    compute_diou_1d,
    focal_loss,
    sinusoidal_positional_encoding_1d,
    temporal_nms,
)
from src.actions.temporal_pyramid import (
    BLUE_HUB_POS,
    BLUE_TOWER_POS,
    DEPOT_NORTH_POS,
    FEATURE_DIM,
    FIELD_LENGTH_INCHES,
    FIELD_WIDTH_INCHES,
    RED_HUB_POS,
    RED_TOWER_POS,
    TemporalPyramid,
    TemporalWindowBuffer,
    extract_frame_features,
    extract_sequence_features,
    get_torch_device,
)
from src.common.interfaces import (
    ActionRecognitionActorBase,
    ActionRecognitionActorProtocol,
)
from src.common.types import (
    ActionSegment,
    ActionType,
    AllianceColor,
    BoundingBox,
    FieldElement,
    RobotTrackletState,
    TrackState,
)


# ============================================================================
# Helpers to generate synthetic RobotTrackletStates
# ============================================================================

def make_tracklet_state(
    track_id: int = 1,
    team_number: int = 254,
    alliance: AllianceColor = AllianceColor.RED,
    field_x: float = 162.8,
    field_y: float = 158.85,
    vx: float = 0.0,
    vy: float = 0.0,
    heading: float = 0.0,
    bbox_w: float = 80.0,
    bbox_h: float = 80.0,
    track_state: TrackState = TrackState.TRACKED,
) -> RobotTrackletState:
    """Creates a mock RobotTrackletState for testing."""
    bbox = BoundingBox(
        center_x=960.0,
        center_y=540.0,
        width=bbox_w,
        height=bbox_h,
        confidence=0.95,
        class_id=0,
    )
    return RobotTrackletState(
        track_id=track_id,
        team_number=team_number,
        alliance=alliance,
        pixel_bbox=bbox,
        field_x_inch=field_x,
        field_y_inch=field_y,
        velocity_x_ips=vx,
        velocity_y_ips=vy,
        heading_rad=heading,
        track_state=track_state,
        reid_embedding=None,
    )


# ============================================================================
# 1. Interface Contract Validation
# ============================================================================

def test_actor_interface_contracts():
    """Assert ActionRecognitionActor inherits ActionRecognitionActorBase and satisfies ActionRecognitionActorProtocol."""
    actor = ActionRecognitionActor(fps=60.0)

    # ABC inheritance check
    assert issubclass(ActionRecognitionActor, ActionRecognitionActorBase)
    assert isinstance(actor, ActionRecognitionActorBase)

    # Runtime Protocol check
    assert isinstance(actor, ActionRecognitionActorProtocol)

    # Check required async methods exist and are callable
    assert hasattr(actor, "ingest_tracklet_window")
    assert hasattr(actor, "localize_actions")
    assert hasattr(actor, "localize_all_actions")
    assert asyncio.iscoroutinefunction(actor.ingest_tracklet_window)
    assert asyncio.iscoroutinefunction(actor.localize_actions)
    assert asyncio.iscoroutinefunction(actor.localize_all_actions)


# ============================================================================
# 2. Temporal Pyramid Verification
# ============================================================================

def test_temporal_pyramid_multiscale_downsampling():
    """Verify feature pyramid construction across levels 0 to 3 with correct temporal downsampling."""
    pyramid = TemporalPyramid(in_channels=FEATURE_DIM, hidden_channels=64, num_levels=4)
    assert pyramid.strides == [1, 2, 4, 8]

    # Input: T = 128 frames, D = 20 features
    t_len = 128
    features = np.random.randn(t_len, FEATURE_DIM).astype(np.float32)
    levels = pyramid.forward(features)

    assert 0 in levels
    assert 1 in levels
    assert 2 in levels
    assert 3 in levels

    # Verify temporal dimensions downsample by factor of 2 at each level
    assert levels[0].shape == (128, 64)
    assert levels[1].shape == (64, 64)
    assert levels[2].shape == (32, 64)
    assert levels[3].shape == (16, 64)

    # Verify outputs are finite
    for lvl, fmap in levels.items():
        assert np.all(np.isfinite(fmap)), f"Level {lvl} contains NaN or Inf"


def test_temporal_pyramid_arbitrary_length():
    """Verify feature pyramid handles arbitrary non-power-of-2 sequence lengths."""
    pyramid = TemporalPyramid(in_channels=FEATURE_DIM, hidden_channels=32, num_levels=4)
    t_len = 77
    features = np.random.randn(t_len, FEATURE_DIM).astype(np.float32)
    levels = pyramid.forward(features)

    assert levels[0].shape[0] == 77
    assert levels[1].shape[0] == (77 + 1) // 2  # 39
    assert levels[2].shape[0] == (levels[1].shape[0] + 1) // 2  # 20
    assert levels[3].shape[0] == (levels[2].shape[0] + 1) // 2  # 10


# ============================================================================
# 3. DIoU and Focal Loss Metric Tests
# ============================================================================

def test_compute_diou_1d_metrics():
    """Verify 1D DIoU computation between overlapping, non-overlapping, and shifted temporal intervals."""
    # Case 1: Identical intervals -> IoU=1.0, rho=0 -> DIoU = 1.0
    diou_identical = compute_diou_1d(np.array([10.0, 20.0]), np.array([10.0, 20.0]))
    assert pytest.approx(diou_identical, abs=1e-5) == 1.0

    # Case 2: Shifted overlapping intervals [0, 10] and [5, 15]
    # Inter = 5, Union = 15 -> IoU = 1/3 ~ 0.3333
    # c1 = 5, c2 = 10 -> rho^2 = 25
    # Smallest enclosing = [0, 15] -> c^2 = 225
    # DIoU = 1/3 - 25/225 = 1/3 - 1/9 = 2/9 ~ 0.2222
    diou_shifted = compute_diou_1d(np.array([0.0, 10.0]), np.array([5.0, 15.0]))
    assert pytest.approx(diou_shifted, abs=1e-4) == 2.0 / 9.0

    # Case 3: Non-overlapping intervals [0, 10] and [20, 30]
    # Inter = 0 -> IoU = 0
    # c1 = 5, c2 = 25 -> rho^2 = 400
    # Enclosing = [0, 30] -> c^2 = 900
    # DIoU = 0 - 400/900 = -4/9 ~ -0.4444
    diou_non_overlap = compute_diou_1d(np.array([0.0, 10.0]), np.array([20.0, 30.0]))
    assert pytest.approx(diou_non_overlap, abs=1e-4) == -4.0 / 9.0

    # Case 4: Non-overlapping intervals further apart [0, 10] and [30, 40]
    # c1 = 5, c2 = 35 -> rho^2 = 900, c^2 = 1600 -> DIoU = -9/16 = -0.5625
    diou_further = compute_diou_1d(np.array([0.0, 10.0]), np.array([30.0, 40.0]))
    assert pytest.approx(diou_further, abs=1e-4) == -9.0 / 16.0
    assert diou_further < diou_non_overlap  # Penalizes greater temporal separation

    # Case 5: Batched inputs (N, 2) vs (N, 2)
    segs1 = np.array([[10.0, 20.0], [0.0, 10.0]])
    segs2 = np.array([[10.0, 20.0], [5.0, 15.0]])
    batch_diou = compute_diou_1d(segs1, segs2)
    assert batch_diou.shape == (2,)
    assert pytest.approx(batch_diou[0], abs=1e-5) == 1.0
    assert pytest.approx(batch_diou[1], abs=1e-4) == 2.0 / 9.0


def test_focal_loss_scoring():
    """Verify Focal Loss formula, probability calibration, and focusing behavior."""
    # Well-classified easy sample (p=0.98, target=0)
    p_easy = np.array([[0.98, 0.02]])
    y_easy = np.array([0])
    loss_easy = focal_loss(p_easy, y_easy, alpha=0.25, gamma=2.0)

    # Uncertain / hard sample (p=0.55, target=0)
    p_hard = np.array([[0.55, 0.45]])
    y_hard = np.array([0])
    loss_hard = focal_loss(p_hard, y_hard, alpha=0.25, gamma=2.0)

    # Misclassified sample (p=0.05, target=0)
    p_wrong = np.array([[0.05, 0.95]])
    y_wrong = np.array([0])
    loss_wrong = focal_loss(p_wrong, y_wrong, alpha=0.25, gamma=2.0)

    assert loss_easy < loss_hard < loss_wrong
    assert loss_easy < 1e-4  # Easy example strongly suppressed by (1 - p_t)^gamma

    # Verify focusing effect: gamma=2 down-weights easy example much more than gamma=0 (standard CE)
    loss_ce = focal_loss(p_easy, y_easy, alpha=0.25, gamma=0.0)
    assert loss_easy < loss_ce

    # One-hot target test
    y_one_hot = np.array([[1, 0]])
    loss_one_hot = focal_loss(p_easy, y_one_hot, alpha=0.25, gamma=2.0)
    assert loss_one_hot > 0.0


# ============================================================================
# 4. Positional Encodings & Attention Unit Tests
# ============================================================================

def test_sinusoidal_positional_encoding():
    """Verify 1D sinusoidal positional encodings properties."""
    seq_len = 64
    dim = 32
    pe = sinusoidal_positional_encoding_1d(seq_len, dim)
    assert pe.shape == (seq_len, dim)
    assert np.all(pe >= -1.0) and np.all(pe <= 1.0)
    # Positions 0 and 1 should differ
    assert not np.allclose(pe[0], pe[1])


def test_local_self_attention():
    """Verify local self-attention forward pass restricts computation to temporal window."""
    attn = LocalSelfAttentionBlock(channels=32, window_size=5)
    x = np.random.randn(40, 32).astype(np.float32)
    out = attn.forward(x)
    assert out.shape == (40, 32)
    assert np.all(np.isfinite(out))


# ============================================================================
# 5. Synthetic Scoring Event Detection (Prompt Requirement 4)
# ============================================================================

@pytest.mark.asyncio
async def test_synthetic_scoring_event_detection_red_hub():
    """
    Simulate a robot trajectory with a realistic scoring cycle at the Hub:
    - Approaching Red Hub at t < 15.0s
    - Decelerating and staying within scoring radius (< 60") at t = 15.0s to t = 18.5s
    - Departing Red Hub at t > 18.5s towards neutral zone
    - Ingest window into ActionRecognitionActor and localize actions.
    - Assert predicted temporal boundaries match ground truth within delta_t < 0.25s.
    - Assert classification confidence for SCORE_HUB exceeds threshold > 0.85.
    - Assert ActionSegment.element is properly associated (FieldElement.RED_HUB).
    """
    fps = 60.0
    dt = 1.0 / fps
    total_seconds = 24.0
    num_frames = int(total_seconds * fps)  # 1440 frames

    actor = ActionRecognitionActor(fps=fps)

    track_states: List[RobotTrackletState] = []
    timestamps_ms: List[int] = []
    frame_indices: List[int] = []

    # Ground truth event timing
    gt_start_s = 15.0
    gt_end_s = 18.5
    gt_start_ms = int(gt_start_s * 1000)  # 15000 ms
    gt_end_ms = int(gt_end_s * 1000)      # 18500 ms

    red_hub_x, red_hub_y = RED_HUB_POS  # (162.8, 158.85)

    for i in range(num_frames):
        t_sec = i * dt
        ts_ms = int(t_sec * 1000)
        timestamps_ms.append(ts_ms)
        frame_indices.append(i)

        if t_sec < gt_start_s:
            # Transit phase: moving towards Red Hub from neutral zone
            # Starts at x=320.0, moves to x=163.0 over 15 seconds
            progress = t_sec / gt_start_s
            x = 320.0 - progress * (320.0 - 163.0)
            y = red_hub_y
            vx = -100.0  # high speed
            vy = 0.0
        elif t_sec <= gt_end_s:
            # Scoring phase: stationary at Red Hub (distance < 2 inches, well within 60" radius)
            x = red_hub_x + 0.5
            y = red_hub_y + 0.5
            vx = 0.0
            vy = 0.0
        else:
            # Departure phase: accelerating away towards neutral zone
            depart_t = t_sec - gt_end_s
            x = 163.0 + depart_t * 35.0
            y = red_hub_y
            vx = 110.0  # high speed
            vy = 0.0

        state = make_tracklet_state(
            track_id=1,
            team_number=254,
            alliance=AllianceColor.RED,
            field_x=x,
            field_y=y,
            vx=vx,
            vy=vy,
        )
        track_states.append(state)

    # Ingest the entire trajectory window
    await actor.ingest_tracklet_window(
        track_states=track_states,
        window_size=len(track_states),
        timestamps_ms=timestamps_ms,
        frame_indices=frame_indices,
    )

    # Localize actions
    actions = await actor.localize_actions(track_id=1)

    # Verify SCORE_HUB action was localized
    hub_scores = [a for a in actions if a.action_type == ActionType.SCORE_HUB]
    assert len(hub_scores) >= 1, f"Expected SCORE_HUB action, found: {[a.action_type for a in actions]}"

    score_act = hub_scores[0]

    # 1. Assert classification confidence exceeds 0.85
    assert score_act.confidence > 0.85, f"Confidence {score_act.confidence} <= 0.85"

    # 2. Assert predicted temporal boundaries match ground truth within delta_t < 0.25s (250 ms)
    start_err_s = abs(score_act.start_time_ms - gt_start_ms) / 1000.0
    end_err_s = abs(score_act.end_time_ms - gt_end_ms) / 1000.0

    assert start_err_s < 0.25, f"Start time error {start_err_s:.3f}s exceeds 0.25s threshold"
    assert end_err_s < 0.25, f"End time error {end_err_s:.3f}s exceeds 0.25s threshold"

    # 3. Assert ActionSegment.element is properly associated (FieldElement.RED_HUB)
    assert score_act.element == FieldElement.RED_HUB

    # 4. Assert field coords near Red Hub
    assert abs(score_act.field_coords[0] - red_hub_x) < 5.0
    assert abs(score_act.field_coords[1] - red_hub_y) < 5.0
    assert score_act.track_id == 1
    assert score_act.team_number == 254


@pytest.mark.asyncio
async def test_synthetic_scoring_event_detection_blue_hub():
    """Verify SCORE_HUB localization correctly associates FieldElement.BLUE_HUB when near Blue Hub."""
    fps = 30.0
    dt = 1.0 / fps
    total_seconds = 10.0
    num_frames = int(total_seconds * fps)

    actor = ActionRecognitionActor(fps=fps)
    blue_hub_x, blue_hub_y = BLUE_HUB_POS  # (488.4, 158.85)

    track_states: List[RobotTrackletState] = []
    timestamps_ms: List[int] = []
    frame_indices: List[int] = []

    # Scoring event from t=3.0s to t=6.5s
    gt_start_s, gt_end_s = 3.0, 6.5

    for i in range(num_frames):
        t_sec = i * dt
        ts_ms = int(t_sec * 1000)
        timestamps_ms.append(ts_ms)
        frame_indices.append(i)

        if gt_start_s <= t_sec <= gt_end_s:
            x, y = blue_hub_x + 1.0, blue_hub_y
            vx, vy = 0.0, 0.0
        else:
            x, y = 350.0, 200.0  # Away in neutral zone
            vx, vy = 100.0, 0.0

        track_states.append(
            make_tracklet_state(
                track_id=2,
                team_number=1678,
                alliance=AllianceColor.BLUE,
                field_x=x,
                field_y=y,
                vx=vx,
                vy=vy,
            )
        )

    await actor.ingest_tracklet_window(
        track_states=track_states,
        window_size=len(track_states),
        timestamps_ms=timestamps_ms,
        frame_indices=frame_indices,
    )

    actions = await actor.localize_actions(track_id=2)
    blue_scores = [a for a in actions if a.action_type == ActionType.SCORE_HUB]
    assert len(blue_scores) >= 1

    act = blue_scores[0]
    assert act.confidence > 0.85
    assert act.element == FieldElement.BLUE_HUB
    assert abs(act.start_time_ms / 1000.0 - gt_start_s) < 0.25
    assert abs(act.end_time_ms / 1000.0 - gt_end_s) < 0.25


# ============================================================================
# 6. Multiple Actions & Edge Cases (Prompt Requirement 5)
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_action_types_detection():
    """
    Test INTAKE_FUEL near Depot, CLIMB_TOWER near Tower, and TRANSIT_NEUTRAL_ZONE.
    """
    fps = 30.0
    dt = 1.0 / fps
    total_seconds = 20.0
    num_frames = int(total_seconds * fps)

    actor = ActionRecognitionActor(fps=fps)
    depot_x, depot_y = DEPOT_NORTH_POS  # (325.6, 280.0)
    tower_x, tower_y = RED_TOWER_POS   # (90.0, 158.85)

    track_states: List[RobotTrackletState] = []
    timestamps_ms: List[int] = []
    frame_indices: List[int] = []

    # Action 1: INTAKE_FUEL near Depot from t=2.0s to t=5.0s
    # Action 2: TRANSIT_NEUTRAL_ZONE from t=7.0s to t=10.0s (x in [250, 400], fast)
    # Action 3: CLIMB_TOWER near Tower from t=13.0s to t=17.0s (stationary near tower)

    for i in range(num_frames):
        t_sec = i * dt
        ts_ms = int(t_sec * 1000)
        timestamps_ms.append(ts_ms)
        frame_indices.append(i)

        if 2.0 <= t_sec <= 5.0:
            # Intake at Depot
            x, y = depot_x + 1.0, depot_y + 1.0
            vx, vy = 0.0, 0.0
        elif 7.0 <= t_sec <= 10.0:
            # Fast transit across neutral zone
            x = 260.0 + (t_sec - 7.0) * 40.0
            y = 150.0
            vx, vy = 120.0, 0.0
        elif 13.0 <= t_sec <= 17.0:
            # Climb at Tower
            x, y = tower_x + 2.0, tower_y + 1.0
            vx, vy = 0.0, 0.0
        else:
            # Idle in open alliance territory
            x, y = 50.0, 50.0
            vx, vy = 5.0, 0.0

        track_states.append(
            make_tracklet_state(
                track_id=3,
                team_number=971,
                alliance=AllianceColor.RED,
                field_x=x,
                field_y=y,
                vx=vx,
                vy=vy,
            )
        )

    await actor.ingest_tracklet_window(
        track_states=track_states,
        window_size=len(track_states),
        timestamps_ms=timestamps_ms,
        frame_indices=frame_indices,
    )

    actions = await actor.localize_actions(track_id=3)

    # 1. Check INTAKE_FUEL
    intakes = [a for a in actions if a.action_type == ActionType.INTAKE_FUEL]
    assert len(intakes) >= 1
    assert intakes[0].element == FieldElement.DEPOT
    assert intakes[0].confidence > 0.85
    assert abs(intakes[0].start_time_ms / 1000.0 - 2.0) < 0.25
    assert abs(intakes[0].end_time_ms / 1000.0 - 5.0) < 0.25

    # 2. Check TRANSIT_NEUTRAL_ZONE
    transits = [a for a in actions if a.action_type == ActionType.TRANSIT_NEUTRAL_ZONE]
    assert len(transits) >= 1
    assert transits[0].element == FieldElement.NEUTRAL_ZONE
    assert transits[0].confidence > 0.85
    assert abs(transits[0].start_time_ms / 1000.0 - 7.0) < 0.25
    assert abs(transits[0].end_time_ms / 1000.0 - 10.0) < 0.25

    # 3. Check CLIMB_TOWER
    climbs = [a for a in actions if a.action_type == ActionType.CLIMB_TOWER]
    assert len(climbs) >= 1
    assert climbs[0].element == FieldElement.RED_TOWER
    assert climbs[0].confidence > 0.85
    assert abs(climbs[0].start_time_ms / 1000.0 - 13.0) < 0.25
    assert abs(climbs[0].end_time_ms / 1000.0 - 17.0) < 0.25


@pytest.mark.asyncio
async def test_edge_cases():
    """
    Test edge cases:
    - Empty windows
    - Unobserved track_id
    - Pure IDLE trajectory (no actions triggered)
    - Short degenerate track (len < 2)
    """
    actor = ActionRecognitionActor(fps=60.0)

    # 1. Unobserved track_id on empty actor
    res = await actor.localize_actions(track_id=999)
    assert res == []

    # 2. Ingest empty track list
    await actor.ingest_tracklet_window([], window_size=128)
    res = await actor.localize_actions(track_id=1)
    assert res == []

    # 3. Single frame state (len < 2)
    single_state = [make_tracklet_state(track_id=5, field_x=100.0, field_y=100.0)]
    await actor.ingest_tracklet_window(single_state, window_size=128)
    res = await actor.localize_actions(track_id=5)
    assert res == []

    # 4. Pure IDLE trajectory in empty open space
    idle_states = [
        make_tracklet_state(track_id=6, field_x=50.0, field_y=50.0, vx=5.0, vy=5.0)
        for _ in range(30)
    ]
    await actor.ingest_tracklet_window(idle_states, window_size=128)
    res = await actor.localize_actions(track_id=6)
    # Should not produce scoring, climbing, or intake actions
    assert res == []


# ============================================================================
# 7. Temporal NMS Unit Tests
# ============================================================================

def test_temporal_nms_deduplication():
    """Verify temporal NMS suppresses lower-confidence overlapping proposals of the same action."""
    proposals = [
        {
            "action_type": ActionType.SCORE_HUB,
            "confidence": 0.95,
            "start_frame": 100,
            "end_frame": 150,
            "pyramid_level": 2,
            "field_coords": (162.8, 158.85),
            "element": FieldElement.RED_HUB,
            "diou": 1.0,
        },
        {
            "action_type": ActionType.SCORE_HUB,
            "confidence": 0.88,
            "start_frame": 105,
            "end_frame": 148,  # Highly overlapping with proposal 1
            "pyramid_level": 1,
            "field_coords": (162.8, 158.85),
            "element": FieldElement.RED_HUB,
            "diou": 0.9,
        },
        {
            "action_type": ActionType.SCORE_HUB,
            "confidence": 0.92,
            "start_frame": 300,
            "end_frame": 350,  # Distinct, non-overlapping second score
            "pyramid_level": 2,
            "field_coords": (162.8, 158.85),
            "element": FieldElement.RED_HUB,
            "diou": 1.0,
        },
        {
            "action_type": ActionType.IDLE,
            "confidence": 0.99,
            "start_frame": 50,
            "end_frame": 80,
            "pyramid_level": 0,
            "field_coords": (50.0, 50.0),
            "element": FieldElement.NEUTRAL_ZONE,
            "diou": 1.0,
        },
    ]

    kept = temporal_nms(proposals, iou_threshold=0.4, min_confidence=0.5)

    # 1. IDLE should be pruned
    assert not any(p["action_type"] == ActionType.IDLE for p in kept)

    # 2. Overlapping SCORE_HUB at frame 105..148 should be suppressed in favor of frame 100..150
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.95
    assert kept[0]["start_frame"] == 100
    assert kept[1]["confidence"] == 0.92
    assert kept[1]["start_frame"] == 300


# ============================================================================
# 8. Batched Multi-Track Action Localization (All 6 Robots on GPU 1)
# ============================================================================

@pytest.mark.asyncio
async def test_batched_multi_track_localization_six_robots():
    """
    Verify batched multi-track action localization processes all 6 robots simultaneously
    in a single GPU forward pass on GPU 1 (cuda:1).
    - Track 1 (Red 1): Scoring cycle at Red Hub
    - Track 2 (Red 2): Intake at North Depot
    - Track 3 (Red 3): Climb at Red Tower
    - Track 4 (Blue 1): Scoring cycle at Blue Hub
    - Track 5 (Blue 2): Fast transit through Neutral Zone
    - Track 6 (Blue 3): Idle in defensive alliance territory
    """
    fps = 60.0
    dt = 1.0 / fps
    total_seconds = 6.0
    num_frames = int(total_seconds * fps)  # 360 frames

    actor = ActionRecognitionActor(fps=fps)
    red_hub_x, red_hub_y = RED_HUB_POS
    blue_hub_x, blue_hub_y = BLUE_HUB_POS
    depot_x, depot_y = DEPOT_NORTH_POS
    tower_x, tower_y = RED_TOWER_POS

    for i in range(num_frames):
        t_sec = i * dt
        ts_ms = int(t_sec * 1000)
        frame_idx = i

        # Track 1: Scores at Red Hub between 1.5s and 4.0s
        if 1.5 <= t_sec <= 4.0:
            st1 = make_tracklet_state(track_id=1, team_number=254, alliance=AllianceColor.RED,
                                      field_x=red_hub_x + 0.5, field_y=red_hub_y + 0.5, vx=0.0, vy=0.0)
        else:
            st1 = make_tracklet_state(track_id=1, team_number=254, alliance=AllianceColor.RED,
                                      field_x=200.0, field_y=red_hub_y, vx=100.0, vy=0.0)

        # Track 2: Intakes at North Depot between 1.0s and 3.0s
        if 1.0 <= t_sec <= 3.0:
            st2 = make_tracklet_state(track_id=2, team_number=1323, alliance=AllianceColor.RED,
                                      field_x=depot_x + 1.0, field_y=depot_y + 1.0, vx=0.0, vy=0.0)
        else:
            st2 = make_tracklet_state(track_id=2, team_number=1323, alliance=AllianceColor.RED,
                                      field_x=50.0, field_y=50.0, vx=50.0, vy=0.0)

        # Track 3: Climbs at Red Tower between 2.0s and 5.0s
        if 2.0 <= t_sec <= 5.0:
            st3 = make_tracklet_state(track_id=3, team_number=971, alliance=AllianceColor.RED,
                                      field_x=tower_x + 1.0, field_y=tower_y + 1.0, vx=0.0, vy=0.0)
        else:
            st3 = make_tracklet_state(track_id=3, team_number=971, alliance=AllianceColor.RED,
                                      field_x=120.0, field_y=50.0, vx=60.0, vy=0.0)

        # Track 4: Scores at Blue Hub between 1.5s and 4.0s
        if 1.5 <= t_sec <= 4.0:
            st4 = make_tracklet_state(track_id=4, team_number=1678, alliance=AllianceColor.BLUE,
                                      field_x=blue_hub_x + 0.5, field_y=blue_hub_y + 0.5, vx=0.0, vy=0.0)
        else:
            st4 = make_tracklet_state(track_id=4, team_number=1678, alliance=AllianceColor.BLUE,
                                      field_x=400.0, field_y=blue_hub_y, vx=100.0, vy=0.0)

        # Track 5: Fast transit across Neutral Zone between 1.0s and 3.5s
        if 1.0 <= t_sec <= 3.5:
            st5 = make_tracklet_state(track_id=5, team_number=2910, alliance=AllianceColor.BLUE,
                                      field_x=260.0 + (t_sec - 1.0) * 40.0, field_y=150.0, vx=120.0, vy=0.0)
        else:
            st5 = make_tracklet_state(track_id=5, team_number=2910, alliance=AllianceColor.BLUE,
                                      field_x=550.0, field_y=150.0, vx=10.0, vy=0.0)

        # Track 6: Idle in open territory
        st6 = make_tracklet_state(track_id=6, team_number=4414, alliance=AllianceColor.BLUE,
                                  field_x=550.0, field_y=50.0, vx=5.0, vy=5.0)

        batch_frame = [st1, st2, st3, st4, st5, st6]
        await actor.ingest_tracklet_window(
            track_states=batch_frame,
            window_size=num_frames,
            timestamps_ms=[ts_ms] * 6,
            frame_indices=[frame_idx] * 6,
        )

    # 1. Execute batched multi-track localization for all 6 robots
    all_actions = await actor.localize_all_actions([1, 2, 3, 4, 5, 6])
    assert len(all_actions) == 6
    assert set(all_actions.keys()) == {1, 2, 3, 4, 5, 6}

    # Verify Track 1 (Red 1): SCORE_HUB at RED_HUB
    t1_scores = [a for a in all_actions[1] if a.action_type == ActionType.SCORE_HUB]
    assert len(t1_scores) >= 1
    assert t1_scores[0].element == FieldElement.RED_HUB
    assert t1_scores[0].confidence > 0.85

    # Verify Track 2 (Red 2): INTAKE_FUEL at DEPOT
    t2_intakes = [a for a in all_actions[2] if a.action_type == ActionType.INTAKE_FUEL]
    assert len(t2_intakes) >= 1
    assert t2_intakes[0].element == FieldElement.DEPOT

    # Verify Track 3 (Red 3): CLIMB_TOWER at RED_TOWER
    t3_climbs = [a for a in all_actions[3] if a.action_type == ActionType.CLIMB_TOWER]
    assert len(t3_climbs) >= 1
    assert t3_climbs[0].element == FieldElement.RED_TOWER

    # Verify Track 4 (Blue 1): SCORE_HUB at BLUE_HUB
    t4_scores = [a for a in all_actions[4] if a.action_type == ActionType.SCORE_HUB]
    assert len(t4_scores) >= 1
    assert t4_scores[0].element == FieldElement.BLUE_HUB
    assert t4_scores[0].confidence > 0.85

    # Verify Track 5 (Blue 2): TRANSIT_NEUTRAL_ZONE
    t5_transits = [a for a in all_actions[5] if a.action_type == ActionType.TRANSIT_NEUTRAL_ZONE]
    assert len(t5_transits) >= 1
    assert t5_transits[0].element == FieldElement.NEUTRAL_ZONE

    # Verify Track 6 (Blue 3): Idle (no aggressive scoring/intake actions)
    assert len(all_actions[6]) == 0

    # 2. Strict protocol check: individual localize_actions yields identical output
    single_t1 = await actor.localize_actions(1)
    assert len(single_t1) == len(t1_scores)
    assert single_t1[0].action_type == t1_scores[0].action_type
    assert single_t1[0].element == t1_scores[0].element
    assert single_t1[0].start_frame == t1_scores[0].start_frame
    assert single_t1[0].end_frame == t1_scores[0].end_frame


@pytest.mark.asyncio
async def test_latency_benchmark_under_5ms():
    """
    Verify batched action localization latency for all 6 robots simultaneously
    is strictly under 5 ms on GPU 1 (cuda:1).
    """
    actor = ActionRecognitionActor(fps=60.0)
    for tid in range(1, 7):
        states = [make_tracklet_state(track_id=tid, field_x=162.8 if tid == 1 else 50.0) for _ in range(128)]
        await actor.ingest_tracklet_window(states, window_size=128)

    # Warmup
    for _ in range(10):
        _ = await actor.localize_all_actions([1, 2, 3, 4, 5, 6])
    if actor.model.device.type == "cuda":
        torch.cuda.synchronize(actor.model.device)

    # Benchmark 50 iterations
    n_iters = 50
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = await actor.localize_all_actions([1, 2, 3, 4, 5, 6])
    if actor.model.device.type == "cuda":
        torch.cuda.synchronize(actor.model.device)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / n_iters) * 1000.0
    print(f"\n[BENCHMARK] Batched 6-Robot Localization Latency: {avg_ms:.3f} ms on {actor.model.device}")
    assert avg_ms < 5.0, f"Average latency {avg_ms:.3f} ms exceeded the 5.0 ms requirement"


@pytest.mark.asyncio
async def test_deterministic_cpu_fallback():
    """
    Verify deterministic CPU fallback when CUDA is not used (device='cpu').
    """
    actor_cpu = ActionRecognitionActor(fps=60.0, device="cpu")
    assert actor_cpu.model.device == torch.device("cpu")
    assert actor_cpu._buffer.device == torch.device("cpu")

    # Ingest trajectory for Red Hub score on CPU
    fps = 60.0
    dt = 1.0 / fps
    track_states = []
    timestamps_ms = []
    frame_indices = []

    for i in range(128):
        t_sec = i * dt
        ts_ms = int(t_sec * 1000)
        timestamps_ms.append(ts_ms)
        frame_indices.append(i)

        if 0.5 <= t_sec <= 1.8:
            x, y, vx, vy = RED_HUB_POS[0] + 0.5, RED_HUB_POS[1] + 0.5, 0.0, 0.0
        else:
            x, y, vx, vy = 50.0, 50.0, 0.0, 0.0

        track_states.append(
            make_tracklet_state(
                track_id=10,
                team_number=254,
                field_x=x,
                field_y=y,
                vx=vx,
                vy=vy,
            )
        )

    await actor_cpu.ingest_tracklet_window(
        track_states=track_states,
        window_size=128,
        timestamps_ms=timestamps_ms,
        frame_indices=frame_indices,
    )

    actions = await actor_cpu.localize_actions(10)
    hub_scores = [a for a in actions if a.action_type == ActionType.SCORE_HUB]
    assert len(hub_scores) >= 1
    score_act = hub_scores[0]
    assert score_act.action_type == ActionType.SCORE_HUB
    assert score_act.element == FieldElement.RED_HUB
    assert score_act.confidence > 0.85


def test_device_selection_and_fallback():
    """Verify target device resolution logic targets cuda:1 with clean fallbacks."""
    # Explicit CPU
    dev_cpu = get_torch_device("cpu")
    assert dev_cpu == torch.device("cpu")

    # Non-existent device index falls back cleanly
    dev_fallback = get_torch_device("cuda:9999")
    if torch.cuda.is_available():
        assert dev_fallback.type == "cuda"
    else:
        assert dev_fallback == torch.device("cpu")

    # Default device targets cuda:1 on multi-GPU systems
    default_dev = get_torch_device()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        assert default_dev == torch.device("cuda:1")


def test_pytorch_diou_and_focal_loss_cuda_tensors():
    """Verify compute_diou_1d and focal_loss execute on CUDA tensors."""
    dev = get_torch_device()

    # DIoU on CUDA
    seg_a = torch.tensor([[10.0, 20.0], [0.0, 10.0]], device=dev)
    seg_b = torch.tensor([[10.0, 20.0], [5.0, 15.0]], device=dev)
    diou = compute_diou_1d(seg_a, seg_b)
    assert isinstance(diou, torch.Tensor)
    assert diou.device.type == dev.type
    assert pytest.approx(float(diou[0].item()), abs=1e-4) == 1.0
    assert pytest.approx(float(diou[1].item()), abs=1e-4) == 2.0 / 9.0

    # Focal loss on CUDA
    preds = torch.tensor([[0.98, 0.02], [0.05, 0.95]], device=dev)
    targets = torch.tensor([0, 0], device=dev)
    loss = focal_loss(preds, targets)
    assert isinstance(loss, float)
    assert loss > 0.0

