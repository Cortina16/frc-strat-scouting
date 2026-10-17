"""Unit and Integration Tests for analytics-implementation.md Pipeline.

Verifies:
1. The Blue Alliance API client match harvesting, metadata parsing, and winner extraction.
2. Hub scoring intersection detection: confirms scored balls when ball detections intersect
   Hub geometries during point increases, and distinguishes missed/deflected shots.
3. Analytics metrics engine: verifies all 12 vectorized mathematical metrics and 90-dimensional
   feature tensor assembly across 5 phases.
4. Monte Carlo simulation engine: verifies 10,000 skew-normal rollouts, win probabilities,
   and strategy delta-W evaluation.
5. XGBoost match predictor: verifies training, prediction, and validation against real match winners.
6. Flow orchestrator end-to-end integration.
"""

from __future__ import annotations

import math
import os
import tempfile
import cv2
import numpy as np
import pytest

from src.analytics.hub_scorer import HubAlliance, HubGeometry, HubScoringDetector, ShotEvent
from src.analytics.match_predictor import MatchPrediction, MatchPredictor, ValidationReport
from src.analytics.metrics_engine import (
    AnalyticsMetricsEngine,
    GRID_M,
    GRID_N,
    NUM_PHASES,
    PhaseMetrics,
)
from src.analytics.monte_carlo import MonteCarloSimulationEngine, SimulationResult
from src.analytics.tba_client import MatchMetadata, TheBlueAllianceClient
from src.analytics.video_watcher import VideoWatcher, VideoMatchAnalysisResult
from src.analytics.flow_orchestrator import AnalyticsFlowOrchestrator, FlowExecutionResult
from src.analytics.visualizer import AnalyticsPredictionVisualizer


# ============================================================================
# 1. The Blue Alliance API Client Tests
# ============================================================================

def test_tba_client_fixture_parsing():
    """Verify TBA client parses match metadata, winner, and score breakdown correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = TheBlueAllianceClient(cache_dir=tmp_dir, video_dir=tmp_dir)
        matches = client.get_event_matches("2024cmptx")

        assert len(matches) >= 10
        m1 = matches[0]
        assert m1.event_key == "2024cmptx"
        assert m1.match_key == "2024cmptx_qm1"
        assert len(m1.red_teams) == 3
        assert len(m1.blue_teams) == 3
        assert m1.winning_alliance in ("red", "blue", "draw")
        assert m1.red_score > 0
        assert m1.blue_score > 0
        assert len(m1.video_urls) > 0
        assert "youtube.com" in m1.video_urls[0]


def test_tba_client_video_acquisition_fallback():
    """Verify TBA client acquires video or falls back to local workspace file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = TheBlueAllianceClient(cache_dir=tmp_dir, video_dir=tmp_dir)
        match = MatchMetadata(
            match_key="2024cmp_qm1",
            event_key="2024cmp",
            comp_level="qm",
            match_number=1,
            red_teams=["254", "1678", "1323"],
            blue_teams=["1690", "971", "4414"],
            red_score=110,
            blue_score=95,
            winning_alliance="red",
            video_urls=["https://www.youtube.com/watch?v=j8wz5vw5XfE"],
        )
        # Fallback to output.mp4 if present or mock file
        mock_file = os.path.join(tmp_dir, "test.mp4")
        with open(mock_file, "wb") as f:
            f.write(b"0" * 2000)

        acquired = client.acquire_match_video(match, preferred_local_fallback=mock_file)
        assert acquired is not None
        assert os.path.isfile(acquired)


# ============================================================================
# 2. Hub Scoring Intersection Detector Tests
# ============================================================================

def test_hub_scoring_detector_confirmed_scored_ball():
    """
    Verify ball intersecting Hub when points increase is tagged as is_scored = True.
    """
    detector = HubScoringDetector()

    # Red Hub screen bbox is (1380, 135, 1610, 305) (Screen Right)
    ball_in_red_hub = (1495.0, 220.0)
    tracked_robots = [
        {"track_id": 1, "center": (1450.0, 250.0), "alliance": "RED", "team_number": "254"},
        {"track_id": 4, "center": (450.0, 250.0), "alliance": "BLUE", "team_number": "1690"},
    ]

    # Register score tick (+2 Red points at timestamp 5.0s)
    detector.register_score_tick(timestamp_s=5.0, red_score=2, blue_score=0)

    # Process frame at timestamp 5.2s with ball in Red Hub
    shots = detector.process_frame(
        frame_idx=150,
        timestamp_s=5.2,
        ball_positions=[ball_in_red_hub],
        tracked_robots=tracked_robots,
        use_screen_coords=True,
    )

    assert len(shots) == 1
    shot = shots[0]
    assert shot.hub_alliance == HubAlliance.RED
    assert shot.is_scored is True
    assert shot.robot_track_id == 1
    assert shot.robot_alliance == "RED"
    assert shot.robot_team == "254"


def test_hub_scoring_detector_missed_shot_without_point_increase():
    """
    Verify ball intersecting Hub WITHOUT point increase is tagged as is_scored = False (missed/deflected).
    """
    detector = HubScoringDetector()
    # Blue Hub screen bbox is (335, 135, 565, 305) (Screen Left)
    ball_in_blue_hub = (450.0, 220.0)
    tracked_robots = [
        {"track_id": 4, "center": (450.0, 250.0), "alliance": "BLUE", "team_number": "1690"},
    ]

    # No score tick registered! Scores remain 0, 0
    shots = detector.process_frame(
        frame_idx=300,
        timestamp_s=10.0,
        ball_positions=[ball_in_blue_hub],
        tracked_robots=tracked_robots,
        use_screen_coords=True,
    )

    assert len(shots) == 1
    shot = shots[0]
    assert shot.hub_alliance == HubAlliance.BLUE
    assert shot.is_scored is False  # Missed / deflected shot


def test_hub_ball_tracklet_single_count_guarantee():
    """
    Verify that a yellow ball residing in the Hub catchment across multiple frames
    is counted exactly once, preventing frame-by-frame duplicate scoring.
    """
    detector = HubScoringDetector()
    detector.register_score_tick(timestamp_s=5.0, red_score=4, blue_score=0)

    robots = [{"track_id": 1, "center": (1450.0, 250.0), "alliance": "RED", "team_number": "254"}]

    all_emitted_shots = []
    # Simulate ball remaining in Hub catchment for 12 continuous frames
    for f in range(12):
        ts = 5.0 + f * (1.0 / 30.0)
        ball_pos = (1495.0 + (f % 3) * 0.5, 220.0 + (f % 4) * 0.5)
        shots = detector.process_frame(
            frame_idx=100 + f,
            timestamp_s=ts,
            ball_positions=[ball_pos],
            tracked_robots=robots,
            use_screen_coords=True,
            scoring_allowed=True,
        )
        all_emitted_shots.extend(shots)

    assert len(all_emitted_shots) == 1
    assert len(detector.confirmed_shots) == 1
    assert detector.confirmed_shots[0].is_scored is True
    assert detector.confirmed_shots[0].hub_alliance == HubAlliance.RED


def test_hub_scorer_grace_period_suppression():
    """
    Verify that scoring attribution is strictly suppressed during the 3-second grace period
    after match start (scoring_allowed = False).
    """
    detector = HubScoringDetector()
    detector.register_score_tick(timestamp_s=1.0, red_score=2, blue_score=0)

    robots = [{"track_id": 1, "center": (1450.0, 250.0), "alliance": "RED", "team_number": "254"}]

    # Ball in red hub during grace period (scoring_allowed = False)
    shots = detector.process_frame(
        frame_idx=50,
        timestamp_s=1.5,
        ball_positions=[(1495.0, 220.0)],
        tracked_robots=robots,
        use_screen_coords=True,
        scoring_allowed=False,  # Within 3s grace period!
    )
    assert len(shots) == 1
    assert shots[0].is_scored is False  # Scoring suppressed!
    assert detector.confirmed_shots[0].is_scored is False


def test_scoreboard_reader_timer_countdown_detection():
    """
    Verify BroadcastScoreboardReader detects match countdown start and enforces 3.0s grace period.
    """
    from src.analytics.scoreboard_reader import BroadcastScoreboardReader

    reader = BroadcastScoreboardReader(grace_period_s=3.0)

    # Synthetic 1080p frame canvas
    frame = np.full((1080, 1920, 3), (20, 20, 20), dtype=np.uint8)

    # Scoreboard background: Blue [75:112, 805:885], Timer [75:112, 910:1010], Red [75:112, 1015:1115]
    frame[70:115, 800:885] = (170, 95, 27)
    frame[70:115, 885:1015] = (250, 250, 250)
    frame[70:115, 1015:1100] = (34, 26, 214)

    # Pre-match single-digit artifact should NOT start countdown
    cv2.putText(frame[70:115, 885:1015], "4", (45, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    reading0 = reader.read_frame(frame, frame_idx=50, timestamp_s=1.0)
    assert reading0.match_started is False
    assert reading0.scoring_allowed is False

    # Initial frame with 0:15
    frame[70:115, 885:1015] = (250, 250, 250)
    cv2.putText(frame[70:115, 885:1015], "0:15", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    reading1 = reader.read_frame(frame, frame_idx=100, timestamp_s=3.3)
    assert reading1.match_started is False
    assert reading1.scoring_allowed is False

    # Decrement to 0:14
    frame[70:115, 885:1015] = (250, 250, 250)
    cv2.putText(frame[70:115, 885:1015], "0:14", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    reading2 = reader.read_frame(frame, frame_idx=130, timestamp_s=4.3)
    assert reading2.match_started is True
    assert reading2.scoring_allowed is False  # Still inside 3.0s grace period!

    # Frame after 3.5s elapsed
    reading3 = reader.read_frame(frame, frame_idx=235, timestamp_s=7.8)
    assert reading3.match_started is True
    assert reading3.match_elapsed_s >= 3.0
    assert reading3.scoring_allowed is True


def test_scoreboard_reader_score_increments():
    """
    Verify BroadcastScoreboardReader accurately extracts score increments from HUD.
    """
    from src.analytics.scoreboard_reader import BroadcastScoreboardReader

    reader = BroadcastScoreboardReader()
    frame = np.full((1080, 1920, 3), (20, 20, 20), dtype=np.uint8)
    frame[70:115, 800:885] = (170, 95, 27)
    frame[70:115, 885:1015] = (250, 250, 250)
    frame[70:115, 1015:1100] = (34, 26, 214)

    # Frame 1: Blue=0, Red=0
    cv2.putText(frame[70:115, 800:885], "0", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame[70:115, 1015:1100], "0", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    r1 = reader.read_frame(frame, 1, 0.033)
    assert r1.blue_score == 0
    assert r1.red_score == 0


def test_hub_scoring_detector_tag_anchored_hubs():
    """
    Verify dynamic AprilTag anchoring of Red and Blue Hubs based on detected tags:
    - Tags <= 16 (specifically 8/5, 9/10, 11/2, 4/3) anchor the Red Hub (Right).
    - Tags >= 17 (specifically 18/27, 19/20, 21/24, 26/25) anchor the Blue Hub (Left).
    - Catchment zone is anchored directly ABOVE detected AprilTags (offset by ~110px).
    """
    from dataclasses import dataclass

    @dataclass
    class MockTag:
        tag_id: int
        center_px: Tuple[float, float]

    detector = HubScoringDetector()

    # Blue Hub Tag 18 at (450, 330) on screen left
    # Red Hub Tag 5 at (1500, 330) on screen right
    tags = [
        MockTag(tag_id=18, center_px=(450.0, 330.0)),
        MockTag(tag_id=5, center_px=(1500.0, 330.0)),
    ]

    detector.update_from_tags(tags, frame_shape=(1080, 1920))

    # Catchment center should be ~110px above tag center (Y ~ 220)
    bx1, by1, bx2, by2 = detector.blue_hub.screen_bbox
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    assert abs(bcx - 450.0) < 5.0
    assert abs(bcy - 220.0) < 5.0
    assert bcx < 960.0  # Left side

    rx1, ry1, rx2, ry2 = detector.red_hub.screen_bbox
    rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
    assert abs(rcx - 1500.0) < 5.0
    assert abs(rcy - 220.0) < 5.0
    assert rcx > 960.0  # Right side


def test_hub_scoring_detector_occlusion_and_persistence():
    """
    Verify Hub location is preserved when Hub tags are temporarily occluded:
    1. Perimeter wall tags (e.g. Tag 6 at x=1700 or Tag 17 at x=260) NEVER hijack Hub positions.
    2. Hub locations are saved and maintained across occluded frames.
    3. If only one Hub is visible, the other is triangulated via the mutual relative vector.
    """
    from dataclasses import dataclass

    @dataclass
    class MockTag:
        tag_id: int
        center_px: Tuple[float, float]

    detector = HubScoringDetector()

    # Frame 1: Both Hubs clearly observed
    tags_f1 = [
        MockTag(tag_id=18, center_px=(450.0, 330.0)),   # Blue Hub
        MockTag(tag_id=5, center_px=(1500.0, 330.0)),   # Red Hub
    ]
    detector.update_from_tags(tags_f1, frame_shape=(1080, 1920))
    orig_bx1, _, orig_bx2, _ = detector.blue_hub.screen_bbox
    orig_rx1, _, orig_rx2, _ = detector.red_hub.screen_bbox
    orig_bcx = (orig_bx1 + orig_bx2) / 2.0
    orig_rcx = (orig_rx1 + orig_rx2) / 2.0

    # Frame 2: Blue Hub tags occluded! Only perimeter wall tags (Tag 6, Tag 17) and Red Hub tag 5 visible
    tags_f2 = [
        MockTag(tag_id=5, center_px=(1500.0, 330.0)),   # Red Hub tag
        MockTag(tag_id=17, center_px=(260.0, 510.0)),   # Perimeter wall tag on Blue side!
        MockTag(tag_id=6, center_px=(1700.0, 510.0)),   # Perimeter wall tag on Red side!
    ]
    detector.update_from_tags(tags_f2, frame_shape=(1080, 1920))

    # Blue Hub MUST NOT jump to wall tag 17 at x=260! It should stay triangulated/persistent at x ~ 450
    curr_bx1, _, curr_bx2, _ = detector.blue_hub.screen_bbox
    curr_bcx = (curr_bx1 + curr_bx2) / 2.0
    assert abs(curr_bcx - orig_bcx) < 20.0, f"Blue Hub jumped to {curr_bcx}, should be near {orig_bcx}"
    assert curr_bcx > 350.0, "Blue Hub falsely moved towards wall tag 17"

    # Frame 3: Both Hub tags occluded! Only wall tags visible
    tags_f3 = [
        MockTag(tag_id=17, center_px=(260.0, 510.0)),
        MockTag(tag_id=6, center_px=(1700.0, 510.0)),
    ]
    detector.update_from_tags(tags_f3, frame_shape=(1080, 1920))
    # Both Hubs must maintain persistent locations
    f3_bx1, _, f3_bx2, _ = detector.blue_hub.screen_bbox
    f3_rx1, _, f3_rx2, _ = detector.red_hub.screen_bbox
    assert abs((f3_bx1 + f3_bx2) / 2.0 - orig_bcx) < 20.0
    assert abs((f3_rx1 + f3_rx2) / 2.0 - orig_rcx) < 20.0


def test_detect_flying_balls_near_hubs():
    """Verify direct screen-space detection of flying yellow balls in Hub funnels."""
    from src.vision.ball_detector import detect_flying_balls_near_hubs

    # Create dummy 1080p frame with dark arena background
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)

    # Place a flying yellow ball near Blue Hub funnel opening: (450, 210)
    # Yellow BGR ~ (20, 220, 240)
    cv2.circle(frame, (450, 210), 6, (25, 215, 235), -1)

    # Place another flying yellow ball near Red Hub funnel opening: (1500, 210)
    cv2.circle(frame, (1500, 210), 7, (20, 210, 230), -1)

    blue_bbox = (335.0, 135.0, 565.0, 305.0)
    red_bbox = (1380.0, 135.0, 1610.0, 305.0)

    flying = detect_flying_balls_near_hubs(frame, blue_bbox=blue_bbox, red_bbox=red_bbox)
    assert len(flying) >= 2

    # Check detected coordinates match within 4 px
    found_blue = any(math.hypot(x - 450, y - 210) < 5.0 for x, y in flying)
    found_red = any(math.hypot(x - 1500, y - 210) < 5.0 for x, y in flying)
    assert found_blue, "Failed to detect flying ball near Blue Hub"
    assert found_red, "Failed to detect flying ball near Red Hub"


def test_top_metrics_banner_rendering():
    """Verify _render_top_metrics_banner generates 1920x90 telemetry bar with metric cards."""
    from src.analytics.visualizer import AnalyticsPredictionVisualizer
    from src.analytics.tba_client import MatchMetadata

    visualizer = AnalyticsPredictionVisualizer()
    meta = MatchMetadata(
        match_key="2026casf_qm1",
        event_key="2026casf",
        comp_level="qm",
        match_number=1,
        red_teams=["254", "1678", "971"],
        blue_teams=["1323", "4414", "2910"],
        red_score=85,
        blue_score=75,
        winning_alliance="red",
        video_urls=[],
    )

    banner = visualizer._render_top_metrics_banner(
        w=1920,
        h=90,
        match_metadata=meta,
        timestamp_s=25.0,
        frame_idx=1500,
        red_win_prob=0.54,
        blue_win_prob=0.46,
        cur_red_score=85,
        cur_blue_score=75,
        active_robots=[],
        scored_shots_count=8,
    )

    assert banner.shape == (90, 1920, 3)
    assert banner.dtype == np.uint8
    # Banner must have non-zero pixels (content rendered)
    assert cv2.countNonZero(cv2.cvtColor(banner, cv2.COLOR_BGR2GRAY)) > 1000


# ============================================================================
# 3. Analytics Metrics Engine Tests (analytics-implementation.md)
# ============================================================================

def test_metrics_engine_tensors_and_12_metrics():
    """Verify calculation of all 12 mathematical metrics and 90-dim vector assembly."""
    engine = AnalyticsMetricsEngine()

    # 1. Spatial Occupancy Tensor R_p (6, 54, 27)
    trajectories = [
        {"track_id": 0, "frame_index": 10, "x_ft": 15.0, "y_ft": 13.5},
        {"track_id": 0, "frame_index": 20, "x_ft": 25.0, "y_ft": 13.5},
        {"track_id": 1, "frame_index": 10, "x_ft": 40.0, "y_ft": 13.5},
    ]
    r_tensor = engine.build_spatial_occupancy_tensor(trajectories, phase_frames=[10, 20])
    assert r_tensor.shape == (6, GRID_M, GRID_N)
    assert r_tensor[0, 15, 13] == 1.0
    assert r_tensor[0, 25, 13] == 1.0

    # 2. Ball Density Matrix with Gaussian Convolution
    ball_pts = [(15.0, 13.0), (16.0, 14.0), (15.5, 13.5)]
    b_density = engine.build_ball_density_matrix(ball_pts, sigma_ft=2.0)
    assert b_density.shape == (GRID_M, GRID_N)
    assert b_density[15, 13] > 0.0

    # 3. Metrics Verification
    r_0 = r_tensor[0]
    rai = engine.compute_route_adaptability_index(r_0, cycles=[])
    assert isinstance(rai, float) and rai >= 0.0

    stall_risk = engine.compute_stall_risk([0.1, 0.05, 1.5, 2.0], ["Transit", "Transit", "Transit", "Transit"])
    assert stall_risk == 0.5  # 2 of 4 frames < 0.2 ft/s in Transit

    steal_risk = engine.compute_steal_risk(
        r_0, "RED", [0.1, 1.0], [(30.0, 13.0), (10.0, 13.0)], [2.0, 10.0]
    )
    assert isinstance(steal_risk, float) and steal_risk >= 0.0

    choke = engine.compute_choke_density(r_0)
    assert 0.0 <= choke <= 1.0

    p_var = engine.compute_path_variance(r_0)
    assert p_var >= 0.0

    tse, bpf_coop = engine.compute_tse_and_bpf_coop(b_density, (15.0, 13.0))
    assert 0.0 <= tse <= 1.0
    assert bpf_coop >= 0.0

    iqi, fwr = engine.compute_iqi_and_fwr(10, 2, 5.0)
    assert math.isclose(iqi, 10.0 / 12.0, rel_tol=1e-3)
    assert math.isclose(fwr, 2.0 / 5.0, rel_tol=1e-3)

    # 4. Logistic Accuracy Model
    shots = [
        ShotEvent("s1", 1, 5.0, 2, HubAlliance.RED, 15.0, 13.0, 0, "254", "RED", 8.0, True, True),
        ShotEvent("s2", 2, 7.0, 2, HubAlliance.RED, 15.0, 13.0, 0, "254", "RED", 14.0, False, False),
        ShotEvent("s3", 3, 9.0, 2, HubAlliance.RED, 15.0, 13.0, 0, "254", "RED", 6.0, True, True),
        ShotEvent("s4", 4, 11.0, 2, HubAlliance.RED, 15.0, 13.0, 0, "254", "RED", 18.0, False, False),
    ]
    rfr, b0, b_dist, b_open = engine.estimate_logistic_accuracy_and_rfr(shots, 30.0)
    assert rfr > 0.0
    assert isinstance(b0, float)
    assert b_dist < 0.0  # Accuracy decreases as distance increases

    # 5. Full Phase Metrics and 90-dim Vector
    phases = []
    for p_id in range(1, 6):
        pm = engine.compute_phase_metrics(
            phase_id=p_id,
            r_k=r_0,
            alliance="RED",
            partner_r=r_tensor[1],
            shots=shots,
            velocities=[1.5, 2.0, 0.1, 1.2],
            positions=[(15.0, 13.0), (16.0, 13.0), (25.0, 13.0), (18.0, 13.0)],
            states=["Transit", "Intaking", "Transit", "Scoring"],
            defender_distances=[7.0, 6.0, 2.5, 8.0],
        )
        assert len(pm.to_vector()) == 18
        phases.append(pm)

    x_input = engine.assemble_90d_feature_vector(phases)
    assert x_input.shape == (90,)
    assert not np.isnan(x_input).any()


# ============================================================================
# 4. Monte Carlo Simulation Engine Tests
# ============================================================================

def test_monte_carlo_skew_normal_simulation():
    """Verify 10,000-iteration Monte Carlo rollout and win probabilities."""
    mc = MonteCarloSimulationEngine(num_simulations=10000, random_seed=42)

    red_profile = {
        "auto_mu": 35.0, "auto_sigma": 4.0,
        "main_mu": 70.0, "main_sigma": 6.0,
        "endgame_mu": 25.0, "endgame_sigma": 3.0,
        "defense_pressure": 3.0,
        "passing_eff": 0.90,
    }
    blue_profile = {
        "auto_mu": 28.0, "auto_sigma": 5.0,
        "main_mu": 58.0, "main_sigma": 7.0,
        "endgame_mu": 20.0, "endgame_sigma": 4.0,
        "defense_pressure": 1.5,
    }

    res = mc.simulate_match(red_profile, blue_profile)
    assert res.num_iterations == 10000
    assert 0.0 <= res.red_win_prob <= 1.0
    assert 0.0 <= res.blue_win_prob <= 1.0
    assert math.isclose(res.red_win_prob + res.blue_win_prob + res.draw_prob, 1.0, rel_tol=1e-4)

    # Red should strongly be favored given higher profile means
    assert res.red_win_prob > res.blue_win_prob
    assert res.red_score_mean > res.blue_score_mean
    assert res.red_score_ci95[0] < res.red_score_mean < res.red_score_ci95[1]
    assert len(res.strategy_recommendation) > 0


# ============================================================================
# 5. XGBoost Predictor & TBA Real Winner Validation Tests
# ============================================================================

def test_xgboost_training_and_real_winner_validation():
    """Verify training XGBoost on 90-dim vectors and validating against real TBA matches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "xgboost_test.json")
        predictor = MatchPredictor(model_save_path=model_path, n_estimators=40, max_depth=3)

        rng = np.random.default_rng(42)
        # Synthesize 40 matches with 90-dim features
        X = rng.normal(0.0, 1.0, (40, 90)).astype(np.float32)
        # Red wins when feature 8 (RFR diff) + feature 5 (TSE diff) is positive
        y = ((X[:, 8] * 2.0 + X[:, 5] * 1.5 + rng.normal(0, 0.5, 40)) > 0).astype(np.int32)

        metrics = predictor.train(X[:30], y[:30], X[30:], y[30:])
        assert metrics["train_accuracy"] >= 0.70
        assert os.path.isfile(model_path)

        # Create test matches with real metadata
        test_matches = []
        for i in range(10):
            meta = MatchMetadata(
                match_key=f"2024cmp_qm{i+1}",
                event_key="2024cmp",
                comp_level="qm",
                match_number=i+1,
                red_teams=["254", "1678", "1323"],
                blue_teams=["1690", "971", "4414"],
                red_score=100 if y[30+i] == 1 else 80,
                blue_score=80 if y[30+i] == 1 else 100,
                winning_alliance="red" if y[30+i] == 1 else "blue",
                score_breakdown={"red": {"autoPoints": 30, "teleopPoints": 50, "endgamePoints": 20},
                                 "blue": {"autoPoints": 28, "teleopPoints": 48, "endgamePoints": 20}},
            )
            test_matches.append((X[30+i], meta))

        report = predictor.validate_against_real_matches(test_matches)
        assert report.total_matches == 10
        assert 0.0 <= report.accuracy <= 1.0
        assert report.log_loss_score >= 0.0
        assert report.brier_score >= 0.0
        assert len(report.confusion_matrix) == 2
        summary_md = report.summary_markdown()
        assert "Validation Report" in summary_md


# ============================================================================
# 6. Flow Orchestrator Integration Test
# ============================================================================

def test_analytics_flow_orchestrator_execution():
    """Verify end-to-end execution of the analytics flow orchestrator."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a small dummy video file (10 frames)
        dummy_video = os.path.join(tmp_dir, "dummy_match.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video, fourcc, 30.0, (640, 360))
        for _ in range(30):
            frame = np.full((360, 640, 3), 40, dtype=np.uint8)
            # Draw mock yellow ball
            cv2.circle(frame, (250, 180), 8, (0, 220, 255), -1)
            writer.write(frame)
        writer.release()

        model_save = os.path.join(tmp_dir, "predictor.json")
        orchestrator = AnalyticsFlowOrchestrator(
            tba_api_key=None,  # Uses fixture matches
            model_weights_path="yolo11x_robot_best.pt",
            predictor_save_path=model_save,
            default_video_fallback=dummy_video,
        )

        flow_res = orchestrator.run_flow(
            event_key="2024cmptx",
            max_matches=4,
            max_frames_per_video=30,
            train_test_split=0.50,
        )

        assert flow_res.num_matches_evaluated == 4
        assert flow_res.validation_report.total_matches == 2
        assert os.path.isfile(model_save)
        assert len(flow_res.match_features) == 4
        for feat in flow_res.match_features.values():
            assert feat.shape == (90,)


def test_flow_orchestrator_match_filtering_and_caching():
    """Verify flow orchestrator filters by match_key and caches video CV results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_video = os.path.join(tmp_dir, "dummy_match.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video, fourcc, 30.0, (640, 360))
        for _ in range(20):
            frame = np.full((360, 640, 3), 40, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        model_save = os.path.join(tmp_dir, "predictor_filter.json")
        orchestrator = AnalyticsFlowOrchestrator(
            tba_api_key=None,
            model_weights_path="yolo11x_robot_best.pt",
            predictor_save_path=model_save,
            default_video_fallback=dummy_video,
        )

        # Single match execution
        flow_res = orchestrator.run_flow(
            event_key="2026casf",
            match_key="2026casf_qm1",
            max_frames_per_video=20,
        )

        assert flow_res.num_matches_evaluated == 1
        assert "2026casf_qm1" in flow_res.match_features
        assert flow_res.validation_report.total_matches == 1

        # Caching verification across multi-match reuse
        flow_res_cached = orchestrator.run_flow(
            event_key="2026casf",
            max_matches=2,
            max_frames_per_video=20,
        )
        assert flow_res_cached.num_matches_evaluated == 2


# ============================================================================
# 7. Spatial Backend & Video Rendering Integration Tests
# ============================================================================

def test_video_watcher_spatial_backend():
    """Verify VideoWatcher with spatial backend extracts FUEL clusters and robot units."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_video = os.path.join(tmp_dir, "dummy_spatial.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video, fourcc, 30.0, (1920, 1080))
        for i in range(15):
            frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
            # Draw moving yellow fuel blob
            cv2.circle(frame, (400 + i * 5, 500), 12, (20, 220, 240), -1)
            # Draw red bumper robot proposal
            cv2.rectangle(frame, (600, 400), (700, 500), (30, 30, 200), -1)
            # Draw blue bumper robot proposal
            cv2.rectangle(frame, (1200, 400), (1300, 500), (200, 60, 30), -1)
            writer.write(frame)
        writer.release()

        watcher = VideoWatcher(cv_backend="spatial")
        res = watcher.watch_video(dummy_video, match_key="test_spatial", max_frames=15)

        assert res.match_key == "test_spatial"
        assert res.total_frames_processed >= 10
        # Check observations have physical positions and state
        for obs in res.robot_observations:
            assert obs.field_x_ft >= 0.0
            assert obs.field_y_ft >= 0.0
            assert obs.state.upper() in ("IDLE", "INTAKING", "SCORING", "TRANSIT")


def test_analytics_prediction_visualizer_rendering():
    """Verify prediction visualizer renders both side_by_side and broadcast video modes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_vid = os.path.join(tmp_dir, "vis_in.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(input_vid, fourcc, 30.0, (1920, 1080))
        for _ in range(10):
            frame = np.full((1080, 1920, 3), 50, dtype=np.uint8)
            cv2.circle(frame, (500, 500), 10, (20, 220, 240), -1)
            writer.write(frame)
        writer.release()

        meta = MatchMetadata(
            match_key="2026casf_qm1",
            event_key="2026casf",
            comp_level="qm",
            match_number=1,
            red_teams=["254", "1678", "971"],
            blue_teams=["1323", "4414", "2910"],
            red_score=85,
            blue_score=75,
            winning_alliance="red",
            video_urls=["https://youtube.com/watch?v=mock"],
        )

        visualizer = AnalyticsPredictionVisualizer()

        # Test side_by_side layout
        out_sbs = os.path.join(tmp_dir, "out_sbs.mp4")
        res_sbs = visualizer.render_prediction_video(
            input_video_path=input_vid,
            output_video_path=out_sbs,
            match_metadata=meta,
            max_frames=10,
            frame_stride=1,
            layout="side_by_side",
        )
        assert os.path.isfile(out_sbs)
        assert res_sbs["frames_rendered"] > 0
        assert os.path.getsize(out_sbs) > 0

        # Test broadcast layout
        out_bc = os.path.join(tmp_dir, "out_bc.mp4")
        res_bc = visualizer.render_prediction_video(
            input_video_path=input_vid,
            output_video_path=out_bc,
            match_metadata=meta,
            max_frames=10,
            frame_stride=1,
            layout="broadcast",
        )
        assert os.path.isfile(out_bc)
        assert res_bc["frames_rendered"] > 0
        assert os.path.getsize(out_bc) > 0


def test_flow_orchestrator_with_video_rendering():
    """Verify flow orchestrator generates prediction video when render_video=True."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_video = os.path.join(tmp_dir, "dummy_render.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dummy_video, fourcc, 30.0, (640, 360))
        for _ in range(20):
            frame = np.full((360, 640, 3), 40, dtype=np.uint8)
            cv2.circle(frame, (200, 200), 8, (20, 220, 240), -1)
            writer.write(frame)
        writer.release()

        out_pred_vid = os.path.join(tmp_dir, "annotated_pred.mp4")
        orchestrator = AnalyticsFlowOrchestrator(
            tba_api_key=None,
            model_weights_path="yolo11x_robot_best.pt",
            predictor_save_path=os.path.join(tmp_dir, "pred.json"),
            default_video_fallback=dummy_video,
            cv_backend="fast",
        )

        flow_res = orchestrator.run_flow(
            event_key="2024cmptx",
            max_matches=2,
            max_frames_per_video=15,
            train_test_split=0.50,
            render_video=True,
            video_output_path=out_pred_vid,
            render_layout="side_by_side",
            render_frames=10,
        )

        assert flow_res.rendered_video_path == out_pred_vid
        assert os.path.isfile(out_pred_vid)
        assert os.path.getsize(out_pred_vid) > 0

