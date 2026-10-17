"""Unit tests for Scoreboard OCR Foul Reconciliation and Attributed Score Synchronization.

Tests:
1. Clean target score calculation (GUI score - foul points).
2. Reconciliation deficit allocation to active shooter robots.
3. Reconciliation surplus pruning for low-confidence events.
4. Robot attribution sum matching clean target score invariant.
5. Scoreboard reader clean score calculation.
6. Automatic reconciliation via process_frame().
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.hub_scorer import HubAlliance, HubScoringDetector, ShotEvent
from src.analytics.scoreboard_reader import BroadcastScoreboardReader, ScoreboardReading


def test_clean_target_calculation() -> None:
    """Verifies clean target score calculation: max(0, gui_score - foul_points)."""
    scorer = HubScoringDetector()
    scorer.set_foul_points(red_fouls=5, blue_fouls=10)

    # Test standard case: Red GUI 42 - Foul 5 = 37, Blue GUI 50 - Foul 10 = 40
    attr = scorer.get_attributed_scores(gui_red_score=42, gui_blue_score=50)
    assert attr["RED"].gui_ocr_score == 42
    assert attr["RED"].foul_points == 5
    assert attr["RED"].clean_target_score == 37
    assert attr["BLUE"].gui_ocr_score == 50
    assert attr["BLUE"].foul_points == 10
    assert attr["BLUE"].clean_target_score == 40

    # Test edge case: foul points exceed GUI score
    attr_edge = scorer.get_attributed_scores(gui_red_score=3, gui_blue_score=8)
    assert attr_edge["RED"].clean_target_score == 0
    assert attr_edge["BLUE"].clean_target_score == 0


def test_reconciliation_deficit_allocation() -> None:
    """
    Verifies that when visual tracking under-counts (deficit = clean_target - current > 0),
    reconciliation allocates missing points to the active shooter robot at the Hub.
    """
    scorer = HubScoringDetector()
    scorer.set_hub_screen_bboxes(
        red_bbox=(900.0, 400.0, 1100.0, 560.0),
        blue_bbox=(300.0, 400.0, 500.0, 560.0),
    )
    scorer.set_foul_points(red_fouls=4, blue_fouls=0)

    # Simulate 3 existing scored shots for Red (3 points)
    for i in range(3):
        scorer.confirmed_shots.append(
            ShotEvent(
                event_id=f"initial_red_{i}",
                frame_index=100 + i,
                timestamp_s=10.0 + i * 0.1,
                phase_id=1,
                hub_alliance=HubAlliance.RED,
                ball_x=1000.0,
                ball_y=480.0,
                robot_track_id=1,
                robot_team="frc254",
                robot_alliance="RED",
                distance_ft=8.0,
                is_open=True,
                is_scored=True,
                score_increment=1,
                confidence=0.90,
            )
        )

    # GUI Red Score is 12, with 4 foul points -> Clean target = 8.
    # Current attributed = 3. Deficit = 8 - 3 = 5.
    tracked_robots = [
        {
            "track_id": 2,
            "team_number": "frc1678",
            "alliance": "RED",
            "center": (1010.0, 520.0),  # Close to Red Hub center (1000, 480)
            "state": "SCORING",
        },
        {
            "track_id": 3,
            "team_number": "frc971",
            "alliance": "RED",
            "center": (1500.0, 800.0),  # Far away
            "state": "DEFENSE",
        },
    ]

    attr = scorer.reconcile_score_attribution(
        gui_red_score=12,
        gui_blue_score=0,
        tracked_robots=tracked_robots,
        frame_idx=150,
        timestamp_s=15.0,
    )

    red_attr = attr["RED"]
    assert red_attr.clean_target_score == 8
    assert red_attr.current_attributed_score == 8
    assert red_attr.deficit_or_surplus == 0

    # 5 deficit points should have been attributed to frc1678 (active in SCORING state)
    assert red_attr.robot_scores["frc254"] == 3
    assert red_attr.robot_scores["frc1678"] == 5
    assert sum(red_attr.robot_scores.values()) == 8


def test_reconciliation_surplus_pruning() -> None:
    """
    Verifies that when visual tracking over-counts (surplus = current - clean_target > 0),
    reconciliation prunes the lowest confidence scored events.
    """
    scorer = HubScoringDetector()
    scorer.set_foul_points(red_fouls=2, blue_fouls=0)

    # Simulate 5 scored shots with varying confidences
    confidences = [0.95, 0.40, 0.85, 0.50, 0.90]
    for i, conf in enumerate(confidences):
        scorer.confirmed_shots.append(
            ShotEvent(
                event_id=f"red_shot_{i}",
                frame_index=200 + i,
                timestamp_s=20.0 + i,
                phase_id=2,
                hub_alliance=HubAlliance.RED,
                ball_x=1000.0,
                ball_y=480.0,
                robot_track_id=1,
                robot_team="frc254",
                robot_alliance="RED",
                distance_ft=7.5,
                is_open=True,
                is_scored=True,
                score_increment=1,
                confidence=conf,
            )
        )

    # Current scored = 5.
    # GUI Red Score is 5, with 2 fouls -> Clean target = 3.
    # Surplus = 2 shots to prune (lowest confidences: 0.40 and 0.50).
    attr = scorer.reconcile_score_attribution(
        gui_red_score=5,
        gui_blue_score=0,
        tracked_robots=[],
        frame_idx=210,
        timestamp_s=21.0,
    )

    red_attr = attr["RED"]
    assert red_attr.clean_target_score == 3
    assert red_attr.current_attributed_score == 3
    assert red_attr.deficit_or_surplus == 0
    assert red_attr.robot_scores["frc254"] == 3

    # Check that shots with conf 0.40 (idx 1) and 0.50 (idx 3) were pruned
    shot_1 = next(s for s in scorer.confirmed_shots if s.event_id == "red_shot_1")
    shot_3 = next(s for s in scorer.confirmed_shots if s.event_id == "red_shot_3")
    assert not shot_1.is_scored
    assert not shot_3.is_scored


def test_robot_attribution_sum_equals_clean_target() -> None:
    """Verifies that sum(robot_scores.values()) strictly matches clean_target_score."""
    scorer = HubScoringDetector()
    scorer.set_hub_screen_bboxes(
        red_bbox=(900.0, 400.0, 1100.0, 560.0),
        blue_bbox=(300.0, 400.0, 500.0, 560.0),
    )
    scorer.set_foul_points(red_fouls=6, blue_fouls=3)

    # Add initial shots for 2 robots on Blue
    scorer.confirmed_shots.extend(
        [
            ShotEvent(
                event_id="blue_1",
                frame_index=10,
                timestamp_s=5.0,
                phase_id=1,
                hub_alliance=HubAlliance.BLUE,
                ball_x=400.0,
                ball_y=480.0,
                robot_track_id=10,
                robot_team="frc1323",
                robot_alliance="BLUE",
                distance_ft=9.0,
                is_open=True,
                is_scored=True,
                score_increment=2,
                confidence=0.89,
            ),
            ShotEvent(
                event_id="blue_2",
                frame_index=20,
                timestamp_s=8.0,
                phase_id=1,
                hub_alliance=HubAlliance.BLUE,
                ball_x=400.0,
                ball_y=480.0,
                robot_track_id=11,
                robot_team="frc2910",
                robot_alliance="BLUE",
                distance_ft=11.0,
                is_open=True,
                is_scored=True,
                score_increment=2,
                confidence=0.91,
            ),
        ]
    )

    tracked_bots = [
        {"track_id": 10, "team_number": "frc1323", "alliance": "BLUE", "center": (410.0, 500.0), "state": "SCORING"},
        {"track_id": 11, "team_number": "frc2910", "alliance": "BLUE", "center": (420.0, 510.0), "state": "INTAKE"},
    ]

    # Blue GUI 15, Blue Fouls 3 -> Clean target = 12.
    # Current attributed = 4. Deficit = 8.
    attr = scorer.reconcile_score_attribution(
        gui_red_score=0,
        gui_blue_score=15,
        tracked_robots=tracked_bots,
        frame_idx=30,
        timestamp_s=10.0,
    )

    blue_attr = attr["BLUE"]
    assert blue_attr.clean_target_score == 12
    assert blue_attr.current_attributed_score == 12
    assert sum(blue_attr.robot_scores.values()) == 12


def test_scoreboard_reader_clean_scores() -> None:
    """Verifies BroadcastScoreboardReader clean score computation."""
    reader = BroadcastScoreboardReader()
    reader.set_foul_points(red_fouls=12, blue_fouls=7)

    # Mock an empty frame
    mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    reading = reader.read_frame(mock_frame, frame_idx=0, timestamp_s=0.0)

    # Initial zero scores
    assert reading.clean_red_score == 0
    assert reading.clean_blue_score == 0
    assert reading.red_foul_points == 12
    assert reading.blue_foul_points == 7

    # Simulate updated current scores
    reader._current_red_score = 45
    reader._current_blue_score = 30
    reading_updated = reader.read_frame(mock_frame, frame_idx=1, timestamp_s=0.033)

    assert reading_updated.clean_red_score == 45 - 12  # 33
    assert reading_updated.clean_blue_score == 30 - 7   # 23


def test_process_frame_auto_reconcile() -> None:
    """Verifies that passing foul_points and gui_total_scores to process_frame auto-reconciles."""
    scorer = HubScoringDetector()
    scorer.set_hub_screen_bboxes(
        red_bbox=(900.0, 400.0, 1100.0, 560.0),
        blue_bbox=(300.0, 400.0, 500.0, 560.0),
    )

    robots = [
        {"track_id": 1, "team_number": "frc254", "alliance": "RED", "center": (980.0, 470.0), "state": "SCORING"},
    ]

    # Process frame with GUI totals (Red 10, Blue 0) and fouls (Red 3, Blue 0) -> Clean Red target = 7
    scorer.process_frame(
        frame_idx=10,
        timestamp_s=10.0,
        ball_positions=[],
        tracked_robots=robots,
        gui_total_scores=(10, 0),
        foul_points=(3, 0),
        auto_reconcile=True,
    )

    attr = scorer.get_attributed_scores()
    assert attr["RED"].clean_target_score == 7
    assert attr["RED"].current_attributed_score == 7
    assert attr["RED"].deficit_or_surplus == 0
    assert attr["RED"].robot_scores["frc254"] == 7
