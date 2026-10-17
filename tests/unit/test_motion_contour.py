"""
Unit tests for OpenCV Dynamic Motion Contouring and Motion-Guided YOLO Verification.
"""

import numpy as np
import pytest

from src.vision.motion_contour import (
    DynamicMotionResult,
    DynamicMotionSegmenter,
    MotionGuidedYOLOVerifier,
)


def test_motion_segmenter_static_scene():
    """Verifies that an unchanging static background scene produces zero moving contours."""
    segmenter = DynamicMotionSegmenter(history=30, var_threshold=25.0)

    # 1080p top field slice (500x1920)
    static_frame = np.full((500, 1920, 3), 100, dtype=np.uint8)
    # Add static structures (e.g. gray line, hub funnel shape)
    static_frame[200:300, 800:1100] = (60, 60, 60)

    # Apply across 15 frames to let background model settle
    res = None
    for _ in range(15):
        res = segmenter.apply(static_frame, offset_y=80)

    assert res is not None
    assert isinstance(res, DynamicMotionResult)
    # No moving objects in static scene
    assert len(res.robot_motion_regions) == 0
    assert len(res.ball_motion_regions) == 0
    # Static mask covers the entire frame
    assert np.mean(res.static_mask == 255) > 0.95


def test_motion_segmenter_moving_robot():
    """Verifies that a moving rectangular colored body is detected as a robot motion region."""
    segmenter = DynamicMotionSegmenter(history=30, var_threshold=20.0, min_robot_area=800.0)

    bg = np.full((500, 1920, 3), 90, dtype=np.uint8)

    # Warm up background
    for _ in range(10):
        segmenter.apply(bg, offset_y=80)

    # Move a robot box (120x80) from x=500 to x=600 across frames
    res = None
    for step in range(8):
        frame = bg.copy()
        cur_x = 500 + step * 15
        cur_y = 300
        # Draw robot chassis + bumper
        frame[cur_y : cur_y + 80, cur_x : cur_x + 120] = (30, 30, 220)  # Red robot
        res = segmenter.apply(frame, offset_y=80)

    assert res is not None
    assert len(res.robot_motion_regions) >= 1
    box, area, energy = res.robot_motion_regions[0]
    # Check that the box covers the moving robot's position
    # Robot is moving across x=500..725, y=300..380 (+80 global offset)
    assert box[0] >= 480.0 and box[2] <= 760.0 and box[2] > box[0]
    assert box[1] >= 340.0 and box[3] <= 500.0 and box[3] > box[1]
    assert area >= 800.0
    assert energy > 0.10


def test_motion_segmenter_ball_vs_robot_separation():
    """Verifies that small rolling balls are segregated from large moving robots."""
    segmenter = DynamicMotionSegmenter(
        history=30,
        var_threshold=20.0,
        min_robot_area=1000.0,
        max_ball_area=1100.0,
        min_ball_area=15.0,
    )

    bg = np.full((500, 1920, 3), 90, dtype=np.uint8)
    for _ in range(10):
        segmenter.apply(bg, offset_y=80)

    # Frame with both a moving robot (130x85) and a small moving ball (radius 8, area ~200)
    res = None
    for step in range(8):
        frame = bg.copy()
        # Moving robot at x=1000..1130
        rx = 1000 + step * 15
        frame[280:365, rx : rx + 130] = (220, 110, 30)  # Blue robot

        # Moving ball at x=400..420
        bx = 400 + step * 25
        by = 250 - step * 10
        import cv2
        cv2.circle(frame, (bx, by), 9, (20, 230, 250), -1)  # Yellow ball

        res = segmenter.apply(frame, offset_y=80)

    assert res is not None
    # Robot regions should contain the large body
    assert len(res.robot_motion_regions) >= 1
    # Ball regions should contain the small sphere
    assert len(res.ball_motion_regions) >= 1

    ball_box, ball_area, _ = res.ball_motion_regions[0]
    assert ball_area <= 1100.0


def test_motion_history_decay_and_pause():
    """Verifies that a temporarily stationary robot remains active in motion history map."""
    segmenter = DynamicMotionSegmenter(history=30, motion_decay=0.96)

    bg = np.full((500, 1920, 3), 90, dtype=np.uint8)
    for _ in range(10):
        segmenter.apply(bg, offset_y=80)

    # Robot moves for 5 frames then stops for 10 frames
    for step in range(5):
        frame = bg.copy()
        frame[300:380, 500 + step * 10 : 620 + step * 10] = (30, 30, 220)
        segmenter.apply(frame, offset_y=80)

    # Robot pauses at x=550..670, y=300..380
    paused_frame = bg.copy()
    paused_frame[300:380, 550:670] = (30, 30, 220)

    test_box = np.array([550.0, 380.0, 670.0, 460.0], dtype=np.float32)  # global coords (y+80)

    # Check that energy persists after 10 frames of pause
    for _ in range(10):
        res = segmenter.apply(paused_frame, offset_y=80)

    energy = segmenter.get_motion_energy_at(test_box, offset_y=80)
    # Energy should still be significantly above zero due to decay memory
    assert energy > 0.08


def test_static_object_suppression():
    """Verifies that MotionGuidedYOLOVerifier suppresses static false positives with 0 motion."""
    segmenter = DynamicMotionSegmenter(history=30)
    verifier = MotionGuidedYOLOVerifier(min_dynamic_energy=0.03)

    bg = np.full((500, 1920, 3), 90, dtype=np.uint8)
    # Static hub logo drawn at x=1400..1500, y=300..370
    bg[300:370, 1400:1500] = (30, 30, 220)  # Red text on static background

    # Let background model absorb the static logo
    for _ in range(25):
        segmenter.apply(bg, offset_y=80)

    # Moving robot at x=700..820
    moving_frame = bg.copy()
    moving_frame[320:400, 700:820] = (220, 110, 30)
    segmenter.apply(moving_frame, offset_y=80)

    # Candidates: Candidate 1 is static logo (low conf 0.25), Candidate 2 is moving robot (conf 0.45)
    static_cand = (np.array([1400.0, 380.0, 1500.0, 450.0], dtype=np.float32), 0.25, "RED")
    moving_cand = (np.array([700.0, 400.0, 820.0, 480.0], dtype=np.float32), 0.45, "BLUE")

    filtered = verifier.filter_static_proposals(
        [static_cand, moving_cand],
        motion_segmenter=segmenter,
        offset_y=80,
    )

    # Static candidate should be suppressed; moving candidate retained!
    assert len(filtered) == 1
    assert filtered[0][2] == "BLUE"
    assert filtered[0][1] == 0.45
