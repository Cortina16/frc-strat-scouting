"""
Unit test suite for Dynamic Robot Detection, Dual-Tile Extraction,
Alliance-Decoupled NMS, and Bumper-Guided Proposal Fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.types import BoundingBox
from src.vision.detector_trt import (
    classify_bumper_fabric,
    dynamic_waterfall_selection,
    extract_top_screen_tiles,
    recover_missing_bumper_proposals,
    suppress_duplicate_boxes,
)


def test_extract_top_screen_tiles() -> None:
    """Verifies that top-screen dual-tile extraction creates overlapping tiles with correct offsets."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tiles, offsets = extract_top_screen_tiles(
        frame,
        roi_y=(80, 660),
        tile_left_x=(0, 1100),
        tile_right_x=(820, 1920),
    )

    assert len(tiles) == 2
    assert len(offsets) == 2

    # Left tile: height = 660 - 80 = 580, width = 1100
    assert tiles[0].shape == (580, 1100, 3)
    assert offsets[0] == (0, 80)

    # Right tile: height = 580, width = 1920 - 820 = 1100
    assert tiles[1].shape == (580, 1100, 3)
    assert offsets[1] == (820, 80)


def test_classify_bumper_fabric() -> None:
    """Verifies colorimetric classification of Red and Blue bumper fabric patches."""
    # Create synthetic red bumper patch (BGR: high R, low B and G)
    red_patch = np.zeros((30, 80, 3), dtype=np.uint8)
    red_patch[:, :, 2] = 220  # Red channel
    red_patch[:, :, 1] = 25   # Green channel
    red_patch[:, :, 0] = 25   # Blue channel

    r_score, r_al = classify_bumper_fabric(red_patch)
    assert r_al == "RED"
    assert r_score > 0.50

    # Create synthetic blue bumper patch (BGR: high B, low R and G)
    blue_patch = np.zeros((30, 80, 3), dtype=np.uint8)
    blue_patch[:, :, 0] = 220  # Blue channel
    blue_patch[:, :, 1] = 40   # Green channel
    blue_patch[:, :, 2] = 25   # Red channel

    b_score, b_al = classify_bumper_fabric(blue_patch)
    assert b_al == "BLUE"
    assert b_score > 0.50

    # Gray neutral carpet patch
    gray_patch = np.full((30, 80, 3), 100, dtype=np.uint8)
    _, g_al = classify_bumper_fabric(gray_patch)
    assert g_al == "UNKNOWN"


def test_suppress_duplicate_boxes_alliance_aware() -> None:
    """
    Verifies that duplicate detections on the same robot are suppressed,
    while touching robots of opposing alliances in scrums/defense are BOTH preserved.
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Draw Red robot at (450, 400) - box: [400..500]
    frame[360:440, 400:500] = (30, 30, 30)
    frame[415:440, 400:475] = (20, 20, 220)  # Red bumper on left side

    # Draw Blue robot at (490, 400) - box: [440..540], overlapping Red robot (IoU ~ 0.52)
    frame[360:440, 490:540] = (30, 30, 30)
    frame[415:440, 475:540] = (220, 30, 20)  # Blue bumper on right side

    # Red box: center_x=450
    box_red = BoundingBox(center_x=450.0, center_y=400.0, width=100.0, height=80.0, confidence=0.85, class_id=0)
    # Duplicate red box: center_x=455 (IoU > 0.75)
    box_red_dup = BoundingBox(center_x=455.0, center_y=402.0, width=98.0, height=78.0, confidence=0.70, class_id=0)
    # Blue box overlapping red box: center_x=490 (IoU ~ 0.52)
    box_blue = BoundingBox(center_x=490.0, center_y=400.0, width=100.0, height=80.0, confidence=0.80, class_id=0)

    boxes = [box_red, box_red_dup, box_blue]
    suppressed = suppress_duplicate_boxes(boxes, iou_threshold=0.45, max_boxes=6, frame_bgr=frame)

    # box_red_dup should be suppressed, but box_blue and box_red must BOTH be preserved!
    assert len(suppressed) == 2
    assert box_red in suppressed
    assert box_blue in suppressed
    assert box_red_dup not in suppressed


def test_dynamic_waterfall_selection() -> None:
    """
    Verifies that dynamic waterfall selection enforces alliance quotas
    (up to 3 Red, up to 3 Blue, total 6 robots).
    """
    # 5 Red candidates with varying confidences
    red_cands = [
        (BoundingBox(center_x=100.0 * i, center_y=300.0, width=60.0, height=50.0, confidence=0.90 - i * 0.1, class_id=0),
         0.90 - i * 0.1, "RED")
        for i in range(1, 6)
    ]

    # 3 Blue candidates with lower confidences
    blue_cands = [
        (BoundingBox(center_x=1000.0 + 100.0 * i, center_y=300.0, width=60.0, height=50.0, confidence=0.40 - i * 0.1, class_id=0),
         0.40 - i * 0.1, "BLUE")
        for i in range(1, 4)
    ]

    all_cands = red_cands + blue_cands
    selected = dynamic_waterfall_selection(all_cands, target_red=3, target_blue=3, max_total=6)

    # Exactly 6 robots should be selected: top 3 Red and all 3 Blue
    assert len(selected) == 6

    # Verify that exactly 3 Red candidates were chosen (not 5 Red)
    red_selected = [b for b in selected if b.center_x < 800.0]
    blue_selected = [b for b in selected if b.center_x >= 800.0]

    assert len(red_selected) == 3
    assert len(blue_selected) == 3


def test_recover_missing_bumper_proposals() -> None:
    """Verifies that colorimetric bumper segmentation recovers missing robot bounding boxes."""
    frame = np.full((1080, 1920, 3), 80, dtype=np.uint8)  # Gray carpet

    # Draw Blue bumper on the field (y in [250, 280], x in [400, 500])
    frame[250:280, 400:500, 0] = 210  # High B
    frame[250:280, 400:500, 1] = 40
    frame[250:280, 400:500, 2] = 20

    existing_boxes = [
        BoundingBox(center_x=1200.0, center_y=400.0, width=100.0, height=80.0, confidence=0.80, class_id=0)
    ]

    recovered = recover_missing_bumper_proposals(
        frame_bgr=frame,
        existing_boxes=existing_boxes,
        target_alliance="BLUE",
        field_roi_y=(150, 650),
    )

    assert len(recovered) >= 1
    # Check that recovered box is centered near (450, 265)
    rec_box = recovered[0]
    assert abs(rec_box.center_x - 450.0) < 30.0
    assert abs(rec_box.center_y - 265.0) < 30.0
    assert rec_box.width >= 80.0
    assert rec_box.height >= 60.0
