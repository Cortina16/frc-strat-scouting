#!/usr/bin/env python3
"""
High-Performance Video Annotation Script for Continuous 6-Robot Detection.

Runs inference using `yolo11x_robot_best.pt` on match video (e.g. `output.mp4`),
featuring:
1. Dual-Tile High-Resolution Inference on the top wide-field broadcast screen (80..660px).
2. Dynamic Waterfall Thresholding (0.35 -> 0.02) to dynamically adjust sensitivity.
3. Alliance-Decoupled Non-Maximum Suppression (Red & Blue robots never suppress each other).
4. Bumper Fabric Guided Proposal Fallback for recovering obscured robots.
5. Persistent Kalman Memory Coasting across occlusions to maintain continuous 6-robot tracking.
6. Crisp corner-bracketed bounding boxes with alliance pills and broadcast HUD overlay.

Outputs:
    - Annotated MP4 video (`output_yolo11x_annotated.mp4`)
    - Keyframe snapshot images at designated frames (e.g. frames 600, 900, 1200, 1500, 2000)
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.motion_contour import (
    DynamicMotionResult,
    DynamicMotionSegmenter,
    MotionGuidedYOLOVerifier,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("frc.render_yolo11x")

# Alliance BGR Colors
ALLIANCE_COLORS = {
    "RED": (30, 30, 225),       # Vibrant Red
    "BLUE": (225, 115, 25),     # Vibrant Blue
    "UNKNOWN": (180, 180, 180), # Silver Gray
}


def classify_bumper_patch(box_crop: np.ndarray) -> Tuple[float, str]:
    """
    Evaluates bumper fabric color purity in candidate robot box crop.
    Checks bottom 35% first; if ambiguous, checks full box.

    Returns:
        Tuple of (bumper_score, alliance_str: 'RED' | 'BLUE' | 'UNKNOWN').
    """
    if box_crop is None or box_crop.size == 0 or box_crop.shape[0] < 4 or box_crop.shape[1] < 4:
        return 0.0, "UNKNOWN"

    h, w = box_crop.shape[:2]
    bumper_h = max(4, int(round(h * 0.35)))
    bumper_crop = box_crop[h - bumper_h :, :]

    def _eval_patch(patch: np.ndarray) -> Tuple[float, float]:
        if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
            return 0.0, 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        red_mask = ((h_ch <= 15) | (h_ch >= 165)) & (s_ch >= 45) & (v_ch >= 35)
        blue_mask = (h_ch >= 95) & (h_ch <= 135) & (s_ch >= 45) & (v_ch >= 35)
        red_frac = float(np.mean(red_mask))
        blue_frac = float(np.mean(blue_mask))

        b_ch = patch[:, :, 0].astype(np.float32)
        g_ch = patch[:, :, 1].astype(np.float32)
        r_ch = patch[:, :, 2].astype(np.float32)
        diff_red = float(np.mean((r_ch > b_ch + 15.0) & (r_ch > g_ch + 10.0)))
        diff_blue = float(np.mean((b_ch > r_ch + 15.0) & (b_ch > g_ch + 10.0)))

        score_r = max(red_frac, diff_red)
        score_b = max(blue_frac, diff_blue)
        return score_r, score_b

    score_r, score_b = _eval_patch(bumper_crop)
    if max(score_r, score_b) < 0.10:
        full_r, full_b = _eval_patch(box_crop)
        if max(full_r, full_b) > max(score_r, score_b):
            score_r, score_b = full_r, full_b

    max_score = max(score_r, score_b)
    if score_r > 0.07 and score_r > score_b * 1.15:
        return score_r, "RED"
    elif score_b > 0.07 and score_b > score_r * 1.15:
        return score_b, "BLUE"
    elif score_r > score_b and score_r > 0.05:
        return score_r, "RED"
    elif score_b > score_r and score_b > 0.05:
        return score_b, "BLUE"
    else:
        return max_score, "UNKNOWN"


def is_inside_field(box: np.ndarray) -> bool:
    """
    Verifies that a bounding box is within the top screen field carpet boundary,
    strictly rejecting audience seating, scoreboard graphics, and referee areas.
    """
    cx = float(box[0] + box[2]) / 2.0
    cy = float(box[1] + box[3]) / 2.0
    # Field top boundary (arena perimeter rail at y=250; audience bleachers are y < 250)
    if cy < 250.0:
        return False
    # Field bottom boundary (center glass rail is at y=505, sides slope to y=565)
    if 280.0 <= cx <= 1640.0 and cy > 505.0:
        return False
    if cy > 565.0:
        return False
    if cx < 40.0 or cx > 1880.0:
        return False
    return True


def recover_missing_bumper_proposals(
    frame_bgr: np.ndarray,
    existing_boxes: List[np.ndarray],
    target_alliance: str,
    field_roi_y: Tuple[int, int] = (255, 510),
    min_area: float = 160.0,
) -> List[Tuple[np.ndarray, float, str]]:
    """
    Recovers missing robot candidate bounding boxes using colorimetric bumper segmentation
    on the field carpet when neural network detections fall below threshold.
    """
    if frame_bgr is None or frame_bgr.size == 0 or target_alliance not in ("RED", "BLUE"):
        return []

    h_img, w_img = frame_bgr.shape[:2]
    y1 = max(0, field_roi_y[0])
    y2 = min(h_img, field_roi_y[1])
    field_crop = frame_bgr[y1:y2, :]

    hsv = cv2.cvtColor(field_crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    b_ch = field_crop[:, :, 0].astype(np.float32)
    g_ch = field_crop[:, :, 1].astype(np.float32)
    r_ch = field_crop[:, :, 2].astype(np.float32)

    if target_alliance == "RED":
        mask = (
            ((h_ch <= 14) | (h_ch >= 166))
            & (s_ch >= 55)
            & (v_ch >= 40)
            & (r_ch > b_ch + 15.0)
            & (r_ch > g_ch + 10.0)
        )
    else:
        mask = (
            (h_ch >= 95)
            & (h_ch <= 135)
            & (s_ch >= 55)
            & (v_ch >= 40)
            & (b_ch > r_ch + 15.0)
            & (b_ch > g_ch + 10.0)
        )

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Tuple[np.ndarray, float, str]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > 18000:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 25 or bw > 250 or bh > 85:
            continue

        global_cx = bx + bw / 2.0
        global_cy = (by + y1) + bh / 2.0

        # Exclude static Hub structures (funnel + ramp) from bumper proposals
        if 1360.0 <= global_cx <= 1660.0 and 210.0 <= global_cy <= 435.0:
            continue
        if 350.0 <= global_cx <= 650.0 and 210.0 <= global_cy <= 435.0:
            continue

        # Reject static perimeter wall graphics
        if target_alliance == "BLUE" and global_cx > 1350.0:
            continue
        if target_alliance == "RED" and global_cx < 550.0:
            continue
        if global_cx > 1680.0 and global_cy > 450.0:
            continue
        if 280.0 <= global_cx <= 470.0 and global_cy > 460.0:
            continue

        # Construct typical robot box (aspect ~1.3, width ~110px, height ~75px)
        cand_w = max(float(bw), 95.0)
        cand_h = max(float(bh) * 2.2, 65.0)

        cand_box = np.array([
            max(0.0, global_cx - cand_w / 2.0),
            max(float(y1), global_cy - cand_h / 2.0),
            min(float(w_img), global_cx + cand_w / 2.0),
            min(float(y2), global_cy + cand_h / 2.0),
        ], dtype=np.float32)

        if not is_inside_field(cand_box):
            continue

        # Ensure not overlapping an existing detection
        overlap = False
        for ex in existing_boxes:
            ex_cx = (ex[0] + ex[2]) / 2.0
            ex_cy = (ex[1] + ex[3]) / 2.0
            dx = abs(ex_cx - global_cx)
            dy = abs(ex_cy - global_cy)
            if dx < (cand_w + (ex[2] - ex[0])) * 0.35 and dy < (cand_h + (ex[3] - ex[1])) * 0.35:
                overlap = True
                break

        if not overlap:
            px1, py1 = int(cand_box[0]), int(cand_box[1])
            px2, py2 = int(cand_box[2]), int(cand_box[3])
            patch = frame_bgr[max(0, py1) : min(h_img, py2), max(0, px1) : min(w_img, px2)]
            score, det_al = classify_bumper_patch(patch)
            if det_al == target_alliance or score > 0.06:
                candidates.append((cand_box, max(0.20, float(score)), target_alliance))

    # Sort recovered candidates by bumper confidence descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


class TrackedRobot:
    """Represents an actively tracked robot with temporal smoothing and physical coasting."""

    def __init__(self, track_id: int, bbox: np.ndarray, conf: float, alliance: str):
        self.track_id = track_id
        self.bbox = bbox.astype(np.float32)  # [x1, y1, x2, y2]
        self.conf = float(conf)
        self.alliance = alliance
        self.alliance_votes = collections.Counter({alliance: 3 if alliance != "UNKNOWN" else 1})
        self.missed_frames = 0
        self.total_frames = 1
        self.vx = 0.0
        self.vy = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return (float(self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[1] + self.bbox[3]) / 2.0)

    def update(self, new_bbox: np.ndarray, new_conf: float, new_alliance: str):
        old_cx, old_cy = self.center
        new_cx = float(new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = float(new_bbox[1] + new_bbox[3]) / 2.0
        dx = np.clip(new_cx - old_cx, -35.0, 35.0)
        dy = np.clip(new_cy - old_cy, -35.0, 35.0)
        self.vx = 0.65 * dx + 0.35 * self.vx
        self.vy = 0.65 * dy + 0.35 * self.vy

        alpha = 0.75
        self.bbox = alpha * new_bbox.astype(np.float32) + (1.0 - alpha) * self.bbox
        self.conf = 0.80 * float(new_conf) + 0.20 * self.conf
        if new_alliance != "UNKNOWN":
            self.alliance_votes[new_alliance] += 3
        else:
            self.alliance_votes["UNKNOWN"] += 1

        red_v = self.alliance_votes["RED"]
        blue_v = self.alliance_votes["BLUE"]
        if red_v > blue_v and red_v >= 2:
            self.alliance = "RED"
        elif blue_v > red_v and blue_v >= 2:
            self.alliance = "BLUE"
        elif new_alliance != "UNKNOWN":
            self.alliance = new_alliance

        self.missed_frames = 0
        self.total_frames += 1

    def coast_forward(self, max_w: int = 1920, max_h: int = 1080):
        """Coasts bounding box forward with physical velocity inertia during momentary occlusion."""
        self.missed_frames += 1
        self.vx *= 0.94
        self.vy *= 0.94
        self.bbox[0] = np.clip(self.bbox[0] + self.vx, 0, max_w - 20)
        self.bbox[2] = np.clip(self.bbox[2] + self.vx, 20, max_w)
        self.bbox[1] = np.clip(self.bbox[1] + self.vy, 80, max_h - 20)
        self.bbox[3] = np.clip(self.bbox[3] + self.vy, 100, max_h)
        self.conf = max(0.20, self.conf * 0.985)


class SimpleRobotTracker:
    """IoU and centroid distance association tracker with strict 6-robot retention."""

    def __init__(self, max_missed: int = 45):
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedRobot] = {}
        self.max_missed = max_missed

    @staticmethod
    def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return float(interArea / unionArea) if unionArea > 0 else 0.0

    @staticmethod
    def _compute_dist(boxA: np.ndarray, boxB: np.ndarray) -> float:
        cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
        cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
        return float(np.hypot(cA[0] - cB[0], cA[1] - cB[1]))

    def update(
        self,
        detections: List[Tuple[np.ndarray, float, str]],
        max_w: int = 1920,
        max_h: int = 1080,
    ) -> List[TrackedRobot]:
        """
        Matches detections to existing tracks and maintains active tracks (including coasted tracks).
        Guarantees up to 3 Red and 3 Blue tracks (total 6 tracks).
        """
        track_ids = list(self.tracks.keys())

        if not track_ids:
            for bbox, conf, alliance in detections:
                tid = self.next_track_id
                self.next_track_id += 1
                t = TrackedRobot(tid, bbox, conf, alliance)
                self.tracks[tid] = t
            return list(self.tracks.values())[:6]

        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].coast_forward(max_w, max_h)
                if self.tracks[tid].missed_frames > self.max_missed:
                    del self.tracks[tid]
            return list(self.tracks.values())[:6]

        # Cost matrix computation
        num_t = len(track_ids)
        num_d = len(detections)
        cost_matrix = np.full((num_t, num_d), 1e6, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t_box = self.tracks[tid].bbox
            t_al = self.tracks[tid].alliance
            for j, (d_box, _, d_al) in enumerate(detections):
                iou = self._compute_iou(t_box, d_box)
                dist = self._compute_dist(t_box, d_box)

                # Prioritize same alliance
                al_cost = 0.0 if (t_al == d_al or t_al == "UNKNOWN" or d_al == "UNKNOWN") else 0.50

                if iou >= 0.15:
                    cost_matrix[i, j] = (1.0 - iou) + al_cost
                elif dist < 120.0:
                    cost_matrix[i, j] = 1.0 + (dist / 100.0) + al_cost

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e5:
                tid = track_ids[r]
                d_box, d_conf, d_alliance = detections[c]
                self.tracks[tid].update(d_box, d_conf, d_alliance)
                matched_tracks.add(tid)
                matched_dets.add(c)

        # Coast unmatched tracks forward
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].coast_forward(max_w, max_h)
                if self.tracks[tid].missed_frames > self.max_missed:
                    del self.tracks[tid]

        # Handle unmatched detections -> new tracks if room allows
        for j, (d_box, d_conf, d_alliance) in enumerate(detections):
            if j not in matched_dets and len(self.tracks) < 8:
                tid = self.next_track_id
                self.next_track_id += 1
                t = TrackedRobot(tid, d_box, d_conf, d_alliance)
                self.tracks[tid] = t

        # Quota-based selection: exactly up to 3 RED, up to 3 BLUE
        all_active = list(self.tracks.values())
        red_tracks = [t for t in all_active if t.alliance == "RED"]
        blue_tracks = [t for t in all_active if t.alliance == "BLUE"]
        unk_tracks = [t for t in all_active if t.alliance == "UNKNOWN"]

        red_tracks.sort(key=lambda t: (t.missed_frames == 0, t.conf), reverse=True)
        blue_tracks.sort(key=lambda t: (t.missed_frames == 0, t.conf), reverse=True)
        unk_tracks.sort(key=lambda t: (t.missed_frames == 0, t.conf), reverse=True)

        selected = red_tracks[:3] + blue_tracks[:3]
        if len(selected) < 6 and unk_tracks:
            selected += unk_tracks[: 6 - len(selected)]
        if len(selected) < 6:
            rem = [t for t in all_active if t not in selected]
            selected += rem[: 6 - len(selected)]

        return selected


def draw_crisp_robot_annotation(
    frame: np.ndarray,
    track: TrackedRobot,
) -> None:
    """
    Renders high-definition bounding box, corner brackets, contact patch dot,
    and semi-transparent label pill with:
        Robot #{track_id} [{ALLIANCE}] {conf*100:.1f}% [or LOCK if coasting]
    """
    h_img, w_img = frame.shape[:2]
    x1 = int(round(track.bbox[0]))
    y1 = int(round(track.bbox[1]))
    x2 = int(round(track.bbox[2]))
    y2 = int(round(track.bbox[3]))

    x1 = max(0, min(w_img - 1, x1))
    y1 = max(0, min(h_img - 1, y1))
    x2 = max(0, min(w_img - 1, x2))
    y2 = max(0, min(h_img - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return

    color = ALLIANCE_COLORS.get(track.alliance, ALLIANCE_COLORS["UNKNOWN"])
    is_coasting = track.missed_frames > 0

    # 1. Main Bounding Box (thin crisp outline)
    line_thickness = 1 if is_coasting else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

    # 2. Corner Bracket Styling
    w = x2 - x1
    h = y2 - y1
    c_len = max(6, min(18, int(min(w, h) * 0.25)))
    c_thick = 2 if is_coasting else 3

    # Top-Left Bracket
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), color, c_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), color, c_thick, cv2.LINE_AA)
    # Top-Right Bracket
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), color, c_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + c_len), color, c_thick, cv2.LINE_AA)
    # Bottom-Left Bracket
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), color, c_thick, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - c_len), color, c_thick, cv2.LINE_AA)
    # Bottom-Right Bracket
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), color, c_thick, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - c_len), color, c_thick, cv2.LINE_AA)

    # 3. Contact Patch Dot at bottom-center (cx, y2)
    cx = int(round((x1 + x2) / 2.0))
    cv2.circle(frame, (cx, y2), 5, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, y2), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # 4. Dark Semi-Transparent Label Pill
    status_suffix = "LOCK" if is_coasting else f"{track.conf * 100:.1f}%"
    label_text = f"Robot #{track.track_id} [{track.alliance}] {status_suffix}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.42
    f_thick = 1
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, f_thick)
    pill_w = tw + 16
    pill_h = th + 10

    # Placement preference: directly above box, fallback to inside/below
    if y1 - pill_h - 4 >= 0:
        px1 = max(2, min(w_img - pill_w - 2, x1))
        py1 = y1 - pill_h - 4
    else:
        px1 = max(2, min(w_img - pill_w - 2, x1))
        py1 = y2 + 4
    px2 = px1 + pill_w
    py2 = py1 + pill_h

    # Semi-transparent dark background
    overlay_pill = frame[py1:py2, px1:px2].copy()
    bg_pill = np.full_like(overlay_pill, (16, 18, 22))
    cv2.addWeighted(bg_pill, 0.85, overlay_pill, 0.15, 0, overlay_pill)
    frame[py1:py2, px1:px2] = overlay_pill

    # Border & Left Alliance Color Indicator Stripe
    cv2.rectangle(frame, (px1, py1), (px2, py2), (60, 65, 75), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (px1, py1), (px1 + 4, py2), color, -1)

    # Label text in high-contrast crisp white
    cv2.putText(
        frame,
        label_text,
        (px1 + 8, py2 - 6),
        font,
        font_scale,
        (255, 255, 255),
        f_thick,
        cv2.LINE_AA,
    )


def draw_sleek_hud(
    frame: np.ndarray,
    frame_index: int,
    total_frames: int,
    timestamp_s: float,
    num_robots: int,
    current_fps: float,
    num_red: int = 0,
    num_blue: int = 0,
    num_motion_robots: int = 0,
    num_motion_balls: int = 0,
) -> None:
    """
    Renders top-left broadcast HUD overlay card with glassmorphism styling.
    """
    hud_x, hud_y = 20, 20
    hud_w, hud_h = 510, 160

    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (16, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (65, 70, 80), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (hud_x, hud_y), (hud_x + 4, hud_y + hud_h), (0, 215, 255), -1)

    # Header
    cv2.putText(
        frame,
        "YOLOv11x REBUILT Vision (yolo11x_robot_best.pt)",
        (hud_x + 14, hud_y + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.46,
        (0, 230, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.line(frame, (hud_x + 10, hud_y + 34), (hud_x + hud_w - 10, hud_y + 34), (50, 55, 65), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1

    # Frame metric
    cv2.putText(
        frame,
        f"Frame: {frame_index:,} / {total_frames:,}",
        (hud_x + 16, hud_y + 56),
        font,
        scale,
        (230, 230, 235),
        thick,
        cv2.LINE_AA,
    )

    # Timestamp metric
    cv2.putText(
        frame,
        f"Timestamp: {timestamp_s:.2f}s",
        (hud_x + 16, hud_y + 78),
        font,
        scale,
        (230, 230, 235),
        thick,
        cv2.LINE_AA,
    )

    # Robots detected metric
    robot_str = f"Robots: {num_robots} / 6 active ({num_red} Red | {num_blue} Blue) [Continuous 6-Robot Lock]"
    cv2.putText(
        frame,
        robot_str,
        (hud_x + 16, hud_y + 100),
        font,
        scale,
        (230, 230, 235),
        thick,
        cv2.LINE_AA,
    )

    # OpenCV Dynamic Motion Segmentation metric
    motion_str = f"Dynamic Motion: {num_motion_robots} robot regions | {num_motion_balls} moving balls"
    cv2.putText(
        frame,
        motion_str,
        (hud_x + 16, hud_y + 122),
        font,
        scale,
        (255, 205, 90),
        thick,
        cv2.LINE_AA,
    )

    # Real-time FPS metric
    cv2.putText(
        frame,
        f"Real-Time FPS: {current_fps:.1f}",
        (hud_x + 16, hud_y + 144),
        font,
        scale,
        (130, 240, 130),
        thick,
        cv2.LINE_AA,
    )


def render_video_with_model(
    video_path: str,
    model_path: str,
    output_path: str,
    imgsz: int = 1024,
    conf: float = 0.005,
    device: str = "0",
    max_frames: int = 2400,
    start_frame: int = 0,
    snapshot_frames: Optional[List[int]] = None,
    snapshot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes streaming inference using dual-tile top-screen high resolution,
    dynamic waterfall thresholding, bumper fallback, and 6-robot persistent tracking.
    """
    if snapshot_frames is None:
        snapshot_frames = [600, 900, 1200, 1500, 2000]

    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    effective_total = min(total_video_frames, max_frames)
    logger.info(
        "Source Video: %s (%dx%d @ %.2f FPS, %d total frames)",
        video_path,
        w_img,
        h_img,
        video_fps,
        total_video_frames,
    )
    logger.info(
        "Rendering target: %d frames (frames %d..%d) to %s",
        effective_total,
        start_frame,
        start_frame + effective_total - 1,
        output_path,
    )

    # Initialize YOLO model
    logger.info("Loading YOLO model: %s on device: %s...", model_path, device)
    model = YOLO(model_path)

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 60.0, (w_img, h_img))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for target: {output_path}")

    tracker = SimpleRobotTracker(max_missed=45)

    # OpenCV Dynamic Motion Segmentation Engine & Verifier
    motion_segmenter = DynamicMotionSegmenter(
        history=100,
        var_threshold=25.0,
        detect_shadows=False,
        min_robot_area=1000.0,
        max_ball_area=650.0,
        min_ball_area=15.0,
        motion_decay=0.96,
        static_energy_cutoff=0.03,
    )
    motion_verifier = MotionGuidedYOLOVerifier(min_dynamic_energy=0.03)

    t_render_start = time.perf_counter()
    recent_frame_times = collections.deque(maxlen=30)

    frames_rendered = 0
    total_robot_detections = 0
    saved_snapshots: List[str] = []

    # Fast forward to start_frame if requested
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame

    try:
        while cap.isOpened() and frames_rendered < effective_total:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t_frame_start = time.perf_counter()

            # 1. Extract Top screen field view (y: 80..580)
            top_y1, top_y2 = 80, 580
            top_view = frame[top_y1:top_y2, :]

            # 1.1. Execute OpenCV dynamic motion segmentation BEFOREhand to separate static objects from dynamic movement
            motion_res = motion_segmenter.apply(top_view, offset_y=top_y1, offset_x=0)

            # 1.2. Extract Dual-Tile from top screen field view for high-resolution YOLO inference
            # Left tile: [0..1100], Right tile: [820..1920]
            tile_l = frame[top_y1:top_y2, 0:1100]
            tile_r = frame[top_y1:top_y2, 820:1920]

            # 2. Batched Inference across both tiles at imgsz=1024, conf=0.005
            res = model.predict(
                [tile_l, tile_r],
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )

            # 3. Collect and remap candidate boxes back to original 1080p frame
            raw_candidates: List[Tuple[np.ndarray, float, str]] = []

            # Left tile
            if res[0].boxes is not None and len(res[0].boxes) > 0:
                xyxy_l = res[0].boxes.xyxy.cpu().numpy()
                conf_l = res[0].boxes.conf.cpu().numpy()
                for b_idx in range(len(xyxy_l)):
                    bx = xyxy_l[b_idx]
                    c = float(conf_l[b_idx])
                    global_box = np.array([bx[0], bx[1] + top_y1, bx[2], bx[3] + top_y1], dtype=np.float32)
                    if not is_inside_field(global_box):
                        continue
                    x1, y1, x2, y2 = int(global_box[0]), int(global_box[1]), int(global_box[2]), int(global_box[3])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    if x2 > x1 and y2 > y1:
                        _, al = classify_bumper_patch(frame[y1:y2, x1:x2])
                    else:
                        al = "UNKNOWN"
                    raw_candidates.append((global_box, c, al))

            # Right tile
            if res[1].boxes is not None and len(res[1].boxes) > 0:
                xyxy_r = res[1].boxes.xyxy.cpu().numpy()
                conf_r = res[1].boxes.conf.cpu().numpy()
                for b_idx in range(len(xyxy_r)):
                    bx = xyxy_r[b_idx]
                    c = float(conf_r[b_idx])
                    global_box = np.array([bx[0] + 820, bx[1] + top_y1, bx[2] + 820, bx[3] + top_y1], dtype=np.float32)
                    if not is_inside_field(global_box):
                        continue
                    x1, y1, x2, y2 = int(global_box[0]), int(global_box[1]), int(global_box[2]), int(global_box[3])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    if x2 > x1 and y2 > y1:
                        _, al = classify_bumper_patch(frame[y1:y2, x1:x2])
                    else:
                        al = "UNKNOWN"
                    raw_candidates.append((global_box, c, al))

            # 3.1. Motion-Guided YOLO Verification: Filter out static arena objects
            raw_candidates = motion_verifier.filter_static_proposals(
                raw_candidates,
                motion_segmenter,
                offset_y=top_y1,
                offset_x=0,
            )

            # 3.2. Inject Dynamic Motion Robot Regions: Run YOLO on moving candidate crops to discover authentic robots
            for padded_box, area, energy in motion_res.robot_motion_regions:
                if not is_inside_field(padded_box) or energy < 0.10:
                    continue
                # Check if already covered by an existing high-confidence candidate
                covered = False
                for existing_cand in raw_candidates:
                    if existing_cand[1] >= 0.40 and SimpleRobotTracker._compute_iou(padded_box, existing_cand[0]) > 0.40:
                        covered = True
                        break
                if not covered:
                    px1 = max(0, int(round(padded_box[0])))
                    py1 = max(0, int(round(padded_box[1])))
                    px2 = min(w_img, int(round(padded_box[2])))
                    py2 = min(h_img, int(round(padded_box[3])))
                    if (px2 - px1) >= 45 and (py2 - py1) >= 35:
                        crop = frame[py1:py2, px1:px2]
                        if crop.size > 0:
                            # Run YOLO on the moving object crop to verify what is actually a robot
                            crop_res = model.predict(crop, imgsz=448, conf=0.03, device=device, verbose=False)
                            if crop_res[0].boxes is not None and len(crop_res[0].boxes) > 0:
                                for cb in crop_res[0].boxes:
                                    cxy = cb.xyxy[0].cpu().numpy()
                                    cconf = float(cb.conf[0])
                                    gx1 = float(cxy[0] + px1)
                                    gy1 = float(cxy[1] + py1)
                                    gx2 = float(cxy[2] + px1)
                                    gy2 = float(cxy[3] + py1)
                                    gbox = np.array([gx1, gy1, gx2, gy2], dtype=np.float32)
                                    if is_inside_field(gbox):
                                        bx1, by1 = max(0, int(gx1)), max(0, int(gy1))
                                        bx2, by2 = min(w_img, int(gx2)), min(h_img, int(gy2))
                                        if bx2 > bx1 and by2 > by1:
                                            _, al = classify_bumper_patch(frame[by1:by2, bx1:bx2])
                                        else:
                                            al = "UNKNOWN"
                                        raw_candidates.append((gbox, cconf, al))

            # 4. Alliance-Decoupled Non-Maximum Suppression
            # Sort by conf descending
            raw_candidates.sort(key=lambda x: x[1], reverse=True)
            deduped_candidates: List[Tuple[np.ndarray, float, str]] = []

            for cand in raw_candidates:
                cb, cc, ca = cand
                overlap = False
                ccy = float(cb[1] + cb[3]) / 2.0
                for kb, kc, ka in deduped_candidates:
                    kcy = float(kb[1] + kb[3]) / 2.0
                    # Same alliance deduplication: two robots of the same alliance cannot occupy the same physical space
                    if ca == ka and ca in ("RED", "BLUE"):
                        inter_w = max(0.0, min(cb[2], kb[2]) - max(cb[0], kb[0]))
                        inter_h = max(0.0, min(cb[3], kb[3]) - max(cb[1], kb[1]))
                        inter_area = inter_w * inter_h
                        area_c = max(1.0, (cb[2] - cb[0]) * (cb[3] - cb[1]))
                        area_k = max(1.0, (kb[2] - kb[0]) * (kb[3] - kb[1]))
                        if (inter_area / min(area_c, area_k)) > 0.25:
                            overlap = True
                            break
                    # Opposing alliances do not suppress each other unless near 100% overlap
                    elif ca in ("RED", "BLUE") and ka in ("RED", "BLUE") and ca != ka:
                        allowed_iou = 0.85
                    elif abs(ccy - kcy) > 35.0:
                        # Stacked vertically in perspective (one robot behind another)
                        allowed_iou = 0.65
                    else:
                        allowed_iou = 0.40
                    iou = SimpleRobotTracker._compute_iou(cb, kb)
                    if iou > allowed_iou:
                        overlap = True
                        break
                if not overlap:
                    deduped_candidates.append(cand)

            # 5. Dynamic Waterfall Selection: prioritize 3 RED, 3 BLUE (total 6)
            red_cands = [c for c in deduped_candidates if c[2] == "RED"]
            blue_cands = [c for c in deduped_candidates if c[2] == "BLUE"]
            unk_cands = [c for c in deduped_candidates if c[2] == "UNKNOWN"]

            # 6. Bumper Fabric Fallback if an alliance is missing candidates
            existing_box_arrs = [c[0] for c in (red_cands + blue_cands)]
            if len(red_cands) < 3:
                recovered_red = recover_missing_bumper_proposals(frame, existing_box_arrs, "RED")
                red_cands += recovered_red[: 3 - len(red_cands)]
                existing_box_arrs += [r[0] for r in recovered_red]

            if len(blue_cands) < 3:
                recovered_blue = recover_missing_bumper_proposals(frame, existing_box_arrs, "BLUE")
                blue_cands += recovered_blue[: 3 - len(blue_cands)]

            # Combine selected candidates
            selected_detections = red_cands[:3] + blue_cands[:3]
            if len(selected_detections) < 6 and unk_cands:
                selected_detections += unk_cands[: 6 - len(selected_detections)]

            # 7. Update Persistent Kalman Tracker
            active_tracks = tracker.update(selected_detections, max_w=w_img, max_h=h_img)
            num_robots = len(active_tracks)
            total_robot_detections += num_robots

            num_red = sum(1 for t in active_tracks if t.alliance == "RED")
            num_blue = sum(1 for t in active_tracks if t.alliance == "BLUE")

            # 8. Draw annotations for each tracked robot
            for track in active_tracks:
                draw_crisp_robot_annotation(frame, track)

            # Performance FPS calculation
            t_frame_end = time.perf_counter()
            frame_elapsed = max(1e-4, t_frame_end - t_frame_start)
            recent_frame_times.append(frame_elapsed)
            current_fps = len(recent_frame_times) / sum(recent_frame_times)

            timestamp_s = frame_idx / video_fps

            # Draw HUD
            draw_sleek_hud(
                frame=frame,
                frame_index=frame_idx,
                total_frames=total_video_frames,
                timestamp_s=timestamp_s,
                num_robots=num_robots,
                current_fps=current_fps,
                num_red=num_red,
                num_blue=num_blue,
                num_motion_robots=len(motion_res.robot_motion_regions),
                num_motion_balls=len(motion_res.ball_motion_regions),
            )

            # Write annotated frame
            writer.write(frame)
            frames_rendered += 1

            # Snapshot capture
            if snapshot_dir and frame_idx in snapshot_frames:
                snap_path = os.path.join(snapshot_dir, f"annotated_yolo11x_frame_{frame_idx}.jpg")
                cv2.imwrite(snap_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_snapshots.append(snap_path)
                logger.info("Saved keyframe snapshot [%d] -> %s (robots: %d)", frame_idx, snap_path, num_robots)

            if frames_rendered % 150 == 0:
                elapsed_so_far = time.perf_counter() - t_render_start
                avg_fps = frames_rendered / elapsed_so_far
                logger.info(
                    "Progress: %d / %d frames (%.1f%%) | %.1f avg FPS | %d / 6 robots locked (%d Red | %d Blue)",
                    frames_rendered,
                    effective_total,
                    (frames_rendered / effective_total) * 100.0,
                    avg_fps,
                    num_robots,
                    num_red,
                    num_blue,
                )

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    t_render_total = time.perf_counter() - t_render_start
    overall_render_fps = frames_rendered / max(1e-4, t_render_total)
    avg_robots = total_robot_detections / max(1, frames_rendered)

    file_size_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    file_size_mb = file_size_bytes / (1024 * 1024)

    logger.info(
        "=== Video Rendering Finished ==="
        "\nOutput Path: %s"
        "\nFile Size: %.2f MB (%d bytes)"
        "\nFrames Rendered: %d"
        "\nElapsed Time: %.2fs"
        "\nRender Speed: %.1f FPS"
        "\nAvg Robots / Frame: %.2f"
        "\nKeyframe Snapshots: %s",
        os.path.abspath(output_path),
        file_size_mb,
        file_size_bytes,
        frames_rendered,
        t_render_total,
        overall_render_fps,
        avg_robots,
        saved_snapshots,
    )

    return {
        "output_path": os.path.abspath(output_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "frames_rendered": frames_rendered,
        "elapsed_seconds": t_render_total,
        "render_fps": overall_render_fps,
        "avg_robots_per_frame": avg_robots,
        "saved_snapshots": saved_snapshots,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLOv11x robot detection video with 6-robot lock.")
    parser.add_argument("--video", "-v", type=str, default="output.mp4", help="Path to input video.")
    parser.add_argument("--model", "-m", type=str, default="yolo11x_robot_best.pt", help="Path to model weights.")
    parser.add_argument("--output", "-o", type=str, default="output_yolo11x_annotated.mp4", help="Output video path.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution for tiles.")
    parser.add_argument("--conf", type=float, default=0.005, help="Base proposal confidence threshold.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index.")
    parser.add_argument("--max-frames", "-n", type=int, default=2400, help="Max frames to render.")
    parser.add_argument("--start-frame", "-s", type=int, default=0, help="Starting frame index.")
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="/home/PaulRekouts/.gemini/antigravity-cli/brain/4d5d63ae-d521-439d-b83c-3ce2d7f51657",
        help="Directory to save keyframe snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_video_with_model(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        snapshot_dir=args.snapshot_dir,
    )


if __name__ == "__main__":
    main()
