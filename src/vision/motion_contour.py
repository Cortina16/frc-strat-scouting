"""
OpenCV Dynamic Motion Contouring and Motion-Guided YOLO Verification.

Separates static arena objects (Hub funnel, towers, perimeter glass, volunteer seating)
from dynamic movement of robots and balls on the playing field using high-speed
background subtraction, temporal motion history accumulation, and contour classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("frc.vision.motion_contour")


@dataclass
class DynamicMotionResult:
    """Encapsulates the output of OpenCV dynamic motion segmentation."""

    fg_mask: np.ndarray  # Binary uint8 mask (0 or 255)
    motion_history: np.ndarray  # Float32 energy map [0.0, 1.0]
    robot_motion_regions: List[Tuple[np.ndarray, float, float]]  # (box [x1,y1,x2,y2], area, energy)
    ball_motion_regions: List[Tuple[np.ndarray, float, float]]  # (box [x1,y1,x2,y2], area, energy)
    static_mask: np.ndarray  # Binary uint8 mask of static regions (255 where static, 0 where moving)


class DynamicMotionSegmenter:
    """
    High-performance OpenCV motion segmentation engine.
    Uses MOG2 background modeling combined with temporal motion history accumulation
    to separate stationary field structures from moving robots and game pieces.
    """

    def __init__(
        self,
        history: int = 100,
        var_threshold: float = 25.0,
        detect_shadows: bool = False,
        min_robot_area: float = 1000.0,
        max_ball_area: float = 650.0,
        min_ball_area: float = 15.0,
        motion_decay: float = 0.96,
        static_energy_cutoff: float = 0.05,
    ) -> None:
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.min_robot_area = min_robot_area
        self.max_ball_area = max_ball_area
        self.min_ball_area = min_ball_area
        self.motion_decay = motion_decay
        self.static_energy_cutoff = static_energy_cutoff

        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self._motion_history: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count: int = 0

    def reset(self) -> None:
        """Resets background model and motion history."""
        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        self._motion_history = None
        self._prev_gray = None
        self._frame_count = 0

    def apply(
        self,
        frame_roi: np.ndarray,
        offset_y: int = 0,
        offset_x: int = 0,
    ) -> DynamicMotionResult:
        """
        Processes a video frame ROI (e.g. top-screen wide field) and extracts
        motion masks, motion history energy, and classified dynamic contours.

        Args:
            frame_roi: BGR frame array of shape (H, W, 3).
            offset_y: Vertical pixel offset of ROI relative to full 1080p frame.
            offset_x: Horizontal pixel offset of ROI relative to full 1080p frame.

        Returns:
            DynamicMotionResult containing segmented dynamic regions and energy maps.
        """
        if frame_roi is None or frame_roi.size == 0:
            empty = np.zeros((1, 1), dtype=np.uint8)
            empty_float = np.zeros((1, 1), dtype=np.float32)
            return DynamicMotionResult(
                fg_mask=empty,
                motion_history=empty_float,
                robot_motion_regions=[],
                ball_motion_regions=[],
                static_mask=empty,
            )

        h, w = frame_roi.shape[:2]

        # 1. Update MOG2 background subtractor
        raw_fg = self._fgbg.apply(frame_roi)

        # 2. Complement with frame differencing to catch rapid motion
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, diff_thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            combined_fg = cv2.bitwise_or(raw_fg, diff_thresh)
        else:
            combined_fg = raw_fg
        self._prev_gray = gray

        # Frame 0 initializes the background model; ignore initial MOG2 warm-up flash
        if self._frame_count == 0:
            combined_fg = np.zeros((h, w), dtype=np.uint8)

        # 3. Morphological filtering to join chassis, bumpers, and intake superstructures
        fg_closed = cv2.morphologyEx(combined_fg, cv2.MORPH_CLOSE, self._close_kernel)
        fg_clean = cv2.morphologyEx(fg_closed, cv2.MORPH_DILATE, self._dilate_kernel)

        # 4. Update Motion History Energy Map
        if self._motion_history is None or self._motion_history.shape != (h, w):
            self._motion_history = np.zeros((h, w), dtype=np.float32)

        motion_instant = (fg_clean > 0).astype(np.float32)
        # Exponential temporal decay with instantaneous peak hold
        self._motion_history = np.maximum(
            self._motion_history * self.motion_decay,
            motion_instant,
        )

        # 5. Static Mask (areas where motion energy is below cutoff)
        static_mask = (self._motion_history <= self.static_energy_cutoff).astype(np.uint8) * 255

        # 6. Extract Connected Dynamic Contours
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        robot_regions: List[Tuple[np.ndarray, float, float]] = []
        ball_regions: List[Tuple[np.ndarray, float, float]] = []

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_ball_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            global_x1 = float(bx + offset_x)
            global_y1 = float(by + offset_y)
            global_x2 = float(bx + bw + offset_x)
            global_y2 = float(by + bh + offset_y)
            box = np.array([global_x1, global_y1, global_x2, global_y2], dtype=np.float32)

            # Compute motion energy inside contour bounding box
            energy_patch = self._motion_history[by : by + bh, bx : bx + bw]
            avg_energy = float(np.mean(energy_patch)) if energy_patch.size > 0 else 0.0

            # Scale-based classification
            # Robot moving contour: substantial area or robot-like dimensions
            if (area >= self.min_robot_area or (bw >= 60 and bh >= 45)) and (bw >= 40 and bh >= 30):
                # Add 20px padding to encompass full chassis and contact patch
                pad_w = 20.0
                pad_h = 20.0
                padded_box = np.array([
                    max(0.0, global_x1 - pad_w),
                    max(0.0, global_y1 - pad_h),
                    global_x2 + pad_w,
                    global_y2 + pad_h,
                ], dtype=np.float32)
                robot_regions.append((padded_box, area, avg_energy))
            # Ball moving contour: compact dimensions (including high-speed motion streaks)
            elif bw <= 75 and bh <= 75 and area <= self.max_ball_area:
                ball_regions.append((box, area, avg_energy))

        # Sort robot regions by area descending
        robot_regions.sort(key=lambda x: x[1], reverse=True)
        self._frame_count += 1

        return DynamicMotionResult(
            fg_mask=fg_clean,
            motion_history=self._motion_history,
            robot_motion_regions=robot_regions,
            ball_motion_regions=ball_regions,
            static_mask=static_mask,
        )

    def get_motion_energy_at(
        self,
        box: np.ndarray,
        offset_y: int = 0,
        offset_x: int = 0,
    ) -> float:
        """
        Evaluates dynamic motion energy [0.0, 1.0] within a global bounding box [x1, y1, x2, y2].
        """
        if self._motion_history is None or self._motion_history.size == 0:
            return 0.0

        h_hist, w_hist = self._motion_history.shape[:2]
        x1 = max(0, min(w_hist - 1, int(round(box[0] - offset_x))))
        y1 = max(0, min(h_hist - 1, int(round(box[1] - offset_y))))
        x2 = max(0, min(w_hist, int(round(box[2] - offset_x))))
        y2 = max(0, min(h_hist, int(round(box[3] - offset_y))))

        if x2 <= x1 or y2 <= y1:
            return 0.0

        patch = self._motion_history[y1:y2, x1:x2]
        return float(np.mean(patch)) if patch.size > 0 else 0.0


class MotionGuidedYOLOVerifier:
    """
    Cross-references candidate detections with the dynamic motion field,
    suppressing static arena artifacts and prioritizing active moving robots.
    """

    def __init__(self, min_dynamic_energy: float = 0.03) -> None:
        self.min_dynamic_energy = min_dynamic_energy

    def filter_static_proposals(
        self,
        candidates: List[Tuple[np.ndarray, float, str]],
        motion_segmenter: DynamicMotionSegmenter,
        offset_y: int = 0,
        offset_x: int = 0,
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        Suppresses candidate bounding boxes that reside in permanently static zones
        with near-zero motion energy.

        Args:
            candidates: List of (bbox [x1,y1,x2,y2], conf, alliance).
            motion_segmenter: Segmenter instance holding the motion history energy map.
            offset_y: Vertical pixel offset of ROI relative to full 1080p frame.
            offset_x: Horizontal pixel offset of ROI relative to full 1080p frame.

        Returns:
            Filtered list of candidates with static objects removed.
        """
        if not candidates:
            return []

        verified: List[Tuple[np.ndarray, float, str]] = []
        for box, conf, alliance in candidates:
            energy = motion_segmenter.get_motion_energy_at(box, offset_y, offset_x)
            # High-confidence neural detections (>= 0.65) are retained even if temporarily still
            if conf >= 0.65 or energy >= self.min_dynamic_energy:
                verified.append((box, conf, alliance))
            else:
                logger.debug(
                    "Suppressed static false positive at [%.1f, %.1f, %.1f, %.1f] (energy=%.4f, conf=%.3f)",
                    box[0], box[1], box[2], box[3], energy, conf,
                )

        return verified
