"""Broadcast Scoreboard and Match Timer Perception Reader.

Extracts real-time match state directly from 1080p broadcast video HUD:
1. Match Timer Countdown: Tracks remaining seconds ('M:SS'), detects when countdown starts.
2. Match Start Synchronization: Detects initial timer countdown decrement ($0:15 -> 0:14).
3. 3-Second Scoring Suppression: Strictly prevents scoring attribution during the first 3.0s
   after the match starts ($t_{match} < 3.0\\text{s}$).
4. Live Alliance Scores: Reads Blue and Red team scores from HUD via topological digit template matching,
   providing live $\\Delta \\text{score}$ values to validate and bound CV-detected scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("frc.analytics.scoreboard")


@dataclass(slots=True)
class ScoreboardReading:
    """State extracted from the broadcast match HUD for a single video frame."""
    frame_idx: int
    timestamp_s: float
    blue_score: int
    red_score: int
    blue_score_delta: int
    red_score_delta: int
    timer_seconds: Optional[int]
    timer_display: str
    is_counting_down: bool
    match_started: bool
    match_start_timestamp_s: Optional[float]
    match_elapsed_s: float
    scoring_allowed: bool
    blue_foul_points: int = 0
    red_foul_points: int = 0
    clean_blue_score: int = 0
    clean_red_score: int = 0


class BroadcastScoreboardReader:
    """
    Reads the broadcast match scoreboard HUD (timer, blue score, red score)
    and synchronizes match timing and scoring validity.
    """

    def __init__(
        self,
        grace_period_s: float = 3.0,
        expected_resolution: Tuple[int, int] = (1920, 1080),
    ) -> None:
        self.grace_period_s = grace_period_s
        self.expected_w, self.expected_h = expected_resolution

        # State tracking
        self._match_started: bool = False
        self._match_start_timestamp_s: Optional[float] = None
        self._match_start_frame: Optional[int] = None
        self._prev_timer_seconds: Optional[int] = None
        self._prev_timer_patch: Optional[np.ndarray] = None
        self._timer_change_count: int = 0

        self._current_blue_score: int = 0
        self._current_red_score: int = 0
        self._prev_emitted_blue: int = 0
        self._prev_emitted_red: int = 0

        # Foul / penalty points for each alliance
        self._blue_foul_points: int = 0
        self._red_foul_points: int = 0

        # Recent readings for outlier smoothing
        self._recent_blue_reads: List[int] = []
        self._recent_red_reads: List[int] = []

        # Precompute normalized digit templates for 0-9
        self._templates: Dict[int, List[np.ndarray]] = self._build_canonical_templates()

    def set_foul_points(self, red_fouls: int = 0, blue_fouls: int = 0) -> None:
        """Sets known or official foul/penalty points for each alliance."""
        self._red_foul_points = max(0, int(red_fouls))
        self._blue_foul_points = max(0, int(blue_fouls))

    def reset(self) -> None:
        """Resets all internal scoreboard tracking state."""
        self._match_started = False
        self._match_start_timestamp_s = None
        self._match_start_frame = None
        self._prev_timer_seconds = None
        self._prev_timer_patch = None
        self._timer_change_count = 0
        self._current_blue_score = 0
        self._current_red_score = 0
        self._prev_emitted_blue = 0
        self._prev_emitted_red = 0
        self._blue_foul_points = 0
        self._red_foul_points = 0
        self._recent_blue_reads.clear()
        self._recent_red_reads.clear()

    def read_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
    ) -> ScoreboardReading:
        """
        Parses scoreboard ROI from frame and returns structured ScoreboardReading.
        """
        h_frame, w_frame = frame.shape[:2]
        scale_x = w_frame / float(self.expected_w)
        scale_y = h_frame / float(self.expected_h)

        # 1. Extract Timer Region
        # Standard 1080p coordinates: y in [75:112], x in [885:1015]
        ty1, ty2 = int(round(75 * scale_y)), int(round(112 * scale_y))
        tx1, tx2 = int(round(885 * scale_x)), int(round(1015 * scale_x))
        timer_crop = frame[ty1:ty2, tx1:tx2]

        timer_str, timer_sec, is_active_pulse = self._parse_timer(timer_crop)

        # 2. Match Start Countdown Detection
        # Check if timer has started counting down
        if not self._match_started:
            if self._prev_timer_seconds is not None and timer_sec is not None:
                # Canonical Auto start (15 -> 14)
                if self._prev_timer_seconds == 15 and timer_sec == 14:
                    self._timer_change_count += 2
                # Valid countdown step within Auto (<= 15)
                elif 0 <= timer_sec < self._prev_timer_seconds <= 15 and (self._prev_timer_seconds - timer_sec) <= 2:
                    self._timer_change_count += 1
                elif timer_sec == 15 and self._prev_timer_seconds > 15:
                    # Transition to Auto start
                    self._timer_change_count += 1

            # Latch match started when active countdown is confirmed
            if self._timer_change_count >= 2:
                self._match_started = True
                self._match_start_timestamp_s = timestamp_s
                self._match_start_frame = frame_idx
                logger.info(
                    "Broadcast Match Countdown Start Detected @ frame %d (%.2fs) | Timer: %s",
                    frame_idx, timestamp_s, timer_str,
                )

        if timer_sec is not None:
            self._prev_timer_seconds = timer_sec

        # 3. Match Elapsed Time & 3-Second Grace Period
        if self._match_started and self._match_start_timestamp_s is not None:
            match_elapsed_s = max(0.0, timestamp_s - self._match_start_timestamp_s)
            scoring_allowed = (match_elapsed_s >= self.grace_period_s)
        else:
            match_elapsed_s = 0.0
            scoring_allowed = False

        # 4. Extract Alliance Scores
        # Blue score: y in [75:112], x in [805:885]
        by1, by2 = int(round(75 * scale_y)), int(round(112 * scale_y))
        bx1, bx2 = int(round(805 * scale_x)), int(round(885 * scale_x))
        blue_crop = frame[by1:by2, bx1:bx2]
        blue_score_read = self._parse_score_box(blue_crop, is_red=False)

        # Red score: y in [75:112], x in [1015:1115]
        ry1, ry2 = int(round(75 * scale_y)), int(round(112 * scale_y))
        rx1, rx2 = int(round(1015 * scale_x)), int(round(1115 * scale_x))
        red_crop = frame[ry1:ry2, rx1:rx2]
        red_score_read = self._parse_score_box(red_crop, is_red=True)

        # Apply monotonic score filtering (FRC scores do not decrease during match)
        if blue_score_read is not None:
            self._recent_blue_reads.append(blue_score_read)
            if len(self._recent_blue_reads) > 5:
                self._recent_blue_reads.pop(0)
            med_b = int(np.median(self._recent_blue_reads))
            if med_b >= self._current_blue_score and (med_b - self._current_blue_score) <= 35:
                self._current_blue_score = med_b

        if red_score_read is not None:
            self._recent_red_reads.append(red_score_read)
            if len(self._recent_red_reads) > 5:
                self._recent_red_reads.pop(0)
            med_r = int(np.median(self._recent_red_reads))
            if med_r >= self._current_red_score and (med_r - self._current_red_score) <= 35:
                self._current_red_score = med_r

        blue_delta = max(0, self._current_blue_score - self._prev_emitted_blue)
        red_delta = max(0, self._current_red_score - self._prev_emitted_red)
        self._prev_emitted_blue = self._current_blue_score
        self._prev_emitted_red = self._current_red_score

        clean_blue = max(0, self._current_blue_score - self._blue_foul_points)
        clean_red = max(0, self._current_red_score - self._red_foul_points)

        return ScoreboardReading(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            blue_score=self._current_blue_score,
            red_score=self._current_red_score,
            blue_score_delta=blue_delta,
            red_score_delta=red_delta,
            timer_seconds=timer_sec,
            timer_display=timer_str,
            is_counting_down=self._match_started,
            match_started=self._match_started,
            match_start_timestamp_s=self._match_start_timestamp_s,
            match_elapsed_s=match_elapsed_s,
            scoring_allowed=scoring_allowed,
            blue_foul_points=self._blue_foul_points,
            red_foul_points=self._red_foul_points,
            clean_blue_score=clean_blue,
            clean_red_score=clean_red,
        )

    def _parse_timer(self, timer_crop: np.ndarray) -> Tuple[str, Optional[int], bool]:
        """Parses timer digits and checks for active frame change pulse."""
        if timer_crop.size == 0 or timer_crop.shape[0] < 10 or timer_crop.shape[1] < 20:
            return "--:--", None, False

        gray = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2GRAY)

        # Pulse detection against previous frame
        is_pulse = False
        if self._prev_timer_patch is not None and self._prev_timer_patch.shape == gray.shape:
            diff = cv2.absdiff(gray, self._prev_timer_patch)
            if float(diff.mean()) > 10.0:
                is_pulse = True
        self._prev_timer_patch = gray.copy()

        # Binarize dark digits on light background
        _, th = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [
            cv2.boundingRect(c)
            for c in cnts
            if cv2.contourArea(c) > 15
            and cv2.boundingRect(c)[3] >= 15
            and cv2.boundingRect(c)[0] > 4
            and cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] < (timer_crop.shape[1] - 4)
        ]
        boxes = sorted(boxes, key=lambda b: b[0])

        if not boxes:
            return "--:--", None, is_pulse

        digits = []
        for x, y, w, h in boxes:
            patch = th[y : y + h, x : x + w]
            d, conf = self._match_digit(patch)
            if d != -1:
                digits.append(d)

        if not digits:
            return "--:--", None, is_pulse

        if len(digits) == 1:
            if not self._match_started:
                # Reject transient single digit before match start (e.g. graphic transitions)
                return "--:--", None, is_pulse
            sec = digits[0]
            return f"0:{sec:02d}", sec, is_pulse
        elif len(digits) == 2:
            sec = digits[0] * 10 + digits[1]
            return f"0:{sec:02d}", sec, is_pulse
        elif len(digits) >= 3:
            minutes = digits[0]
            sec = digits[1] * 10 + digits[2]
            total_sec = minutes * 60 + sec
            return f"{minutes}:{sec:02d}", total_sec, is_pulse

        return "--:--", None, is_pulse

    def _parse_score_box(self, crop: np.ndarray, is_red: bool) -> Optional[int]:
        """Extracts score integer from Blue or Red HUD box."""
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 15:
            return None

        # Digits are bright white on colored background
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [
            cv2.boundingRect(c)
            for c in cnts
            if cv2.contourArea(c) > 15 and cv2.boundingRect(c)[3] >= 15
        ]
        boxes = sorted(boxes, key=lambda b: b[0])

        if not boxes:
            return None

        digits = []
        for x, y, w, h in boxes:
            patch = th[y : y + h, x : x + w]
            d, conf = self._match_digit(patch)
            if d != -1:
                digits.append(d)

        if not digits:
            return None

        score_val = 0
        for d in digits:
            score_val = score_val * 10 + d
        return score_val

    def _match_digit(self, patch: np.ndarray) -> Tuple[int, float]:
        """
        Classifies digit using aspect-ratio preserved normalized correlation
        and internal hole/topology filtering.
        """
        if patch.shape[0] < 8 or patch.shape[1] < 3:
            return -1, 0.0

        p_norm = self._pad_and_resize(patch)
        holes = self._count_holes(patch)
        h, w = patch.shape[:2]
        aspect = float(w) / float(max(1, h))

        # Hard topological filters
        # Digit 1: very narrow stroke (aspect < 0.45) and 0 holes
        if aspect < 0.45 and holes == 0:
            return 1, 0.95

        best_d = -1
        best_score = -1.0

        for d, variants in self._templates.items():
            # Check topology compatibility
            if holes == 2 and d != 8:
                continue
            if holes == 1 and d not in (0, 4, 6, 9):
                continue
            if holes == 0 and d in (0, 8):
                continue

            for v in variants:
                score = float((p_norm * v).sum())
                if score > best_score:
                    best_score = score
                    best_d = d

        if best_score < 0.35:
            # Fallback to unrestricted search
            for d, variants in self._templates.items():
                for v in variants:
                    score = float((p_norm * v).sum())
                    if score > best_score:
                        best_score = score
                        best_d = d

        return best_d, best_score

    @staticmethod
    def _count_holes(binary_patch: np.ndarray) -> int:
        """Counts internal holes via contour hierarchy."""
        cnts, hier = cv2.findContours(binary_patch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(cnts) == 0:
            return 0
        holes = 0
        for h in hier[0]:
            if h[3] != -1:  # Has parent contour -> internal hole
                holes += 1
        return holes

    @staticmethod
    def _pad_and_resize(patch: np.ndarray, target_h: int = 36, target_w: int = 26) -> np.ndarray:
        """Resizes patch preserving aspect ratio into normalized target canvas."""
        h, w = patch.shape[:2]
        scale = min((target_h - 4) / max(1, h), (target_w - 4) / max(1, w))
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        resized = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w), dtype=np.float32)
        dy = (target_h - nh) // 2
        dx = (target_w - nw) // 2
        canvas[dy : dy + nh, dx : dx + nw] = resized.astype(np.float32) / 255.0
        norm = np.linalg.norm(canvas)
        if norm > 1e-6:
            canvas /= norm
        return canvas

    def _build_canonical_templates(self) -> Dict[int, List[np.ndarray]]:
        """Precomputes multi-scale font templates across digits 0-9."""
        templates: Dict[int, List[np.ndarray]] = {}
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
        ]

        for d in range(10):
            d_str = str(d)
            variants = []
            for font in fonts:
                for scale in [0.75, 0.85, 0.95]:
                    for thick in [2, 3]:
                        canvas = np.zeros((48, 48), dtype=np.uint8)
                        (tw, th), _ = cv2.getTextSize(d_str, font, scale, thick)
                        tx = max(0, (48 - tw) // 2)
                        ty = max(th, (48 + th) // 2)
                        cv2.putText(canvas, d_str, (tx, ty), font, scale, 255, thick, cv2.LINE_AA)

                        pts = np.where(canvas > 0)
                        if len(pts[0]) > 0:
                            y1, y2 = pts[0].min(), pts[0].max() + 1
                            x1, x2 = pts[1].min(), pts[1].max() + 1
                            v = self._pad_and_resize(canvas[y1:y2, x1:x2])
                            variants.append(v)

            templates[d] = variants
        return templates
