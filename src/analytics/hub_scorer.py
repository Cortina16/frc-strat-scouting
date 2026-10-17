"""Hub Scoring Intersection and Shot Verification Engine.

Detects when yellow balls ('FUEL') are actually scored by checking spatial intersection
between ball detections and Alliance Hub catchment geometries specifically at the moments
official match score increments occur.
Distinguishes confirmed scored balls from deflected or missed shots, attributes scoring
events to individual robots, and produces shot outcomes for logistic accuracy modeling.
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("frc.analytics.hub_scorer")


class HubAlliance(str, Enum):
    RED = "RED"
    BLUE = "BLUE"


@dataclass(slots=True)
class AllianceScoreAttribution:
    """Consolidated score attribution comparing CV-detected score against GUI OCR minus fouls."""
    alliance: HubAlliance
    gui_ocr_score: int
    foul_points: int
    clean_target_score: int
    current_attributed_score: int
    deficit_or_surplus: int  # clean_target - current_attributed (>0: undercount, <0: overcount)
    robot_scores: Dict[str, int] = field(default_factory=dict)  # {unit_id / team_number: points}


@dataclass(slots=True)
class HubGeometry:
    """Represents an Alliance Hub in field inches and image screen pixels."""
    alliance: HubAlliance
    # Field CAD center (inches)
    field_x_inch: float
    field_y_inch: float
    field_radius_inch: float = 60.0  # 5.0 feet interaction radius
    # Broadcast screen bounding box [x1, y1, x2, y2] in 1080p
    screen_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def intersects_point_field(self, x_inch: float, y_inch: float, tolerance_inch: float = 0.0) -> bool:
        """Checks if field coordinate point falls within the Hub catchment radius."""
        dist = math.hypot(x_inch - self.field_x_inch, y_inch - self.field_y_inch)
        return dist <= (self.field_radius_inch + tolerance_inch)

    def intersects_point_screen(
        self,
        px_x: float,
        px_y: float,
        margin_x: float = 50.0,
        margin_y_top: float = 75.0,
        margin_y_bottom: float = 35.0,
    ) -> bool:
        """
        Checks if image pixel coordinate falls within the Hub screen boundary.
        Uses liberal asymmetric margins: higher top margin to reliably capture incoming ballistic flight arcs.
        """
        x1, y1, x2, y2 = self.screen_bbox
        return (x1 - margin_x <= px_x <= x2 + margin_x) and (y1 - margin_y_top <= px_y <= y2 + margin_y_bottom)

    def intersects_point_screen_approach(self, px_x: float, px_y: float) -> bool:
        """
        Outer approach zone (loosened to capture incoming ballistic flight arcs).
        """
        return self.intersects_point_screen(
            px_x, px_y, margin_x=65.0, margin_y_top=110.0, margin_y_bottom=40.0
        )

    def intersects_point_screen_catchment(self, px_x: float, px_y: float) -> bool:
        """
        Inner scoring zone (tightened to physical funnel catchment).
        """
        return self.intersects_point_screen(
            px_x, px_y, margin_x=25.0, margin_y_top=40.0, margin_y_bottom=15.0
        )


@dataclass(slots=True)
class HubBallTracklet:
    """Kalman-filtered trajectory state for a ball in/near an Alliance Hub."""
    track_id: int
    alliance: HubAlliance
    state: np.ndarray  # [x, y, vx, vy]^T shape (4, 1)
    covariance: np.ndarray  # (4, 4)
    history: List[Tuple[float, float]] = field(default_factory=list)
    frames_seen: int = 1
    frames_lost: int = 0
    has_scored: bool = False
    entry_frame: int = 0
    entry_timestamp_s: float = 0.0
    scoring_frame: Optional[int] = None
    scoring_timestamp_s: Optional[float] = None
    associated_shot_event_id: Optional[str] = None
    attributed_robot: Optional[Dict[str, Any]] = None

    def predict(self, dt: float = 0.033) -> Tuple[float, float]:
        """Predicts position via constant-velocity model."""
        dt = max(1e-4, min(0.2, dt))
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        Q = np.diag([2.0, 2.0, 15.0, 15.0]).astype(np.float32)
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, meas_x: float, meas_y: float) -> None:
        """Updates Kalman state from detected measurement."""
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        R = np.diag([5.0, 5.0]).astype(np.float32)
        z = np.array([[meas_x], [meas_y]], dtype=np.float32)
        y = z - H @ self.state
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        I4 = np.eye(4, dtype=np.float32)
        self.covariance = (I4 - K @ H) @ self.covariance

        cx = float(self.state[0, 0])
        cy = float(self.state[1, 0])
        self.history.append((cx, cy))
        self.frames_seen += 1
        self.frames_lost = 0

    @property
    def current_pos(self) -> Tuple[float, float]:
        return float(self.state[0, 0]), float(self.state[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.state[2, 0]), float(self.state[3, 0])


@dataclass(slots=True)
class ShotEvent:
    """Discrete shot attempt event evaluated for Hub scoring conversion."""
    event_id: str
    frame_index: int
    timestamp_s: float
    phase_id: int  # 1: Auto, 2: Early, 3: Mid, 4: Late, 5: Endgame
    hub_alliance: HubAlliance
    ball_x: float
    ball_y: float
    robot_track_id: Optional[int]
    robot_team: Optional[str]
    robot_alliance: str
    distance_ft: float
    is_open: bool  # Defender > 4.0 ft away
    is_scored: bool  # True if intersected Hub during point increment
    score_increment: int = 1
    confidence: float = 1.0


class HubScoringDetector:
    """
    Watches ball detections, robot tracking states, and real-time score increments
    to detect authentic scoring events at the Alliance Hubs.
    """

    def __init__(
        self,
        red_hub_field: Tuple[float, float] = (162.8, 158.85),
        blue_hub_field: Tuple[float, float] = (488.4, 158.85),
        red_hub_screen: Tuple[float, float, float, float] = (1380.0, 135.0, 1610.0, 305.0),
        blue_hub_screen: Tuple[float, float, float, float] = (335.0, 135.0, 565.0, 305.0),
        scoring_window_s: float = 2.25,  # Liberal correlation window between ball intersection and score increment
    ) -> None:
        self.red_hub = HubGeometry(
            alliance=HubAlliance.RED,
            field_x_inch=red_hub_field[0],
            field_y_inch=red_hub_field[1],
            field_radius_inch=60.0,
            screen_bbox=red_hub_screen,
        )
        self.blue_hub = HubGeometry(
            alliance=HubAlliance.BLUE,
            field_x_inch=blue_hub_field[0],
            field_y_inch=blue_hub_field[1],
            field_radius_inch=60.0,
            screen_bbox=blue_hub_screen,
        )
        self.scoring_window_s = scoring_window_s

        # Persistent Hub Tag History and Mutual Triangulation Memory
        self._hub_tag_history_red: Dict[int, List[Tuple[float, float]]] = collections.defaultdict(list)
        self._hub_tag_history_blue: Dict[int, List[Tuple[float, float]]] = collections.defaultdict(list)
        self._persistent_red_bbox: Optional[Tuple[float, float, float, float]] = None
        self._persistent_blue_bbox: Optional[Tuple[float, float, float, float]] = None
        self._hub_relative_delta_xy: Optional[Tuple[float, float]] = None  # (dx, dy) from Blue Hub to Red Hub

        # History buffers for temporal correlation
        # Each entry: (frame, timestamp_s, [(x, y, is_screen)], hub_candidate)
        self._ball_history: collections.deque[Tuple[int, float, List[Tuple[float, float]], HubAlliance]] = collections.deque(maxlen=180)
        # Score state buffer: [(timestamp_s, red_score, blue_score, red_delta, blue_delta)]
        self._score_history: List[Tuple[float, int, int, int, int]] = []
        # Active ball tracklets in Hub zones: {track_id: HubBallTracklet}
        self.active_ball_tracklets_red: Dict[int, HubBallTracklet] = {}
        self.active_ball_tracklets_blue: Dict[int, HubBallTracklet] = {}
        self._next_track_id: int = 1
        self._last_processed_ts: Optional[float] = None
        self._last_red_score: int = 0
        self._last_blue_score: int = 0

        # Foul points and latest GUI scores for clean attribution reconciliation
        self._foul_points_red: int = 0
        self._foul_points_blue: int = 0
        self._latest_gui_red: int = 0
        self._latest_gui_blue: int = 0

        # Output shot events
        self.confirmed_shots: List[ShotEvent] = []

    def reset(self) -> None:
        """Resets dynamic ball tracking and score histories."""
        self.active_ball_tracklets_red.clear()
        self.active_ball_tracklets_blue.clear()
        self._next_track_id = 1
        self._last_processed_ts = None
        self._ball_history.clear()
        self._score_history.clear()
        self.confirmed_shots.clear()
        self._last_red_score = 0
        self._last_blue_score = 0
        self._foul_points_red = 0
        self._foul_points_blue = 0
        self._latest_gui_red = 0
        self._latest_gui_blue = 0

    def set_foul_points(self, red_fouls: int = 0, blue_fouls: int = 0) -> None:
        """Sets known or official foul/penalty points for each alliance."""
        self._foul_points_red = max(0, int(red_fouls))
        self._foul_points_blue = max(0, int(blue_fouls))

    def get_attributed_scores(
        self,
        gui_red_score: Optional[int] = None,
        gui_blue_score: Optional[int] = None,
    ) -> Dict[str, AllianceScoreAttribution]:
        """
        Calculates current attributed scores for Red and Blue alliances and active robots,
        comparing against clean target scores (GUI OCR score - foul points).
        """
        red_gui = gui_red_score if gui_red_score is not None else self._latest_gui_red
        blue_gui = gui_blue_score if gui_blue_score is not None else self._latest_gui_blue

        target_red = max(0, red_gui - self._foul_points_red)
        target_blue = max(0, blue_gui - self._foul_points_blue)

        red_bot_scores: Dict[str, int] = collections.defaultdict(int)
        blue_bot_scores: Dict[str, int] = collections.defaultdict(int)

        attr_red = 0
        attr_blue = 0

        for s in self.confirmed_shots:
            if s.is_scored:
                pts = s.score_increment
                bot_key = s.robot_team or (f"Bot_{s.robot_track_id}" if s.robot_track_id is not None else f"{s.hub_alliance.value}_Unknown")
                if s.hub_alliance == HubAlliance.RED:
                    attr_red += pts
                    red_bot_scores[bot_key] += pts
                elif s.hub_alliance == HubAlliance.BLUE:
                    attr_blue += pts
                    blue_bot_scores[bot_key] += pts

        return {
            "RED": AllianceScoreAttribution(
                alliance=HubAlliance.RED,
                gui_ocr_score=red_gui,
                foul_points=self._foul_points_red,
                clean_target_score=target_red,
                current_attributed_score=attr_red,
                deficit_or_surplus=target_red - attr_red,
                robot_scores=dict(red_bot_scores),
            ),
            "BLUE": AllianceScoreAttribution(
                alliance=HubAlliance.BLUE,
                gui_ocr_score=blue_gui,
                foul_points=self._foul_points_blue,
                clean_target_score=target_blue,
                current_attributed_score=attr_blue,
                deficit_or_surplus=target_blue - attr_blue,
                robot_scores=dict(blue_bot_scores),
            ),
        }

    def reconcile_score_attribution(
        self,
        gui_red_score: int,
        gui_blue_score: int,
        tracked_robots: Sequence[Dict[str, Any]],
        frame_idx: int = 0,
        timestamp_s: float = 0.0,
        phase_id: int = 2,
    ) -> Dict[str, AllianceScoreAttribution]:
        """
        Reconciles the current attributed score against (GUI OCR score - alliance foul points).

        - Deficit (target > current): Vision under-counted balls.
          Allocates missing deficit points to the robot actively in the Hub zone
          (or closest friendly robot in SCORING state), creating confirmed reconciled ShotEvents.
        - Surplus (target < current): Vision over-counted balls.
          Marks lowest confidence excess events as unconfirmed to bound the sum exactly.
        """
        self._latest_gui_red = max(0, gui_red_score)
        self._latest_gui_blue = max(0, gui_blue_score)

        for alliance, hub, gui_sc, foul_pts in [
            (HubAlliance.RED, self.red_hub, self._latest_gui_red, self._foul_points_red),
            (HubAlliance.BLUE, self.blue_hub, self._latest_gui_blue, self._foul_points_blue),
        ]:
            target_clean = max(0, gui_sc - foul_pts)
            current_shots = [s for s in self.confirmed_shots if s.hub_alliance == alliance and s.is_scored]
            current_attributed = sum(s.score_increment for s in current_shots)

            diff = target_clean - current_attributed

            if diff > 0:
                # Under-counting deficit: Allocate diff points to active shooter robot
                friendly_robots = [r for r in tracked_robots if r.get("alliance") == alliance.value]
                if not friendly_robots:
                    friendly_robots = list(tracked_robots)

                best_bot: Optional[Dict[str, Any]] = None
                best_dist = 12.0
                if friendly_robots:
                    hx = (hub.screen_bbox[0] + hub.screen_bbox[2]) / 2.0
                    hy = (hub.screen_bbox[1] + hub.screen_bbox[3]) / 2.0
                    scoring_bots = [r for r in friendly_robots if str(r.get("state", "")).upper() == "SCORING"]
                    pool = scoring_bots if scoring_bots else friendly_robots
                    scored_candidates = []
                    for r in pool:
                        rc = r.get("center", (0.0, 0.0))
                        d = math.hypot(rc[0] - hx, rc[1] - hy)
                        scored_candidates.append((d, r))
                    scored_candidates.sort(key=lambda x: x[0])
                    best_dist = scored_candidates[0][0] / 30.0  # approximate ft
                    best_bot = scored_candidates[0][1]

                for k in range(diff):
                    recon_event = ShotEvent(
                        event_id=f"recon_{alliance.value.lower()}_{frame_idx}_{k}_{len(self.confirmed_shots)}",
                        frame_index=frame_idx,
                        timestamp_s=timestamp_s,
                        phase_id=phase_id,
                        hub_alliance=alliance,
                        ball_x=float((hub.screen_bbox[0] + hub.screen_bbox[2]) / 2.0),
                        ball_y=float((hub.screen_bbox[1] + hub.screen_bbox[3]) / 2.0),
                        robot_track_id=best_bot.get("track_id") if best_bot else None,
                        robot_team=best_bot.get("team_number") if best_bot else None,
                        robot_alliance=best_bot.get("alliance", alliance.value) if best_bot else alliance.value,
                        distance_ft=float(max(4.0, best_dist)),
                        is_open=True,
                        is_scored=True,
                        score_increment=1,
                        confidence=0.88,
                    )
                    self.confirmed_shots.append(recon_event)

            elif diff < 0:
                # Over-counting surplus: Prune surplus lowest-confidence scored events
                surplus = abs(diff)
                current_shots.sort(key=lambda s: s.confidence)
                pruned_count = 0
                for s in current_shots:
                    if pruned_count >= surplus:
                        break
                    s.is_scored = False
                    pruned_count += s.score_increment

        return self.get_attributed_scores()

    def set_hub_screen_bboxes(
        self,
        red_bbox: Optional[Tuple[float, float, float, float]] = None,
        blue_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Dynamically updates the screen catchment bounding boxes for both Hubs."""
        if red_bbox is not None:
            self.red_hub.screen_bbox = red_bbox
            self._persistent_red_bbox = red_bbox
        if blue_bbox is not None:
            self.blue_hub.screen_bbox = blue_bbox
            self._persistent_blue_bbox = blue_bbox

    def update_from_tags(
        self,
        detected_tags: Sequence[Any],
        frame_shape: Tuple[int, int] = (1080, 1920),
    ) -> None:
        """
        Dynamically updates Hub catchment bounding boxes based strictly on detected Hub AprilTags.

        Rules:
        - Red Hub structure tags: paired faces 8/5, 9/10, 11/2, 4/3 ({8, 5, 9, 10, 11, 2, 4, 3}).
        - Blue Hub structure tags: paired faces 18/27, 19/20, 21/24, 26/25 ({18, 27, 19, 20, 21, 24, 26, 25}).
        - Perimeter / driver station tags are NEVER used to position Hubs.
        - The scoring catchment area is directly ABOVE the Hub AprilTags.
        - If Hub tags are temporarily occluded, mutual relative triangulation or persistent memory is used.
        """
        h_frame, w_frame = frame_shape
        w_box, h_box = 230.0, 170.0

        red_struct = {8, 5, 9, 10, 11, 2, 4, 3}
        blue_struct = {18, 27, 19, 20, 21, 24, 26, 25}

        tag_items = []
        for t in detected_tags:
            tid = getattr(t, "tag_id", None)
            c = getattr(t, "center_px", None) or getattr(t, "image_center_px", None)
            if tid is not None and c is not None:
                tag_items.append((int(tid), float(c[0]), float(c[1])))

        # Filter strictly for tags on the Hub structures (never fall back to perimeter wall tags)
        red_tags = [t for t in tag_items if t[0] in red_struct]
        blue_tags = [t for t in tag_items if t[0] in blue_struct]

        # 1. Update tag sighting history
        for t in red_tags:
            self._hub_tag_history_red[t[0]].append((t[1], t[2]))
        for t in blue_tags:
            self._hub_tag_history_blue[t[0]].append((t[1], t[2]))

        # 2. Compute observed bounding boxes
        red_bbox_obs = None
        if red_tags:
            cx = float(np.mean([t[1] for t in red_tags]))
            top_y = float(min(t[2] for t in red_tags))
            center_y = top_y - 110.0
            x1 = np.clip(cx - w_box / 2.0, 0, w_frame - 1)
            y1 = np.clip(center_y - h_box / 2.0, 0, h_frame - 1)
            x2 = np.clip(cx + w_box / 2.0, 0, w_frame - 1)
            y2 = np.clip(center_y + h_box / 2.0, 0, h_frame - 1)
            red_bbox_obs = (float(x1), float(y1), float(x2), float(y2))
            if self._persistent_red_bbox is None:
                self._persistent_red_bbox = red_bbox_obs
            else:
                self._persistent_red_bbox = tuple(
                    0.85 * p + 0.15 * c for p, c in zip(self._persistent_red_bbox, red_bbox_obs)
                )

        blue_bbox_obs = None
        if blue_tags:
            cx = float(np.mean([t[1] for t in blue_tags]))
            top_y = float(min(t[2] for t in blue_tags))
            center_y = top_y - 110.0
            x1 = np.clip(cx - w_box / 2.0, 0, w_frame - 1)
            y1 = np.clip(center_y - h_box / 2.0, 0, h_frame - 1)
            x2 = np.clip(cx + w_box / 2.0, 0, w_frame - 1)
            y2 = np.clip(center_y + h_box / 2.0, 0, h_frame - 1)
            blue_bbox_obs = (float(x1), float(y1), float(x2), float(y2))
            if self._persistent_blue_bbox is None:
                self._persistent_blue_bbox = blue_bbox_obs
            else:
                self._persistent_blue_bbox = tuple(
                    0.85 * p + 0.15 * c for p, c in zip(self._persistent_blue_bbox, blue_bbox_obs)
                )

        # 3. Update mutual triangulation displacement vector between Blue and Red Hubs
        if red_bbox_obs is not None and blue_bbox_obs is not None:
            rcx = (red_bbox_obs[0] + red_bbox_obs[2]) / 2.0
            rcy = (red_bbox_obs[1] + red_bbox_obs[3]) / 2.0
            bcx = (blue_bbox_obs[0] + blue_bbox_obs[2]) / 2.0
            bcy = (blue_bbox_obs[1] + blue_bbox_obs[3]) / 2.0
            delta_xy = (rcx - bcx, rcy - bcy)
            if self._hub_relative_delta_xy is None:
                self._hub_relative_delta_xy = delta_xy
            else:
                self._hub_relative_delta_xy = (
                    0.90 * self._hub_relative_delta_xy[0] + 0.10 * delta_xy[0],
                    0.90 * self._hub_relative_delta_xy[1] + 0.10 * delta_xy[1],
                )

        # 4. Finalize Red Hub screen bbox (observation -> triangulation -> persistent)
        if red_bbox_obs is not None:
            self.red_hub.screen_bbox = red_bbox_obs
        elif blue_bbox_obs is not None and self._hub_relative_delta_xy is not None:
            bcx = (blue_bbox_obs[0] + blue_bbox_obs[2]) / 2.0
            bcy = (blue_bbox_obs[1] + blue_bbox_obs[3]) / 2.0
            rcx = bcx + self._hub_relative_delta_xy[0]
            rcy = bcy + self._hub_relative_delta_xy[1]
            self.red_hub.screen_bbox = (
                float(np.clip(rcx - w_box / 2.0, 0, w_frame - 1)),
                float(np.clip(rcy - h_box / 2.0, 0, h_frame - 1)),
                float(np.clip(rcx + w_box / 2.0, 0, w_frame - 1)),
                float(np.clip(rcy + h_box / 2.0, 0, h_frame - 1)),
            )
        elif self._persistent_red_bbox is not None:
            self.red_hub.screen_bbox = self._persistent_red_bbox

        # 5. Finalize Blue Hub screen bbox (observation -> triangulation -> persistent)
        if blue_bbox_obs is not None:
            self.blue_hub.screen_bbox = blue_bbox_obs
        elif red_bbox_obs is not None and self._hub_relative_delta_xy is not None:
            rcx = (red_bbox_obs[0] + red_bbox_obs[2]) / 2.0
            rcy = (red_bbox_obs[1] + red_bbox_obs[3]) / 2.0
            bcx = rcx - self._hub_relative_delta_xy[0]
            bcy = rcy - self._hub_relative_delta_xy[1]
            self.blue_hub.screen_bbox = (
                float(np.clip(bcx - w_box / 2.0, 0, w_frame - 1)),
                float(np.clip(bcy - h_box / 2.0, 0, h_frame - 1)),
                float(np.clip(bcx + w_box / 2.0, 0, w_frame - 1)),
                float(np.clip(bcy + h_box / 2.0, 0, h_frame - 1)),
            )
        elif self._persistent_blue_bbox is not None:
            self.blue_hub.screen_bbox = self._persistent_blue_bbox

    def get_hub_tag_most_common_locations(self) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """Returns the empirical median (most common) screen locations for all observed Hub structure tags."""
        res: Dict[str, Dict[int, Tuple[float, float]]] = {"red": {}, "blue": {}}
        for tid, pts in self._hub_tag_history_red.items():
            if pts:
                med_x = float(np.median([p[0] for p in pts]))
                med_y = float(np.median([p[1] for p in pts]))
                res["red"][tid] = (med_x, med_y)
        for tid, pts in self._hub_tag_history_blue.items():
            if pts:
                med_x = float(np.median([p[0] for p in pts]))
                med_y = float(np.median([p[1] for p in pts]))
                res["blue"][tid] = (med_x, med_y)
        return res

    def register_score_tick(
        self,
        timestamp_s: float,
        red_score: int,
        blue_score: int,
    ) -> Tuple[int, int]:
        """
        Registers an updated score reading from TBA / FMS.
        Returns (red_increment, blue_increment).
        """
        red_delta = max(0, red_score - self._last_red_score)
        blue_delta = max(0, blue_score - self._last_blue_score)

        if red_delta > 0 or blue_delta > 0:
            self._score_history.append((timestamp_s, red_score, blue_score, red_delta, blue_delta))
            logger.debug(
                "Score increment @ %.2fs: +%d Red (now %d), +%d Blue (now %d)",
                timestamp_s, red_delta, red_score, blue_delta, blue_score,
            )

        self._last_red_score = red_score
        self._last_blue_score = blue_score
        return red_delta, blue_delta

    def process_frame(
        self,
        frame_idx: int,
        timestamp_s: float,
        ball_positions: Sequence[Tuple[float, float]],  # Pixel (cx, cy) or field (x, y)
        tracked_robots: Sequence[Dict[str, Any]],  # [{track_id, center, alliance, ...}]
        use_screen_coords: bool = True,
        is_autonomous: bool = False,
        scoring_allowed: bool = True,
        gui_score_delta: Optional[Tuple[int, int]] = None,
        gui_total_scores: Optional[Tuple[int, int]] = None,
        foul_points: Optional[Tuple[int, int]] = None,
        auto_reconcile: bool = True,
    ) -> List[ShotEvent]:
        """
        Evaluates ball detections in the current frame using Kalman ball tracklets.
        Enforces single-count scoring per ball across frames, suppresses scoring during grace periods,
        syncs against live GUI score increments, and reconciles attributed scores to GUI OCR minus fouls.
        """
        if foul_points is not None:
            self.set_foul_points(red_fouls=foul_points[0], blue_fouls=foul_points[1])
        # Determine match phase (1..5)
        if is_autonomous or timestamp_s < 15.0:
            phase_id = 1  # Auto
        elif timestamp_s < 50.0:
            phase_id = 2  # Early Teleop
        elif timestamp_s < 90.0:
            phase_id = 3  # Mid Teleop
        elif timestamp_s < 120.0:
            phase_id = 4  # Late Teleop
        else:
            phase_id = 5  # Endgame

        dt = (timestamp_s - self._last_processed_ts) if self._last_processed_ts is not None else 0.033
        dt = max(1e-4, min(0.25, dt))
        self._last_processed_ts = timestamp_s

        # Determine recent score increments
        recent_red_delta = 0
        recent_blue_delta = 0
        if gui_score_delta is not None:
            recent_red_delta = max(0, gui_score_delta[0])
            recent_blue_delta = max(0, gui_score_delta[1])
        else:
            for ts, _, _, r_d, b_d in self._score_history:
                if abs(timestamp_s - ts) <= self.scoring_window_s:
                    recent_red_delta += r_d
                    recent_blue_delta += b_d

        # Separate candidates into Red and Blue approach zones
        red_candidates: List[Tuple[float, float]] = []
        blue_candidates: List[Tuple[float, float]] = []
        for bx, by in ball_positions:
            if use_screen_coords:
                if self.red_hub.intersects_point_screen_approach(bx, by):
                    red_candidates.append((bx, by))
                elif self.blue_hub.intersects_point_screen_approach(bx, by):
                    blue_candidates.append((bx, by))
            else:
                if self.red_hub.intersects_point_field(bx, by, tolerance_inch=24.0):
                    red_candidates.append((bx, by))
                elif self.blue_hub.intersects_point_field(bx, by, tolerance_inch=24.0):
                    blue_candidates.append((bx, by))

        new_events: List[ShotEvent] = []

        # Process Red Hub
        red_events = self._update_alliance_tracklets(
            alliance=HubAlliance.RED,
            hub=self.red_hub,
            candidates=red_candidates,
            tracked_robots=tracked_robots,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            dt=dt,
            phase_id=phase_id,
            scoring_allowed=scoring_allowed,
            recent_delta=recent_red_delta,
            use_screen_coords=use_screen_coords,
        )
        new_events.extend(red_events)

        # Process Blue Hub
        blue_events = self._update_alliance_tracklets(
            alliance=HubAlliance.BLUE,
            hub=self.blue_hub,
            candidates=blue_candidates,
            tracked_robots=tracked_robots,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            dt=dt,
            phase_id=phase_id,
            scoring_allowed=scoring_allowed,
            recent_delta=recent_blue_delta,
            use_screen_coords=use_screen_coords,
        )
        new_events.extend(blue_events)

        if auto_reconcile and gui_total_scores is not None:
            self.reconcile_score_attribution(
                gui_red_score=gui_total_scores[0],
                gui_blue_score=gui_total_scores[1],
                tracked_robots=tracked_robots,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                phase_id=phase_id,
            )

        return new_events

    def _update_alliance_tracklets(
        self,
        alliance: HubAlliance,
        hub: HubGeometry,
        candidates: Sequence[Tuple[float, float]],
        tracked_robots: Sequence[Dict[str, Any]],
        frame_idx: int,
        timestamp_s: float,
        dt: float,
        phase_id: int,
        scoring_allowed: bool,
        recent_delta: int,
        use_screen_coords: bool,
    ) -> List[ShotEvent]:
        """
        Updates Kalman ball tracklets for an alliance hub, enforces single-count guarantees,
        and attributes confirmed scoring events.
        """
        tracklets = (
            self.active_ball_tracklets_red
            if alliance == HubAlliance.RED
            else self.active_ball_tracklets_blue
        )
        new_events: List[ShotEvent] = []

        # 1. Kalman Predict step for all active tracklets
        for tr in tracklets.values():
            tr.predict(dt)

        # 2. Association between predicted tracklet positions and detected candidates
        max_dist = 48.0 if use_screen_coords else 28.0  # px or inches
        matched_tr_ids = set()
        matched_cand_indices = set()

        if tracklets and candidates:
            tr_list = list(tracklets.values())
            pair_dists: List[Tuple[float, int, int]] = []
            for tr in tr_list:
                tx, ty = tr.current_pos
                for c_idx, (cx, cy) in enumerate(candidates):
                    d = math.hypot(tx - cx, ty - cy)
                    if d <= max_dist:
                        pair_dists.append((d, tr.track_id, c_idx))

            pair_dists.sort(key=lambda p: p[0])
            for d, tid, c_idx in pair_dists:
                if tid not in matched_tr_ids and c_idx not in matched_cand_indices:
                    matched_tr_ids.add(tid)
                    matched_cand_indices.add(c_idx)
                    cx, cy = candidates[c_idx]
                    tracklets[tid].update(cx, cy)

        # 3. Handle unmatched candidates -> spawn new tracklets
        for c_idx, (cx, cy) in enumerate(candidates):
            if c_idx not in matched_cand_indices:
                tid = self._next_track_id
                self._next_track_id += 1
                new_tr = HubBallTracklet(
                    track_id=tid,
                    alliance=alliance,
                    state=np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32),
                    covariance=np.diag([10.0, 10.0, 30.0, 30.0]).astype(np.float32),
                    history=[(cx, cy)],
                    frames_seen=1,
                    frames_lost=0,
                    has_scored=False,
                    entry_frame=frame_idx,
                    entry_timestamp_s=timestamp_s,
                )
                tracklets[tid] = new_tr
                matched_tr_ids.add(tid)

        # 4. Handle unmatched existing tracklets -> coast or prune
        to_prune = []
        for tid, tr in tracklets.items():
            if tid not in matched_tr_ids:
                tr.frames_lost += 1
                if tr.frames_lost > 6:
                    to_prune.append(tid)
        for tid in to_prune:
            del tracklets[tid]

        # 5. Evaluate Scoring & Single-Count Guarantee
        for tr in tracklets.values():
            tx, ty = tr.current_pos
            in_catchment = (
                hub.intersects_point_screen_catchment(tx, ty)
                if use_screen_coords
                else hub.intersects_point_field(tx, ty)
            )

            if in_catchment:
                # If this tracklet has NOT been evaluated for scoring yet
                if tr.associated_shot_event_id is None:
                    is_scored = scoring_allowed and (recent_delta > 0)
                    shooter, dist_ft, is_open = self._attribute_shooter(
                        hub=hub,
                        ball_loc=(tx, ty),
                        tracked_robots=tracked_robots,
                        target_alliance=alliance.value,
                        use_screen_coords=use_screen_coords,
                    )
                    event_id = f"shot_{frame_idx}_{len(self.confirmed_shots) + len(new_events)}"
                    event = ShotEvent(
                        event_id=event_id,
                        frame_index=frame_idx,
                        timestamp_s=timestamp_s,
                        phase_id=phase_id,
                        hub_alliance=alliance,
                        ball_x=tx,
                        ball_y=ty,
                        robot_track_id=shooter.get("track_id") if shooter else None,
                        robot_team=shooter.get("team_number") if shooter else None,
                        robot_alliance=shooter.get("alliance", alliance.value) if shooter else alliance.value,
                        distance_ft=dist_ft,
                        is_open=is_open,
                        is_scored=is_scored,
                    )
                    tr.associated_shot_event_id = event_id
                    tr.has_scored = is_scored
                    if is_scored:
                        tr.scoring_frame = frame_idx
                        tr.scoring_timestamp_s = timestamp_s
                    new_events.append(event)
                    self.confirmed_shots.append(event)
                elif not tr.has_scored:
                    # Upgrade previously unconfirmed shot if score increment arrives while ball is in catchment
                    if scoring_allowed and recent_delta > 0:
                        tr.has_scored = True
                        tr.scoring_frame = frame_idx
                        tr.scoring_timestamp_s = timestamp_s
                        for s in self.confirmed_shots:
                            if s.event_id == tr.associated_shot_event_id:
                                s.is_scored = True
                                break
        return new_events

    def _attribute_shooter(
        self,
        hub: HubGeometry,
        ball_loc: Tuple[float, float],
        tracked_robots: Sequence[Dict[str, Any]],
        target_alliance: str,
        use_screen_coords: bool,
    ) -> Tuple[Optional[Dict[str, Any]], float, bool]:
        """
        Finds the most likely shooting robot on target_alliance that delivered the ball.
        Computes shot distance (ft) and whether the robot was open (defender > 4.0 ft away).
        """
        candidates = [r for r in tracked_robots if r.get("alliance") == target_alliance]
        if not candidates:
            # Fallback to any robot
            candidates = list(tracked_robots)

        if not candidates:
            return None, 12.0, True

        # Find closest friendly robot to Hub
        best_robot = None
        min_dist_to_hub = float("inf")

        hub_center = (
            ((hub.screen_bbox[0] + hub.screen_bbox[2]) / 2.0, (hub.screen_bbox[1] + hub.screen_bbox[3]) / 2.0)
            if use_screen_coords else
            (hub.field_x_inch / 12.0, hub.field_y_inch / 12.0)
        )

        for bot in candidates:
            bot_pos = bot.get("center", (0.0, 0.0))
            if not use_screen_coords:
                bot_pos = (bot.get("field_x_inch", 0.0) / 12.0, bot.get("field_y_inch", 0.0) / 12.0)

            d = math.hypot(bot_pos[0] - hub_center[0], bot_pos[1] - hub_center[1])
            if d < min_dist_to_hub:
                min_dist_to_hub = d
                best_robot = bot

        # Approximate distance in feet (if screen coords, scale by ~0.035 ft/px)
        distance_ft = min_dist_to_hub if not use_screen_coords else max(4.0, min_dist_to_hub * 0.038)

        # Check defender proximity
        is_open = True
        if best_robot:
            opp_alliance = "BLUE" if target_alliance == "RED" else "RED"
            opponents = [r for r in tracked_robots if r.get("alliance") == opp_alliance]
            b_center = best_robot.get("center", (0.0, 0.0))
            for opp in opponents:
                o_center = opp.get("center", (0.0, 0.0))
                opp_d_px = math.hypot(b_center[0] - o_center[0], b_center[1] - o_center[1])
                opp_d_ft = opp_d_px * 0.038 if use_screen_coords else opp_d_px / 12.0
                if opp_d_ft <= 4.0:
                    is_open = False
                    break

        return best_robot, distance_ft, is_open

    def get_phase_shot_stats(self, phase_id: int, alliance: Optional[str] = None) -> Dict[str, Any]:
        """Calculates shot counts, scoring accuracy, and distance breakdown for a phase."""
        shots = [s for s in self.confirmed_shots if s.phase_id == phase_id]
        if alliance:
            shots = [s for s in shots if s.robot_alliance == alliance]

        total_shots = len(shots)
        scored_shots = sum(1 for s in shots if s.is_scored)
        accuracy = (scored_shots / total_shots) if total_shots > 0 else 0.0

        open_shots = [s for s in shots if s.is_open]
        def_shots = [s for s in shots if not s.is_open]

        open_acc = (sum(1 for s in open_shots if s.is_scored) / len(open_shots)) if open_shots else 0.0
        def_acc = (sum(1 for s in def_shots if s.is_scored) / len(def_shots)) if def_shots else 0.0

        return {
            "total_shots": total_shots,
            "scored_shots": scored_shots,
            "missed_shots": total_shots - scored_shots,
            "accuracy": accuracy,
            "open_shots": len(open_shots),
            "open_accuracy": open_acc,
            "defended_shots": len(def_shots),
            "defended_accuracy": def_acc,
            "accuracy_dropoff": open_acc - def_acc,
            "shots": shots,
        }
