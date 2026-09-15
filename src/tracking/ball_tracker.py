"""
Multi-Object Ball Tracking & Occlusion Management Engine.

Tracks up to 504 individual yellow game pieces ("FUEL" / balls) per frame featuring:
- 6-State Kalman Filter with carpet friction dampening: [xc, yc, vx, vy, r, vr]
- Sub-millisecond spatial association via scipy.spatial.cKDTree
- Robot bumper/intake occlusion linking and possession state detection
- Scoring hub aperture transit detection
- Velocity-directed forward coasting across occlusions
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.common.types import (
    BallCluster,
    BallState,
    BallTrackletState,
    BoundingBox,
    HomographyEstimate,
)
from src.vision.homography import project_pixel_to_field_3d

logger = logging.getLogger("frc.tracking.ball_tracker")

BALL_CLASS_ID: int = 1
FUEL_DIAMETER_INCH: float = 5.91
FUEL_RADIUS_INCH: float = 2.955
CARPET_CENTER_ELEVATION_INCH: float = 2.955


class SingleBallKalman:
    """
    6-State Kalman filter for an individual game piece:
    x = [xc, yc, vx, vy, r, vr]^T
    Supports parabolic ballistic arcs under downward gravity during flight,
    and carpet rolling friction dampening when on carpet.
    """

    __slots__ = ("_x", "_y", "_vx", "_vy", "_r", "_vr", "friction", "g_px", "dt")

    def __init__(self, cx: float, cy: float, radius: float, dt: float = 1.0 / 60.0, g_px: float = 1.2) -> None:
        self.dt = dt
        self.friction = 0.96  # Carpet friction decay per frame for rolling balls
        self.g_px = g_px      # Downward vertical gravity acceleration (~386 in/s^2)
        self._x = float(cx)
        self._y = float(cy)
        self._vx = 0.0
        self._vy = 0.0
        self._r = float(radius)
        self._vr = 0.0

    @property
    def x(self) -> np.ndarray:
        return np.array([self._x, self._y, self._vx, self._vy, self._r, self._vr], dtype=np.float32)

    def predict(self, state: BallState = BallState.ON_CARPET) -> Tuple[float, float]:
        """
        Projects state forward by dt.
        Applies parabolic gravity acceleration for in-flight balls,
        and friction dampening for rolling carpet balls.
        """
        if state == BallState.IN_FLIGHT:
            self._vy += self.g_px
            self._vx *= 0.995  # Aerodynamic drag
            self._vy *= 0.995
            self._x += self._vx
            self._y += self._vy
        else:
            self._x += self._vx
            self._y += self._vy
            self._vx *= self.friction
            self._vy *= self.friction

        self._r = float(min(16.0, max(3.0, self._r + self._vr)))
        self._vr *= self.friction
        return self._x, self._y

    def update(self, cx: float, cy: float, radius: float, confidence: float = 1.0) -> None:
        """Updates Kalman state with new measurement, scaling gains with confidence."""
        conf_clamped = max(0.15, min(1.0, confidence))
        beta = 0.70 * conf_clamped
        err_x = float(cx) - self._x
        err_y = float(cy) - self._y

        # Update position directly to measurement while smoothing velocity
        self._x = float(cx)
        self._y = float(cy)
        self._vx = self._vx * self.friction + beta * err_x
        self._vy = self._vy * self.friction + beta * err_y
        self._r = float(min(16.0, max(3.0, radius)))
        self._vr = 0.0


class BallTrack:
    """Individual ball track lifecycle record."""

    def __init__(
        self,
        ball_id: int,
        cx: float,
        cy: float,
        radius: float,
        confidence: float,
        frame_index: int,
        dt: float = 1.0 / 60.0,
    ) -> None:
        self.ball_id = ball_id
        self.kalman = SingleBallKalman(cx, cy, radius, dt=dt)
        self.confidence = confidence
        self.start_frame = frame_index
        self.last_frame = frame_index
        self.coast_frames = 0
        self.state: BallState = BallState.ON_CARPET
        self.occluded_by_robot_id: Optional[int] = None

        # Pixel state
        self.center_x = cx
        self.center_y = cy
        self.radius = radius
        self.vx_px = 0.0
        self.vy_px = 0.0

        # Field state (inches)
        self.field_x = 0.0
        self.field_y = 0.0
        self.field_vx = 0.0
        self.field_vy = 0.0

    def predict(self) -> Tuple[float, float]:
        px, py = self.kalman.predict(state=self.state)
        self.center_x = px
        self.center_y = py
        self.vx_px = self.kalman._vx
        self.vy_px = self.kalman._vy
        self.radius = self.kalman._r
        return px, py

    def update_matched(
        self,
        cx: float,
        cy: float,
        radius: float,
        confidence: float,
        frame_index: int,
    ) -> None:
        self.kalman.update(cx, cy, radius, confidence=confidence)
        self.center_x = self.kalman._x
        self.center_y = self.kalman._y
        self.vx_px = self.kalman._vx
        self.vy_px = self.kalman._vy
        self.radius = self.kalman._r
        self.confidence = confidence
        self.last_frame = frame_index
        self.coast_frames = 0
        self.occluded_by_robot_id = None

        # Speed check for in-flight projectile state (scaled to per-frame velocity at 60 FPS)
        speed_px = np.hypot(self.vx_px, self.vy_px)
        if speed_px > 6.0 or (self.vy_px < -2.0 and speed_px > 3.0):
            self.state = BallState.IN_FLIGHT
        else:
            self.state = BallState.ON_CARPET


class BallTracker:
    """
    High-capacity multi-object tracker managing up to 504 concurrent yellow ball tracks.
    """

    def __init__(
        self,
        max_balls: int = 504,
        distance_match_thresh_px: float = 25.0,
        max_coast_frames: int = 20,
        target_fps: float = 60.0,
    ) -> None:
        self.max_balls = max_balls
        self.distance_match_thresh = distance_match_thresh_px
        self.max_coast_frames = max_coast_frames
        self.dt = 1.0 / target_fps

        self._next_id: int = 1
        self._tracks: Dict[int, BallTrack] = {}

    @property
    def active_track_count(self) -> int:
        return len([t for t in self._tracks.values() if t.state != BallState.REMOVED])

    def update(
        self,
        detections: List[BoundingBox],
        frame_index: int,
        homography: Optional[HomographyEstimate] = None,
        robot_boxes: Optional[Sequence[BoundingBox]] = None,
        clusters: Optional[Sequence[BallCluster]] = None,
    ) -> List[BallTrackletState]:
        """
        Processes new detections, executes k-d tree spatial matching, applies occlusion guards,
        and returns the current state of all active ball tracklets and ambient clusters.

        Args:
            detections: List of BoundingBox detections for the current frame.
            frame_index: Current frame index.
            homography: Optional HomographyEstimate for field coordinate projection.
            robot_boxes: Optional sequence of active robot bounding boxes for occlusion linking.
            clusters: Optional sequence of ambient BallCluster instances for bulk carpet tracking.

        Returns:
            List of BallTrackletState objects.
        """
        active_tracks = [t for t in self._tracks.values() if t.state != BallState.REMOVED]

        # 1. Predict next state for all active tracks
        for t in active_tracks:
            t.predict()

        # 2. Spatial Association via scipy.spatial.cKDTree
        det_pts = np.array(
            [[b.center_x, b.center_y] for b in detections],
            dtype=np.float32,
        ) if detections else np.empty((0, 2), dtype=np.float32)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        if len(active_tracks) > 0 and len(det_pts) > 0:
            tree = cKDTree(det_pts)
            track_pts = np.array(
                [[t.center_x, t.center_y] for t in active_tracks],
                dtype=np.float32,
            )

            distances, indices = tree.query(
                track_pts,
                distance_upper_bound=self.distance_match_thresh,
            )

            valid_mask = (indices < len(detections)) & np.isfinite(distances)
            best_trk_for_det: Dict[int, Tuple[float, int]] = {}
            for trk_idx in np.where(valid_mask)[0]:
                d_idx = int(indices[trk_idx])
                dist = float(distances[trk_idx])
                if d_idx not in best_trk_for_det or dist < best_trk_for_det[d_idx][0]:
                    best_trk_for_det[d_idx] = (dist, trk_idx)

            for d_idx, (_, trk_idx) in best_trk_for_det.items():
                trk = active_tracks[trk_idx]
                det = detections[d_idx]
                trk.update_matched(
                    cx=det.center_x,
                    cy=det.center_y,
                    radius=det.width / 2.0,
                    confidence=det.confidence,
                    frame_index=frame_index,
                )
                matched_tracks.add(trk.ball_id)
                matched_detections.add(d_idx)

        # 3. Handle Unmatched Tracks with Occlusion Methods
        for trk in active_tracks:
            if trk.ball_id in matched_tracks:
                continue

            trk.coast_frames += 1

            # Check Robot Under-Chassis Occlusion / Intake Linking
            occluding_robot_id: Optional[int] = None
            if robot_boxes:
                for r_box in robot_boxes:
                    half_w = (r_box.width * 0.6)  # Generous intake catchment zone
                    half_h = (r_box.height * 0.6)
                    rx1, rx2 = r_box.center_x - half_w, r_box.center_x + half_w
                    ry1, ry2 = r_box.center_y - half_h, r_box.center_y + half_h

                    if rx1 <= trk.center_x <= rx2 and ry1 <= trk.center_y <= ry2:
                        # Ball has passed beneath this robot
                        occluding_robot_id = getattr(r_box, "class_id", 0)
                        break

            if occluding_robot_id is not None:
                trk.occluded_by_robot_id = occluding_robot_id
                trk.state = BallState.INTAKED
            elif trk.state == BallState.IN_FLIGHT:
                # In-flight ball coasting through parabolic ballistic arc
                if trk.center_y >= 540.0:  # Reached floor carpet plane
                    trk.state = BallState.ON_CARPET
            else:
                trk.state = BallState.LOST

            # Prune tracks that exceeded max coast frames
            if trk.coast_frames > self.max_coast_frames:
                trk.state = BallState.REMOVED

        # 4. Initialize New Tracks for Unmatched Detections (Up to max_balls)
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue
            if self.active_track_count >= self.max_balls:
                break

            new_track = BallTrack(
                ball_id=self._next_id,
                cx=det.center_x,
                cy=det.center_y,
                radius=det.width / 2.0,
                confidence=det.confidence,
                frame_index=frame_index,
                dt=self.dt,
            )
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        # 5. Project to Field Coordinates (using dynamic Homography matrix if present)
        results: List[BallTrackletState] = []
        h_mat = homography.h_matrix if (homography and homography.h_matrix is not None) else None
        h_inv = np.linalg.inv(h_mat) if h_mat is not None else None

        for t in self._tracks.values():
            if t.state == BallState.REMOVED:
                continue

            # Field projection (3D ray projection at carpet ball center elevation Z=2.955" or in-flight 3D elevation)
            ball_z = CARPET_CENTER_ELEVATION_INCH
            ball_depth = 0.0
            if homography is not None and homography.camera_center is not None and homography.camera_pose_rvec is not None:
                f_cam = homography.focal_length or 1250.6
                K_cam = np.array([[f_cam, 0.0, 960.0], [0.0, f_cam, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)

                # 1. 3D optical ray in world space from camera optical center
                R_cam, _ = cv2.Rodrigues(homography.camera_pose_rvec)
                C_cam = homography.camera_center
                p_cam = np.linalg.inv(K_cam) @ np.array([t.center_x, t.center_y, 1.0], dtype=np.float64)
                d_world = (R_cam.T @ p_cam).ravel()
                ray_norm = np.linalg.norm(d_world)
                d_unit = d_world / ray_norm if ray_norm > 1e-9 else np.array([0.0, 0.0, -1.0])

                # 2. Floor carpet plane intersection (Z = 2.955")
                c_3d_carpet, depth_carpet = project_pixel_to_field_3d(
                    pixel_u=t.center_x,
                    pixel_v=t.center_y,
                    elevation_inch=CARPET_CENTER_ELEVATION_INCH,
                    rvec=homography.camera_pose_rvec,
                    tvec=homography.camera_pose_tvec,
                    camera_matrix=K_cam,
                    camera_center=homography.camera_center,
                    h_floor=homography.h_matrix,
                )

                # 3. Known FUEL sphere 5.91" diameter apparent depth calculation:
                # Apparent depth: lambda = (f * 2.955) / r_px
                r_obs = max(1.5, t.radius)
                lambda_pinhole = float(f_cam * FUEL_RADIUS_INCH / r_obs)
                P_pinhole = C_cam + lambda_pinhole * d_unit
                z_pinhole = float(P_pinhole[2])

                # 4. In-Flight vs Carpet classification:
                # If apparent elevation Z > 4.5" (and within playing envelope Z <= 120") and within field bounds, or track is in flight:
                if (z_pinhole > 4.5 and z_pinhole <= 120.0 and 0.0 <= P_pinhole[0] <= 651.2 and 0.0 <= P_pinhole[1] <= 317.7) or t.state == BallState.IN_FLIGHT:
                    t.state = BallState.IN_FLIGHT
                    t.field_x = float(P_pinhole[0])
                    t.field_y = float(P_pinhole[1])
                    ball_z = float(np.clip(z_pinhole, 4.5, 95.0))
                    ball_depth = float(lambda_pinhole)
                else:
                    t.field_x = float(c_3d_carpet[0])
                    t.field_y = float(c_3d_carpet[1])
                    ball_z = CARPET_CENTER_ELEVATION_INCH
                    ball_depth = float(depth_carpet)
            elif h_inv is not None:
                p_px = np.array([t.center_x, t.center_y, 1.0], dtype=np.float64)
                p_field = h_inv @ p_px
                if abs(p_field[2]) > 1e-6:
                    fx = float(p_field[0] / p_field[2])
                    fy = float(p_field[1] / p_field[2])
                    t.field_x = fx
                    t.field_y = fy

            # Scale velocities to inches/sec based on camera scale approximation
            t.field_vx = float(t.vx_px * 0.15)
            t.field_vy = float(t.vy_px * 0.15)

            bbox = BoundingBox(
                center_x=t.center_x,
                center_y=t.center_y,
                width=t.radius * 2.0,
                height=t.radius * 2.0,
                confidence=t.confidence,
                class_id=BALL_CLASS_ID,
            )

            cr_x = float(t.field_x - 325.6)
            cr_y = float(t.field_y - 158.85)
            cr_z = float(ball_z)

            results.append(
                BallTrackletState(
                    ball_id=t.ball_id,
                    pixel_bbox=bbox,
                    field_x_inch=t.field_x,
                    field_y_inch=t.field_y,
                    velocity_x_ips=t.field_vx,
                    velocity_y_ips=t.field_vy,
                    radius_px=t.radius,
                    state=t.state,
                    field_z_inch=ball_z,
                    depth_inch=ball_depth,
                    center_rel_x_inch=cr_x,
                    center_rel_y_inch=cr_y,
                    center_rel_z_inch=cr_z,
                    occluded_by_robot_id=t.occluded_by_robot_id,
                    coast_frames=t.coast_frames,
                    is_cluster=False,
                    cluster_ball_count=1,
                )
            )

        # 6. Append Ambient Cluster representations if provided, strictly respecting max_balls ceiling
        if clusters:
            cur_balls = len(results)
            remaining_quota = max(0, self.max_balls - cur_balls)
            for c in clusters:
                if remaining_quota <= 0:
                    break
                c_field_x, c_field_y, c_field_z = 0.0, 0.0, CARPET_CENTER_ELEVATION_INCH
                c_depth = 0.0
                if homography is not None and homography.camera_center is not None and homography.camera_pose_rvec is not None:
                    f_cam = homography.focal_length or 1250.6
                    K_cam = np.array([[f_cam, 0.0, 960.0], [0.0, f_cam, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
                    c_3d, c_depth = project_pixel_to_field_3d(
                        pixel_u=c.centroid_x,
                        pixel_v=c.centroid_y,
                        elevation_inch=CARPET_CENTER_ELEVATION_INCH,
                        rvec=homography.camera_pose_rvec,
                        tvec=homography.camera_pose_tvec,
                        camera_matrix=K_cam,
                        camera_center=homography.camera_center,
                        h_floor=homography.h_matrix,
                    )
                    c_field_x = c_3d[0]
                    c_field_y = c_3d[1]
                    c_field_z = c_3d[2]
                elif h_inv is not None:
                    p_px = np.array([c.centroid_x, c.centroid_y, 1.0], dtype=np.float64)
                    p_field = h_inv @ p_px
                    if abs(p_field[2]) > 1e-6:
                        c_field_x = float(p_field[0] / p_field[2])
                        c_field_y = float(p_field[1] / p_field[2])

                c_radius = max(6.0, float(np.sqrt(c.area_px / np.pi)))
                est_count = min(remaining_quota, c.estimated_ball_count)
                remaining_quota -= est_count
                results.append(
                    BallTrackletState(
                        ball_id=100000 + c.cluster_id,
                        pixel_bbox=c.bbox,
                        field_x_inch=c_field_x,
                        field_y_inch=c_field_y,
                        velocity_x_ips=0.0,
                        velocity_y_ips=0.0,
                        radius_px=c_radius,
                        state=BallState.ON_CARPET,
                        field_z_inch=c_field_z,
                        depth_inch=c_depth,
                        center_rel_x_inch=float(c_field_x - 325.6),
                        center_rel_y_inch=float(c_field_y - 158.85),
                        center_rel_z_inch=float(c_field_z),
                        occluded_by_robot_id=None,
                        coast_frames=0,
                        is_cluster=True,
                        cluster_ball_count=est_count,
                    )
                )

        return results
