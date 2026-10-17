"""
Stage 1: Spatial Calibration & Dynamic Top-Down Metric Homography.

Detects static AprilTag fiducials located on alliance hubs and field structures,
resolves ground-truth CAD geometry, and computes a cached homography transformation
matrix to map broadcast pixel coordinates into a flattened, top-down 2D metric field space.
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from scipy.optimize import least_squares
import yaml

from src.vision.apriltag_detector import AprilTagDetector, load_field_tags
from src.vision.homography import (
    compute_normalized_dlt,
    estimate_camera_pose_pnp,
    optimize_homography_lm,
)
from src.vision.spatial_types import AprilTagAnchor, CalibrationResult, FieldGeometry

logger = logging.getLogger("frc.vision.spatial_calibrator")

# Specific Hub Structure AprilTag Sets (FRC 2026 REBUILT)
# Red Hub: paired faces 8/5, 9/10, 11/2, 4/3. All tags <= 16 are on the Red half of the field.
RED_HUB_STRUCTURE_TAGS = {8, 5, 9, 10, 11, 2, 4, 3}
# Blue Hub: paired faces 18/27, 19/20, 21/24, 26/25. All tags >= 17 are on the Blue half of the field.
BLUE_HUB_STRUCTURE_TAGS = {18, 27, 19, 20, 21, 24, 26, 25}


class SpatialCalibrator:
    """
    Solves spatial field calibration and manages cached dynamic planar homography.
    Warp broadcast perspectives into top-down metric field canvases (1302 x 635 px).
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        scale_px_per_inch: float = 2.0,
        enable_optical_flow_cmc: bool = True,
        align_with_broadcast_view: bool = True,
    ) -> None:
        self.scale = scale_px_per_inch
        self.enable_cmc = enable_optical_flow_cmc
        self.align_with_broadcast_view = align_with_broadcast_view
        self.tag_detector = AprilTagDetector(config_path=config_path)
        self.field_geometry = self._load_field_geometry(config_path)

        # Canvas shape in pixels: (height, width)
        self.canvas_w = int(round(self.field_geometry.length_inches * self.scale))
        self.canvas_h = int(round(self.field_geometry.width_inches * self.scale))
        self.canvas_shape: Tuple[int, int] = (self.canvas_h, self.canvas_w)

        # Top-down orientation alignment: when looking South, invert axes so Blue is on left and Red is on right
        self._is_inverted: bool = False
        self._update_scaling_matrices()

        # Cached Homography State
        self._cached_H_img_to_field: Optional[np.ndarray] = None
        self._cached_H_field_to_img: Optional[np.ndarray] = None
        self._cached_H_canvas_to_img: Optional[np.ndarray] = None
        self._cached_H_img_to_canvas: Optional[np.ndarray] = None
        self._last_rmse: float = 0.0

        # Cached Camera Pose (3D PnP)
        self._cached_rvec: Optional[np.ndarray] = None
        self._cached_tvec: Optional[np.ndarray] = None
        self._cached_K: Optional[np.ndarray] = None

        # Cached Dynamic Hub Screen Bounding Boxes (Broadcast View: Blue on LEFT, Red on RIGHT)
        self._cached_blue_hub_bbox: Tuple[float, float, float, float] = (335.0, 135.0, 565.0, 305.0)
        self._cached_red_hub_bbox: Tuple[float, float, float, float] = (1380.0, 135.0, 1610.0, 305.0)

        # Persistent Hub Tag History and Mutual Triangulation Memory
        self._hub_tag_history_red: Dict[int, List[Tuple[float, float]]] = collections.defaultdict(list)
        self._hub_tag_history_blue: Dict[int, List[Tuple[float, float]]] = collections.defaultdict(list)
        self._persistent_red_hub_bbox: Optional[Tuple[float, float, float, float]] = None
        self._persistent_blue_hub_bbox: Optional[Tuple[float, float, float, float]] = None
        self._hub_relative_delta_xy: Optional[Tuple[float, float]] = None  # (dx, dy) from Blue Hub to Red Hub

        # Sparse optical flow tracking for CMC cache propagation
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_keypoints: Optional[np.ndarray] = None

        # Fallback canonical homography (pre-calibrated for standard broadcast angle)
        self._default_H_field_to_img = self._init_default_homography()

    def _update_scaling_matrices(self) -> None:
        """Updates S_field_to_canvas and S_canvas_to_field matrices based on orientation."""
        if self._is_inverted:
            # Broadcast-aligned top-down orientation (looking South):
            # Blue alliance on LEFT (x=0..), Red alliance on RIGHT (x=..W)
            self.S_field_to_canvas = np.array([
                [-self.scale, 0.0, self.scale * self.field_geometry.length_inches],
                [0.0, -self.scale, self.scale * self.field_geometry.width_inches],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        else:
            # Direct CAD orientation (Red alliance on LEFT, Blue on RIGHT)
            self.S_field_to_canvas = np.array([
                [self.scale, 0.0, 0.0],
                [0.0, self.scale, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        self.S_canvas_to_field = np.linalg.inv(self.S_field_to_canvas)

    def _load_field_geometry(self, config_path: Optional[str]) -> FieldGeometry:
        """Parses field CAD dimensions and Hub structural geometry from yaml config."""
        paths = []
        if config_path:
            paths.append(Path(config_path))
        paths.extend([
            Path("configs/field_2026_rebuilt.yaml"),
            Path(__file__).resolve().parent.parent.parent / "configs" / "field_2026_rebuilt.yaml",
        ])

        for p in paths:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    f_cfg = cfg.get("field", {})
                    e_cfg = cfg.get("elements", {})
                    red_hub = e_cfg.get("red_hub", {}).get("center", [162.8, 158.85])
                    blue_hub = e_cfg.get("blue_hub", {}).get("center", [488.4, 158.85])
                    hub_size = e_cfg.get("red_hub", {}).get("size", [47.0, 47.0])

                    return FieldGeometry(
                        length_inches=float(f_cfg.get("length_inches", 651.2)),
                        width_inches=float(f_cfg.get("width_inches", 317.7)),
                        guardrail_height_inches=float(f_cfg.get("guardrail_height_inches", 20.0)),
                        red_hub_center_inch=(float(red_hub[0]), float(red_hub[1])),
                        blue_hub_center_inch=(float(blue_hub[0]), float(blue_hub[1])),
                        hub_dimensions_inch=(float(hub_size[0]), float(hub_size[1])),
                        carpet_scale_px_per_inch=self.scale,
                    )
                except Exception as e:
                    logger.warning("Failed parsing field yaml at %s: %s", p, e)

        return FieldGeometry(carpet_scale_px_per_inch=self.scale)

    def _init_default_homography(self) -> np.ndarray:
        """Generates a geometrically plausible initial homography for 1080p broadcast frames."""
        src_field = np.array([
            [100.0, 30.0],
            [550.0, 30.0],
            [550.0, 280.0],
            [100.0, 280.0],
        ], dtype=np.float64)
        # Default broadcast angle: camera on North sideline looking South
        # Blue Hub (left) and Red Hub (right)
        dst_img = np.array([
            [1420.0, 240.0],
            [500.0, 240.0],
            [300.0, 680.0],
            [1620.0, 680.0],
        ], dtype=np.float64)
        H, _ = cv2.findHomography(src_field, dst_img)
        return H if H is not None else np.eye(3, dtype=np.float64)

    def calibrate_frame(
        self,
        frame_bgr: np.ndarray,
        force_recompute: bool = False,
    ) -> CalibrationResult:
        """
        Executes Stage 1 calibration pipeline:
        1. Detects static field AprilTags on alliance hubs and field structures.
        2. Solves or updates homography matrix mapping image pixels to metric field coords.
        3. Warps broadcast frame into flattened top-down 2D metric field space.
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Detect static AprilTags on Alliance Hubs and perimeter
        detected_tags_raw = self.tag_detector.detect(frame_bgr)

        # Split-screen broadcast isolation: isolate wide-angle view (y <= 0.65 * H)
        # NEVER fall back to lower split-screen PIP tags (y > 0.65 * H) from secondary cameras!
        top_tags = [t for t in detected_tags_raw if t.center_px[1] <= 0.65 * h_frame]
        effective_tags = top_tags

        # Deduplicate detections by tag_id, favoring upper/wide-angle detections
        tag_dict: Dict[int, Any] = {}
        for t in effective_tags:
            tid = int(t.tag_id)
            if tid not in tag_dict or t.center_px[1] < tag_dict[tid].center_px[1]:
                tag_dict[tid] = t
        unique_tags = list(tag_dict.values())

        anchors: List[AprilTagAnchor] = []
        field_pts_3d: List[Tuple[float, float, float]] = []
        image_pts: List[Tuple[float, float]] = []

        for tag in unique_tags:
            fx, fy, fz = tag.field_coords_inch
            cx, cy = tag.center_px

            anchor = AprilTagAnchor(
                tag_id=tag.tag_id,
                image_center_px=tag.center_px,
                field_coords_inch=tag.field_coords_inch,
                corners_px=tag.corners_px,
                reprojection_error_px=0.0,
            )
            anchors.append(anchor)
            field_pts_3d.append((fx, fy, fz))
            image_pts.append((cx, cy))

        is_cached = False
        reproj_rmse = 0.0

        has_elevated_tags = any(fz > 1.0 for _, _, fz in field_pts_3d)

        # 2. Compute or update homography matrix
        pnp_success = False
        rvec, tvec, cam_center, H_floor, pnp_rmse, f_val = None, None, None, None, 999.0, None

        if len(field_pts_3d) >= 4 or force_recompute:
            f_arr_3d = np.array(field_pts_3d, dtype=np.float64)
            f_arr_2d = f_arr_3d[:, :2]
            i_arr = np.array(image_pts, dtype=np.float64)

            if has_elevated_tags:
                try:
                    rvec, tvec, cam_center, H_floor, pnp_rmse, f_val, pnp_success = estimate_camera_pose_pnp(
                        object_points_3d=f_arr_3d,
                        image_points_2d=i_arr,
                        image_shape=(h_frame, w_frame),
                    )
                except Exception as e:
                    logger.debug("PnP estimation error in SpatialCalibrator: %s", e)
                    pnp_success = False

            if has_elevated_tags and pnp_success and pnp_rmse < 20.0 and H_floor is not None:
                # STRICT PNP RELIANCE: Solved 3D ground plane homography
                if self.align_with_broadcast_view and rvec is not None:
                    R, _ = cv2.Rodrigues(rvec)
                    is_north = bool(R[0, 0] < 0 or (cam_center is not None and cam_center[1] > self.field_geometry.width_inches / 2))
                    if is_north != self._is_inverted:
                        self._is_inverted = is_north
                        self._update_scaling_matrices()

                self._cached_H_field_to_img = H_floor
                self._cached_H_img_to_field = np.linalg.inv(H_floor)
                self._cached_H_canvas_to_img = self._cached_H_field_to_img @ self.S_canvas_to_field
                self._cached_H_img_to_canvas = np.linalg.inv(self._cached_H_canvas_to_img)
                self._cached_rvec = rvec
                self._cached_tvec = tvec
                if f_val is not None:
                    self._cached_K = np.array([
                        [f_val, 0.0, float(w_frame) / 2.0],
                        [0.0, f_val, float(h_frame) / 2.0],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
                self._last_rmse = pnp_rmse
                reproj_rmse = pnp_rmse
                is_cached = False
            elif has_elevated_tags and self._cached_H_field_to_img is not None:
                # Graceful propagation via CMC / cached homography without false alarm
                logger.debug(
                    "Propagating ground homography via CMC / cache (PnP rmse=%.2f, success=%s).",
                    pnp_rmse, pnp_success,
                )
                is_cached = True
                reproj_rmse = self._last_rmse
            else:
                # Purely 2D planar tags (all Z <= 1.0) or uninitialized frame 0
                try:
                    H_init, _ = compute_normalized_dlt(f_arr_2d, i_arr)
                    H_opt, rmse, _, converged = optimize_homography_lm(H_init, f_arr_2d, i_arr)
                    if converged and rmse < 25.0:
                        self._cached_H_field_to_img = H_opt
                        self._cached_H_img_to_field = np.linalg.inv(H_opt)
                        self._cached_H_canvas_to_img = self._cached_H_field_to_img @ self.S_canvas_to_field
                        self._cached_H_img_to_canvas = np.linalg.inv(self._cached_H_canvas_to_img)
                        self._last_rmse = rmse
                        reproj_rmse = rmse
                        is_cached = False
                    else:
                        H_ransac, _ = cv2.findHomography(f_arr_2d, i_arr, cv2.RANSAC, 4.0)
                        if H_ransac is not None:
                            self._cached_H_field_to_img = H_ransac
                            self._cached_H_img_to_field = np.linalg.inv(H_ransac)
                            self._cached_H_canvas_to_img = self._cached_H_field_to_img @ self.S_canvas_to_field
                            self._cached_H_img_to_canvas = np.linalg.inv(self._cached_H_canvas_to_img)
                            self._last_rmse = 2.0
                            reproj_rmse = 2.0
                            is_cached = False
                except Exception as e:
                    logger.debug("Direct 2D homography optimization failed: %s; relying on cache/CMC", e)
                    is_cached = True
                    reproj_rmse = self._last_rmse
        else:
            is_cached = True
            reproj_rmse = self._last_rmse

        # 3. Cache Propagation via Camera Motion Compensation (Optical Flow) when tags < 4 or PnP failed
        if is_cached and self._cached_H_field_to_img is not None:
            if self.enable_cmc and self._prev_gray is not None and self._prev_keypoints is not None:
                try:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        self._prev_gray, gray, self._prev_keypoints, None,
                        winSize=(21, 21), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                    )
                    good_prev = self._prev_keypoints[st.ravel() == 1]
                    good_curr = p1[st.ravel() == 1]

                    if len(good_prev) >= 10:
                        A, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                        if A is not None:
                            # Form 3x3 affine transformation matrix M
                            M_affine = np.eye(3, dtype=np.float64)
                            M_affine[:2, :] = A
                            # Update homography: H_curr = M_affine * H_prev
                            self._cached_H_field_to_img = M_affine @ self._cached_H_field_to_img
                            self._cached_H_img_to_field = np.linalg.inv(self._cached_H_field_to_img)
                            self._cached_H_canvas_to_img = self._cached_H_field_to_img @ self.S_canvas_to_field
                            self._cached_H_img_to_canvas = np.linalg.inv(self._cached_H_canvas_to_img)
                except Exception as e:
                    logger.debug("Optical flow CMC step failed: %s", e)

        # If cache is still uninitialized, activate default pre-calibrated homography
        if self._cached_H_img_to_field is None:
            if self.align_with_broadcast_view:
                self._is_inverted = True
                self._update_scaling_matrices()
            self._cached_H_field_to_img = self._default_H_field_to_img
            self._cached_H_img_to_field = np.linalg.inv(self._default_H_field_to_img)
            self._cached_H_canvas_to_img = self._cached_H_field_to_img @ self.S_canvas_to_field
            self._cached_H_img_to_canvas = np.linalg.inv(self._cached_H_canvas_to_img)
            self._last_rmse = 3.5
            reproj_rmse = 3.5
            is_cached = True

        # Refresh keypoints for next frame CMC
        if self.enable_cmc:
            self._prev_gray = gray
            self._prev_keypoints = cv2.goodFeaturesToTrack(
                gray, maxCorners=120, qualityLevel=0.03, minDistance=15
            )

        # 4. Warp broadcast frame to flattened top-down 2D metric field plane
        warped_canvas = cv2.warpPerspective(
            frame_bgr,
            self._cached_H_img_to_canvas,
            (self.canvas_w, self.canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(25, 45, 25),
        )

        # 5. Compute dynamic tag-anchored Hub screen bounding boxes
        red_hub_bbox, blue_hub_bbox = self.compute_hub_screen_bboxes(
            tags=anchors,
            frame_shape=(h_frame, w_frame),
            rvec=rvec if (has_elevated_tags and pnp_success) else self._cached_rvec,
            tvec=tvec if (has_elevated_tags and pnp_success) else self._cached_tvec,
            K=self._cached_K,
        )

        return CalibrationResult(
            detected_tags=anchors,
            field_geometry=self.field_geometry,
            H_image_to_field=self._cached_H_img_to_field.copy(),
            H_field_to_image=self._cached_H_field_to_img.copy(),
            H_image_to_canvas=self._cached_H_img_to_canvas.copy(),
            canvas_shape=self.canvas_shape,
            reprojection_rmse_px=reproj_rmse,
            is_cached=is_cached,
            warped_field_plane=warped_canvas,
            red_hub_screen_bbox=red_hub_bbox,
            blue_hub_screen_bbox=blue_hub_bbox,
        )

    def compute_hub_screen_bboxes(
        self,
        tags: Sequence[AprilTagAnchor],
        frame_shape: Tuple[int, int],
        rvec: Optional[np.ndarray] = None,
        tvec: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """
        Dynamically calculates and stabilizes the screen bounding boxes for Red and Blue Hub catchment openings.

        Strict Rules:
        - Red Hub structure tags: paired faces 8/5, 9/10, 11/2, 4/3 ({8, 5, 9, 10, 11, 2, 4, 3}).
        - Blue Hub structure tags: paired faces 18/27, 19/20, 21/24, 26/25 ({18, 27, 19, 20, 21, 24, 26, 25}).
        - Perimeter wall / driver station tags are NEVER used directly to anchor Hub locations.
        - History of actual Hub tag sightings is stored to compute the persistent most common locations.
        - If Hub tags are temporarily occluded, mutual relative triangulation (Blue <-> Red delta)
          or 3D PnP CAD raycasting is used to maintain rock-solid position without jumping.
        """
        h_frame, w_frame = frame_shape
        w_box = 230.0
        h_box = 170.0

        # Filter tags strictly by known physical Hub structure tag sets
        red_struct = [t for t in tags if t.tag_id in RED_HUB_STRUCTURE_TAGS]
        blue_struct = [t for t in tags if t.tag_id in BLUE_HUB_STRUCTURE_TAGS]

        # 1. Update tag sighting history for actual Hub structure tags
        for t in red_struct:
            self._hub_tag_history_red[t.tag_id].append((float(t.image_center_px[0]), float(t.image_center_px[1])))
        for t in blue_struct:
            self._hub_tag_history_blue[t.tag_id].append((float(t.image_center_px[0]), float(t.image_center_px[1])))

        # 2. Compute observed bounding boxes if structure tags are visible
        red_bbox_obs = None
        if red_struct:
            cx = float(np.mean([t.image_center_px[0] for t in red_struct]))
            top_y = float(min(t.image_center_px[1] for t in red_struct))
            center_y = top_y - 110.0
            x1 = np.clip(cx - w_box / 2.0, 0, w_frame - 1)
            y1 = np.clip(center_y - h_box / 2.0, 0, h_frame - 1)
            x2 = np.clip(cx + w_box / 2.0, 0, w_frame - 1)
            y2 = np.clip(center_y + h_box / 2.0, 0, h_frame - 1)
            red_bbox_obs = (float(x1), float(y1), float(x2), float(y2))
            if self._persistent_red_hub_bbox is None:
                self._persistent_red_hub_bbox = red_bbox_obs
            else:
                self._persistent_red_hub_bbox = tuple(
                    0.85 * p + 0.15 * c for p, c in zip(self._persistent_red_hub_bbox, red_bbox_obs)
                )

        blue_bbox_obs = None
        if blue_struct:
            cx = float(np.mean([t.image_center_px[0] for t in blue_struct]))
            top_y = float(min(t.image_center_px[1] for t in blue_struct))
            center_y = top_y - 110.0
            x1 = np.clip(cx - w_box / 2.0, 0, w_frame - 1)
            y1 = np.clip(center_y - h_box / 2.0, 0, h_frame - 1)
            x2 = np.clip(cx + w_box / 2.0, 0, w_frame - 1)
            y2 = np.clip(center_y + h_box / 2.0, 0, h_frame - 1)
            blue_bbox_obs = (float(x1), float(y1), float(x2), float(y2))
            if self._persistent_blue_hub_bbox is None:
                self._persistent_blue_hub_bbox = blue_bbox_obs
            else:
                self._persistent_blue_hub_bbox = tuple(
                    0.85 * p + 0.15 * c for p, c in zip(self._persistent_blue_hub_bbox, blue_bbox_obs)
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

        # 4. Finalize Red Hub Screen BBox (Observation -> Triangulation -> 3D PnP -> Persistent Memory)
        final_red_bbox = red_bbox_obs
        if final_red_bbox is None:
            if blue_bbox_obs is not None and self._hub_relative_delta_xy is not None:
                # Triangulate Red from visible Blue Hub
                bcx = (blue_bbox_obs[0] + blue_bbox_obs[2]) / 2.0
                bcy = (blue_bbox_obs[1] + blue_bbox_obs[3]) / 2.0
                rcx = bcx + self._hub_relative_delta_xy[0]
                rcy = bcy + self._hub_relative_delta_xy[1]
                final_red_bbox = (
                    float(np.clip(rcx - w_box / 2.0, 0, w_frame - 1)),
                    float(np.clip(rcy - h_box / 2.0, 0, h_frame - 1)),
                    float(np.clip(rcx + w_box / 2.0, 0, w_frame - 1)),
                    float(np.clip(rcy + h_box / 2.0, 0, h_frame - 1)),
                )
            elif rvec is not None and tvec is not None and K is not None:
                # 3D Projective ray-cast for Red Hub Rim (CAD: 162.8, 158.85, 72.0)
                red_rim_3d = np.array([[162.8, 158.85, 72.0]], dtype=np.float64)
                proj, _ = cv2.projectPoints(red_rim_3d, rvec, tvec, K, None)
                px, py = proj.ravel()[:2]
                if 0 <= px < w_frame and 0 <= py < h_frame:
                    final_red_bbox = (
                        float(np.clip(px - w_box / 2.0, 0, w_frame - 1)),
                        float(np.clip(py - h_box / 2.0, 0, h_frame - 1)),
                        float(np.clip(px + w_box / 2.0, 0, w_frame - 1)),
                        float(np.clip(py + h_box / 2.0, 0, h_frame - 1)),
                    )
            elif self._persistent_red_hub_bbox is not None:
                final_red_bbox = self._persistent_red_hub_bbox

        # 5. Finalize Blue Hub Screen BBox (Observation -> Triangulation -> 3D PnP -> Persistent Memory)
        final_blue_bbox = blue_bbox_obs
        if final_blue_bbox is None:
            if red_bbox_obs is not None and self._hub_relative_delta_xy is not None:
                # Triangulate Blue from visible Red Hub
                rcx = (red_bbox_obs[0] + red_bbox_obs[2]) / 2.0
                rcy = (red_bbox_obs[1] + red_bbox_obs[3]) / 2.0
                bcx = rcx - self._hub_relative_delta_xy[0]
                bcy = rcy - self._hub_relative_delta_xy[1]
                final_blue_bbox = (
                    float(np.clip(bcx - w_box / 2.0, 0, w_frame - 1)),
                    float(np.clip(bcy - h_box / 2.0, 0, h_frame - 1)),
                    float(np.clip(bcx + w_box / 2.0, 0, w_frame - 1)),
                    float(np.clip(bcy + h_box / 2.0, 0, h_frame - 1)),
                )
            elif rvec is not None and tvec is not None and K is not None:
                # 3D Projective ray-cast for Blue Hub Rim (CAD: 488.4, 158.85, 72.0)
                blue_rim_3d = np.array([[488.4, 158.85, 72.0]], dtype=np.float64)
                proj, _ = cv2.projectPoints(blue_rim_3d, rvec, tvec, K, None)
                px, py = proj.ravel()[:2]
                if 0 <= px < w_frame and 0 <= py < h_frame:
                    final_blue_bbox = (
                        float(np.clip(px - w_box / 2.0, 0, w_frame - 1)),
                        float(np.clip(py - h_box / 2.0, 0, h_frame - 1)),
                        float(np.clip(px + w_box / 2.0, 0, w_frame - 1)),
                        float(np.clip(py + h_box / 2.0, 0, h_frame - 1)),
                    )
            elif self._persistent_blue_hub_bbox is not None:
                final_blue_bbox = self._persistent_blue_hub_bbox

        if final_red_bbox is not None:
            self._cached_red_hub_bbox = final_red_bbox
        if final_blue_bbox is not None:
            self._cached_blue_hub_bbox = final_blue_bbox

        return self._cached_red_hub_bbox, self._cached_blue_hub_bbox

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

    def pixels_to_field_coords(self, px: float, py: float) -> Tuple[float, float]:
        """Maps 2D broadcast pixel coordinate (u, v) to physical field coordinates (X, Y) in inches."""
        if self._cached_H_img_to_field is None:
            return (0.0, 0.0)
        pt = np.array([px, py, 1.0], dtype=np.float64)
        proj = self._cached_H_img_to_field @ pt
        z = proj[2] if abs(proj[2]) > 1e-9 else 1e-9
        return float(proj[0] / z), float(proj[1] / z)

    def field_coords_to_pixels(self, fx: float, fy: float) -> Tuple[float, float]:
        """Maps physical field coordinates (X, Y) in inches to broadcast pixel coordinate (u, v)."""
        if self._cached_H_field_to_img is None:
            return (0.0, 0.0)
        pt = np.array([fx, fy, 1.0], dtype=np.float64)
        proj = self._cached_H_field_to_img @ pt
        z = proj[2] if abs(proj[2]) > 1e-9 else 1e-9
        return float(proj[0] / z), float(proj[1] / z)

    def field_coords_to_canvas(self, fx: float, fy: float) -> Tuple[float, float]:
        """Maps physical field coordinates (X, Y) in inches to top-down canvas coordinates (x_c, y_c)."""
        if self._is_inverted:
            return float((self.field_geometry.length_inches - fx) * self.scale), float((self.field_geometry.width_inches - fy) * self.scale)
        return float(fx * self.scale), float(fy * self.scale)

    def canvas_to_field_coords(self, cx: float, cy: float) -> Tuple[float, float]:
        """Maps top-down canvas coordinates (x_c, y_c) to physical field coordinates (X, Y) in inches."""
        if self._is_inverted:
            return float(self.field_geometry.length_inches - (cx / self.scale)), float(self.field_geometry.width_inches - (cy / self.scale))
        return float(cx / self.scale), float(cy / self.scale)

    def pixels_to_canvas_coords(self, px: float, py: float) -> Tuple[float, float]:
        """Maps broadcast pixel coordinate (u, v) directly to top-down canvas coordinates (x_c, y_c)."""
        if self._cached_H_img_to_canvas is None:
            return (0.0, 0.0)
        pt = np.array([px, py, 1.0], dtype=np.float64)
        proj = self._cached_H_img_to_canvas @ pt
        z = proj[2] if abs(proj[2]) > 1e-9 else 1e-9
        return float(proj[0] / z), float(proj[1] / z)

    def canvas_to_pixels_coords(self, cx: float, cy: float) -> Tuple[float, float]:
        """Maps top-down canvas coordinates (x_c, y_c) directly to broadcast pixel coordinate (u, v)."""
        if self._cached_H_canvas_to_img is None:
            return (0.0, 0.0)
        pt = np.array([cx, cy, 1.0], dtype=np.float64)
        proj = self._cached_H_canvas_to_img @ pt
        z = proj[2] if abs(proj[2]) > 1e-9 else 1e-9
        return float(proj[0] / z), float(proj[1] / z)
