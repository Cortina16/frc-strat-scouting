"""
Multi-Stage Computer Vision & Spatial Analytics Pipeline for FRC 2026 REBUILT.

Orchestrates:
1. Spatial Calibration & Top-Down Metric Homography (AprilTags on Alliance Hubs)
2. Game Piece (Yellow FUEL) Quantification & Proportional Mass Allocation (504 max capacity)
3. Deterministic Robot Localization & Identification (Strict 3v3 Cardinality)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.common.types import BoundingBox
from src.vision.bumper_detector import BumperDetector
from src.vision.deterministic_robot_tracker import DeterministicRobotTracker
from src.vision.fuel_quantifier import FuelQuantifier
from src.vision.spatial_calibrator import SpatialCalibrator
from src.vision.spatial_types import (
    CalibrationResult,
    FieldGeometry,
    GamePieceQuantificationResult,
    RobotLocalizationResult,
    SpatialPipelineResult,
    TrackedRobotUnit,
    YellowBlobCluster,
)

logger = logging.getLogger("frc.vision.spatial_pipeline")


class SpatialAnalyticsPipeline:
    """
    Unified end-to-end multi-stage pipeline coordinator.
    Processes video frames and returns fully reconciled spatial telemetry and annotated visualizations.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        scale_px_per_inch: float = 2.0,
        max_fuel_capacity: int = 504,
        enable_cmc: bool = True,
        bumper_model_path: Optional[str] = "weights/bumper_best.pt",
        enable_bumper_model: bool = True,
    ) -> None:
        self.calibrator = SpatialCalibrator(
            config_path=config_path,
            scale_px_per_inch=scale_px_per_inch,
            enable_optical_flow_cmc=enable_cmc,
        )
        self.geom = self.calibrator.field_geometry
        self.fuel_quantifier = FuelQuantifier(
            max_field_capacity=max_fuel_capacity,
            scale_px_per_inch=scale_px_per_inch,
        )
        self.robot_tracker = DeterministicRobotTracker(
            field_geometry=self.geom,
        )
        self.bumper_detector: Optional[BumperDetector] = (
            BumperDetector(model_path=bumper_model_path) if enable_bumper_model else None
        )

    def reset(self) -> None:
        """Resets all sub-pipeline stateful trackers and accumulators."""
        self.fuel_quantifier.reset()
        self.robot_tracker.reset()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
        external_proposals: Optional[Sequence[BoundingBox]] = None,
        render_visualization: bool = True,
    ) -> SpatialPipelineResult:
        """
        Executes full 3-stage sequence:
        1. Spatial Calibration & Homography
        2. Game Piece Quantification & Proportional Mass Allocation
        3. Deterministic 3v3 Robot Localization
        """
        # ====================================================================
        # Stage 1: Spatial Calibration & Homography
        # ====================================================================
        calib_res: CalibrationResult = self.calibrator.calibrate_frame(frame_bgr)

        # ====================================================================
        # Stage 2: Game Piece (Yellow FUEL) Quantification
        # ====================================================================
        fuel_res: GamePieceQuantificationResult = self.fuel_quantifier.quantify_frame(
            calib_res.warped_field_plane,
            isolate_dynamic_only=True,
            calibrator=self.calibrator,
        )

        # ====================================================================
        # Stage 3: Deterministic Robot Localization & Identification
        # ====================================================================
        proposals_to_track = list(external_proposals) if external_proposals is not None else None
        if self.bumper_detector is not None and self.bumper_detector.is_available:
            bumpers = self.bumper_detector.detect_bumpers(
                frame_bgr=frame_bgr,
                H_image_to_field=calib_res.H_image_to_field,
            )
            if proposals_to_track is not None:
                proposals_to_track = self.bumper_detector.fuse_proposals(proposals_to_track, bumpers)
            elif bumpers:
                proposals_to_track = [self.bumper_detector.extrapolate_robot_box(b) for b in bumpers]

        robot_res: RobotLocalizationResult = self.robot_tracker.track_frame(
            frame_bgr=frame_bgr,
            H_image_to_field=calib_res.H_image_to_field,
            external_proposals=proposals_to_track,
            fg_mask=fuel_res.foreground_mask,
            calibrator=self.calibrator,
        )

        # ====================================================================
        # Visual Rendering (Optional)
        # ====================================================================
        vis_broadcast: Optional[np.ndarray] = None
        vis_topdown: Optional[np.ndarray] = None

        if render_visualization:
            vis_broadcast = self._render_broadcast_overlay(
                frame_bgr=frame_bgr,
                calib=calib_res,
                robots=robot_res,
                frame_index=frame_index,
            )
            vis_topdown = self._render_top_down_overlay(
                warped_plane=calib_res.warped_field_plane,
                calib=calib_res,
                fuel=fuel_res,
                robots=robot_res,
                frame_index=frame_index,
            )

        return SpatialPipelineResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            calibration=calib_res,
            game_pieces=fuel_res,
            robots=robot_res,
            annotated_broadcast_frame=vis_broadcast,
            annotated_top_down_plane=vis_topdown,
        )

    def _render_broadcast_overlay(
        self,
        frame_bgr: np.ndarray,
        calib: CalibrationResult,
        robots: RobotLocalizationResult,
        frame_index: int,
    ) -> np.ndarray:
        """Renders HUD telemetry on the original broadcast frame."""
        vis = frame_bgr.copy()

        # 1. Draw detected AprilTags on Hubs
        for tag in calib.detected_tags:
            pts = tag.corners_px.astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cx, cy = int(tag.image_center_px[0]), int(tag.image_center_px[1])
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                vis, f"Tag #{tag.tag_id}", (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA,
            )

        # 2. Draw 6 Tracked Robot Units (3 Red, 3 Blue)
        for unit in robots.all_units:
            color = (30, 30, 225) if unit.alliance == "RED" else (225, 80, 25)
            b = unit.pixel_bbox
            x1 = int(b.center_x - b.width / 2.0)
            y1 = int(b.center_y - b.height / 2.0)
            x2 = int(b.center_x + b.width / 2.0)
            y2 = int(b.center_y + b.height / 2.0)

            # Box outline
            thickness = 2 if not unit.is_projected else 1
            line_type = cv2.LINE_AA
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness, line_type)

            # Label banner
            proj_str = " [PROJ]" if unit.is_projected else ""
            label = f"{unit.unit_id}{proj_str} ({unit.field_x_inch:.0f}\", {unit.field_y_inch:.0f}\")"
            cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + len(label) * 8 + 10, y1), color, -1)
            cv2.putText(
                vis, label, (x1 + 4, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # 3. Top HUD Banner
        cache_status = "CACHED (CMC)" if calib.is_cached else "DLT+LM OPT"
        hud_text = (
            f"FRAME {frame_index:04d} | TAGS: {len(calib.detected_tags)} | "
            f"HOMOGRAPHY: {cache_status} (RMSE: {calib.reprojection_rmse_px:.2f}px) | "
            f"UNITS: 3 RED / 3 BLUE"
        )
        cv2.rectangle(vis, (15, 15), (780, 45), (10, 10, 10), -1)
        cv2.rectangle(vis, (15, 15), (780, 45), (100, 100, 100), 1)
        cv2.putText(vis, hud_text, (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return vis

    def _render_top_down_overlay(
        self,
        warped_plane: np.ndarray,
        calib: CalibrationResult,
        fuel: GamePieceQuantificationResult,
        robots: RobotLocalizationResult,
        frame_index: int,
    ) -> np.ndarray:
        """Renders the top-down 2D metric field plane with structural elements and game pieces."""
        vis = warped_plane.copy()
        scale = self.geom.carpet_scale_px_per_inch

        # 1. Draw Field CAD Structures (Alliance Hubs)
        half_hub = 23.5 * scale

        # Red Hub: 47" x 47" centered at (162.8, 158.85)
        rx_c, ry_c = self.calibrator.field_coords_to_canvas(
            self.geom.red_hub_center_inch[0], self.geom.red_hub_center_inch[1]
        )
        rx1, ry1 = int(rx_c - half_hub), int(ry_c - half_hub)
        rx2, ry2 = int(rx_c + half_hub), int(ry_c + half_hub)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 0, 180), 2)
        cv2.putText(vis, "RED HUB (72\")", (rx1 - 10, ry1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1)

        # Blue Hub: 47" x 47" centered at (488.4, 158.85)
        bx_c, by_c = self.calibrator.field_coords_to_canvas(
            self.geom.blue_hub_center_inch[0], self.geom.blue_hub_center_inch[1]
        )
        bx1, by1 = int(bx_c - half_hub), int(by_c - half_hub)
        bx2, by2 = int(bx_c + half_hub), int(by_c + half_hub)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), (180, 50, 0), 2)
        cv2.putText(vis, "BLUE HUB (72\")", (bx1 - 10, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 120, 0), 1)

        # 2. Draw Yellow Game Piece Clusters & Proportional Mass Allocations
        for blob in fuel.blob_clusters:
            # Draw contour boundary
            cv2.drawContours(vis, [blob.contour_px], -1, (0, 255, 255), 1)
            cx, cy = int(blob.canvas_centroid_px[0]), int(blob.canvas_centroid_px[1])
            cv2.circle(vis, (cx, cy), 3, (0, 200, 255), -1)

            # Display proportional ball allocation if significant
            if blob.estimated_count >= 1:
                lbl = f"#{blob.blob_id}: {blob.estimated_count} ({blob.percentage_total_area:.1f}%)"
                cv2.putText(
                    vis, lbl, (cx + 5, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA,
                )

        # 3. Draw 6 Tracked Robot Units on the Metric Plane
        for unit in robots.all_units:
            color = (20, 20, 240) if unit.alliance == "RED" else (240, 100, 20)
            cx_c, cy_c = self.calibrator.field_coords_to_canvas(unit.field_x_inch, unit.field_y_inch)
            cx = int(cx_c)
            cy = int(cy_c)

            # Draw robot chassis footprint (30" x 30" -> 60 x 60 px)
            chassis_half = int(15.0 * scale)
            cv2.rectangle(
                vis,
                (cx - chassis_half, cy - chassis_half),
                (cx + chassis_half, cy + chassis_half),
                color,
                2 if not unit.is_projected else 1,
            )
            # Centroid
            cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(vis, (cx, cy), 3, color, -1)

            # Label
            tag_str = f"{unit.unit_id}" + ("*" if unit.is_projected else "")
            cv2.putText(vis, tag_str, (cx - 20, cy - chassis_half - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        # 4. Top Telemetry Banner
        header = (
            f"2D METRIC FIELD PLANE (651.2\" x 317.7\") | "
            f"TOTAL YELLOW AREA: {fuel.total_yellow_pixel_area:.0f} px | "
            f"EST. FUEL: {fuel.total_estimated_count} / {fuel.max_field_capacity} | "
            f"CLUSTERS: {len(fuel.blob_clusters)}"
        )
        cv2.rectangle(vis, (10, 10), (820, 38), (15, 15, 15), -1)
        cv2.putText(vis, header, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        return vis
