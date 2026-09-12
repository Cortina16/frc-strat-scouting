"""
High-Speed Yellow Ball ("FUEL") Detector.

Extracts individual yellow game pieces from broadcast frames up to a maximum of 504
detections per frame, featuring:
- Ultra-fast HSV + RGB chromaticity segmentation (< 8ms)
- Euclidean distance transform peak finding for dense touching clusters
- Sub-pixel plateau centroid clustering and Euclidean Spatial Non-Maximum Suppression (NMS)
  to prevent 1-2 pixel duplicate overlaps
- Dual-Zone Detection:
  * Focal Zone (near/over active robots): discrete individual ball detection and sub-pixel centroids
  * Ambient Zone (bulk field carpet): connected component blob detection with area-based
    percentage and count approximations (N_est = Area / (A_nominal * eta))
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.common.types import (
    BallCluster,
    BoundingBox,
    DualZoneDetectionResult,
    RawDecodedFrame,
)

logger = logging.getLogger("frc.vision.ball_detector")


class YellowBallDetector:
    """
    High-throughput detector for FRC yellow game pieces (balls / FUEL).
    Capacitated to detect up to 504 individual balls per frame with zero sub-pixel overlap.
    """

    BALL_CLASS_ID: int = 1

    def __init__(
        self,
        max_balls: int = 504,
        min_radius_px: float = 2.5,
        max_radius_px: float = 30.0,
        min_separation_px: float = 7.0,
        field_roi_y: Tuple[float, float] = (0.12, 0.65),
        field_bounds_screen: Tuple[float, float, float, float] = (0.02, 0.98, 0.13, 0.62),
        nominal_ball_radius_px: float = 6.5,
        packing_efficiency: float = 0.82,
        robot_catchment_margin: float = 40.0,
    ) -> None:
        self.max_balls = max_balls
        self.min_radius_px = min_radius_px
        self.max_radius_px = max_radius_px
        self.min_separation_px = min_separation_px
        self.field_roi_y = field_roi_y
        self.field_bounds_screen = field_bounds_screen
        self.nominal_ball_radius_px = nominal_ball_radius_px
        self.nominal_ball_area = float(np.pi * (nominal_ball_radius_px ** 2))
        self.packing_efficiency = packing_efficiency
        self.robot_catchment_margin = robot_catchment_margin

        # Pre-allocate morphological structuring elements
        self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _segment_yellow(self, roi: np.ndarray) -> np.ndarray:
        """
        Fast dual-space color segmentation combining HSV yellow hue and RGB chromaticity.
        Executes in < 8ms on CPU, suppressing green field carpet (R < 70) and red/blue bumpers.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([18, 80, 75], dtype=np.uint8)
        upper_yellow = np.array([36, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_yellow, upper_yellow)

        b_ch, g_ch, r_ch = cv2.split(roi)
        # Yellow has strong red and green, but low blue. Green carpet has R < 70.
        mask_rgb = (r_ch >= 135) & (g_ch >= 135) & (b_ch <= 125)
        mask = mask_hsv & mask_rgb.astype(np.uint8) * 255

        # Spatial carpet boundary masking within ROI
        roi_h, roi_w = roi.shape[:2]
        if self.field_bounds_screen is not None:
            bx1 = int(self.field_bounds_screen[0] * roi_w)
            bx2 = int(self.field_bounds_screen[1] * roi_w)
            y_span = max(1e-5, (self.field_roi_y[1] - self.field_roi_y[0]))
            h_full = int(roi_h / y_span)
            y1_global = int(self.field_roi_y[0] * h_full)
            by1_roi = max(0, int(self.field_bounds_screen[2] * h_full) - y1_global)
            by2_roi = min(roi_h, int(self.field_bounds_screen[3] * h_full) - y1_global)

            spatial_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            spatial_mask[by1_roi:by2_roi, bx1:bx2] = 255
            mask = cv2.bitwise_and(mask, spatial_mask)

        # Morphological opening to eliminate single-pixel noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        return mask

    def _extract_peaks_with_nms(
        self,
        dist: np.ndarray,
        y_offset: float = 0.0,
        x_offset: float = 0.0,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extracts distance transform local peaks, collapses multi-pixel plateaus into
        exact centroids, and executes Euclidean Spatial NMS to eliminate overlapping detections.

        Returns:
            List of (cx, cy, radius, confidence)
        """
        dilated = cv2.dilate(dist, self._peak_kernel)
        peaks = (
            (dist == dilated)
            & (dist >= self.min_radius_px)
            & (dist <= self.max_radius_px)
        )

        num_peaks, _, _, centroids = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
        if num_peaks <= 1:
            return []

        candidates: List[Tuple[float, float, float, float]] = []
        for i in range(1, num_peaks):
            cx_local, cy_local = centroids[i]
            ix, iy = int(round(cx_local)), int(round(cy_local))
            iy = max(0, min(dist.shape[0] - 1, iy))
            ix = max(0, min(dist.shape[1] - 1, ix))
            r_val = float(dist[iy, ix])
            conf = float(np.clip(r_val / 8.0, 0.20, 0.99))
            candidates.append((float(cx_local + x_offset), float(cy_local + y_offset), r_val, conf))

        # Sort candidates by radius/prominence descending
        candidates.sort(key=lambda c: c[2], reverse=True)

        # Euclidean Spatial Non-Maximum Suppression
        kept: List[Tuple[float, float, float, float]] = []
        for cx, cy, r_val, conf in candidates:
            too_close = False
            for kx, ky, kr, _ in kept:
                sep_thresh = max(self.min_separation_px, max(r_val, kr) * 0.85)
                if (cx - kx) ** 2 + (cy - ky) ** 2 < (sep_thresh ** 2):
                    too_close = True
                    break
            if not too_close:
                kept.append((cx, cy, r_val, conf))

        return kept

    def detect_dual_zone(
        self,
        frame_bgr: np.ndarray,
        robot_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> DualZoneDetectionResult:
        """
        Dual-Zone detection partitioning:
        1. Focal Zone (Over & near detected robots): high-precision distance transform peak finding
           with sub-pixel centroid clustering and Euclidean NMS.
        2. Ambient Zone (Field carpet & piles): connected component blob detection with
           area/density-based ball count and percentage approximations.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return DualZoneDetectionResult(individual_balls=[], clusters=[])

        h, w = frame_bgr.shape[:2]
        y1 = max(0, int(self.field_roi_y[0] * h))
        y2 = min(h, int(self.field_roi_y[1] * h))
        if y2 <= y1:
            return DualZoneDetectionResult(individual_balls=[], clusters=[])

        roi = frame_bgr[y1:y2, :]
        roi_h, roi_w = roi.shape[:2]

        # 1. Color Segmentation
        mask = self._segment_yellow(roi)

        # 2. Partition Focal Zone vs. Ambient Zone
        focal_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        has_robots = robot_boxes is not None and len(robot_boxes) > 0

        if has_robots:
            margin = self.robot_catchment_margin
            for r_box in robot_boxes:
                rx1 = max(0, int(r_box.center_x - r_box.width / 2.0 - margin))
                rx2 = min(roi_w, int(r_box.center_x + r_box.width / 2.0 + margin))
                ry1 = max(0, int(r_box.center_y - r_box.height / 2.0 - margin) - y1)
                ry2 = min(roi_h, int(r_box.center_y + r_box.height / 2.0 + margin) - y1)
                if rx2 > rx1 and ry2 > ry1:
                    focal_mask[ry1:ry2, rx1:rx2] = 255

        individual_boxes: List[BoundingBox] = []

        # 3. Focal Zone: Individual Ball Detection
        if has_robots:
            focal_yellow = cv2.bitwise_and(mask, focal_mask)
            if np.any(focal_yellow):
                dist_focal = cv2.distanceTransform(focal_yellow, cv2.DIST_L2, 3)
                focal_peaks = self._extract_peaks_with_nms(dist_focal, y_offset=float(y1), x_offset=0.0)
                for cx, cy, r_val, conf in focal_peaks:
                    diameter = max(4.0, r_val * 2.0)
                    individual_boxes.append(
                        BoundingBox(
                            center_x=cx,
                            center_y=cy,
                            width=diameter,
                            height=diameter,
                            confidence=conf,
                            class_id=self.BALL_CLASS_ID,
                        )
                    )
        else:
            # When no robots are present, run full field peak extraction with NMS
            dist_all = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            all_peaks = self._extract_peaks_with_nms(dist_all, y_offset=float(y1), x_offset=0.0)
            for cx, cy, r_val, conf in all_peaks[: self.max_balls]:
                diameter = max(4.0, r_val * 2.0)
                individual_boxes.append(
                    BoundingBox(
                        center_x=cx,
                        center_y=cy,
                        width=diameter,
                        height=diameter,
                        confidence=conf,
                        class_id=self.BALL_CLASS_ID,
                    )
                )

        # 4. Ambient Zone: Connected Component Blob Area & Percentage Approximation
        clusters: List[BallCluster] = []
        if has_robots:
            ambient_yellow = cv2.bitwise_and(mask, cv2.bitwise_not(focal_mask))
        else:
            ambient_yellow = np.zeros_like(mask)

        if np.any(ambient_yellow):
            num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(ambient_yellow, connectivity=8)
            effective_ball_area = self.nominal_ball_area * self.packing_efficiency

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 16:  # filter noise speckles
                    continue

                cx = float(centroids[i, 0])
                cy = float(centroids[i, 1] + y1)
                bw = float(stats[i, cv2.CC_STAT_WIDTH])
                bh = float(stats[i, cv2.CC_STAT_HEIGHT])

                est_count = max(1, int(round(area / effective_ball_area)))
                pct = float((est_count / float(self.max_balls)) * 100.0)

                clusters.append(
                    BallCluster(
                        cluster_id=i,
                        centroid_x=cx,
                        centroid_y=cy,
                        bbox=BoundingBox(
                            center_x=cx,
                            center_y=cy,
                            width=bw,
                            height=bh,
                            confidence=0.85,
                            class_id=self.BALL_CLASS_ID,
                        ),
                        area_px=float(area),
                        estimated_ball_count=est_count,
                        inventory_pct=pct,
                    )
                )

        # Cap individual boxes to max_balls
        if len(individual_boxes) > self.max_balls:
            individual_boxes = individual_boxes[: self.max_balls]

        # Cap clusters so total balls across focal and ambient never exceeds max_balls ceiling
        remaining_balls = max(0, self.max_balls - len(individual_boxes))
        capped_clusters: List[BallCluster] = []
        for c in clusters:
            if remaining_balls <= 0:
                break
            if c.estimated_ball_count > remaining_balls:
                c.estimated_ball_count = remaining_balls
                c.inventory_pct = float((remaining_balls / float(self.max_balls)) * 100.0)
            capped_clusters.append(c)
            remaining_balls -= c.estimated_ball_count

        return DualZoneDetectionResult(
            individual_balls=individual_boxes,
            clusters=capped_clusters,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
        robot_boxes: Optional[Sequence[BoundingBox]] = None,
    ) -> List[BoundingBox]:
        """
        Standard detection entrypoint. Returns discrete individual ball bounding boxes
        conforming to the top-504 ceiling with Euclidean Spatial NMS.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        dual_result = self.detect_dual_zone(frame_bgr, robot_boxes=robot_boxes)
        results = list(dual_result.individual_balls)

        if len(results) < self.max_balls and dual_result.clusters:
            remaining_quota = self.max_balls - len(results)
            for c in dual_result.clusters:
                if remaining_quota <= 0:
                    break
                results.append(
                    BoundingBox(
                        center_x=c.centroid_x,
                        center_y=c.centroid_y,
                        width=self.nominal_ball_radius_px * 2.0,
                        height=self.nominal_ball_radius_px * 2.0,
                        confidence=0.80,
                        class_id=self.BALL_CLASS_ID,
                    )
                )
                remaining_quota -= 1

        return results[: self.max_balls]

    def detect_frame(self, frame: RawDecodedFrame) -> List[BoundingBox]:
        """Detects yellow balls from a RawDecodedFrame."""
        if frame is None or frame.image_bgr is None:
            return []
        return self.detect(frame.image_bgr)

    def detect_batch(self, frames: Sequence[RawDecodedFrame]) -> List[List[BoundingBox]]:
        """Batched ball detection across a sequence of frames."""
        results: List[List[BoundingBox]] = []
        for f in frames:
            results.append(self.detect_frame(f))
        return results
