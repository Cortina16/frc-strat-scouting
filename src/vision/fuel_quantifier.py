"""
Stage 2: Game Piece (Yellow FUEL) Quantification & Proportional Mass Allocation.

Processes flattened top-down field planes, isolates dynamic elements via background subtraction,
applies dual-space HSV + RGB yellow chromaticity segmentation, eliminates noise via morphological
opening/closing, extracts cluster contours, and allocates proportional mass out of 504 maximum capacity.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.vision.spatial_types import (
    GamePieceQuantificationResult,
    YellowBlobCluster,
)

logger = logging.getLogger("frc.vision.fuel_quantifier")


class FuelQuantifier:
    """
    High-accuracy quantifier for FRC 2026 REBUILT yellow game pieces (FUEL).
    Allocates estimated individual counts out of 504 maximum arena pieces
    proportionally based on segmented pixel areas.
    """

    def __init__(
        self,
        max_field_capacity: int = 504,
        scale_px_per_inch: float = 2.0,
        min_blob_area_px: float = 6.0,
        history: int = 60,
        var_threshold: float = 18.0,
    ) -> None:
        self.max_capacity = max_field_capacity
        self.scale = scale_px_per_inch
        self.min_blob_area = min_blob_area_px

        # Background subtractor for top-down canvas
        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )

        # Morphological structuring elements
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._frame_count: int = 0

    def reset(self) -> None:
        """Resets temporal background subtraction history."""
        self._fgbg = cv2.createBackgroundSubtractorMOG2(
            history=60,
            varThreshold=18.0,
            detectShadows=False,
        )
        self._frame_count = 0

    def quantify_frame(
        self,
        warped_field_plane: np.ndarray,
        isolate_dynamic_only: bool = True,
        calibrator: Optional[Any] = None,
    ) -> GamePieceQuantificationResult:
        """
        Executes Stage 2 quantification pipeline on a top-down flattened field plane:
        1. Background subtraction to isolate dynamic movement.
        2. Optimized dual-space HSV + RGB yellow chromaticity masking.
        3. Morphological opening and closing noise reduction.
        4. Contour extraction and centroid calculation.
        5. Proportional mass allocation out of 504 maximum capacity.
        """
        if warped_field_plane is None or warped_field_plane.size == 0:
            return GamePieceQuantificationResult(
                total_yellow_pixel_area=0.0,
                total_estimated_count=0,
                max_field_capacity=self.max_capacity,
                blob_clusters=[],
                foreground_mask=None,
                yellow_mask=None,
            )

        self._frame_count += 1
        canvas_h, canvas_w = warped_field_plane.shape[:2]

        # 1. Background subtraction on the flattened field plane
        fg_mask = self._fgbg.apply(warped_field_plane)

        # 2. Dual-Space Yellow Chromaticity Mask (HSV + RGB Contrast)
        hsv = cv2.cvtColor(warped_field_plane, cv2.COLOR_BGR2HSV)
        # Yellow Hue range: 16 to 38, Saturation: 75 to 255, Value: 70 to 255
        lower_yellow = np.array([16, 75, 70], dtype=np.uint8)
        upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_yellow, upper_yellow)

        b_ch, g_ch, r_ch = cv2.split(warped_field_plane)
        # Yellow has strong red and green, low blue. Green carpet has R < 80.
        r_f = r_ch.astype(np.float32)
        g_f = g_ch.astype(np.float32)
        b_f = b_ch.astype(np.float32)

        # Chromaticity contrast: (R + G) / (2 * max(1, B)) >= 1.30
        rgb_ratio = (r_f + g_f) / (2.0 * np.maximum(1.0, b_f))
        mask_rgb = (r_ch >= 120) & (g_ch >= 120) & (b_ch <= 130) & (rgb_ratio >= 1.30)

        # Raw yellow chromaticity mask
        yellow_chroma_mask = mask_hsv & (mask_rgb.astype(np.uint8) * 255)

        # 3. Dynamic element isolation
        if isolate_dynamic_only and self._frame_count > 5:
            # Dilate foreground motion mask slightly to ensure edges of moving balls are captured
            fg_dilated = cv2.dilate(fg_mask, self._kernel_open, iterations=1)
            # Combine yellow color with dynamic motion, while allowing high-confidence carpet clusters
            combined_mask = cv2.bitwise_and(yellow_chroma_mask, fg_dilated)
            # If dynamic mask is too sparse due to stationary balls, fall back to pure yellow chroma
            if cv2.countNonZero(combined_mask) < 20 and cv2.countNonZero(yellow_chroma_mask) > 0:
                combined_mask = yellow_chroma_mask
        else:
            combined_mask = yellow_chroma_mask

        # 4. Morphological opening/closing noise reduction
        # Opening removes isolated single-pixel noise
        mask_opened = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self._kernel_open)
        # Closing bridges small internal gaps in touching ball clusters
        clean_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, self._kernel_close)

        # 5. Extract contours for all yellow game piece clusters
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter valid blob clusters and compute raw pixel areas
        valid_blobs: List[Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int, int, int]]] = []
        total_pixel_area = 0.0

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_blob_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            # Compute centroid from spatial moments
            M = cv2.moments(c)
            if M["m00"] > 1e-6:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + w / 2.0)
                cy = float(y + h / 2.0)

            valid_blobs.append((c, area, (cx, cy), (x, y, w, h)))
            total_pixel_area += area

        # 6. Proportional Mass Allocation (assigning estimated count out of 504)
        blob_clusters: List[YellowBlobCluster] = []
        allocated_count_sum = 0

        for idx, (cnt, area, (cx, cy), bbox) in enumerate(valid_blobs):
            if total_pixel_area > 0.0:
                pct = float((area / total_pixel_area) * 100.0)
                est_count = int(round((area / total_pixel_area) * self.max_capacity))
            else:
                pct = 0.0
                est_count = 0

            # Convert canvas centroid (pixels) to metric field coordinates (inches)
            if calibrator is not None and hasattr(calibrator, "canvas_to_field_coords"):
                field_x, field_y = calibrator.canvas_to_field_coords(cx, cy)
            else:
                field_x = float(cx / self.scale)
                field_y = float(cy / self.scale)

            blob = YellowBlobCluster(
                blob_id=idx + 1,
                contour_px=cnt,
                pixel_area=area,
                percentage_total_area=pct,
                estimated_count=est_count,
                canvas_centroid_px=(cx, cy),
                field_coords_inch=(field_x, field_y),
                bounding_box_canvas=bbox,
            )
            blob_clusters.append(blob)
            allocated_count_sum += est_count

        # Bound total count to max capacity
        total_count = min(self.max_capacity, allocated_count_sum)

        return GamePieceQuantificationResult(
            total_yellow_pixel_area=total_pixel_area,
            total_estimated_count=total_count,
            max_field_capacity=self.max_capacity,
            blob_clusters=blob_clusters,
            foreground_mask=fg_mask,
            yellow_mask=clean_mask,
        )
