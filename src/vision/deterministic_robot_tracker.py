"""
Stage 3: Deterministic Robot Localization & Identification.

Filters candidate proposals using spatial field boundaries, aspect ratio, and physical footprint
constraints (discarding human referees and media personnel), classifies bumper fabric color purity
(Red vs Blue), and enforces a strict 3v3 cardinality constraint via bipartite Hungarian matching
and spatial persistence across frame occlusions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.common.types import BoundingBox
from src.vision.detector_trt import classify_bumper_fabric
from src.vision.robot_ekf import RobotEKF, emulate_iou, fuse_bounding_boxes
from src.vision.spatial_types import (
    FieldGeometry,
    RobotCandidateProposal,
    RobotLocalizationResult,
    TrackedRobotUnit,
)

logger = logging.getLogger("frc.vision.deterministic_robot_tracker")


class DeterministicRobotTracker:
    """
    Deterministic localization and tracking engine.
    Strictly outputs exactly 3 Red robots and 3 Blue robots in every frame.
    """

    def __init__(
        self,
        field_geometry: Optional[FieldGeometry] = None,
        min_aspect_ratio: float = 0.52,
        max_aspect_ratio: float = 2.10,
        min_pixel_area: float = 1200.0,
        max_pixel_area: float = 45000.0,
        carpet_margin_inch: float = 6.0,
        min_track_confidence: float = 0.50,
        min_bumper_score: float = 0.22,
        max_matching_cost: float = 0.70,
        max_gating_dist_inch: float = 240.0,
        grouping_proximity_px: float = 160.0,
        grouping_proximity_inch: float = 65.0,
        grouping_emul_threshold: float = 0.02,
    ) -> None:
        self.geom = field_geometry if field_geometry is not None else FieldGeometry()
        self.min_ar = min_aspect_ratio
        self.max_ar = max_aspect_ratio
        self.min_area = min_pixel_area
        self.max_area = max_pixel_area
        self.carpet_margin = carpet_margin_inch
        self.min_track_confidence = min_track_confidence
        self.min_bumper_score = min_bumper_score
        self.max_matching_cost = max_matching_cost
        self.max_gating_dist_inch = max_gating_dist_inch
        self.grouping_proximity_px = grouping_proximity_px
        self.grouping_proximity_inch = grouping_proximity_inch
        self.grouping_emul_threshold = grouping_emul_threshold

        # Persistent track states for exactly 3 Red and 3 Blue units
        self._red_tracks: List[TrackedRobotUnit] = []
        self._blue_tracks: List[TrackedRobotUnit] = []
        self._initialized: bool = False

        # Previous frame velocity vectors (vx, vy) in field inches/sec
        self._velocities: Dict[str, Tuple[float, float]] = {}

        # Dedicated Extended Kalman Filter (EKF) per canonical robot unit
        self._ekfs: Dict[str, RobotEKF] = {}

    def reset(self) -> None:
        """Resets persistent track memory."""
        self._red_tracks.clear()
        self._blue_tracks.clear()
        self._velocities.clear()
        self._ekfs.clear()
        self._initialized = False

    def _bootstrap_tracks(self) -> None:
        """Initializes canonical 3v3 units at standard FRC alliance starting line locations."""
        red_starts = [(120.0, 70.0), (120.0, 160.0), (120.0, 250.0)]
        blue_starts = [(530.0, 70.0), (530.0, 160.0), (530.0, 250.0)]

        self._red_tracks = [
            TrackedRobotUnit(
                unit_id=f"RED_{i+1}",
                alliance="RED",
                pixel_bbox=BoundingBox(
                    center_x=350.0, center_y=350.0 + i * 80.0, width=90.0, height=70.0, confidence=0.85, class_id=0
                ),
                canvas_bbox=(int(x * self.geom.carpet_scale_px_per_inch - 30), int(y * self.geom.carpet_scale_px_per_inch - 30), 60, 60),
                field_x_inch=x,
                field_y_inch=y,
                color_confidence=0.90,
                persistence_score=0.95,
                is_projected=True,
            )
            for i, (x, y) in enumerate(red_starts)
        ]

        self._blue_tracks = [
            TrackedRobotUnit(
                unit_id=f"BLUE_{i+1}",
                alliance="BLUE",
                pixel_bbox=BoundingBox(
                    center_x=1400.0, center_y=350.0 + i * 80.0, width=90.0, height=70.0, confidence=0.85, class_id=0
                ),
                canvas_bbox=(int(x * self.geom.carpet_scale_px_per_inch - 30), int(y * self.geom.carpet_scale_px_per_inch - 30), 60, 60),
                field_x_inch=x,
                field_y_inch=y,
                color_confidence=0.90,
                persistence_score=0.95,
                is_projected=True,
            )
            for i, (x, y) in enumerate(blue_starts)
        ]

        for u in self._red_tracks + self._blue_tracks:
            self._velocities[u.unit_id] = (0.0, 0.0)
            self._ekfs[u.unit_id] = RobotEKF(
                init_field_x=u.field_x_inch,
                init_field_y=u.field_y_inch,
                init_w_px=u.pixel_bbox.width,
                init_h_px=u.pixel_bbox.height,
                init_vx=0.0,
                init_vy=0.0,
                carpet_length_inch=self.geom.length_inches,
                carpet_width_inch=self.geom.width_inches,
            )

        self._initialized = True

    def _extract_regional_contour_proposals(
        self,
        frame_bgr: np.ndarray,
        fg_mask: Optional[np.ndarray] = None,
    ) -> List[BoundingBox]:
        """Extracts candidate regional contours from frame and motion cues."""
        h_frame, w_frame = frame_bgr.shape[:2]
        proposals: List[BoundingBox] = []

        # Color-saliency map (suppressing green carpet to highlight robot chassis and bumpers)
        b, g, r = cv2.split(frame_bgr)
        r_f = r.astype(np.float32)
        g_f = g.astype(np.float32)
        b_f = b.astype(np.float32)

        # High red or high blue relative to green
        bumper_saliency = np.maximum(r_f - g_f, b_f - g_f)
        saliency_mask = (bumper_saliency > 25.0).astype(np.uint8) * 255

        if fg_mask is not None:
            if fg_mask.shape[:2] != (h_frame, w_frame):
                fg_resized = cv2.resize(fg_mask, (w_frame, h_frame), interpolation=cv2.INTER_NEAREST)
            else:
                fg_resized = fg_mask
            saliency_mask = cv2.bitwise_or(saliency_mask, fg_resized)

        # Morphological dilation to group bumper parts and intake mechanisms into single robot chassis
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        grouped = cv2.dilate(saliency_mask, kernel, iterations=3)

        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = float(w * h)
            if area >= self.min_area:
                proposals.append(
                    BoundingBox(
                        center_x=float(x + w / 2.0),
                        center_y=float(y + h / 2.0),
                        width=float(w),
                        height=float(h),
                        confidence=0.75,
                        class_id=0,
                    )
                )

        return proposals

    def track_frame(
        self,
        frame_bgr: np.ndarray,
        H_image_to_field: np.ndarray,
        external_proposals: Optional[Sequence[BoundingBox]] = None,
        fg_mask: Optional[np.ndarray] = None,
        calibrator: Optional[Any] = None,
    ) -> RobotLocalizationResult:
        """
        Executes Stage 3 localization and tracking:
        1. Accumulates candidate proposals from regional contours or external detectors.
        2. Filters out human referees, media staff, and non-robot artifacts.
        3. Classifies candidate bumper fabric purity (Red vs Blue).
        4. Solves bipartite matching with Hungarian algorithm for exact 3v3 cardinality.
        """
        if not self._initialized:
            self._bootstrap_tracks()

        h_frame, w_frame = frame_bgr.shape[:2]

        # 1. Gather raw proposals
        raw_boxes: List[BoundingBox] = []
        if external_proposals and len(external_proposals) > 0:
            raw_boxes.extend(external_proposals)
        else:
            raw_boxes.extend(self._extract_regional_contour_proposals(frame_bgr, fg_mask))

        candidate_proposals: List[RobotCandidateProposal] = []
        surviving_red_proposals: List[Tuple[BoundingBox, float, Tuple[float, float]]] = []
        surviving_blue_proposals: List[Tuple[BoundingBox, float, Tuple[float, float]]] = []

        # 2. Geometric, contextual, and footprint filtering
        for idx, box in enumerate(raw_boxes):
            w = box.width
            h = box.height
            cx = box.center_x
            cy = box.center_y
            area = w * h
            ar = w / max(1.0, h)

            # Map candidate base to physical field ground plane
            # Evaluate ground contact point (bottom-center of bounding box)
            pt_ground = np.array([cx, cy + h * 0.40, 1.0], dtype=np.float64)
            proj = H_image_to_field @ pt_ground
            z = proj[2] if abs(proj[2]) > 1e-9 else 1e-9
            fx = float(proj[0] / z)
            fy = float(proj[1] / z)

            # Check field boundary (ignoring human referees & media outside perimeter)
            is_inside_field = (
                (-self.carpet_margin <= fx <= self.geom.length_inches + self.carpet_margin)
                and (-self.carpet_margin <= fy <= self.geom.width_inches + self.carpet_margin)
            )

            filter_reason: Optional[str] = None
            if box.confidence < self.min_track_confidence:
                filter_reason = f"CONFIDENCE_TOO_LOW ({box.confidence:.2f} < {self.min_track_confidence:.2f})"
            elif not is_inside_field:
                filter_reason = "OUTSIDE_FIELD_PERIMETER (Referee / Media / Spectator)"
            elif ar < self.min_ar:
                filter_reason = f"ASPECT_RATIO_TOO_SLENDER ({ar:.2f} < {self.min_ar}: Human upright figure)"
            elif ar > self.max_ar:
                filter_reason = f"ASPECT_RATIO_TOO_WIDE ({ar:.2f} > {self.max_ar})"
            elif area < self.min_area:
                filter_reason = f"AREA_TOO_SMALL ({area:.0f} < {self.min_area})"
            elif area > self.max_area:
                filter_reason = f"AREA_TOO_LARGE ({area:.0f} > {self.max_area})"

            prop = RobotCandidateProposal(
                proposal_id=idx + 1,
                pixel_bbox=box,
                field_coords_inch=(fx, fy),
                aspect_ratio=ar,
                pixel_area=area,
                is_inside_field=is_inside_field,
                filter_reason=filter_reason,
            )
            candidate_proposals.append(prop)

            if filter_reason is not None:
                continue

            # 3. Bumper Colorimetric Classification on surviving candidate
            if box.class_id == 1:
                # Pre-verified RED bumper proposal
                surviving_red_proposals.append((box, max(0.85, box.confidence), (fx, fy)))
            elif box.class_id == 2:
                # Pre-verified BLUE bumper proposal
                surviving_blue_proposals.append((box, max(0.85, box.confidence), (fx, fy)))
            else:
                # Extract lower 35% bounding box crop
                x1 = max(0, int(cx - w / 2.0))
                y1 = max(0, int(cy - h / 2.0))
                x2 = min(w_frame, int(cx + w / 2.0))
                y2 = min(h_frame, int(cy + h / 2.0))

                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                crop = frame_bgr[y1:y2, x1:x2]
                score, alliance = classify_bumper_fabric(crop)

                if alliance == "RED":
                    surviving_red_proposals.append((box, score, (fx, fy)))
                elif alliance == "BLUE":
                    surviving_blue_proposals.append((box, score, (fx, fy)))
                else:
                    # Ambiguous detection: only assign if sufficient bumper score
                    if score >= self.min_bumper_score:
                        if fx < self.geom.length_inches / 2.0:
                            surviving_red_proposals.append((box, score, (fx, fy)))
                        else:
                            surviving_blue_proposals.append((box, score, (fx, fy)))

        # 4. Strict 3v3 Bipartite Hungarian Assignment with Spatial Persistence
        updated_red_units = self._assign_alliance_units(
            tracks=self._red_tracks,
            proposals=surviving_red_proposals,
            alliance="RED",
            H_image_to_field=H_image_to_field,
            calibrator=calibrator,
        )
        self._red_tracks = updated_red_units

        updated_blue_units = self._assign_alliance_units(
            tracks=self._blue_tracks,
            proposals=surviving_blue_proposals,
            alliance="BLUE",
            H_image_to_field=H_image_to_field,
            calibrator=calibrator,
        )
        self._blue_tracks = updated_blue_units

        all_units = self._red_tracks + self._blue_tracks

        return RobotLocalizationResult(
            raw_proposals_count=len(raw_boxes),
            filtered_proposals_count=len(candidate_proposals) - (len(surviving_red_proposals) + len(surviving_blue_proposals)),
            candidate_proposals=candidate_proposals,
            red_units=self._red_tracks,
            blue_units=self._blue_tracks,
            all_units=all_units,
        )

    def _assign_alliance_units(
        self,
        tracks: List[TrackedRobotUnit],
        proposals: List[Tuple[BoundingBox, float, Tuple[float, float]]],
        alliance: str,
        H_image_to_field: np.ndarray,
        calibrator: Optional[Any] = None,
    ) -> List[TrackedRobotUnit]:
        """
        Executes Hungarian bipartite matching on 3 alliance tracks against M candidate proposals.
        Guarantees STRICTLY 3 output units for this alliance.

        Integrates:
        1. RobotEKF per unit with predictive kinematic motion propagation.
        2. Proximity-emulated IoU for cost evaluation (preventing match dropouts when boxes don't intersect).
        3. Multi-detection clustering: unassigned candidate detections within proximity to a matched bot
           act as additional detections, updating the bot's EKF sequentially and shrinking covariance.
        4. Fused bounding box geometry encompassing all corroborating detections.
        """
        num_tracks = 3
        num_props = len(proposals)
        dt = 1.0 / 30.0

        # Ensure EKF exists for each unit and predict motion
        for i in range(num_tracks):
            unit_id = f"{alliance}_{i+1}"
            prev_t = tracks[i]
            if unit_id not in self._ekfs:
                self._ekfs[unit_id] = RobotEKF(
                    init_field_x=prev_t.field_x_inch,
                    init_field_y=prev_t.field_y_inch,
                    init_w_px=prev_t.pixel_bbox.width,
                    init_h_px=prev_t.pixel_bbox.height,
                    init_vx=self._velocities.get(unit_id, (0.0, 0.0))[0],
                    init_vy=self._velocities.get(unit_id, (0.0, 0.0))[1],
                    carpet_length_inch=self.geom.length_inches,
                    carpet_width_inch=self.geom.width_inches,
                )
            self._ekfs[unit_id].predict(dt=dt)

        # Build Cost Matrix (size: 3 x max(3, num_props))
        cols = max(num_tracks, num_props)
        cost_matrix = np.full((num_tracks, cols), 1.0, dtype=np.float64)

        for i in range(num_tracks):
            unit_id = f"{alliance}_{i+1}"
            ekf = self._ekfs[unit_id]
            t = tracks[i]
            for j in range(num_props):
                box, score, (fx, fy) = proposals[j]
                # Spatial distance penalty (normalized by 300 inches)
                dist = np.hypot(ekf.field_x - fx, ekf.field_y - fy)
                dist_cost = min(1.0, dist / 300.0)

                # Distance-aware emulated IoU (smooth proximity falloff even if boxes do not intersect)
                iou_emul = emulate_iou(
                    box1=t.pixel_bbox,
                    box2=box,
                    proximity_thresh_px=120.0,
                    field_coords1=(ekf.field_x, ekf.field_y),
                    field_coords2=(fx, fy),
                    proximity_thresh_inch=48.0,
                )
                iou_cost = 1.0 - iou_emul

                # Color confidence reward
                color_cost = 1.0 - score

                cost = 0.45 * dist_cost + 0.25 * color_cost + 0.30 * iou_cost
                cost_matrix[i, j] = cost

        # Solve Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_props: Dict[int, int] = {}
        matched_cols = set()
        for r, c in zip(row_ind, col_ind):
            if r < num_tracks and c < num_props:
                cost_val = float(cost_matrix[r, c])
                box, score, (fx, fy) = proposals[c]
                unit_id = f"{alliance}_{r+1}"
                ekf = self._ekfs[unit_id]
                dist_val = float(np.hypot(ekf.field_x - fx, ekf.field_y - fy))

                # Strict Hungarian Gating: reject outliers with excessive cost or distance jump
                if cost_val <= self.max_matching_cost and dist_val <= self.max_gating_dist_inch:
                    matched_props[r] = c
                    matched_cols.add(c)
                else:
                    logger.debug(
                        "Gating rejected assignment for %s to proposal %d (cost=%.2f > %.2f or dist=%.1f > %.1f in)",
                        unit_id, c, cost_val, self.max_matching_cost, dist_val, self.max_gating_dist_inch,
                    )

        # Remaining proposals of this alliance available for proximity multi-detection fusion
        unassigned_indices = [j for j in range(num_props) if j not in matched_cols]

        assigned_units: List[TrackedRobotUnit] = []
        for i in range(num_tracks):
            unit_id = f"{alliance}_{i+1}"
            prev_t = tracks[i]
            ekf = self._ekfs[unit_id]

            if i in matched_props:
                # Confirmed Primary Detection Match
                matched_col = matched_props[i]
                primary_box, primary_score, (primary_fx, primary_fy) = proposals[matched_col]

                # Relaxed Multi-Detection Proximity Grouping:
                # Groups corroborating detections (chassis, intake, mechanisms) within relaxed proximity
                additional_boxes: List[BoundingBox] = []
                additional_fields: List[Tuple[float, float]] = []
                additional_scores: List[float] = []

                clustered_boxes = [primary_box]
                clustered_fields = [(primary_fx, primary_fy)]

                changed = True
                while changed and unassigned_indices:
                    changed = False
                    fused_current = fuse_bounding_boxes(clustered_boxes)
                    for u_idx in list(unassigned_indices):
                        cand_box, cand_score, (cand_fx, cand_fy) = proposals[u_idx]
                        emul_prim = emulate_iou(
                            box1=primary_box,
                            box2=cand_box,
                            proximity_thresh_px=self.grouping_proximity_px,
                            field_coords1=(primary_fx, primary_fy),
                            field_coords2=(cand_fx, cand_fy),
                            proximity_thresh_inch=self.grouping_proximity_inch,
                        )
                        emul_fused = emulate_iou(
                            box1=fused_current,
                            box2=cand_box,
                            proximity_thresh_px=self.grouping_proximity_px,
                            field_coords1=(ekf.field_x, ekf.field_y),
                            field_coords2=(cand_fx, cand_fy),
                            proximity_thresh_inch=self.grouping_proximity_inch,
                        )
                        emul = max(emul_prim, emul_fused)
                        if emul > self.grouping_emul_threshold:
                            additional_boxes.append(cand_box)
                            additional_fields.append((cand_fx, cand_fy))
                            additional_scores.append(cand_score)
                            clustered_boxes.append(cand_box)
                            clustered_fields.append((cand_fx, cand_fy))
                            unassigned_indices.remove(u_idx)
                            changed = True

                # Multi-observation EKF update: shrinks covariance with each additional detection
                ekf.update_multi(
                    primary_box=primary_box,
                    additional_boxes=additional_boxes,
                    H_image_to_field=H_image_to_field,
                    primary_field=(primary_fx, primary_fy),
                    additional_fields=additional_fields,
                )

                fused_fx = ekf.field_x
                fused_fy = ekf.field_y
                vx = ekf.vx
                vy = ekf.vy
                self._velocities[unit_id] = (vx, vy)

                # Consolidate bounding box to encompass all corroborating detections
                fused_bbox = fuse_bounding_boxes([primary_box] + additional_boxes)

                # Blended color confidence and boosted persistence
                blended_score = float(np.mean([primary_score] + additional_scores)) if additional_scores else primary_score
                persistence = min(1.0, prev_t.persistence_score + 0.05 * (1 + len(additional_boxes)))

                if calibrator is not None and hasattr(calibrator, "field_coords_to_canvas"):
                    cx_c, cy_c = calibrator.field_coords_to_canvas(fused_fx, fused_fy)
                    canvas_x, canvas_y = int(cx_c), int(cy_c)
                else:
                    canvas_x = int(fused_fx * self.geom.carpet_scale_px_per_inch)
                    canvas_y = int(fused_fy * self.geom.carpet_scale_px_per_inch)
                canvas_box = (canvas_x - 30, canvas_y - 30, 60, 60)

                assigned_units.append(
                    TrackedRobotUnit(
                        unit_id=unit_id,
                        alliance=alliance,
                        pixel_bbox=fused_bbox,
                        canvas_bbox=canvas_box,
                        field_x_inch=fused_fx,
                        field_y_inch=fused_fy,
                        color_confidence=blended_score,
                        persistence_score=persistence,
                        is_projected=False,
                        additional_detections=additional_boxes,
                    )
                )
            else:
                # Occluded / Missing Detection -> EKF Coasting & Spatial Persistence
                pred_fx = ekf.field_x
                pred_fy = ekf.field_y
                vx = ekf.vx
                vy = ekf.vy
                self._velocities[unit_id] = (vx, vy)

                pred_bbox = ekf.get_pixel_bbox(
                    confidence=prev_t.pixel_bbox.confidence * 0.92,
                    H_image_to_field=H_image_to_field,
                )
                if pred_bbox.center_x == 0.0 and pred_bbox.center_y == 0.0:
                    pred_bbox = prev_t.pixel_bbox

                if calibrator is not None and hasattr(calibrator, "field_coords_to_canvas"):
                    cx_c, cy_c = calibrator.field_coords_to_canvas(pred_fx, pred_fy)
                    canvas_x, canvas_y = int(cx_c), int(cy_c)
                else:
                    canvas_x = int(pred_fx * self.geom.carpet_scale_px_per_inch)
                    canvas_y = int(pred_fy * self.geom.carpet_scale_px_per_inch)
                canvas_box = (canvas_x - 30, canvas_y - 30, 60, 60)

                persistence = max(0.10, prev_t.persistence_score * 0.92)
                assigned_units.append(
                    TrackedRobotUnit(
                        unit_id=unit_id,
                        alliance=alliance,
                        pixel_bbox=pred_bbox,
                        canvas_bbox=canvas_box,
                        field_x_inch=pred_fx,
                        field_y_inch=pred_fy,
                        color_confidence=prev_t.color_confidence * 0.95,
                        persistence_score=persistence,
                        is_projected=True,
                        additional_detections=[],
                    )
                )

        return assigned_units

    @staticmethod
    def _box_iou(b1: BoundingBox, b2: BoundingBox) -> float:
        """Calculates Intersection over Union between two BoundingBoxes."""
        x1 = max(b1.center_x - b1.width / 2.0, b2.center_x - b2.width / 2.0)
        y1 = max(b1.center_y - b1.height / 2.0, b2.center_y - b2.height / 2.0)
        x2 = min(b1.center_x + b1.width / 2.0, b2.center_x + b2.width / 2.0)
        y2 = min(b1.center_y + b1.height / 2.0, b2.center_y + b2.height / 2.0)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = (b1.width * b1.height) + (b2.width * b2.height) - inter
        return float(inter / union) if union > 0 else 0.0
