"""FRC Bumper Detection and Proposal Fusion Module.

Loads trained FRC Bumper YOLO models, isolates bumper fabric patches,
classifies alliance color (RED vs BLUE), registers ground-carpet contact points,
and fuses bumper base proposals with robot superstructure detections.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.common.types import BoundingBox
from src.vision.detector_trt import classify_bumper_fabric

logger = logging.getLogger("frc.vision.bumper_detector")

DEFAULT_BUMPER_MODEL_CANDIDATES = [
    "weights/bumper_best.pt",
    "data/models/bumper_best.pt",
    "/home/RowanH/bumper/FRC-Bumper-Detection-11/runs/detect/train-2/weights/best.pt",
    "/home/RowanH/bumper/FRC-Bumper-Detection-1/runs/detect/train-2/weights/best.pt",
]


def find_best_bumper_model_path(candidate_path: Optional[str] = None) -> Optional[str]:
    """Locates the best available bumper model weights path."""
    if candidate_path and os.path.isfile(candidate_path):
        return str(Path(candidate_path).resolve())

    for p in DEFAULT_BUMPER_MODEL_CANDIDATES:
        if os.path.isfile(p):
            return str(Path(p).resolve())

    return None


@dataclass(slots=True)
class BumperDetection:
    """Detection of an FRC robot bumper with alliance classification."""
    box: BoundingBox
    confidence: float
    alliance: str  # "RED", "BLUE", or "UNKNOWN"
    color_purity: float
    field_coords_inch: Optional[Tuple[float, float]] = None


class BumperDetector:
    """Performs neural network bumper detection and robot proposal fusion."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        imgsz: int = 1280,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model_path = find_best_bumper_model_path(model_path)
        self._model = None

        if self.model_path:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                logger.info("Loaded FRC Bumper YOLO model: %s", self.model_path)
            except Exception as e:
                logger.warning("Failed to load bumper YOLO model (%s): %s", self.model_path, e)
        else:
            logger.info("No bumper model weights found; bumper detector in inactive mode")

    @property
    def is_available(self) -> bool:
        """Returns True if the bumper model is successfully loaded."""
        return self._model is not None

    def detect_bumpers(
        self,
        frame_bgr: np.ndarray,
        H_image_to_field: Optional[np.ndarray] = None,
    ) -> List[BumperDetection]:
        """Detects bumpers in frame and classifies fabric color purity."""
        if not self.is_available or frame_bgr is None or frame_bgr.size == 0:
            return []

        h_img, w_img = frame_bgr.shape[:2]
        detections: List[BumperDetection] = []

        try:
            results = self._model.predict(
                frame_bgr,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
            if not results or results[0].boxes is None:
                return []

            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())

                x1 = max(0, int(xyxy[0]))
                y1 = max(0, int(xyxy[1]))
                x2 = min(w_img, int(xyxy[2]))
                y2 = min(h_img, int(xyxy[3]))

                bw = float(x2 - x1)
                bh = float(y2 - y1)
                if bw < 5.0 or bh < 5.0:
                    continue

                cx = float(x1 + x2) / 2.0
                cy = float(y1 + y2) / 2.0

                crop = frame_bgr[y1:y2, x1:x2]
                score, alliance = classify_bumper_fabric(crop)

                # Class ID encoding: 1 = RED bumper, 2 = BLUE bumper, 0 = UNKNOWN
                cls_id = 1 if alliance == "RED" else (2 if alliance == "BLUE" else 0)

                b_box = BoundingBox(
                    center_x=cx,
                    center_y=cy,
                    width=bw,
                    height=bh,
                    confidence=conf,
                    class_id=cls_id,
                )

                field_pt: Optional[Tuple[float, float]] = None
                if H_image_to_field is not None:
                    # Ground contact point is bottom-center of bumper
                    ground_px = np.array([cx, float(y2), 1.0], dtype=np.float64)
                    proj = H_image_to_field @ ground_px
                    z = proj[2] if abs(proj[2]) > 1e-9 else 1e-9
                    field_pt = (float(proj[0] / z), float(proj[1] / z))

                detections.append(
                    BumperDetection(
                        box=b_box,
                        confidence=conf,
                        alliance=alliance,
                        color_purity=score,
                        field_coords_inch=field_pt,
                    )
                )
        except Exception as e:
            logger.debug("Error in bumper detection inference: %s", e)

        return detections

    def extrapolate_robot_box(
        self,
        bumper: BumperDetection,
    ) -> BoundingBox:
        """Extrapolates full robot bounding box from a bumper detection base."""
        b = bumper.box
        bw = b.width
        bh = b.height
        ground_y = b.center_y + bh / 2.0

        # Robot total height is typically 2.4x to 3.5x bumper height or ~0.9x width
        est_robot_h = max(bh * 2.8, bw * 0.85)
        cy = ground_y - est_robot_h / 2.0

        return BoundingBox(
            center_x=b.center_x,
            center_y=cy,
            width=bw,
            height=est_robot_h,
            confidence=bumper.confidence,
            class_id=b.class_id,
        )

    def fuse_proposals(
        self,
        robot_boxes: Sequence[BoundingBox],
        bumper_detections: Sequence[BumperDetection],
        min_overlap_iob: float = 0.25,
    ) -> List[BoundingBox]:
        """
        Fuses whole-robot proposals with bumper detections:
        1. When a robot box overlaps a bumper, refines ground registration,
           boosts confidence, and injects bumper alliance (class_id 1=RED, 2=BLUE).
        2. When a high-confidence bumper has no matching robot box, synthesizes
           a recovered robot proposal.
        """
        fused: List[BoundingBox] = []
        unmatched_bumpers = list(bumper_detections)

        for r_box in robot_boxes:
            rx1 = r_box.center_x - r_box.width / 2.0
            ry1 = r_box.center_y - r_box.height / 2.0
            rx2 = r_box.center_x + r_box.width / 2.0
            ry2 = r_box.center_y + r_box.height / 2.0

            best_bumper: Optional[BumperDetection] = None
            best_iob = 0.0

            for b_det in unmatched_bumpers:
                b_box = b_det.box
                bx1 = b_box.center_x - b_box.width / 2.0
                by1 = b_box.center_y - b_box.height / 2.0
                bx2 = b_box.center_x + b_box.width / 2.0
                by2 = b_box.center_y + b_box.height / 2.0

                ix1 = max(rx1, bx1)
                iy1 = max(ry1, by1)
                ix2 = min(rx2, bx2)
                iy2 = min(ry2, by2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    bumper_area = (bx2 - bx1) * (by2 - by1)
                    iob = inter_area / max(1.0, bumper_area)
                    if iob > best_iob and iob >= min_overlap_iob:
                        best_iob = iob
                        best_bumper = b_det

            if best_bumper is not None:
                unmatched_bumpers.remove(best_bumper)
                b_ground = best_bumper.box.center_y + best_bumper.box.height / 2.0
                r_top = ry1
                fused_h = max(r_box.height, b_ground - r_top)
                fused_cy = b_ground - fused_h / 2.0
                fused_w = max(r_box.width, best_bumper.box.width)
                fused_conf = min(0.99, float(r_box.confidence * 0.50 + best_bumper.confidence * 0.40 + 0.15))
                cls_id = best_bumper.box.class_id

                fused.append(
                    BoundingBox(
                        center_x=r_box.center_x,
                        center_y=fused_cy,
                        width=fused_w,
                        height=fused_h,
                        confidence=fused_conf,
                        class_id=cls_id,
                    )
                )
            else:
                fused.append(r_box)

        # Recover missing robots from unassigned verified bumpers
        for b_det in unmatched_bumpers:
            if b_det.confidence >= 0.30 and b_det.color_purity >= 0.20 and b_det.alliance in ("RED", "BLUE"):
                recovered_box = self.extrapolate_robot_box(b_det)
                fused.append(recovered_box)

        return fused
