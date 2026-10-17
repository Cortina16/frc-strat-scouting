"""
AprilTag fiducial detector for FRC 2026 REBUILT field geometry.

Targeting AprilTag family tag36h11 across the 32 known field tag locations
specified in configs/field_2026_rebuilt.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.common.types import RawDecodedFrame, TagDetectionResult

# Canonical 32 AprilTag CAD positions (x, y, z) in inches for FRC 2026 REBUILT.
# Origin (0, 0, 0) is at the bottom-left corner of the field carpet.
# Extracted directly from official CAD model field.glb.
CANONICAL_FIELD_TAGS: Dict[int, Tuple[float, float, float]] = {
    # Red Hub & Trench Assembly (Tags 1-12)
    1: (183.33, 25.38, 35.00),
    2: (182.10, 135.10, 44.25),
    3: (205.85, 144.85, 44.25),
    4: (205.62, 158.85, 44.25),
    5: (182.10, 182.60, 44.25),
    6: (183.33, 292.32, 35.00),
    7: (180.64, 292.32, 35.00),
    8: (168.10, 182.36, 44.25),
    9: (158.34, 172.85, 44.25),
    10: (158.58, 158.85, 44.25),
    11: (168.10, 135.34, 44.25),
    12: (180.64, 25.38, 35.00),
    # Red Driver Station Wall (Tags 13-16)
    13: (0.05, 26.23, 21.75),
    14: (0.28, 43.22, 21.75),
    15: (0.06, 147.48, 21.75),
    16: (0.06, 164.48, 21.75),
    # Blue Hub & Trench Assembly (Tags 17-28)
    17: (467.64, 292.32, 35.00),
    18: (469.10, 182.60, 44.25),
    19: (445.35, 172.85, 44.25),
    20: (445.58, 158.85, 44.25),
    21: (469.10, 135.10, 44.25),
    22: (467.87, 25.38, 35.00),
    23: (470.33, 25.38, 35.00),
    24: (483.10, 135.34, 44.25),
    25: (492.62, 144.85, 44.25),
    26: (492.86, 158.85, 44.25),
    27: (483.10, 182.36, 44.25),
    28: (470.33, 292.32, 35.00),
    # Blue Driver Station Wall (Tags 29-32)
    29: (651.15, 291.47, 21.75),
    30: (650.92, 274.48, 21.75),
    31: (651.14, 170.22, 21.75),
    32: (650.90, 153.22, 21.75),
}


def load_field_tags(config_path: Optional[str] = None) -> Dict[int, Tuple[float, float, float]]:
    """Loads 3D field coordinates for AprilTags from configuration or falls back to canonical table."""
    paths_to_check = []
    if config_path:
        paths_to_check.append(Path(config_path))
    paths_to_check.extend([
        Path("configs/field_2026_rebuilt.yaml"),
        Path(__file__).resolve().parent.parent.parent / "configs" / "field_2026_rebuilt.yaml",
    ])

    for p in paths_to_check:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                tags = cfg.get("apriltags", {}).get("tags", [])
                loaded = {
                    int(t["id"]): (float(t["x"]), float(t["y"]), float(t["z"]))
                    for t in tags
                }
                if loaded:
                    return loaded
            except Exception:
                pass

    return dict(CANONICAL_FIELD_TAGS)


class AprilTagDetector:
    """Wrapper around OpenCV ArUco AprilTag detector targeting family tag36h11."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.field_tags = load_field_tags(config_path)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # Zero decimation: full resolution processing, zero downsampling for maximum tag sensitivity
        self.parameters.aprilTagQuadDecimate = 0.0
        self.parameters.aprilTagQuadSigma = 0.0

        # Prefer modern ArucoDetector interface (OpenCV >= 4.7)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector: Optional[cv2.aruco.ArucoDetector] = cv2.aruco.ArucoDetector(
                self.dictionary, self.parameters
            )
        else:
            self._detector = None

    def detect(self, image_bgr: np.ndarray) -> List[TagDetectionResult]:
        """Detects tag36h11 markers in BGR image and maps to CAD field coordinates."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        elif len(image_bgr.shape) == 2:
            gray = image_bgr
        else:
            return []

        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if ids is None or len(ids) == 0:
            return []

        results: List[TagDetectionResult] = []
        for i, tag_arr in enumerate(ids):
            tag_id = int(np.asarray(tag_arr).ravel()[0])
            c = corners[i].reshape(4, 2).astype(np.float32)
            center = (float(c[:, 0].mean()), float(c[:, 1].mean()))
            field_coords = self.field_tags.get(tag_id, (0.0, 0.0, 0.0))

            results.append(
                TagDetectionResult(
                    tag_id=tag_id,
                    corners_px=c,
                    center_px=center,
                    field_coords_inch=field_coords,
                    hamming=0,
                    reprojection_err=0.0,
                )
            )

        return results

    def detect_from_frame(self, frame: RawDecodedFrame) -> List[TagDetectionResult]:
        """Extracts AprilTags from a RawDecodedFrame."""
        if frame is None or frame.image_bgr is None:
            return []
        return self.detect(frame.image_bgr)
