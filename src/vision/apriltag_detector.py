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
CANONICAL_FIELD_TAGS: Dict[int, Tuple[float, float, float]] = {
    1: (0.0, 100.0, 58.5),
    2: (0.0, 158.85, 58.5),
    3: (0.0, 217.7, 58.5),
    4: (40.0, 317.7, 45.0),
    5: (139.3, 158.85, 62.0),
    6: (162.8, 182.35, 62.0),
    7: (186.3, 158.85, 62.0),
    8: (162.8, 135.35, 62.0),
    9: (90.0, 140.0, 70.0),
    10: (90.0, 177.7, 70.0),
    11: (250.0, 0.0, 18.0),
    12: (325.6, 0.0, 18.0),
    13: (400.0, 0.0, 18.0),
    14: (250.0, 317.7, 18.0),
    15: (325.6, 317.7, 18.0),
    16: (400.0, 317.7, 18.0),
    17: (464.9, 158.85, 62.0),
    18: (488.4, 182.35, 62.0),
    19: (511.9, 158.85, 62.0),
    20: (488.4, 135.35, 62.0),
    21: (561.2, 140.0, 70.0),
    22: (561.2, 177.7, 70.0),
    23: (651.2, 100.0, 58.5),
    24: (651.2, 158.85, 58.5),
    25: (651.2, 217.7, 58.5),
    26: (611.2, 317.7, 45.0),
    27: (307.6, 280.0, 30.0),
    28: (343.6, 280.0, 30.0),
    29: (307.6, 37.7, 30.0),
    30: (343.6, 37.7, 30.0),
    31: (100.0, 0.0, 20.0),
    32: (551.2, 0.0, 20.0),
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
