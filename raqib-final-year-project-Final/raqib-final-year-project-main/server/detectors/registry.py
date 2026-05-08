"""
Model registry
==============
Lazy, cached factory for the vehicle detectors (YOLO11, Faster R-CNN,
VisDrone) and the lane detector (YOLOPv2). A detector is loaded on its
first request and kept resident for the rest of the process lifetime —
the registry trades a one-off several-second load per model for
constant-time retrieval thereafter.

This structure lets ``main.py`` dispatch each request to the requested
detector pair without re-loading weights mid-session, while still
letting the user toggle between models from the browser dropdowns.

The earlier RAQIB drafts also wired in RT-DETRv2 (transformer-based
vehicle detector) and UFLD (row-anchor lane detector). They were
removed late in the project: their behaviour was redundant with
YOLO11 / Faster R-CNN / VisDrone (vehicles) and YOLOPv2 (lanes), and
exposing four vehicle and two lane choices in the UI made the
comparison story harder to follow than the contribution warranted.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Canonical names used on the wire (WebSocket message / REST body /
# client UI). Aliases (below) all resolve to one of these for caching.
VEHICLE_MODELS   = ("yolo11", "fasterrcnn", "visdrone")
LANE_MODELS      = ("yolopv2",)
DEFAULT_VEHICLE  = "yolo11"
DEFAULT_LANE     = "yolopv2"

_VEHICLE_ALIASES = {
    "yolo":            "yolo11",
    "yolo11":          "yolo11",
    "yolo11s":         "yolo11",
    "fasterrcnn":      "fasterrcnn",
    "faster-rcnn":     "fasterrcnn",
    "faster_rcnn":     "fasterrcnn",
    "frcnn":           "fasterrcnn",
    "rcnn":            "fasterrcnn",
    "visdrone":        "visdrone",
    "yolov8-visdrone": "visdrone",
    "yolo8-visdrone":  "visdrone",
    "drone":           "visdrone",
    "traffic":         "visdrone",
    "trafficcam":      "visdrone",
}

_LANE_ALIASES = {
    "yolop":       "yolopv2",
    "yolopv2":     "yolopv2",
}


class DetectorRegistry:
    def __init__(self):
        self._vehicle_cache: dict[str, object] = {}
        self._lane_cache:    dict[str, object] = {}

    # ── Name resolution ──────────────────────────────────────────────────
    @staticmethod
    def resolve_vehicle(name: str | None) -> str:
        if not name:
            return DEFAULT_VEHICLE
        return _VEHICLE_ALIASES.get(name.strip().lower(), DEFAULT_VEHICLE)

    @staticmethod
    def resolve_lane(name: str | None) -> str:
        if not name:
            return DEFAULT_LANE
        return _LANE_ALIASES.get(name.strip().lower(), DEFAULT_LANE)

    # ── Loaders (cached) ─────────────────────────────────────────────────
    def get_vehicle(self, name: str | None):
        key = self.resolve_vehicle(name)
        if key in self._vehicle_cache:
            return self._vehicle_cache[key]
        logger.info(f"[Registry] First request for vehicle model '{key}'; loading...")
        if key == "yolo11":
            from .vehicle import VehicleDetector
            self._vehicle_cache[key] = VehicleDetector(confidence_threshold=0.45)
        elif key == "fasterrcnn":
            from .vehicle_fasterrcnn import VehicleDetectorFasterRCNN
            self._vehicle_cache[key] = VehicleDetectorFasterRCNN(confidence_threshold=0.50)
        elif key == "visdrone":
            from .vehicle_visdrone import VehicleDetectorVisDrone
            self._vehicle_cache[key] = VehicleDetectorVisDrone(confidence_threshold=0.30)
        else:  # pragma: no cover — resolve_vehicle prevents this
            raise ValueError(f"Unknown vehicle model: {name}")
        return self._vehicle_cache[key]

    def get_lane(self, name: str | None):
        key = self.resolve_lane(name)
        if key in self._lane_cache:
            return self._lane_cache[key]
        logger.info(f"[Registry] First request for lane model '{key}'; loading...")
        if key == "yolopv2":
            from .lane import LaneDetector
            self._lane_cache[key] = LaneDetector()
        else:  # pragma: no cover
            raise ValueError(f"Unknown lane model: {name}")
        return self._lane_cache[key]

    def preload_defaults(self) -> None:
        """Eagerly load the default pair so the first request is fast."""
        self.get_vehicle(DEFAULT_VEHICLE)
        self.get_lane(DEFAULT_LANE)
