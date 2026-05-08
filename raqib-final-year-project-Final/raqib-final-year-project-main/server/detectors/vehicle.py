"""
RAQIB - Vehicle Detector using YOLO11
--------------------------------------
Uses YOLO11 (Ultralytics, 2024) for object detection. Pre-trained on COCO 2017
with 80 classes; no fine-tuning required because every traffic-relevant class
we care about (car, truck, bus, motorcycle, bicycle, person) is already
represented in COCO.

YOLO11 is the default detector because:

  * It is the model named in the project's interim report, so keeping it as
    the headline choice avoids unnecessary reviewer confusion.
  * Ultralytics publishes official COCO weights as a single .pt file, so the
    system can boot with zero dataset dependency and zero conversion work.
  * The "small" (yolo11s) variant fits comfortably in 2 GB VRAM at 640×640,
    which keeps per-frame latency well inside the 200 ms NFR1 budget on a
    single consumer GPU and leaves headroom for the lane segmentation pass.

Reference:
    Ultralytics (2024). YOLO11.  https://github.com/ultralytics/ultralytics
    Redmon, J. et al. (2016). You Only Look Once: Unified, Real-Time Object
    Detection. CVPR 2016.  https://arxiv.org/abs/1506.02640
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

from .tailgating import TailgatingAnalyzer

logger = logging.getLogger(__name__)

# COCO class IDs for vehicle-related classes we care about.
# (These IDs are fixed in the COCO 2017 annotation schema and are the same
# across all YOLO releases trained on COCO.)
VEHICLE_CLASS_IDS = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    1:  "bicycle",
    0:  "person",       # Pedestrians are relevant for safety
}

# Colour map per class for visualisation (BGR).
CLASS_COLOURS = {
    "car":        (0, 255, 0),       # Green
    "motorcycle": (255, 140, 0),     # Orange
    "bus":        (0, 0, 255),       # Red
    "truck":      (0, 128, 255),     # Orange-red
    "bicycle":    (255, 255, 0),     # Cyan
    "person":     (255, 0, 255),     # Magenta
}


class VehicleDetector:
    """
    Vehicle and pedestrian detection using YOLO11 pre-trained on COCO 2017.

    Ultralytics handles weight download, pre-processing (letterbox + scale),
    and post-processing (NMS, coordinate remapping) internally, so this
    class is a thin adapter that filters to traffic-relevant classes and
    packages results into the JSON-friendly dicts expected by main.py.
    """

    # "s" (small) gives the best accuracy/latency trade for server-side
    # inference on a single RTX 4070 Ti; swap to "n" for CPU-only, or "m"/"l"
    # if accuracy becomes the bottleneck and latency budget allows.
    DEFAULT_WEIGHTS = "yolo11s.pt"

    def __init__(
        self,
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
        weights: str = DEFAULT_WEIGHTS,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[VehicleDetector] Loading YOLO11 ({weights}) on device: {self.device}")
        # Ultralytics downloads the weights on first use and caches them in the
        # user's Ultralytics cache directory.
        self.model = YOLO(weights)
        # Dry-run once so the first real frame doesn't pay the JIT cost.
        _ = self.model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            device=self.device, verbose=False,
        )
        logger.info("[VehicleDetector] YOLO11 loaded successfully.")

        self._tailgating = TailgatingAnalyzer()

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Run YOLO11 inference on a BGR frame (OpenCV convention).

        Args:
            frame_bgr: numpy array (H, W, 3) in BGR.

        Returns:
            List of detection dicts:
            {
                "label":      str,           # e.g. "car"
                "confidence": float,         # 0.0–1.0
                "box":        [x1,y1,x2,y2], # pixel coords
                "class_id":   int,
                "colour":     (B,G,R),
            }
        """
        # Ultralytics accepts BGR numpy arrays directly (it converts internally).
        results = self.model.predict(
            frame_bgr,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        boxes = res.boxes.xyxy.cpu().numpy()      # (N, 4)
        scores = res.boxes.conf.cpu().numpy()     # (N,)
        class_ids = res.boxes.cls.cpu().numpy().astype(int)  # (N,)

        detections: list[dict] = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids):
            if cid not in VEHICLE_CLASS_IDS:
                continue
            label = VEHICLE_CLASS_IDS[cid]
            detections.append({
                "label":      label,
                "confidence": round(float(score), 3),
                "box":        [int(round(x1)), int(round(y1)),
                               int(round(x2)), int(round(y2))],
                "class_id":   int(cid),
                "colour":     CLASS_COLOURS.get(label, (0, 255, 0)),
            })
        return detections

    def analyse_tailgating(self, detections, frame_h, frame_w, **kwargs):
        return self._tailgating.analyse(detections, frame_h, frame_w, **kwargs)
