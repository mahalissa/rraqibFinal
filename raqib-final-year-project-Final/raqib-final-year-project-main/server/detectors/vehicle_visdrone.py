"""
RAQIB - Vehicle Detector for traffic / aerial camera footage (VisDrone)
------------------------------------------------------------------------
The other two vehicle detectors in RAQIB (YOLO11 and Faster R-CNN) are
both COCO-pretrained, and COCO is dominated by street-level / dashcam
viewpoints. On pole-mounted CCTV or other oblique surveillance views
their accuracy degrades sharply (see the TfL benchmark in Ch 6), which
matters because traffic-camera deployment is RAQIB's primary use case.

VisDrone (Zhu et al., 2018) ships ~10k images and ~261k annotated frames
captured from drone-mounted cameras at varying altitudes and oblique angles.
Drone footage is not identical to a fixed traffic camera, but the geometry
is much closer than dashcam — vehicles appear top-down or oblique, occupy
small regions of the frame, and the camera is non-ego — so a VisDrone-trained
detector generalises noticeably better to fixed CCTV than COCO-trained ones.

Weights: ``mshamrai/yolov8s-visdrone`` on HuggingFace (~22 MB, MIT-licensed,
mAP@0.5 ≈ 0.41 on VisDrone val). The .pt file is a standard Ultralytics
YOLOv8 checkpoint, so this wrapper is a near-drop-in clone of VehicleDetector
with two differences:

  1. Weights download via ``huggingface_hub`` rather than Ultralytics' built-in
     download (Ultralytics only knows the URL of its own published assets).
  2. VisDrone uses a 10-class schema, not COCO's 80-class. Class IDs are
     remapped to RAQIB's canonical labels via ``VISDRONE_CLASS_MAP`` below;
     classes that do not correspond to a RAQIB-relevant object (notably
     "tricycle" and "awning-tricycle") are folded into the closest category
     ("motorcycle"), so downstream behaviour analysis sees the same labels
     regardless of which detector produced the frame.

Reference:
    Zhu, P. et al. (2018). VisDrone-DET2018: Vision Meets Drones — A
    Challenge.  https://github.com/VisDrone/VisDrone-Dataset
    mshamrai (2023). yolov8s-visdrone weights.
    https://huggingface.co/mshamrai/yolov8s-visdrone
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .vehicle import CLASS_COLOURS
from .tailgating import TailgatingAnalyzer

logger = logging.getLogger(__name__)

# VisDrone class index → canonical RAQIB label. The order matches the
# data.yaml shipped with mshamrai/yolov8s-visdrone:
#   0 pedestrian, 1 people, 2 bicycle, 3 car, 4 van, 5 truck,
#   6 tricycle, 7 awning-tricycle, 8 bus, 9 motor
VISDRONE_CLASS_MAP = {
    0: "person",
    1: "person",        # "people" in VisDrone = group of pedestrians; collapse to "person".
    2: "bicycle",
    3: "car",
    4: "car",           # "van" — RAQIB has no van class; treat as a car for safety analysis.
    5: "truck",
    6: "motorcycle",    # "tricycle" — closest RAQIB-known label.
    7: "motorcycle",    # "awning-tricycle".
    8: "bus",
    9: "motorcycle",    # "motor" = motorcycle in VisDrone shorthand.
}

_HF_REPO  = "mshamrai/yolov8s-visdrone"
_HF_FILE  = "best.pt"


class VehicleDetectorVisDrone:
    """Vehicle + pedestrian detection using YOLOv8s pre-trained on VisDrone."""

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            f"[VehicleDetectorVisDrone] Loading YOLOv8s-VisDrone "
            f"({_HF_REPO}/{_HF_FILE}) on device: {self.device}"
        )
        weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        self.model = YOLO(weights_path)

        # Warm-up so the first real frame doesn't pay the kernel-compile cost.
        _ = self.model.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            device=self.device, verbose=False,
        )
        logger.info("[VehicleDetectorVisDrone] VisDrone YOLOv8s loaded successfully.")

        self._tailgating = TailgatingAnalyzer()

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
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

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        class_ids = res.boxes.cls.cpu().numpy().astype(int)

        detections: list[dict] = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, class_ids):
            label = VISDRONE_CLASS_MAP.get(int(cid))
            if label is None:
                continue
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
