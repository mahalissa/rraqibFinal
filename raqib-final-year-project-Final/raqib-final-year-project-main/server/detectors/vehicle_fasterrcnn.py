"""
RAQIB - Vehicle Detector using Faster R-CNN
--------------------------------------------
Faster R-CNN (Ren et al., 2015) is the canonical two-stage detector: a
Region Proposal Network (RPN) shares a convolutional backbone with the
detection head, generating candidate regions that are classified and
refined by a Fast R-CNN head. Compared to single-stage detectors it
trades raw throughput for generally higher localisation accuracy,
especially on small or partially occluded objects — a useful third
datapoint in the RAQIB comparison alongside the single-stage CNN family
(YOLO11) and the transformer family (RT-DETRv2).

This module uses the ``fasterrcnn_resnet50_fpn_v2`` weights bundled with
torchvision (COCO-pretrained, identical 91-class schema to YOLO/COCO),
so no vendoring or external download is required. The same six
traffic-relevant classes are filtered out of the 91-class output.

Reference:
    Ren, S. et al. (2015). Faster R-CNN: Towards Real-Time Object
    Detection with Region Proposal Networks. NeurIPS 2015.
    https://arxiv.org/abs/1506.01497
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from .vehicle import CLASS_COLOURS
from .tailgating import TailgatingAnalyzer

logger = logging.getLogger(__name__)

# torchvision Faster R-CNN uses the COCO 91-class schema (not the 80-class
# schema YOLO uses). Class IDs are therefore different.
#   person=1, bicycle=2, car=3, motorcycle=4, bus=6, truck=8
TORCHVISION_COCO_VEHICLES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


def _load_state_dict_with_retries(url: str, *, max_attempts: int = 5):
    """Load torchvision state dict with retries, purging a corrupt cache file."""
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / url.rsplit("/", 1)[-1]

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return torch.hub.load_state_dict_from_url(
                url,
                progress=True,
                check_hash=False,
            )
        except Exception as exc:  # network / partial download / parse failure
            last_error = exc
            if cached_file.exists():
                cached_file.unlink()
            if attempt == max_attempts:
                break
            wait_s = attempt * 2
            logger.warning(
                "[VehicleDetectorFasterRCNN] Failed to load weights on attempt %s/%s (%r); retrying in %ss",
                attempt,
                max_attempts,
                exc,
                wait_s,
            )
            time.sleep(wait_s)

    raise RuntimeError(
        f"Failed to download valid Faster R-CNN weights after {max_attempts} attempts"
    ) from last_error


class VehicleDetectorFasterRCNN:
    """Vehicle + pedestrian detection using Faster R-CNN ResNet-50-FPN v2."""

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[VehicleDetectorFasterRCNN] Loading Faster R-CNN on device: {self.device}")
        # COCO weights auto-downloaded by torchvision. torchvision's strict
        # hash check periodically diverges from the actual file served by
        # download.pytorch.org (a known issue, see torchvision#8257), so we
        # download the state dict with check_hash=False and load it into an
        # unweighted model — same end result, but robust to a stale digest.
        self._weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self._transform = self._weights.transforms()
        state_dict = _load_state_dict_with_retries(self._weights.url)
        self.model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Warm-up so the first real frame doesn't pay the kernel-compile cost.
        dummy = torch.zeros(3, 640, 640, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _ = self.model([dummy])
        logger.info("[VehicleDetectorFasterRCNN] Faster R-CNN loaded successfully.")

        self._tailgating = TailgatingAnalyzer()

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        # torchvision expects RGB float tensors in [0, 1], shape (C, H, W).
        rgb = frame_bgr[:, :, ::-1].copy()  # BGR → RGB
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        )
        # Weights.transforms() handles the canonical torchvision normalisation.
        tensor = self._transform(tensor)

        outputs = self.model([tensor])[0]
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        labels = outputs["labels"].cpu().numpy().astype(int)

        detections: list[dict] = []
        for (x1, y1, x2, y2), score, cid in zip(boxes, scores, labels):
            if score < self.confidence_threshold:
                continue
            if cid not in TORCHVISION_COCO_VEHICLES:
                continue
            label = TORCHVISION_COCO_VEHICLES[cid]
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
