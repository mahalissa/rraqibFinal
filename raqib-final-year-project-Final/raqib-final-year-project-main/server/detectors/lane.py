"""
RAQIB - Lane & Drivable-Area Detector using YOLOPv2
----------------------------------------------------
Uses YOLOPv2 (Han et al., 2022) — a multi-task perception model trained on
BDD100K that produces three heads in a single forward pass:

    1. Object detection      (ignored here; we use YOLO11 for vehicles)
    2. Drivable-area mask    (binary segmentation of the driveable road)
    3. Lane-line mask        (binary segmentation of painted lane markings)

YOLOPv2 is distributed by its authors as a *traced* TorchScript checkpoint
(`yolopv2.pt`, ~38 MB), so it loads with ``torch.jit.load`` without having
to vendor any of their model-code. This keeps the server dependency set
to ``torch`` only for lane inference.

Using segmentation masks (rather than the row-anchor UFLD formulation used
previously) has two concrete advantages for RAQIB:

    * The **drivable-area mask is a natural ego-region proxy**. Lane
      violation simplifies to "is the vehicle's bottom-centre inside the
      drivable-area mask?" — no interpolation of per-anchor points, no
      left/right ego-boundary identification heuristic.
    * The **lane-line mask** remains painted-marking-aware, giving the
      client a faithful overlay without requiring a separate polyline
      reconstruction step.

Reference:
    Han, C. et al. (2022). YOLOPv2: Better, Faster, Stronger for Panoptic
    Driving Perception. arXiv:2208.11434.
    https://github.com/CAIC-AD/YOLOPv2
"""

from __future__ import annotations

import base64
import logging
import shutil
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Model download ────────────────────────────────────────────────────────────
# Official release asset from the YOLOPv2 authors.
_WEIGHTS_URL = (
    "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
)
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "yolopv2.pt"

# ── Input geometry (per the YOLOPv2 demo.py) ─────────────────────────────────
# The traced model expects a letterboxed RGB tensor at 384 × 640 (H × W),
# normalised to [0, 1] in NCHW order.
INPUT_H, INPUT_W = 384, 640

# Thresholds applied to the sigmoid mask logits.
DRIVABLE_THRESHOLD = 0.5
LANE_THRESHOLD     = 0.5


def _download_weights() -> None:
    """Fetch the official YOLOPv2 TorchScript checkpoint if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        logger.info(f"[LaneDetector] Weights already present at {MODEL_PATH}")
        return

    logger.info(f"[LaneDetector] Downloading YOLOPv2 weights from {_WEIGHTS_URL}")
    curl = shutil.which("curl")
    if curl:
        result = subprocess.run(
            [curl, "-L", "-o", str(MODEL_PATH), _WEIGHTS_URL],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl download failed: {result.stderr}")
    else:
        urllib.request.urlretrieve(_WEIGHTS_URL, str(MODEL_PATH))
    logger.info(
        f"[LaneDetector] YOLOPv2 weights downloaded "
        f"({MODEL_PATH.stat().st_size / 1e6:.1f} MB)"
    )


# ── Letterbox (mirrors ultralytics/YOLOP preprocessing) ──────────────────────
def _letterbox(
    img: np.ndarray, new_shape: tuple[int, int] = (INPUT_H, INPUT_W),
    colour: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize keeping aspect ratio and pad to ``new_shape``.
    Returns the padded image, the scale factor used, and the (pad_w, pad_h).
    """
    h0, w0 = img.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour
    )
    return img, r, (left, top)


class LaneDetector:
    """
    YOLOPv2-based lane-line and drivable-area segmentation.

    The class publishes three things to main.py:

      * A lane-line binary mask in the original frame's coordinate space
        (used by the client for a painted-marking overlay).
      * A drivable-area binary mask in the same coordinate space
        (used by lane-violation detection and overlayed on the client).
      * Per-vehicle lane-violation events produced by checking each
        detection's bottom-centre pixel against the drivable-area mask.
    """

    def __init__(self, device: str | None = None):
        _download_weights()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[LaneDetector] Loading YOLOPv2 TorchScript on device: {self.device}")

        # The published checkpoint is a traced TorchScript module, so we load
        # it without any model-code dependency.
        self.model = torch.jit.load(str(MODEL_PATH), map_location=self.device)
        self.model.eval()

        # Warm up so first real frame doesn't pay the kernel-compile cost.
        dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        logger.info("[LaneDetector] YOLOPv2 ready.")

    # ── Pre/post-processing ──────────────────────────────────────────────
    def _preprocess(
        self, frame_bgr: np.ndarray
    ) -> tuple[torch.Tensor, float, tuple[int, int], tuple[int, int]]:
        orig_h, orig_w = frame_bgr.shape[:2]
        img, scale, (pad_w, pad_h) = _letterbox(frame_bgr)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor, scale, (pad_w, pad_h), (orig_h, orig_w)

    def _unletterbox_mask(
        self,
        mask: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_hw: tuple[int, int],
    ) -> np.ndarray:
        """Undo letterbox padding + scale to recover original-frame-size mask."""
        pad_w, pad_h = pad
        orig_h, orig_w = orig_hw
        # Remove padding
        h, w = mask.shape
        mask = mask[
            int(round(pad_h)): h - int(round(pad_h)) or None,
            int(round(pad_w)): w - int(round(pad_w)) or None,
        ]
        if mask.size == 0:
            return np.zeros(orig_hw, dtype=np.uint8)
        # Resize to original frame
        return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _mask_to_base64_png(mask: np.ndarray) -> str:
        """
        Encode a binary uint8 mask as a base64 RGBA PNG (no data-URI prefix).

        The alpha channel is the mask itself (255 where mask is true, 0
        elsewhere). Encoding with alpha is what lets the client compositing
        step (`source-in`) paint the fill colour *only* over the mask
        pixels — a grayscale PNG would be fully opaque everywhere and would
        tint the whole frame instead of just the painted lanes / road.
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        alpha = mask * 255
        rgba[..., 0] = 255          # B
        rgba[..., 1] = 255          # G
        rgba[..., 2] = 255          # R
        rgba[..., 3] = alpha        # A
        ok, buf = cv2.imencode(".png", rgba)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ── Public API ───────────────────────────────────────────────────────
    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        Run YOLOPv2 on a BGR frame and return serialised lane/drivable data.

        Returns:
            {
                "drivable_mask_png":  <base64 PNG>,  # original frame size
                "lane_mask_png":      <base64 PNG>,  # original frame size
                "num_lanes":          int,
                "frame_size":         {"width": W, "height": H},
                # Cached internally (not JSON):
                "_drivable_mask":     np.ndarray (H, W) uint8 0/1
            }
        """
        tensor, scale, pad, orig_hw = self._preprocess(frame_bgr)

        # YOLOPv2 returns  [pred, anchor_grid], seg, ll
        # We only need seg (drivable area) and ll (lane lines).
        _, seg, ll = self.model(tensor)

        # YOLOPv2 heads differ: `seg` (drivable) is a 2-channel logit map
        # we argmax along the channel dim; `ll` (lane lines) is single-channel
        # and the traced model already applies the sigmoid internally, so we
        # threshold the raw output directly. (Applying sigmoid a second time
        # compresses everything into [0.5, 0.73] and destroys separability.)
        drivable_letterboxed = torch.argmax(seg[0], dim=0).cpu().numpy().astype(np.uint8)
        ll_prob = ll[0, 0].cpu().numpy()
        lane_letterboxed = (ll_prob > LANE_THRESHOLD).astype(np.uint8)

        drivable = self._unletterbox_mask(drivable_letterboxed, scale, pad, orig_hw)
        lanes    = self._unletterbox_mask(lane_letterboxed,     scale, pad, orig_hw)

        # Single connected-components pass feeds both the cleaned-up display
        # mask and the num_lanes count. The raw lane mask is noisy (hundreds
        # of tiny speckles at the horizon); we keep only components above
        # ~0.05% of the frame area — roughly a real painted-marking segment.
        frame_area = orig_hw[0] * orig_hw[1]
        min_area = max(800, int(frame_area * 0.0005))
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(lanes, connectivity=8)
        if num_components > 1:
            keep = np.zeros(num_components, dtype=bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
            lanes_clean = keep[labels].astype(np.uint8)
            num_lanes = int(keep.sum())
        else:
            lanes_clean = lanes
            num_lanes = 0

        # Dilate what remains so the painted markings stay visible after the
        # client scales the PNG up to full canvas resolution (raw lines are
        # only 1–2 px wide and anti-alias to near-invisibility after resize).
        lanes_vis = cv2.dilate(
            lanes_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        return {
            "drivable_mask_png": self._mask_to_base64_png(drivable),
            "lane_mask_png":     self._mask_to_base64_png(lanes_vis),
            "num_lanes":         int(num_lanes),
            "frame_size":        {"width": orig_hw[1], "height": orig_hw[0]},
            # Internal (stripped before JSON serialisation): the drivable
            # mask is kept for spatial gating; the cleaned + dilated lane
            # mask is used for the boundary-crossing test.
            "_drivable_mask":    drivable,
            "_lane_mask":        lanes_vis,
        }

    def detect_lane_violation(
        self,
        lane_data: dict,
        vehicle_detections: list[dict],
        frame_w: int,
    ) -> list[dict]:
        """
        Flag vehicles whose footprint straddles a painted lane line.

        This is a **boundary-crossing** test rather than an in/out-of-mask
        test. On a dashcam the drivable-area mask represents only the ego
        lane, so a naive in/out test would mark every forward or adjacent
        car — which are legitimately outside our lane — as a "violation".
        An actual lane violation is defined by the vehicle crossing a
        painted line: its bottom edge must overlap the (dilated) lane-line
        mask by a meaningful fraction of its own width.

        ``frame_w`` retained for API compatibility; not used.
        """
        lane_mask     = lane_data.get("_lane_mask")
        drivable_mask = lane_data.get("_drivable_mask")
        if lane_mask is None or lane_mask.size == 0:
            return []

        h, w = lane_mask.shape
        violations: list[dict] = []
        # Spatial gate: only consider vehicles plausibly inside (or entering)
        # the ego lane. Cars far to the side of the frame that happen to be
        # near their own lane markings aren't "violations" from our POV.
        ego_x_min = int(w * 0.20)
        ego_x_max = int(w * 0.80)
        ego_y_min = int(h * 0.45)
        for det in vehicle_detections:
            if det["label"] not in ("car", "truck", "bus", "motorcycle"):
                continue
            x1, y1, x2, y2 = det["box"]
            x1i, x2i = int(max(0, x1)), int(min(w, x2))
            cx = (x1i + x2i) // 2
            cy = int(min(h - 1, y2 - 2))
            vbox_w = max(1, x2i - x1i)
            if x2i <= x1i:
                continue
            if not (ego_x_min <= cx <= ego_x_max and cy >= ego_y_min):
                continue
            # Look at a thin horizontal strip along the bottom edge of the
            # bounding box. If lane-line pixels occupy ≥15% of the strip
            # width, the vehicle is straddling a line.
            strip_top = max(0, cy - 6)
            strip_bot = min(h, cy + 2)
            strip = lane_mask[strip_top:strip_bot, x1i:x2i]
            if strip.size == 0:
                continue
            # Per-column: does this column contain any lane-line pixel?
            col_has_line = strip.max(axis=0) > 0
            overlap_frac = float(col_has_line.mean())
            if overlap_frac < 0.15:
                continue
            # Which side of the vehicle is over the line? (Informational.)
            line_cols = np.where(col_has_line)[0]
            if line_cols.size == 0:
                side = "unknown"
            else:
                mid = vbox_w / 2
                left_cols  = (line_cols < mid * 0.6).sum()
                right_cols = (line_cols > mid * 1.4).sum()
                if left_cols > right_cols * 1.5:
                    side = "left"
                elif right_cols > left_cols * 1.5:
                    side = "right"
                else:
                    side = "center"
            # Require the centre of the vehicle to actually be inside the
            # ego drivable area — rules out lane-lines of unrelated distant
            # roads when the dashcam sees forks/junctions.
            if drivable_mask is not None and drivable_mask.size:
                if drivable_mask[cy, cx] == 0:
                    # Allow if any column of the strip lies over drivable —
                    # i.e. the vehicle is partially in our lane.
                    drive_strip = drivable_mask[strip_top:strip_bot, x1i:x2i]
                    if drive_strip.max() == 0:
                        continue
            violations.append({
                "behaviour": f"lane_violation_{side}" if side != "center" else "lane_violation",
                "vehicle":   det["label"],
                "box":       det["box"],
                "overlap":   round(overlap_frac, 2),
            })
        return violations
