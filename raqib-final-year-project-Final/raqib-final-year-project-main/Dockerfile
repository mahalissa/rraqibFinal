# ── RAQIB — Real-time Traffic Safety Monitoring ─────────────────────────
#
# Base: PyTorch official image with CUDA runtime. We then explicitly upgrade
# torch/torchvision/torchaudio to cu128 wheels for RTX 50-series support.
# Python 3.11 is already installed; we layer FastAPI, Ultralytics
# (YOLO11 + YOLOv8s-VisDrone) and the app code.
#
# All vehicle and lane model weights are pre-downloaded during build so the
# resulting image is self-contained — no network access needed on the
# target machine other than to serve the WebSocket / REST traffic.
#
# Build:  docker compose build
# Run:    docker compose up
# ────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/root/.cache/torch \
    RAQIB_REQUIRE_CUDA=1 \
    YOLO_CONFIG_DIR=/root/.config/Ultralytics

# OpenCV needs libGL + libglib at runtime even with the headless wheel
# (cv2.imdecode path), and curl is used by the YOLOPv2 weight downloader.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so source edits don't bust this layer.
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/server/requirements.txt

# RTX 50-series GPUs require newer CUDA-compatible PyTorch kernels than the
# base image's torch 2.5/cu124 provides. Reinstall torch/vision/audio from the
# official cu128 index so kernels are available on Blackwell/mobile Blackwell.
RUN pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

COPY server/ /app/server/
COPY client/ /app/client/

WORKDIR /app/server

# ── Pre-fetch model weights so the image is self-contained ──────────────
# Each RUN is its own layer so that a failed stage can resume without
# re-downloading the earlier ones.
RUN python - <<'PY'
# YOLO11s — Ultralytics auto-download to CWD = /app/server.
from ultralytics import YOLO
YOLO("yolo11s.pt")
PY

RUN python - <<'PY'
# Faster R-CNN COCO weights (torchvision's published hash periodically
# diverges from the live file, so we skip the hash check — same workaround
# as the runtime detector).
import os
import time
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as W

url = W.COCO_V1.url
cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
cache_dir.mkdir(parents=True, exist_ok=True)
cached_file = cache_dir / url.rsplit("/", 1)[-1]

last_error = None
strict_prefetch = os.getenv("RAQIB_STRICT_PREFETCH", "0") == "1"
for attempt in range(1, 6):
    try:
        torch.hub.load_state_dict_from_url(url, check_hash=False, progress=False)
        print(f"Faster R-CNN weights ready on attempt {attempt}.")
        break
    except Exception as exc:
        last_error = exc
        if cached_file.exists():
            cached_file.unlink()
        if attempt == 5:
            if strict_prefetch:
                raise RuntimeError(
                    f"Failed to download valid Faster R-CNN weights after {attempt} attempts"
                ) from last_error
            print(
                "WARNING: Could not prefetch Faster R-CNN weights during image build; "
                "continuing (runtime will retry on first detector use)."
            )
            break
        wait_s = attempt * 2
        print(f"Attempt {attempt} failed ({exc!r}); retrying in {wait_s}s...")
        time.sleep(wait_s)
PY

RUN python - <<'PY'
# YOLOPv2 TorchScript (curl-based downloader inside LaneDetector).
from detectors.lane import _download_weights
_download_weights()
PY

RUN python - <<'PY'
# YOLOv8s-VisDrone weights from HuggingFace.
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="mshamrai/yolov8s-visdrone", filename="best.pt")
PY

EXPOSE 8000

# uvicorn with no --reload: production container, restart via compose.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
