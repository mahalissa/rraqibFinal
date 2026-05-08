"""
RAQIB Server – Real-time Traffic Safety Monitoring
====================================================
FastAPI server implementing a client-server architecture for RAQIB.

Architecture:
  • Client (web browser / mobile browser) streams video frames over WebSocket.
  • Server runs two pre-trained deep learning models:
      1. YOLO11 (Ultralytics, 2024) – vehicle / pedestrian detection
      2. YOLOPv2 (arXiv 2208.11434, 2022) – lane-line + drivable-area
         segmentation (multi-task panoptic driving perception)
  • Server returns JSON annotations; client overlays them on the canvas.

This client-server design was chosen over on-device inference because:
  a. Server-grade GPUs provide 10-50× more compute than mobile CPUs/NPUs.
  b. Two-stage detectors (Faster R-CNN) and the larger YOLO variants
     comfortably exceed mobile NPU memory budgets at runtime.
  c. Supervisor feedback (January 2026) explicitly recommended evaluating a
     server-based architecture that "prioritises accuracy over latency."

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000 in a browser on any device on the same network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# TfL JamCam IDs look like ``00002.00865`` — five digits, dot, five digits.
_TFL_SHORT_ID_RE = re.compile(r"\d{1,8}\.\d{1,8}")


def _load_env_file() -> None:
    """Best-effort .env loader so Pusher / TfL keys are picked up.

    We don't pull in python-dotenv because the format we need is trivial
    (KEY=VALUE per line, '#' comments). Existing environment variables
    take precedence, so Docker Compose's env injection still wins when
    the same key is set both ways.
    """
    for candidate in (Path(__file__).parent.parent / ".env",
                      Path(__file__).parent / ".env"):
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        break


_load_env_file()

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse,
    RedirectResponse, StreamingResponse,
)
from pydantic import BaseModel

from detectors.registry import (
    DetectorRegistry,
    VEHICLE_MODELS, LANE_MODELS,
    DEFAULT_VEHICLE, DEFAULT_LANE,
)
from tfl import (
    TflState, run_polling_loop, public_pusher_config,
)
import auth
import violations

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("raqib.server")

# ── Global detector registry (lazy-load, cached per process) ──────────────────
registry: DetectorRegistry = DetectorRegistry()

# ── TfL camera state (shared by polling task + REST handlers) ─────────────────
tfl_state: TflState = TflState()


def _assert_cuda_runtime_ready() -> None:
    """Fail fast if CUDA is required but torch cannot execute kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required but CUDA is not available in this container. "
            "Ensure Docker has NVIDIA GPU access and use a CUDA-compatible "
            "PyTorch build (cu128 for RTX 50-series)."
        )

    try:
        dev_idx = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_idx)
        cc_major, cc_minor = torch.cuda.get_device_capability(dev_idx)
        # Force a real kernel launch (not just capability query) so
        # 'no kernel image is available' fails here during startup.
        x = torch.randn((2048,), device="cuda", dtype=torch.float32)
        _ = (x * 1.0001).sum().item()
        torch.cuda.synchronize()
        logger.info(
            "CUDA ready: device=%s capability=%s.%s torch=%s",
            dev_name,
            cc_major,
            cc_minor,
            torch.__version__,
        )
    except Exception as exc:
        raise RuntimeError(
            "CUDA kernel launch failed during startup. This usually means "
            "the installed torch/CUDA build does not support this GPU "
            "architecture yet (common on RTX 50-series with older cu124 "
            "stacks) or host driver is too old."
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the default detector pair so the first request is fast."""
    require_cuda = os.getenv("RAQIB_REQUIRE_CUDA", "1") == "1"
    if require_cuda:
        _assert_cuda_runtime_ready()
    logger.info("Pre-loading default detection models (YOLO11 + YOLOPv2)…")
    t0 = time.perf_counter()
    registry.preload_defaults()
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Default models ready in {elapsed:.1f}s. "
        f"Other models load lazily on first request."
    )
    # Pre-load VisDrone so the first TfL camera click doesn't block the
    # event loop while the HuggingFace weights download + load.
    try:
        t1 = time.perf_counter()
        registry.get_vehicle("visdrone")
        logger.info(f"VisDrone pre-loaded in {time.perf_counter() - t1:.1f}s.")
    except Exception as exc:
        logger.warning(f"VisDrone pre-load skipped (will load lazily): {exc}")

    tfl_task = asyncio.create_task(run_polling_loop(tfl_state))
    yield
    tfl_task.cancel()
    try:
        await tfl_task
    except asyncio.CancelledError:
        pass
    logger.info("Server shutting down.")


# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="RAQIB – Road Safety Monitoring API",
    description=(
        "Real-time traffic safety detection using YOLO11 (vehicle detection) "
        "and YOLOPv2 (lane-line + drivable-area segmentation)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth gate ─────────────────────────────────────────────────────────────────
# Protect every HTTP route except the open ones (login itself, the health
# probe used by Docker, and the static favicon if any). WebSocket auth is
# handled at handshake time inside the websocket endpoint because Starlette
# middlewares don't fire for WebSocket lifespan.
_OPEN_PATHS = {"/login", "/logout", "/health", "/favicon.ico"}


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    path = request.url.path
    if path in _OPEN_PATHS:
        return await call_next(request)
    cookie = request.cookies.get(auth.COOKIE_NAME)
    if auth.request_is_authed(cookie):
        return await call_next(request)
    # Browser navigations (HTML expected) → bounce to /login.
    accept = request.headers.get("accept", "")
    if "text/html" in accept and request.method == "GET":
        return RedirectResponse(url="/login", status_code=303)
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


# ── Helper: decode base64 frame ───────────────────────────────────────────────
def _decode_frame(b64_data: str) -> np.ndarray | None:
    """Decode a base64-encoded JPEG/PNG string into a BGR numpy array."""
    try:
        # Strip the data-URI prefix if present (e.g. "data:image/jpeg;base64,…")
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as exc:
        logger.error(f"Frame decode error: {exc}")
        return None


CAMERA_PERSPECTIVES = ("traffic", "dashcam")
DEFAULT_PERSPECTIVE = "traffic"


def _resolve_perspective(name: str | None) -> str:
    if not name:
        return DEFAULT_PERSPECTIVE
    n = name.strip().lower()
    if n in ("dashcam", "dash", "vehicle", "ego", "front"):
        return "dashcam"
    return "traffic"


def _filter_caption_bands(
    detections: list[dict[str, Any]],
    frame_h: int,
    top_frac: float = 0.08,
    bot_frac: float = 0.93,
) -> list[dict[str, Any]]:
    """Drop detections whose bbox sits entirely inside the top/bottom strips
    that TfL JamCams use for the timestamp and the camera-name caption.

    VisDrone's small-object head reads those strips as a row of tiny cars
    (the timestamp digits in particular look like a queue of vehicles) and
    the spurious detections then poison the pairwise-tailgating heuristic.
    Filtering geometrically is cheap, conditional on a request flag, and
    preserves any genuine vehicle whose bbox merely *overlaps* the strip
    (kept whenever any pixel of the bbox lies in the road region).
    """
    top_y = frame_h * top_frac
    bot_y = frame_h * bot_frac
    kept: list[dict[str, Any]] = []
    for det in detections:
        _, y1, _, y2 = det["box"]
        if y2 <= top_y or y1 >= bot_y:
            continue  # box is wholly inside a caption strip
        kept.append(det)
    return kept


def _run_pipeline(
    frame: np.ndarray,
    vehicle_model: str | None = None,
    lane_model: str | None = None,
    camera_perspective: str | None = None,
    camera_source: str | None = None,
) -> dict[str, Any]:
    """Run the selected vehicle + lane detectors and behaviour analysis."""
    h, w = frame.shape[:2]
    v_name = registry.resolve_vehicle(vehicle_model)
    l_name = registry.resolve_lane(lane_model)
    p_name = _resolve_perspective(camera_perspective)
    src    = (camera_source or "").strip().lower()
    vehicle = registry.get_vehicle(v_name)
    lane    = registry.get_lane(l_name)

    t0 = time.perf_counter()
    vehicles  = vehicle.detect(frame)
    lane_data = lane.detect(frame)

    # TfL JamCams overlay timestamp + camera-name caption bars on every
    # frame; small-object detectors (notably VisDrone) misread the digits
    # as a row of tiny cars. Suppress before tailgating runs.
    if src == "tfl":
        vehicles = _filter_caption_bands(vehicles, h)

    tailgating = vehicle.analyse_tailgating(vehicles, h, w, perspective=p_name)
    lane_viol  = lane.detect_lane_violation(lane_data, vehicles, w)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Strip internal numpy fields (not JSON-serialisable) before returning.
    lane_payload = {k: v for k, v in lane_data.items() if not k.startswith("_")}

    return {
        "vehicles":           vehicles,
        "lanes":              lane_payload,
        "unsafe":             tailgating + lane_viol,
        "latency_ms":         elapsed_ms,
        "frame_size":         {"width": w, "height": h},
        "vehicle_model":      v_name,
        "lane_model":         l_name,
        "camera_perspective": p_name,
        "camera_source":      src or None,
    }


# ── REST endpoint ─────────────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    image: str                                # base64-encoded JPEG
    vehicle_model: str | None = None          # "yolo11" | "fasterrcnn" | "visdrone"
    lane_model:    str | None = None          # "yolopv2"
    camera_perspective: str | None = None     # "traffic" (default) | "dashcam"
    camera_source: str | None = None          # "tfl" enables caption-band suppression


@app.post("/detect", summary="Detect vehicles and lanes in a single frame")
async def detect(req: DetectRequest):
    """
    POST a base64-encoded JPEG image and receive detection results as JSON.
    Suitable for single-frame analysis or polling-based clients.

    Optionally select which detectors to use via ``vehicle_model`` and
    ``lane_model``; defaults are YOLO11 + YOLOPv2.
    """
    frame = _decode_frame(req.image)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data"})

    # Run blocking ML inference in a thread so the event loop stays free
    # for heartbeats / other connections while a model is loading or running.
    result = await asyncio.to_thread(
        _run_pipeline,
        frame,
        vehicle_model=req.vehicle_model,
        lane_model=req.lane_model,
        camera_perspective=req.camera_perspective,
        camera_source=req.camera_source,
    )
    # Remove internal colour tuples (not JSON-serialisable)
    for v in result["vehicles"]:
        v.pop("colour", None)
    return result


# ── Auth endpoints ────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page():
    return HTMLResponse(
        content=open("../client/login.html", encoding="utf-8").read(),
    )


@app.post("/login", summary="Exchange admin credentials for a session cookie")
async def login(req: LoginRequest):
    if not auth.auth_enabled():
        return JSONResponse(
            status_code=503,
            content={
                "error": "Auth not configured.",
                "hint": "Run `python server/scripts/set_admin_password.py` "
                        "and restart the server.",
            },
        )
    if not auth.check_credentials(req.username.strip(), req.password):
        # Sleep briefly to make brute-force less attractive.
        await asyncio_sleep_short()
        return JSONResponse(status_code=401, content={"error": "Invalid credentials"})
    token = auth.make_token(req.username.strip())
    resp = JSONResponse(content={"ok": True, "user": req.username.strip()})
    resp.set_cookie(
        auth.COOKIE_NAME,
        token,
        max_age=24 * 3600,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return resp


@app.post("/logout", summary="Clear the session cookie")
async def logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(auth.COOKIE_NAME, path="/")
    return resp


@app.get("/auth/status", summary="Who am I, and is auth enabled?")
async def auth_status(req: Request):
    cookie = req.cookies.get(auth.COOKIE_NAME)
    user = auth.verify_token(cookie) if cookie else None
    return {
        "auth_enabled": auth.auth_enabled(),
        "user": user,
    }


async def asyncio_sleep_short():
    await asyncio.sleep(0.5)


# ── Detector / model endpoints ────────────────────────────────────────────────
@app.get("/models", summary="List available vehicle and lane detector models")
async def models():
    return {
        "vehicle":     {"choices": list(VEHICLE_MODELS), "default": DEFAULT_VEHICLE},
        "lane":        {"choices": list(LANE_MODELS),    "default": DEFAULT_LANE},
        "perspective": {"choices": list(CAMERA_PERSPECTIVES), "default": DEFAULT_PERSPECTIVE},
    }


# ── TfL JamCams + Pusher integration ──────────────────────────────────────────
@app.get("/tfl/cameras", summary="Cached list of TfL traffic cameras")
async def tfl_cameras():
    """
    The list is refreshed every ~30s by a background polling task; this
    handler just returns the cached snapshot. ``last_poll_at`` is server
    monotonic time (seconds since process start) so the client can tell
    whether the snapshot is fresh.
    """
    return {
        "count":              len(tfl_state.cameras),
        "last_poll_at":       tfl_state.last_poll_at,
        "last_update_count":  tfl_state.last_update_count,
        "cameras":            tfl_state.cameras,
    }


@app.get("/tfl/config", summary="Public Pusher config (subscriber-only) for the client")
async def tfl_config():
    """
    Returns only the Pusher *key* and *cluster* — the secret stays on the
    server. Clients use this to subscribe to the ``cameras`` channel and
    receive ``cameras-update`` events as TfL refreshes camera footage.
    """
    return public_pusher_config()


# ── Violations browse / serve ─────────────────────────────────────────────────
@app.get("/violations", summary="List saved violation clips, newest first")
async def list_violations():
    return {"clips": violations.list_clips()}


@app.get("/violations/file/{filename}", summary="Stream a saved clip / snapshot")
async def get_violation_file(filename: str):
    p = violations.clip_path(filename)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    media_type = "video/mp4" if filename.lower().endswith(".mp4") else (
        "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else
        "application/json" if filename.lower().endswith(".json") else
        "application/octet-stream"
    )
    return FileResponse(p, media_type=media_type)


@app.get("/tfl/proxy/{short_id}.mp4", summary="Same-origin proxy for TfL JamCams MP4 clips")
async def tfl_proxy(short_id: str):
    """
    Stream a TfL JamCams MP4 clip back to the browser from the same origin
    as the page. This is the only way the client can draw the video onto
    a canvas and still call ``toDataURL`` afterwards — TfL's S3 bucket
    does not send ``Access-Control-Allow-Origin``, so a direct cross-
    origin video taints the canvas and the browser blocks pixel readback.

    ``short_id`` is the TfL JamCam ID without the ``JamCams_`` prefix
    (e.g. ``00002.00865``). We restrict it to digits and a single dot to
    rule out path traversal / SSRF before constructing the upstream URL.
    """
    # Tight whitelist: TfL IDs look like ``NNNNN.NNNNN``. Anything else
    # is a malformed request and should not reach S3.
    if not _TFL_SHORT_ID_RE.fullmatch(short_id):
        raise HTTPException(status_code=400, detail="Invalid TfL camera id")

    upstream = f"https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/{short_id}.mp4"

    import httpx
    client = httpx.AsyncClient(timeout=30.0)
    try:
        upstream_resp = await client.send(
            client.build_request("GET", upstream),
            stream=True,
        )
    except Exception as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"TfL upstream error: {exc}")

    if upstream_resp.status_code != 200:
        await upstream_resp.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=upstream_resp.status_code,
            detail="TfL upstream returned non-200",
        )

    async def _stream():
        try:
            async for chunk in upstream_resp.aiter_bytes(chunk_size=64 * 1024):
                yield chunk
        finally:
            await upstream_resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _stream(),
        media_type=upstream_resp.headers.get("content-type", "video/mp4"),
        # 25-second browser cache: shorter than TfL's ~5-min refresh so a
        # Pusher-driven cache-bust ?_=… still pulls fresh content, long
        # enough that the looping <video> element doesn't re-fetch every
        # restart of the 10-second clip.
        headers={"Cache-Control": "public, max-age=25"},
    )


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Bi-directional WebSocket for real-time video streaming.

    Client sends JSON:  {"image": "<base64 JPEG>", "frame_id": <int>}
    Server replies:     {detection results + "frame_id": <int>}

    Frame rate is client-controlled; the server processes every received frame.
    """
    # Auth at handshake. Starlette doesn't run HTTP middleware for the WS
    # lifespan, so we check the cookie here. Reject before accept() so an
    # unauthorised client gets the standard 403 close instead of an
    # already-open socket that goes silent.
    if auth.auth_enabled():
        cookie = websocket.cookies.get(auth.COOKIE_NAME)
        if not auth.verify_token(cookie):
            await websocket.close(code=4401)
            return

    await websocket.accept()
    client_addr = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket connected from {client_addr}")

    clipper = violations.ClipperSession()

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)

            frame = _decode_frame(payload.get("image", ""))
            if frame is None:
                await websocket.send_text(
                    json.dumps({"error": "Invalid frame", "frame_id": payload.get("frame_id", 0)})
                )
                continue

            result = await asyncio.to_thread(
                _run_pipeline,
                frame,
                vehicle_model=payload.get("vehicle_model"),
                lane_model=payload.get("lane_model"),
                camera_perspective=payload.get("camera_perspective"),
                camera_source=payload.get("camera_source"),
            )
            result["frame_id"] = payload.get("frame_id", 0)

            # Per-session clipper: detects unsafe events and persists
            # an annotated MP4 + snapshot + JSON metadata to disk.
            clipper.update_camera(
                source=payload.get("camera_source"),
                name=payload.get("camera_name"),
                cam_id=payload.get("camera_id"),
            )
            # Manual "Clip now" — bypass cooldown, force a recording
            # window starting from the rolling pre-buffer.
            if payload.get("request_clip"):
                clipper.request_manual()
            saved_id = clipper.push(frame, result)
            if saved_id:
                result["saved_clip_id"] = saved_id
            # Echo the recording state so the client can show "Recording…"
            # feedback in real time rather than just an optimistic guess.
            result["recording"] = clipper._recording_until is not None

            # Colour tuples are not JSON-serialisable – remove before sending
            for v in result["vehicles"]:
                v.pop("colour", None)

            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_addr}")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")


# ── Health / info endpoints ───────────────────────────────────────────────────
@app.get("/health", summary="Server health check")
async def health():
    return {
        "status":  "ok",
        "models":  {
            "vehicle": {
                "choices": list(VEHICLE_MODELS),
                "default": DEFAULT_VEHICLE,
                "loaded":  sorted(registry._vehicle_cache.keys()),
            },
            "lane": {
                "choices": list(LANE_MODELS),
                "default": DEFAULT_LANE,
                "loaded":  sorted(registry._lane_cache.keys()),
            },
        },
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_client():
    """Serve the web client."""
    client_html = open("../client/index.html", encoding="utf-8").read()
    return HTMLResponse(content=client_html)
