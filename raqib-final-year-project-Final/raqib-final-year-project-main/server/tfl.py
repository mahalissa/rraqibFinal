"""
RAQIB - TfL JamCams integration
================================
Polls Transport for London's public JamCams API and broadcasts "this
camera just refreshed" events over Pusher so connected clients can swap
to the new clip without polling TfL themselves.

Architecture
------------
TfL exposes ~900 traffic cameras (CCTV / pole-mounted oblique views) at
``https://api.tfl.gov.uk/Place/Type/JamCam`` as a JSON list. Each camera
has a stable ID, lat/lon, a JPG snapshot URL, and a 10-second MP4 clip
that TfL refreshes roughly every 5 minutes. The MP4 / JPG URLs are AWS
S3 paths whose *content* changes but whose *URL* does not — we have to
diff something else to know when a clip was refreshed.

We hash the relevant ``additionalProperties`` for each camera (the
``available`` flag and the ``timestamp`` field that TfL updates on each
refresh) and emit a Pusher ``cameras-update`` event with the pipe-
joined list of IDs whose hash changed since the previous poll. This
matches the schema documented in Pusher's TfL traffic-camera blog post
so our client code can re-use the standard pattern.

Environment
-----------
``PUSHER_APP_ID``, ``PUSHER_KEY``, ``PUSHER_SECRET`` — required to publish.
``PUSHER_CLUSTER``                                   — defaults to "eu".
``TFL_POLL_INTERVAL_S``                              — defaults to 30.
``TFL_APP_KEY`` (optional)                           — TfL Unified API key
                                                       for higher rate limits.

If the Pusher credentials are not configured, polling still runs and
the cached camera list is still served via ``/tfl/cameras``; only the
real-time push is disabled. This lets the integration degrade gracefully
when running locally without a Pusher account.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TFL_JAMCAMS_URL = "https://api.tfl.gov.uk/Place/Type/JamCam"
PUSHER_CHANNEL  = "cameras"
PUSHER_EVENT    = "cameras-update"


# ── In-memory state (one per server process) ─────────────────────────────
class TflState:
    """Holds the most-recent camera list and the per-camera signature map."""

    def __init__(self) -> None:
        self.cameras: list[dict[str, Any]] = []
        self._signatures: dict[str, str] = {}
        self.last_poll_at: float | None = None
        self.last_update_count: int = 0


def _camera_signature(cam: dict[str, Any]) -> str:
    """Stable per-poll digest used to detect when TfL refreshed a camera."""
    parts: list[str] = []
    for prop in cam.get("additionalProperties", []) or []:
        key = prop.get("key", "")
        if key in ("available", "timestamp", "updated", "videoUrl", "imageUrl"):
            parts.append(f"{key}={prop.get('value', '')}")
    parts.append(f"modified={cam.get('modified', '')}")
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:12]


def _flatten(cam: dict[str, Any]) -> dict[str, Any]:
    """Project a TfL camera record into the small shape the client cares about."""
    props = {p.get("key"): p.get("value") for p in cam.get("additionalProperties", []) or []}
    cam_id = cam.get("id", "")
    short_id = cam_id.replace("JamCams_", "") if cam_id else ""
    return {
        "id":         cam_id,
        "short_id":   short_id,
        "name":       cam.get("commonName", ""),
        "lat":        cam.get("lat"),
        "lon":        cam.get("lon"),
        "available":  str(props.get("available", "")).lower() == "true",
        "image_url":  props.get("imageUrl"),
        "video_url":  props.get("videoUrl"),
        "view":       props.get("view") or props.get("cameraView"),
    }


# ── Pusher (lazy, optional) ──────────────────────────────────────────────
_pusher_client = None


def _get_pusher():
    global _pusher_client
    if _pusher_client is not None:
        return _pusher_client
    try:
        import pusher  # type: ignore
    except ImportError:
        logger.warning("[TfL] pusher package not installed; live updates disabled.")
        return None
    app_id  = os.environ.get("PUSHER_APP_ID")
    key     = os.environ.get("PUSHER_KEY")
    secret  = os.environ.get("PUSHER_SECRET")
    cluster = os.environ.get("PUSHER_CLUSTER", "eu")
    if not (app_id and key and secret):
        logger.warning(
            "[TfL] PUSHER_APP_ID / PUSHER_KEY / PUSHER_SECRET not set; "
            "live updates disabled (cached camera list still served)."
        )
        return None
    _pusher_client = pusher.Pusher(
        app_id=app_id, key=key, secret=secret, cluster=cluster, ssl=True,
    )
    logger.info(f"[TfL] Pusher client initialised (cluster={cluster}).")
    return _pusher_client


def public_pusher_config() -> dict[str, Any]:
    """Subset of Pusher config the browser needs (subscriber-only)."""
    return {
        "key":      os.environ.get("PUSHER_KEY", ""),
        "cluster":  os.environ.get("PUSHER_CLUSTER", "eu"),
        "channel":  PUSHER_CHANNEL,
        "event":    PUSHER_EVENT,
        "enabled":  bool(os.environ.get("PUSHER_KEY")),
    }


# ── Polling loop ────────────────────────────────────────────────────────
async def _fetch_cameras(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    params: dict[str, str] = {}
    app_key = os.environ.get("TFL_APP_KEY")
    if app_key:
        params["app_key"] = app_key
    r = await client.get(TFL_JAMCAMS_URL, params=params, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected TfL response type: {type(data).__name__}")
    return data


async def _poll_once(state: TflState, client: httpx.AsyncClient) -> list[str]:
    raw = await _fetch_cameras(client)
    flattened: list[dict[str, Any]] = []
    new_signatures: dict[str, str] = {}
    changed: list[str] = []

    for cam in raw:
        cam_id = cam.get("id")
        if not cam_id:
            continue
        sig = _camera_signature(cam)
        new_signatures[cam_id] = sig
        flattened.append(_flatten(cam))
        prev = state._signatures.get(cam_id)
        # First poll: don't flood Pusher with ~900 IDs. Treat as baseline.
        if prev is not None and prev != sig:
            changed.append(cam_id)

    state.cameras = flattened
    state._signatures = new_signatures
    state.last_poll_at = asyncio.get_event_loop().time()
    state.last_update_count = len(changed)

    if changed:
        client_p = _get_pusher()
        if client_p is not None:
            payload = "|".join(changed)
            try:
                client_p.trigger(PUSHER_CHANNEL, PUSHER_EVENT, payload)
            except Exception as exc:
                logger.error(f"[TfL] Pusher trigger failed: {exc}")
            else:
                logger.info(f"[TfL] Pushed {len(changed)} camera updates.")

    return changed


async def run_polling_loop(state: TflState) -> None:
    """Background task: poll TfL on a fixed cadence until cancelled."""
    interval = float(os.environ.get("TFL_POLL_INTERVAL_S", "30"))
    logger.info(f"[TfL] Starting polling loop (interval={interval:.0f}s).")
    async with httpx.AsyncClient() as client:
        # First poll establishes the signature baseline; no Pusher emits.
        try:
            await _poll_once(state, client)
            logger.info(f"[TfL] Loaded {len(state.cameras)} cameras (baseline).")
        except Exception as exc:
            logger.error(f"[TfL] Baseline poll failed: {exc}")
        while True:
            try:
                await asyncio.sleep(interval)
                await _poll_once(state, client)
            except asyncio.CancelledError:
                logger.info("[TfL] Polling loop cancelled.")
                raise
            except Exception as exc:
                logger.error(f"[TfL] Poll error: {exc}")
