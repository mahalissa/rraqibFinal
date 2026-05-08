"""
RAQIB - Violation clipper
==========================
Records short MP4 clips (with bounding-box + headway overlay annotations
baked in) whenever the detection pipeline flags an unsafe behaviour, plus
a JSON metadata file alongside.

Architecture
------------
Each WebSocket session owns one ``ClipperSession``. On every processed
frame the session's ``push()`` method is called with the raw BGR frame,
the detection result dict, and a timestamp.

The session keeps a rolling pre-buffer of the last few seconds of frames
(default 3 s). When a frame arrives carrying ``unsafe`` events:

  • If we're not already recording: the pre-buffer becomes the head of
    the new clip and we set ``recording_until = t + post_seconds``.
  • Subsequent frames are appended.
  • Each new unsafe frame extends ``recording_until`` so a sustained
    incident produces one continuous clip rather than dozens.

When the post-window expires the session writes:

  • ``server/violations/clips/<id>.mp4``  — annotated H.264-equivalent
    (OpenCV ``mp4v``) clip, one frame per ``push()`` call.
  • ``server/violations/clips/<id>.jpg``  — annotated single-frame
    "snapshot" of the peak moment (frame with the most flags).
  • ``server/violations/clips/<id>.json`` — metadata.

A 30-second cool-down per camera-source prevents a busy junction from
filling the disk with overlapping clips of the same incident.

Listing
-------
``list_clips()`` returns the metadata JSONs sorted newest-first; the
HTTP layer serves them via ``GET /violations`` and exposes the clip /
snapshot files via ``GET /violations/file/<filename>``.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLIPS_DIR = Path(__file__).resolve().parent / "violation_clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)


# ── Frame annotation (matches client-side overlay style) ────────────────
_LABEL_COLOURS = {
    "car":        (0, 255, 0),
    "truck":      (0, 128, 255),
    "bus":        (0, 0, 255),
    "motorcycle": (255, 140, 0),
    "bicycle":    (0, 255, 255),
    "person":     (255, 0, 255),
}
_FLAG_COLOUR = (0, 0, 255)


def annotate(frame: np.ndarray, result: dict) -> np.ndarray:
    """Paint detection boxes + tailgating markers onto a copy of ``frame``."""
    img = frame.copy()
    for det in result.get("vehicles", []) or []:
        x1, y1, x2, y2 = det.get("box", (0, 0, 0, 0))
        col = _LABEL_COLOURS.get(det.get("label"), (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        label = f"{det.get('label', '?')} {det.get('confidence', 0):.2f}"
        cv2.putText(img, label, (x1, max(8, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    for flag in result.get("unsafe", []) or []:
        if flag.get("behaviour") != "tailgating":
            continue
        x1, y1, x2, y2 = flag.get("box", (0, 0, 0, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), _FLAG_COLOUR, 2)
        thw = flag.get("time_headway_s")
        if thw is not None:
            cv2.putText(img, f"thw={thw}s", (x1, max(8, y1 - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, _FLAG_COLOUR, 1, cv2.LINE_AA)
        lead = flag.get("lead_box")
        if lead:
            lx1, ly1, lx2, ly2 = lead
            cv2.line(img,
                     ((x1 + x2) // 2, (y1 + y2) // 2),
                     ((lx1 + lx2) // 2, (ly1 + ly2) // 2),
                     _FLAG_COLOUR, 1)
    return img


# ── Per-session clipper ─────────────────────────────────────────────────
@dataclass
class ClipperSession:
    pre_seconds:        float = 3.0
    post_seconds:       float = 3.0
    cooldown_seconds:   float = 30.0
    fps_target:         float = 5.0       # used when writing the MP4
    camera_source:      str | None = None
    camera_name:        str | None = None
    camera_id:          str | None = None

    _buffer:            collections.deque = field(default_factory=lambda: collections.deque())
    _recording_frames:  list[tuple[float, np.ndarray, dict]] = field(default_factory=list)
    _recording_until:   float | None = None
    _last_clip_at:      float = 0.0
    _manual_request:    bool = False

    def update_camera(self, source: str | None, name: str | None, cam_id: str | None) -> None:
        # If the user switches camera mid-session, drop any in-flight
        # recording state; the new context isn't related to the old.
        if source != self.camera_source or cam_id != self.camera_id:
            self._buffer.clear()
            self._recording_frames.clear()
            self._recording_until = None
            self._manual_request = False
        self.camera_source = source
        self.camera_name = name
        self.camera_id = cam_id

    def request_manual(self, t: float | None = None) -> None:
        """User clicked 'Clip now'. Force-start (or extend) a recording.

        Bypasses the cooldown and the requirement of an automatic
        violation flag. If a recording is already in progress (because
        of an automatic trigger or a previous manual request) the
        post-window is extended so the user gets at least the full
        ``post_seconds`` of footage *after* the click.
        """
        if t is None:
            t = time.time()
        self._manual_request = True
        if self._recording_until is None:
            # Snapshot the pre-buffer as the head of a new clip. The
            # next push() call appends the current frame and the
            # recording proceeds through the normal post-window flow.
            self._recording_frames = list(self._buffer)
            self._recording_until = t + self.post_seconds
        else:
            self._recording_until = max(
                self._recording_until, t + self.post_seconds,
            )

    def push(self, frame_bgr: np.ndarray, result: dict, t: float | None = None) -> str | None:
        """Returns the clip ID iff a clip was just persisted, else None."""
        if t is None:
            t = time.time()
        unsafe = [u for u in (result.get("unsafe") or [])
                  if u.get("behaviour") == "tailgating"]

        # Maintain pre-buffer window (capped by time, not count).
        self._buffer.append((t, frame_bgr, result))
        cutoff = t - self.pre_seconds
        while self._buffer and self._buffer[0][0] < cutoff:
            if self._recording_until is None:
                self._buffer.popleft()
            else:
                break  # while recording, the buffer is the recording head

        if unsafe:
            if self._recording_until is None and (t - self._last_clip_at) > self.cooldown_seconds:
                # Begin a new recording — start with the contents of the
                # pre-buffer, including the current frame.
                self._recording_frames = list(self._buffer)
                self._recording_until = t + self.post_seconds
            elif self._recording_until is not None:
                # Already recording — append the new frame and extend.
                self._recording_frames.append((t, frame_bgr, result))
                self._recording_until = max(self._recording_until,
                                            t + self.post_seconds)
        else:
            if self._recording_until is not None:
                self._recording_frames.append((t, frame_bgr, result))

        if self._recording_until is not None and t >= self._recording_until:
            return self._flush()
        return None

    def _flush(self) -> str | None:
        frames = list(self._recording_frames)
        manual = self._manual_request
        self._recording_frames.clear()
        self._recording_until = None
        self._manual_request = False
        self._last_clip_at = time.time()
        self._buffer.clear()
        if len(frames) < 2:
            return None

        clip_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]

        try:
            return _write_clip(clip_id, frames, self.fps_target,
                               source=self.camera_source,
                               name=self.camera_name,
                               cam_id=self.camera_id,
                               manual_request=manual)
        except Exception as exc:  # pragma: no cover — best-effort persistence
            logger.exception(f"[violations] failed to write clip {clip_id}: {exc}")
            return None


def _write_clip(
    clip_id: str,
    frames: list[tuple[float, np.ndarray, dict]],
    fps: float,
    source: str | None,
    name: str | None,
    cam_id: str | None,
    manual_request: bool = False,
) -> str | None:
    annotated_frames: list[np.ndarray] = []
    flag_counts:      list[int] = []
    headways:         list[float] = []
    flagged_frames:   int = 0
    for _, frame, result in frames:
        ann = annotate(frame, result)
        annotated_frames.append(ann)
        flags = [u for u in (result.get("unsafe") or [])
                 if u.get("behaviour") == "tailgating"]
        flag_counts.append(len(flags))
        for f in flags:
            if "time_headway_s" in f:
                try:
                    headways.append(float(f["time_headway_s"]))
                except (TypeError, ValueError):
                    pass
        if flags:
            flagged_frames += 1

    h, w = annotated_frames[0].shape[:2]
    mp4_path = CLIPS_DIR / f"{clip_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, max(1.0, fps), (w, h))
    if not writer.isOpened():
        logger.warning(f"[violations] VideoWriter failed for {mp4_path}; skipping.")
        return None
    for af in annotated_frames:
        writer.write(af)
    writer.release()

    # Snapshot = frame with the most flags (peak of the incident).
    peak_idx = max(range(len(flag_counts)), key=lambda i: flag_counts[i])
    snap_path = CLIPS_DIR / f"{clip_id}.jpg"
    cv2.imwrite(str(snap_path), annotated_frames[peak_idx],
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    duration_s = round(frames[-1][0] - frames[0][0], 2)
    # If a violation flagged any frame in the window, that's the headline
    # behaviour even when the user clicked Clip Now (the manual click only
    # extended an already-running recording). Pure manual captures —
    # zero flagged frames — show up as "manual" so the gallery can tag
    # them differently.
    behaviour = "tailgating" if flagged_frames > 0 else (
        "manual" if manual_request else "tailgating"
    )
    meta: dict[str, Any] = {
        "id":             clip_id,
        "created_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "behaviour":      behaviour,
        "manual_request": manual_request,
        "camera_source":  source,
        "camera_name":    name,
        "camera_id":      cam_id,
        "duration_s":     duration_s,
        "frames":         len(frames),
        "frames_flagged": flagged_frames,
        "min_headway_s":  round(min(headways), 2) if headways else None,
        "mean_headway_s": round(sum(headways) / len(headways), 2) if headways else None,
        "frame_size":     {"width": w, "height": h},
        "video":          f"{clip_id}.mp4",
        "snapshot":       f"{clip_id}.jpg",
    }
    (CLIPS_DIR / f"{clip_id}.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        f"[violations] saved clip {clip_id} "
        f"({behaviour}{' / manual' if manual_request and behaviour == 'tailgating' else ''}, "
        f"{duration_s}s, {flagged_frames}/{len(frames)} frames flagged"
        + (f", min headway {min(headways):.2f}s" if headways else "")
        + ")"
    )
    return clip_id


# ── Listing ─────────────────────────────────────────────────────────────
_LIST_LOCK = threading.Lock()


def list_clips(limit: int = 100) -> list[dict[str, Any]]:
    """Return clip metadata JSONs, newest first."""
    with _LIST_LOCK:
        items: list[dict[str, Any]] = []
        for p in sorted(CLIPS_DIR.glob("*.json"), reverse=True):
            try:
                items.append(json.loads(p.read_text()))
            except Exception:
                continue
            if len(items) >= limit:
                break
    return items


def clip_path(filename: str) -> Path | None:
    """Resolve a filename relative to CLIPS_DIR with traversal protection."""
    candidate = (CLIPS_DIR / filename).resolve()
    try:
        candidate.relative_to(CLIPS_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None
