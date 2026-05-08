"""
Benchmark all four vehicle detectors on real TfL JamCams footage.

The point: confirm whether the VisDrone-trained detector (which targets
oblique / overhead viewpoints) actually outperforms the COCO-trained
pair (YOLO11 / Faster R-CNN) on pole-mounted CCTV — RAQIB's headline
claim is that VisDrone is the right default for this view, and this
benchmark is what backs it up with measurements on real TfL footage.

Usage (server does NOT need to be running):
    python tfl_benchmark.py [--cameras 5] [--frames-per-cam 3]

The script:
  * Picks a handful of representative TfL JamCams (roads + intersections)
    by name pattern, falls back to the first N available cameras.
  * Downloads each camera's 10-second MP4 to a temp dir.
  * Decodes ``frames_per_cam`` evenly-spaced frames per clip.
  * Runs every vehicle detector on each frame.
  * Saves an annotated mosaic per (camera, frame) showing all four
    detectors' boxes side-by-side, plus a JSON summary and a markdown
    table of detection counts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.registry import DetectorRegistry, VEHICLE_MODELS  # noqa: E402
from detectors.vehicle import CLASS_COLOURS                       # noqa: E402
from tfl import TflState, _poll_once                              # noqa: E402

import httpx                                                      # noqa: E402

OUTPUT_DIR = ROOT / "tests" / "_inspect" / "tfl_benchmark"
TARGET_CAMERA_HINTS = (
    "piccadilly", "trafalgar", "marble arch", "elephant",
    "vauxhall", "old street", "kings cross", "hyde park",
    "tower bridge", "blackfriars",
)


_FN_SAFE = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_")


def _safe_filename(text: str, n: int = 24) -> str:
    """ASCII-only, filesystem-safe truncation. cv2.imwrite on Windows
    silently fails on non-ASCII paths (no exception raised), so any
    Unicode in the filename has to be stripped before write."""
    cleaned = "".join(c if c in _FN_SAFE else "_" for c in text)
    cleaned = cleaned.strip("_") or "cam"
    return cleaned[:n]


def _draw_boxes(frame: np.ndarray, dets: list[dict], title: str) -> np.ndarray:
    """Return a copy of frame with detection boxes + a title bar drawn on it."""
    out = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        col = det.get("colour") or CLASS_COLOURS.get(det["label"], (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(
            out, f"{det['label']} {det['confidence']:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA,
        )
    # Title strip — opaque so text reads on bright AND dark frames.
    bar_h = 22
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(
        out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


async def _load_tfl_cameras() -> list[dict]:
    """Hit the TfL API directly so this script doesn't need our server up."""
    state = TflState()
    async with httpx.AsyncClient() as client:
        await _poll_once(state, client)
    return state.cameras


def _pick_cameras(cams: list[dict], n: int) -> list[dict]:
    """Try to pick by hint, otherwise fall back to first N available."""
    available = [c for c in cams if c.get("available") and c.get("video_url")]
    chosen: list[dict] = []
    seen: set[str] = set()
    for hint in TARGET_CAMERA_HINTS:
        for c in available:
            if hint in (c["name"] or "").lower() and c["id"] not in seen:
                chosen.append(c)
                seen.add(c["id"])
                break
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for c in available:
            if c["id"] in seen:
                continue
            chosen.append(c)
            seen.add(c["id"])
            if len(chosen) >= n:
                break
    return chosen[:n]


def _download_clip(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.stat().st_size > 1024
    except Exception as exc:
        logging.error(f"download failed for {url}: {exc}")
        return False


def _extract_frames(mp4: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(1, total // (n + 1))
    out: list[np.ndarray] = []
    for i in range(1, n + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=5)
    ap.add_argument("--frames-per-cam", type=int, default=3)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching TfL camera list…")
    cams = asyncio.run(_load_tfl_cameras())
    print(f"  {len(cams)} cameras returned by TfL.")
    chosen = _pick_cameras(cams, args.cameras)
    print(f"  Picked {len(chosen)} cameras for benchmarking:")
    for c in chosen:
        print(f"   • {c['name']}  ({c['view'] or '—'})  {c['id']}")

    print("\nLoading detectors (lazy weights download on first use)…")
    registry = DetectorRegistry()
    detectors = {}
    for name in VEHICLE_MODELS:
        t0 = time.perf_counter()
        try:
            detectors[name] = registry.get_vehicle(name)
            print(f"  {name:11s} loaded in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            print(f"  {name:11s} FAILED: {exc}")

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        for cam in chosen:
            mp4 = Path(td) / f"{cam['short_id']}.mp4"
            print(f"\n→ {cam['name']}  ({cam['id']})")
            if not _download_clip(cam["video_url"], mp4):
                print("  [skip] download failed.")
                continue

            frames = _extract_frames(mp4, args.frames_per_cam)
            if not frames:
                print("  [skip] couldn't decode any frames.")
                continue
            print(f"  decoded {len(frames)} frames at {frames[0].shape[1]}×{frames[0].shape[0]}")

            for f_idx, frame in enumerate(frames):
                h, w = frame.shape[:2]
                tiles: list[np.ndarray] = []
                for name, det in detectors.items():
                    t0 = time.perf_counter()
                    detections = det.detect(frame)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    n_v = sum(1 for d in detections
                              if d["label"] in ("car", "truck", "bus", "motorcycle"))
                    n_p = sum(1 for d in detections if d["label"] == "person")
                    rows.append({
                        "camera":       cam["name"],
                        "camera_id":    cam["id"],
                        "frame":        f_idx,
                        "model":        name,
                        "vehicles":     n_v,
                        "pedestrians":  n_p,
                        "total":        len(detections),
                        "latency_ms":   round(elapsed_ms, 1),
                    })
                    title = (f"{name}  veh={n_v}  ped={n_p}  "
                             f"{elapsed_ms:.0f}ms")
                    tiles.append(_draw_boxes(frame, detections, title))

                # Compose a mosaic so the eye can compare detectors.
                # Three detectors → 1×3 strip; two → side-by-side; four →
                # 2×2 (the original layout, kept in case a future detector
                # variant joins the matrix).
                if len(tiles) == 4:
                    top    = np.hstack(tiles[:2])
                    bottom = np.hstack(tiles[2:])
                    mosaic = np.vstack([top, bottom])
                else:
                    mosaic = np.hstack(tiles) if tiles else None
                if mosaic is None:
                    continue
                cap_label = _safe_filename(cam["name"])
                out_path = OUTPUT_DIR / f"{cap_label}_f{f_idx}.jpg"
                ok = cv2.imwrite(str(out_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 88])
                print(f"  {'wrote' if ok else 'FAILED to write'} {out_path.name}")

    # ── Summary table ────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\n{summary_path} written.")

    # Aggregate per model.
    by_model: dict[str, dict] = {}
    for r in rows:
        m = by_model.setdefault(r["model"], {
            "vehicles": 0, "pedestrians": 0, "total": 0,
            "latency_sum": 0.0, "n": 0,
        })
        m["vehicles"]    += r["vehicles"]
        m["pedestrians"] += r["pedestrians"]
        m["total"]       += r["total"]
        m["latency_sum"] += r["latency_ms"]
        m["n"]           += 1

    print("\n## Per-model totals across all sampled TfL frames\n")
    print("| Model | Frames | Vehicles | Pedestrians | Total dets | Mean latency |")
    print("|---|---|---|---|---|---|")
    for name, m in by_model.items():
        if m["n"] == 0:
            continue
        print(f"| {name} | {m['n']} | {m['vehicles']} | {m['pedestrians']} | "
              f"{m['total']} | {m['latency_sum'] / m['n']:.1f} ms |")

    print(f"\nAnnotated mosaics: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
