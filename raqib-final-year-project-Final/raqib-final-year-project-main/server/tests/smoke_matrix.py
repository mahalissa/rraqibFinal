"""
Smoke test: run every (vehicle, lane) detector pair over the same BDD clip and
record latency + detection counts. Produces a markdown-ready table the
evaluation chapter of the report can cite.

Usage:
    python smoke_matrix.py [video.mov] [--frames N] [--url http://...]
"""
from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import requests

DEFAULT_VIDEO = r"L:\Misc\bdd100k_videos_test_00\bdd100k\videos\test\cabc30fc-e7726578.mov"
DEFAULT_URL   = "http://127.0.0.1:8000"
VEHICLE_MODELS = ("yolo11", "fasterrcnn", "visdrone")
LANE_MODELS    = ("yolopv2",)


def extract_frames(path: str, n: int) -> list[bytes]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(1, total // n)
    frames: list[bytes] = []
    i = 0
    while len(frames) < n:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            break
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frames.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        i += 1
    cap.release()
    return frames


def run_pair(url: str, b64frames: list[bytes], v_model: str, l_model: str) -> dict:
    latencies, n_vehicles, n_unsafe, n_lanes = [], [], [], []
    for img in b64frames:
        r = requests.post(
            f"{url}/detect",
            json={"image": img, "vehicle_model": v_model, "lane_model": l_model},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        latencies.append(data["latency_ms"])
        n_vehicles.append(len(data.get("vehicles", [])))
        n_unsafe.append(len(data.get("unsafe", [])))
        n_lanes.append(data.get("lanes", {}).get("num_lanes", 0))
    return {
        "vehicle":   v_model,
        "lane":      l_model,
        "frames":    len(b64frames),
        "lat_mean":  round(statistics.fmean(latencies), 1),
        "lat_p50":   round(statistics.median(latencies), 1),
        "lat_p95":   round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
        "vehicles":  round(statistics.fmean(n_vehicles), 2),
        "unsafe":    round(statistics.fmean(n_unsafe), 2),
        "lanes":     round(statistics.fmean(n_lanes), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    print(f"Extracting {args.frames} frames from {args.video}…")
    frames = extract_frames(args.video, args.frames)
    print(f"Extracted {len(frames)} frames.\n")

    rows: list[dict] = []
    for v in VEHICLE_MODELS:
        for l in LANE_MODELS:
            t0 = time.perf_counter()
            print(f"  running ({v}, {l})…", end=" ", flush=True)
            try:
                row = run_pair(args.url, frames, v, l)
                rows.append(row)
                print(f"mean {row['lat_mean']} ms  "
                      f"({time.perf_counter() - t0:.1f}s wall)")
            except Exception as exc:
                print(f"FAIL: {exc}")
                rows.append({"vehicle": v, "lane": l, "error": str(exc)})

    print("\n| Vehicle | Lane | Frames | mean ms | p50 ms | p95 ms | veh/frame | unsafe/frame | lanes/frame |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['vehicle']} | {r['lane']} | — | ERROR: {r['error']} |")
            continue
        print(f"| {r['vehicle']} | {r['lane']} | {r['frames']} | "
              f"{r['lat_mean']} | {r['lat_p50']} | {r['lat_p95']} | "
              f"{r['vehicles']} | {r['unsafe']} | {r['lanes']} |")

    out = Path(__file__).parent / "smoke_matrix_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
