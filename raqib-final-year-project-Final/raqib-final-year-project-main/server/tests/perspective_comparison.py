"""
Compare tailgating-flag counts under each (vehicle model × camera perspective)
combination on a fixed traffic-camera clip. Run synchronously via the
DetectorRegistry (no FastAPI process needed) so iteration is fast.

Usage:
    python perspective_comparison.py [video.mov] [--frames 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.registry import DetectorRegistry, VEHICLE_MODELS

DEFAULT_VIDEO = (
    r"L:\Misc\bdd100k_videos_test_00\bdd100k\videos\test\cabc30fc-e7726578.mov"
)
PERSPECTIVES = ("dashcam", "traffic")


def grab_frames(path: str, n: int) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(1, total // n)
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    print(f"Loading {args.frames} frames from {args.video}")
    frames = grab_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames; running matrix...\n")

    registry = DetectorRegistry()
    rows: list[dict] = []
    for v_name in VEHICLE_MODELS:
        try:
            detector = registry.get_vehicle(v_name)
        except Exception as exc:
            print(f"  {v_name:11s}  LOAD FAIL: {exc}")
            continue

        for persp in PERSPECTIVES:
            # New analyzer instance per (model, perspective) pair so the
            # dashcam streak counter doesn't leak between configurations.
            detector._tailgating.reset()
            n_v = 0
            n_t = 0
            for frame in frames:
                h, w = frame.shape[:2]
                vehicles = detector.detect(frame)
                tg = detector.analyse_tailgating(
                    vehicles, h, w, perspective=persp
                )
                n_v += len(vehicles)
                n_t += len(tg)

            rows.append({
                "vehicle":    v_name,
                "perspective": persp,
                "vehicles":   n_v,
                "tailgating": n_t,
            })
            print(f"  {v_name:11s}  {persp:8s}  veh={n_v:4d}  tailgating_flags={n_t}")

    print("\n| Vehicle | Perspective | Vehicles | Tailgating flags | flags/frame |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['vehicle']} | {r['perspective']} | {r['vehicles']} | "
              f"{r['tailgating']} | {r['tailgating'] / max(1, args.frames):.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
