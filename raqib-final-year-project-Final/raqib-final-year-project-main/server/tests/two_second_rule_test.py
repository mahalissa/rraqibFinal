"""
Sanity-check the Highway Code 2-second-rule tailgating heuristic.

Three scenarios are exercised in order, each against the same A406 TfL
clip via the running RAQIB server:

  A) MOVING TRAFFIC — stream the clip's frames in order at ~5 fps. The
     analyzer should accumulate enough velocity history to compute time
     headways; expect non-zero tailgating flags only when vehicles are
     genuinely close and closing.
  B) STATIONARY (single frame repeated) — feed the SAME frame 10× at
     ~5 fps. Every detection has zero pixel velocity, so the analyzer
     must not flag anything (this is the red-light case).
  C) STATIONARY (clip frozen on first frame) — open the clip and rewind
     to frame 0 each iteration. Same expectation as B.

Run the server first (e.g. on port 8001) then launch this script.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import urllib.request as ur

import cv2

DEFAULT_BASE  = "http://127.0.0.1:8001"
TARGET_NAME   = "A406 Billet"
PAYLOAD_TMPL  = {
    "vehicle_model":      "visdrone",
    "lane_model":         "yolopv2",
    "camera_perspective": "traffic",
    "camera_source":      "tfl",
}


def _frames_from_clip(base: str, n: int) -> list:
    cams = json.loads(ur.urlopen(base + "/tfl/cameras").read())["cameras"]
    cam = next(c for c in cams if TARGET_NAME in (c["name"] or ""))
    mp4 = os.path.join(tempfile.gettempdir(), "two_sec_test.mp4")
    ur.urlretrieve(f"{base}/tfl/proxy/{cam['short_id']}.mp4", mp4)
    cap = cv2.VideoCapture(mp4)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(1, total // n)
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out, cam["name"]


def _post(base: str, frame, fps: float) -> dict:
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    body = json.dumps({**PAYLOAD_TMPL,
                       "image": base64.b64encode(jpg.tobytes()).decode()}).encode()
    req = ur.Request(base + "/detect", data=body,
                     headers={"Content-Type": "application/json"})
    return json.loads(ur.urlopen(req).read())


def _scenario(label: str, frames, base: str, fps: float = 5.0) -> dict:
    print(f"\n=== {label} ({len(frames)} frames @ {fps:.0f} fps) ===")
    period = 1.0 / fps
    flags_total = 0
    veh_total   = 0
    flags_per_frame: list[int] = []
    headways: list[float] = []
    for k, frame in enumerate(frames):
        out = _post(base, frame, fps)
        nv = len(out["vehicles"])
        nu = sum(1 for u in out["unsafe"] if u.get("behaviour") == "tailgating")
        veh_total   += nv
        flags_total += nu
        flags_per_frame.append(nu)
        for u in out["unsafe"]:
            if u.get("behaviour") == "tailgating" and "time_headway_s" in u:
                headways.append(u["time_headway_s"])
        print(f"  frame {k:02d}  veh={nv:2d}  tailgating={nu}",
              "  " + ", ".join(f"thw={h:.1f}s"
                               for h in [u.get("time_headway_s")
                                          for u in out["unsafe"]
                                          if u.get("behaviour") == "tailgating"
                                          and "time_headway_s" in u]) if nu else "")
        time.sleep(period)
    return {
        "label":           label,
        "frames":          len(frames),
        "vehicles_total":  veh_total,
        "tailgating_total": flags_total,
        "headways":        headways,
        "per_frame":       flags_per_frame,
    }


def main() -> int:
    global TARGET_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--camera", default=TARGET_NAME,
                    help="substring of TfL camera commonName")
    args = ap.parse_args()

    TARGET_NAME = args.camera
    frames, cam_name = _frames_from_clip(args.base, args.frames)
    print(f"Camera: {cam_name}    frame size: {frames[0].shape[1]}×{frames[0].shape[0]}")

    results = []
    # Reset server-side tracker state between scenarios by switching to
    # a different model briefly — but that's heavy. Easier: just run
    # them in order with a small gap so the 1.5s TTL evicts stale
    # tracks. We sleep 2s between scenarios.
    results.append(_scenario("A) moving traffic", frames, args.base, fps=args.fps))
    time.sleep(2.0)
    results.append(_scenario("B) repeated single frame (red light)",
                             [frames[0]] * args.frames, args.base, fps=args.fps))
    time.sleep(2.0)
    results.append(_scenario("C) middle-frame held still",
                             [frames[len(frames) // 2]] * args.frames,
                             args.base, fps=args.fps))

    print("\n## Summary")
    print("| Scenario | Frames | Total vehicles | Total tailgating flags | Mean headway (s) |")
    print("|---|---|---|---|---|")
    for r in results:
        mh = (sum(r["headways"]) / len(r["headways"])) if r["headways"] else None
        mh_str = f"{mh:.2f}" if mh is not None else "—"
        print(f"| {r['label']} | {r['frames']} | {r['vehicles_total']} | "
              f"{r['tailgating_total']} | {mh_str} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
