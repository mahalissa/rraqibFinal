"""
Drive the /ws/stream endpoint the same way the browser dropdowns do:
for each (vehicle_model, lane_model) pair, stream 5 frames and assert the
server responds with matching `vehicle_model` / `lane_model` fields and a
well-formed payload. This exercises the WebSocket code path that the UI
actually uses (the smoke_matrix tests hit /detect).
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys

import cv2
import websockets

VIDEO = r"L:\Misc\bdd100k_videos_test_00\bdd100k\videos\test\cabc30fc-e7726578.mov"
WS_URL = "ws://127.0.0.1:8000/ws/stream"
PAIRS = [
    ("yolo11",     "yolopv2"),
    ("fasterrcnn", "yolopv2"),
    ("visdrone",   "yolopv2"),
]


def grab_frames(path: str, n: int = 5) -> list[str]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    step = max(1, total // n)
    out: list[str] = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            break
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    cap.release()
    return out


async def main() -> int:
    frames = grab_frames(VIDEO, 5)
    print(f"Loaded {len(frames)} frames.")

    fail = 0
    async with websockets.connect(WS_URL, max_size=16 * 1024 * 1024) as ws:
        for v, l in PAIRS:
            latencies: list[float] = []
            v_echoed: set[str] = set()
            l_echoed: set[str] = set()
            frame_ids: set[int] = set()
            for idx, img in enumerate(frames):
                await ws.send(json.dumps({
                    "image": img,
                    "frame_id": idx,
                    "vehicle_model": v,
                    "lane_model":    l,
                }))
                reply = json.loads(await ws.recv())
                if "error" in reply:
                    print(f"  {v:10s} / {l:8s}  ERROR: {reply['error']}")
                    fail += 1
                    break
                latencies.append(reply["latency_ms"])
                v_echoed.add(reply.get("vehicle_model", "?"))
                l_echoed.add(reply.get("lane_model",    "?"))
                frame_ids.add(reply.get("frame_id", -1))
                # assert mask keys present
                lane_keys = set(reply.get("lanes", {}).keys())
                if not {"drivable_mask_png", "lane_mask_png", "num_lanes"} <= lane_keys:
                    print(f"  {v:10s} / {l:8s}  FAIL: lane payload missing keys {lane_keys}")
                    fail += 1
                    break
            else:
                ok_v = v_echoed == {v}
                ok_l = l_echoed == {l}
                ok_ids = frame_ids == set(range(len(frames)))
                status = "OK" if (ok_v and ok_l and ok_ids) else "MISMATCH"
                mean_ms = sum(latencies) / len(latencies)
                print(f"  {v:10s} / {l:8s}  {status}  mean={mean_ms:5.1f} ms  "
                      f"echo=(v={v_echoed},l={l_echoed})")
                if status != "OK":
                    fail += 1
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
