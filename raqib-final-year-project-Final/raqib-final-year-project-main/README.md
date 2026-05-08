# RAQIB · Real-time Traffic Safety Monitoring

**Final-Year Project 2025/26 · School of Electronic Engineering & Computer Science · Queen Mary University of London**
Student: Maha Alissa · Supervisor: Mr Haris Zia

---

## What it does

RAQIB ingests a video stream — webcam, uploaded clip, or **live London TfL JamCams** — and overlays real-time:

- **Vehicle / pedestrian detection** with one of three runtime-selectable models (YOLO11, Faster R-CNN, **YOLOv8s-VisDrone** for CCTV/aerial views).
- **Lane and drivable-area** segmentation (YOLOPv2).
- **Tailgating flags** based on the **UK Highway Code Rule 126 "two-second rule"** — actual time-headway between paired vehicles, not pixel distance.
- **Lane-violation** detection via boundary-crossing tests against the lane mask.

Detections, headway times, and unsafe-behaviour markers render as a canvas overlay on the live video. When a tailgating event is observed, the server records a **3 s pre + 3 s post annotated MP4 clip** plus a peak-moment snapshot and metadata, all browsable from a saved-violations gallery in the sidebar.

The app is gated by an admin login (PBKDF2-hashed password + signed session cookie). All endpoints — REST, WebSocket, and the violation-clip files — sit behind that gate.

---

## Quick start

### Prerequisites
- Docker Engine ≥ 20.10 with Docker Compose v2 (`docker compose …`)
- NVIDIA GPU + driver ≥ 525
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host

### 1. Set the admin password
```bash
python server/scripts/set_admin_password.py
```
Interactive — prompts for username and password and writes a PBKDF2 hash plus a fresh session secret into `.env`. Pass `--user` / `--password` for non-interactive runs.

### 2. (Optional) Configure Pusher + TfL
If you want live "this camera just refreshed" updates from TfL, copy the keys into `.env`:
```env
PUSHER_APP_ID=...
PUSHER_KEY=...
PUSHER_SECRET=...
PUSHER_CLUSTER=eu
TFL_POLL_INTERVAL_S=30
```
RAQIB still works without Pusher — the polling loop maintains a cached camera list, only the real-time push is disabled.

### 3. Run
```bash
docker compose up --build
```
First build downloads the base PyTorch image, installs deps, and pre-fetches all model weights (~280 MB). Subsequent runs start in seconds.

Open **http://localhost:8000** → redirects to `/login` → sign in with the credentials you set in step 1.

---

## Features

### Runtime-selectable detection models

| Family | Model | Trained on | Notes |
|---|---|---|---|
| Vehicle (default) | **YOLO11s** | COCO 2017 | Fastest CNN single-stage, anchor-based |
| Vehicle | **Faster R-CNN ResNet50-FPN-v2** | COCO 2017 | Two-stage, highest small-object recall on COCO classes |
| Vehicle | **YOLOv8s-VisDrone** | VisDrone 2018 (drone footage) | **Use this for CCTV / oblique cameras** — finds ~7× more vehicles than YOLO11 on TfL JamCams |
| Lane | **YOLOPv2** | BDD100K | Segmentation: drivable area + lane lines |

The client UI exposes dropdowns to switch live; switching auto-loads the chosen model the first time and caches it for the rest of the session.

### Highway Code 2-second-rule tailgating

The traffic-camera tailgating heuristic implements **UK Highway Code Rule 126** directly: *"allow at least a two-second gap between you and the vehicle in front on roads carrying faster-moving traffic."*

- Time, not distance, is the unit of safety. `time_headway = bumper_gap_pixels / lead_vehicle_pixels_per_second`. Pixels cancel out, so no per-camera homography is needed.
- A lightweight tracker keeps per-vehicle velocity history (greedy IoU + centroid-distance association so tracks survive at low frame rates).
- A pair is flagged only if **all** of:
  1. Both vehicles have a velocity estimate (≥ 2 frames of history).
  2. Their velocity vectors are roughly parallel (cosine ≥ 0.85) — rejects oncoming and cross-traffic.
  3. The lead's centroid is *ahead* of the rear in the rear's direction of travel.
  4. The lead lies on the rear's path (perpendicular distance < 0.9 vehicle-lengths) — rejects adjacent-lane false positives.
  5. The lead is actually moving (> 0.3 vehicle-lengths/sec) — **never fires on stationary queues at red lights**.
  6. Time headway < 2.0 s.

The dashcam path is preserved as a separate code path (single-lead detection in the central frame band) and selectable via the **Camera** dropdown.

### Live TfL JamCams + real-time updates via Pusher

A "🚦 TfL Camera" button opens a thumbnail picker for ~880 London traffic cameras. Picking one:
- Streams the JamCam clip through a **same-origin proxy** (`/tfl/proxy/<id>.mp4`) — TfL's S3 bucket sends no `Access-Control-Allow-Origin`, so a direct `<video>.src=` to S3 would taint the canvas and the JPEG-frame-grab loop would fail silently.
- Auto-switches the vehicle dropdown to **VisDrone** and the camera-perspective dropdown to **traffic** (the right pairing for pole-mounted CCTV).
- Subscribes to a Pusher `cameras-update` channel; when TfL refreshes the active camera's clip, the UI hot-swaps `<video>.src` so the new footage plays without dropping the active WebSocket detection stream.
- Applies a server-side **caption-band filter** that drops any detection whose bbox sits entirely inside TfL's top-left timestamp strip or bottom road-name caption — these otherwise look like a queue of small cars to VisDrone and chain into spurious tailgating flags.

A polling task in the FastAPI lifespan refreshes the cached camera list every 30 s; the picker's thumbnails come from each camera's TfL snapshot URL with native `loading="lazy"` so all 880 fit in the DOM without flooding the network.

### Violation clip recording

When the tailgating heuristic fires, the server-side `ClipperSession` (one per WebSocket) builds a continuous annotated MP4:

- **3 s pre-buffer** (rolling, time-keyed) gets prepended.
- Each subsequent flagged frame extends the recording window by 3 s, so a sustained incident produces one continuous clip rather than dozens.
- 30 s cool-down per camera prevents disk-flooding on busy junctions.

Each saved clip writes three files into `server/violation_clips/<id>.{mp4,jpg,json}`:
- **MP4** — annotated frames stitched at the session's nominal fps (boxes + headway labels + rear→lead lines baked in).
- **JPG** — single-frame snapshot of the peak moment (frame with the most flags), used as the gallery thumbnail.
- **JSON** — metadata: camera, duration, frames flagged, min/mean headway, frame size.

Clips appear in the sidebar's "Saved Violations" card with thumbnail + camera name + min-headway. Clicking a tile plays the MP4 in a modal. The directory is bind-mounted as a Docker volume so clips survive `docker compose down`.

### Admin authentication

Every HTTP route except `/login`, `/logout`, `/health`, `/favicon.ico` is gated.

- Password hashing: **PBKDF2-HMAC-SHA256, 200 000 iterations** (stdlib only — no bcrypt/passlib dep).
- Session cookies: `raqib_session`, HttpOnly + SameSite=lax, **HMAC-SHA256-signed JSON payload** with 24 h TTL.
- WebSocket connections check the cookie at handshake time; missing / invalid cookies close with code 4401 instead of leaving an open silent socket.
- Anonymous HTML GETs are 303-redirected to `/login`; anonymous JSON / API requests get 401.

If `ADMIN_PASSWORD_HASH` is unset the gate is *disabled* (with a loud startup warning) so local dev stays frictionless.

---

## Project layout

```
raqib-final-year-project/
├── Dockerfile                         GPU-ready (CUDA 12.4, cuDNN 9), bakes all model weights
├── docker-compose.yml                 nvidia runtime + violation_clips bind mount + .env mount
├── .env.example                       template — copy to `.env`
├── client/
│   ├── index.html                     dashboard (vanilla HTML5 + canvas overlays)
│   └── login.html                     admin login form
└── server/
    ├── main.py                        FastAPI: WebSocket + REST + auth + TfL endpoints
    ├── auth.py                        PBKDF2 + signed session cookies
    ├── violations.py                  per-session ClipperSession + MP4/JPG/JSON writer
    ├── tfl.py                         TfL JamCams polling + Pusher publisher
    ├── requirements.txt
    ├── detectors/
    │   ├── registry.py                lazy-cached detector factory + alias resolution
    │   ├── tailgating.py              perspective-aware analyser (Highway Code 2-sec rule)
    │   ├── vehicle.py                 YOLO11
    │   ├── vehicle_fasterrcnn.py      torchvision Faster R-CNN
    │   ├── vehicle_visdrone.py        YOLOv8s-VisDrone (HF download)
    │   └── lane.py                    YOLOPv2 TorchScript
    ├── scripts/
    │   └── set_admin_password.py      interactive .env updater
    └── tests/
        ├── smoke_matrix.py            REST benchmark across all model pairs
        ├── ws_dropdown_test.py        WebSocket end-to-end smoke test
        ├── perspective_comparison.py  dashcam-vs-traffic flag counts on the same footage
        ├── tfl_benchmark.py           multi-detector mosaic on real TfL clips
        ├── ablation_studies.py        threshold sweeps + caption-band on/off + agreement
        └── two_second_rule_test.py    moving / red-light / frozen-busy verification
```

---

## API

### Auth
| Method | Path | Description |
|---|---|---|
| `GET`  | `/login` | Login page (HTML) |
| `POST` | `/login` | `{username, password}` → sets `raqib_session` cookie |
| `POST` | `/logout` | Clears the cookie |
| `GET`  | `/auth/status` | Returns `{auth_enabled, user}` |

### Detection
| Method | Path | Description |
|---|---|---|
| `WS` | `/ws/stream` | Bi-directional frame streaming |
| `POST` | `/detect` | Single-frame REST detection |
| `GET` | `/models` | Lists vehicle / lane / perspective choices |

WebSocket message body:
```json
{
  "image": "<base64 JPEG>",
  "frame_id": 42,
  "vehicle_model":      "visdrone",   // optional · default yolo11
  "lane_model":         "yolopv2",    // optional · default yolopv2
  "camera_perspective": "traffic",    // optional · default traffic; or "dashcam"
  "camera_source":      "tfl",        // optional · enables caption-band filter
  "camera_name":        "Marble Arch",
  "camera_id":          "JamCams_00001.08850"
}
```

Server response:
```json
{
  "vehicles": [
    {"label":"car", "confidence":0.91, "box":[120,200,350,420], "class_id":2}
  ],
  "lanes":   { "drivable_mask_png":"…", "lane_mask_png":"…", "num_lanes":3 },
  "unsafe":  [
    {
      "behaviour":      "tailgating",
      "box":            [...], "label": "car", "confidence": 0.88,
      "lead_box":       [...],
      "time_headway_s": 0.42,
      "lead_speed_px_s": 73.0,
      "rear_track_id":  17, "lead_track_id": 12
    }
  ],
  "latency_ms":         48.3,
  "frame_size":         {"width":1280, "height":720},
  "vehicle_model":      "visdrone",
  "lane_model":         "yolopv2",
  "camera_perspective": "traffic",
  "saved_clip_id":      "20260428T134312_d99d9420"   // present only when a clip was just persisted
}
```

### TfL
| Method | Path | Description |
|---|---|---|
| `GET` | `/tfl/cameras` | Cached camera list (~880 entries with id, name, lat/lon, image/video URLs) |
| `GET` | `/tfl/config` | Public Pusher subscriber config (key, cluster, channel, event) |
| `GET` | `/tfl/proxy/{short_id}.mp4` | Same-origin streaming proxy for a JamCam MP4 |

### Violations
| Method | Path | Description |
|---|---|---|
| `GET` | `/violations` | Saved clips, newest first, with full metadata |
| `GET` | `/violations/file/{filename}` | Stream a saved MP4 / JPG / JSON |

### Health
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Public probe used by Docker healthcheck |

---

## Configuration

All keys live in `.env` (gitignored). `.env.example` documents the schema.

| Variable | Purpose |
|---|---|
| `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH` | Set via `python server/scripts/set_admin_password.py`. Auth gate is *disabled* when `ADMIN_PASSWORD_HASH` is unset. |
| `SESSION_SECRET` | Long random string used to sign session cookies. Auto-generated by the password script if missing. |
| `SESSION_TTL_S` | Session cookie lifetime in seconds (default `86400`). |
| `PUSHER_APP_ID`, `PUSHER_KEY`, `PUSHER_SECRET`, `PUSHER_CLUSTER` | Optional. Required to publish TfL `cameras-update` events. Without them the polling loop still updates `/tfl/cameras` but real-time pushes are disabled. |
| `TFL_POLL_INTERVAL_S` | TfL JamCams poll cadence in seconds (default `30`). |
| `TFL_APP_KEY` | Optional TfL Unified API key for higher rate limits. |
| `HF_TOKEN` | Optional HuggingFace token (only needed for gated repos; the current model set is public). |
| `RAQIB_REQUIRE_CUDA` | Default `1`. Set to `0` to allow CPU-only startup (slow). |

> **Docker note**: the compose file bind-mounts `./.env` into the container at `/app/.env:ro` rather than using `env_file:`. Compose v2's env-file loader does `$VAR` interpolation on values, which corrupts pbkdf2 hashes (their format is `pbkdf2_sha256$200000$<salt>$<hash>` and the `$<salt>` portion gets expanded to empty). The application's own `_load_env_file()` reads values as literal strings.

---

## Measured performance

### Detector-pair latency (RTX 4070 Ti, BDD100K dashcam, 30 frames @ 1280×720)

| Vehicle | Lane | median ms | p95 ms | veh/frame |
|---|---|---|---|---|
| YOLO11       | YOLOPv2 | 52 | 111 | 3.8 |
| Faster R-CNN | YOLOPv2 | 94 | 103 | 9.5 |
| VisDrone     | YOLOPv2 | 48 |  90 | 6.2 |

All pairings comfortably meet NFR1 (< 200 ms per frame).

### Detector accuracy on real TfL JamCams (15 frames × 5 London cameras)

| Model | Vehicles | Pedestrians | Mean latency |
|---|---|---|---|
| YOLO11       |  22 |  0 | 16.5 ms |
| Faster R-CNN |  47 |  7 | 42.0 ms |
| **VisDrone** | **150** | **15** | **12.3 ms** |

VisDrone finds **~7× more vehicles than YOLO11** *and* runs the fastest, because its training distribution (drone footage at varying altitudes / oblique angles) is much closer to pole-mounted CCTV than COCO is. This is why the TfL picker auto-engages VisDrone.

### 2-second-rule heuristic — moving vs stationary

| Camera | Scenario | Frames | Tailgating flags | Mean headway |
|---|---|---|---|---|
| A406 Billet Upass E | A) moving | 10 | 1 | 0.28 s |
| A406 Billet Upass E | B) repeated frame (red light) | 10 | **0** | — |
| A406 Billet Upass E | C) busy frame held still | 10 | **0** | — |
| Marble Arch | A) moving | 12 | 46 | 0.48 s |
| Marble Arch | B) repeated frame (red light) | 12 | **0** | — |
| Marble Arch | C) busy frame held still | 12 | **0** | — |

Both stationary scenarios fire **zero** flags despite 60-288 detected vehicles — the heuristic correctly recognises queueing as not tailgating.

---

## Development

Run without Docker:
```bash
python -m venv .venv
.venv/Scripts/pip install -r server/requirements.txt
python server/scripts/set_admin_password.py
cd server && ../.venv/Scripts/python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Test scripts (require the server running):
- `python server/tests/smoke_matrix.py` — REST benchmark across all (vehicle, lane) pairs.
- `python server/tests/two_second_rule_test.py --camera "Marble Arch" --frames 12` — 3-scenario verification on a TfL camera.
- `python server/tests/tfl_benchmark.py --cameras 5 --frames-per-cam 3` — annotated multi-detector mosaic on real TfL clips, written to `server/tests/_inspect/tfl_benchmark/`.
- `python server/tests/ablation_studies.py` — confidence-threshold sweeps, caption-band on/off, headway-threshold sensitivity, agreement-with-reference.

---

## References

- Ren, S. et al. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS.
- Han, C. et al. (2022). *YOLOPv2: Better, Faster, Stronger for Panoptic Driving Perception.* arXiv:2208.11434.
- Zhu, P. et al. (2018). *VisDrone-DET2018: Vision Meets Drones — A Challenge.* https://github.com/VisDrone/VisDrone-Dataset
- Ultralytics (2024). *YOLO11.* https://github.com/ultralytics/ultralytics
- UK Department for Transport (2025). *The Highway Code, Rule 126 — Stopping Distances.* https://www.gov.uk/guidance/the-highway-code/general-rules-techniques-and-advice-for-all-drivers-and-riders-103-to-158
- mshamrai (2023). *yolov8s-visdrone.* https://huggingface.co/mshamrai/yolov8s-visdrone
- Pusher Ltd. *Realtime TfL Traffic Camera API* (blog). https://pusher.com/blog/realtime-tfl-traffic-camera-api/
