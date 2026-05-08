"""
Perspective-aware tailgating analyser.

Two camera configurations are supported:

  • "dashcam"  — forward-facing vehicle camera. There is one ego vehicle,
                 and tailgating is its lead vehicle being too close. The
                 lead is approximately centred horizontally, sits low in
                 the frame, and a closer lead has a wider bounding box.

  • "traffic"  — fixed pole-mounted CCTV. There is no ego vehicle: the
                 camera sees many vehicles at once. Tailgating is a
                 *pairwise* relationship between two consecutive vehicles
                 in the same lane.

The traffic-perspective heuristic implements the **two-second rule** from
the UK Highway Code (Rule 126: *"allow at least a two-second gap between
you and the vehicle in front on roads carrying faster-moving traffic"*).
That rule is intrinsically time-based, not distance-based, which matters
because:

  • Two stationary cars 2 m apart at a red light have *infinite* time
    headway — their relative-velocity is zero — and so should not be
    flagged. A pure pixel-distance test fires anyway.
  • Two moving cars in adjacent lanes can be close on screen but are
    not on the same trajectory, so the rear's stopping distance has no
    bearing on the lead's. A trajectory-alignment test is needed.

To compute time headway we need per-vehicle speed, which means tracking
detections across frames. The tracker is intentionally minimal: greedy
IoU association with a 1.5-second TTL and a fixed-size centroid history
per ID. Pixels stay pixels — time headway is dimensionless w.r.t. world
units, so no homography / camera calibration is required.

Reference:
    UK Highway Code, Rule 126 — Stopping distances. Department for
    Transport, 2025. https://www.gov.uk/guidance/the-highway-code/general
    -rules-techniques-and-advice-for-all-drivers-and-riders-103-to-158
"""

from __future__ import annotations

import math
import time


VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")

# Highway Code Rule 126 nominal threshold; doubled in wet conditions and
# up to 10× in icy conditions, but RAQIB has no weather signal so we
# stick with the dry-road default. Future work could expose a weather
# multiplier through the request payload.
HIGHWAY_CODE_HEADWAY_SECONDS = 2.0


# ── Lightweight tracker ──────────────────────────────────────────────────
def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) \
          + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class TailgatingAnalyzer:
    """Streak-gated dashcam heuristic + 2-second-rule traffic heuristic."""

    def __init__(self) -> None:
        # Dashcam-only state.
        self._lead_width_history: list[float] = []
        self._tailgating_streak: int = 0
        # Tracker state for the traffic heuristic.
        self._tracks: dict[int, dict] = {}     # id → {history, last_seen, label}
        self._next_id: int = 1

    def reset(self) -> None:
        self._lead_width_history.clear()
        self._tailgating_streak = 0
        self._tracks.clear()
        self._next_id = 1

    # ── Public entry point ───────────────────────────────────────────────
    def analyse(
        self,
        detections: list[dict],
        frame_h: int,
        frame_w: int,
        perspective: str = "traffic",
        **kwargs,
    ) -> list[dict]:
        if perspective == "dashcam":
            return self._analyse_dashcam(detections, frame_h, frame_w, **kwargs)
        return self._analyse_traffic(detections, frame_h, frame_w, **kwargs)

    # ── Dashcam: pre-existing single-lead heuristic ──────────────────────
    def _analyse_dashcam(
        self,
        detections: list[dict],
        frame_h: int,
        frame_w: int,
        width_ratio_threshold: float = 0.22,
        streak_required: int = 3,
    ) -> list[dict]:
        mid_x_min = frame_w * 0.25
        mid_x_max = frame_w * 0.75
        lead = None
        lead_w = 0.0
        for det in detections:
            if det["label"] not in ("car", "truck", "bus"):
                continue
            x1, y1, x2, y2 = det["box"]
            overlap = max(0, min(x2, mid_x_max) - max(x1, mid_x_min))
            if overlap < 0.4 * (x2 - x1):
                continue
            if y2 < frame_h * 0.55:
                continue
            box_w = x2 - x1
            if box_w > lead_w:
                lead_w = float(box_w)
                lead = det

        self._lead_width_history.append(lead_w / frame_w if lead else 0.0)
        if len(self._lead_width_history) > 6:
            self._lead_width_history.pop(0)

        if lead is None or lead_w / frame_w < width_ratio_threshold:
            self._tailgating_streak = 0
            return []

        self._tailgating_streak += 1
        if self._tailgating_streak < streak_required:
            return []

        return [{
            "behaviour":  "tailgating",
            "box":        lead["box"],
            "label":      lead["label"],
            "confidence": lead["confidence"],
        }]

    # ── Traffic: 2-second-rule pairwise test with motion tracking ────────
    def _associate(
        self,
        boxes: list[tuple[int, int, int, int]],
        t: float,
        iou_min: float = 0.2,
        centroid_max_lengths: float = 2.5,
        scale_ratio_max: float = 1.8,
        ttl_s: float = 1.5,
        history_max: int = 12,
    ) -> list[int]:
        """Greedy association. Returns parallel list of track IDs.

        Each new bbox matches an existing track if **either**:

          • IoU ≥ ``iou_min`` (low-motion case), or
          • centroid distance < ``centroid_max_lengths`` × max bbox
            length AND scale ratio < ``scale_ratio_max`` (high-motion
            case — vehicles can translate further than their own bbox
            length between frames at low fps or at speed, so IoU alone
            silently drops the track).

        Highest-quality candidate wins; tracks unseen for ``ttl_s`` are
        evicted before matching.
        """
        # Evict stale tracks first so we don't waste matches on them.
        stale = [tid for tid, tr in self._tracks.items()
                 if t - tr["last_seen"] > ttl_s]
        for tid in stale:
            del self._tracks[tid]

        ids: list[int] = [-1] * len(boxes)
        claimed: set[int] = set()

        # Build candidates: (score, det_idx, track_id), sorted descending
        # so the greedy match later picks the strongest links first.
        #
        # Scoring is intentionally *not* a single metric. We bias IoU
        # matches above all centroid matches by adding +1.0 to the IoU
        # score. That way, when a slow-moving vehicle's IoU and a fast-
        # moving vehicle's centroid distance both qualify the same
        # detection, the slow-moving (higher-confidence) link wins.
        candidates: list[tuple[float, int, int]] = []
        for det_idx, det_box in enumerate(boxes):
            ax1, ay1, ax2, ay2 = det_box
            a_cx = (ax1 + ax2) / 2
            a_cy = (ay1 + ay2) / 2
            a_len = max(1, max(ax2 - ax1, ay2 - ay1))
            a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            for tid, tr in self._tracks.items():
                bx1, by1, bx2, by2 = tr["last_box"]
                iou = _iou(det_box, tr["last_box"])
                if iou >= iou_min:
                    candidates.append((1.0 + iou, det_idx, tid))
                    continue
                # Centroid + scale fallback for fast motion.
                b_cx = (bx1 + bx2) / 2
                b_cy = (by1 + by2) / 2
                b_len = max(1, max(bx2 - bx1, by2 - by1))
                b_area = max(1, (bx2 - bx1) * (by2 - by1))
                dist = math.hypot(a_cx - b_cx, a_cy - b_cy)
                limit = centroid_max_lengths * max(a_len, b_len)
                if dist > limit:
                    continue
                sr = max(a_area, b_area) / min(a_area, b_area)
                if sr > scale_ratio_max:
                    continue
                # Map distance ∈ [0, limit] → similarity ∈ (0, 1).
                similarity = 1.0 - dist / limit
                candidates.append((similarity, det_idx, tid))
        candidates.sort(reverse=True)

        used_dets: set[int] = set()
        for score, det_idx, tid in candidates:
            if det_idx in used_dets or tid in claimed:
                continue
            ids[det_idx] = tid
            claimed.add(tid)
            used_dets.add(det_idx)

        # Append history for matched tracks; create new tracks for the rest.
        for det_idx, det_box in enumerate(boxes):
            x1, y1, x2, y2 = det_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            tid = ids[det_idx]
            if tid == -1:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "history":  [(t, cx, cy, w, h)],
                    "last_box": det_box,
                    "last_seen": t,
                }
                ids[det_idx] = tid
            else:
                tr = self._tracks[tid]
                tr["history"].append((t, cx, cy, w, h))
                if len(tr["history"]) > history_max:
                    tr["history"] = tr["history"][-history_max:]
                tr["last_box"] = det_box
                tr["last_seen"] = t
        return ids

    @staticmethod
    def _velocity(history: list[tuple], window_s: float = 0.6
                  ) -> tuple[float, float, float] | None:
        """Pixel velocity (vx, vy, speed) over the most recent window.

        Returns None when there isn't enough history to span the window
        or the centroid hasn't moved meaningfully — the caller treats
        either case as "speed unknown" and won't flag the vehicle."""
        if len(history) < 2:
            return None
        t_now, cx_now, cy_now, _, _ = history[-1]
        # Walk back until we cross the window or run out.
        oldest = history[0]
        for entry in reversed(history):
            if t_now - entry[0] >= window_s:
                oldest = entry
                break
        else:
            oldest = history[0]
        t_old, cx_old, cy_old, _, _ = oldest
        dt = t_now - t_old
        if dt <= 0:
            return None
        vx = (cx_now - cx_old) / dt
        vy = (cy_now - cy_old) / dt
        speed = math.hypot(vx, vy)
        return vx, vy, speed

    def _analyse_traffic(
        self,
        detections: list[dict],
        frame_h: int,
        frame_w: int,
        headway_s: float = HIGHWAY_CODE_HEADWAY_SECONDS,
        min_lead_speed_lengths_per_s: float = 0.3,
        cosine_min: float = 0.85,
        max_lateral_lengths: float = 0.9,
        max_scale_ratio: float = 2.0,
    ) -> list[dict]:
        """
        Highway Code Rule 126 — flag a rear vehicle when its time-headway
        to the lead is below ``headway_s`` (default 2.0 s) AND the lead
        is genuinely moving. Stationary queues at signals are *not*
        flagged because the rule applies to "the vehicle in front" being
        moveable; if it stops abruptly you must too. With no relative
        motion, no closing risk exists.

        For each ordered pair of vehicles (rear → lead):

          1. Both must have a velocity estimate (≥ 2 frames of history).
          2. Their velocity vectors must point in similar directions
             (cosine ≥ ``cosine_min``). This rejects oncoming traffic
             and cross-traffic at a junction.
          3. The lead's centroid must lie *ahead* of the rear (positive
             projection of rear→lead onto rear's velocity unit vector).
          4. The lead's centroid must lie close to the rear's path —
             |perpendicular component| < ``max_lateral_lengths`` ×
             rear's bbox length. This is the same-lane test that
             rejects two vehicles in adjacent lanes whose bboxes
             happen to be close on screen.
          5. Lead's speed must exceed ``min_lead_speed_lengths_per_s``
             × lead's bbox length per second. Stationary lead → no flag.
          6. ``time_headway = along_track_gap / lead_speed``. Flag iff
             ``< headway_s``.

        ``max_scale_ratio`` rejects pairs where one bbox is much larger
        than the other (different perspective depths, so almost certainly
        in different lanes despite alignment).
        """
        t = time.perf_counter()
        # Filter to vehicles before tracking — we don't want pedestrians
        # confusing the IoU tracker with their tiny bboxes.
        veh: list[dict] = [d for d in detections if d["label"] in VEHICLE_LABELS]
        if len(veh) < 2:
            # Still tick the tracker so it doesn't keep stale tracks alive
            # forever when the scene briefly empties out.
            self._associate([tuple(d["box"]) for d in veh], t)
            return []

        boxes = [tuple(d["box"]) for d in veh]
        track_ids = self._associate(boxes, t)

        # Per-detection state vector.
        state = []
        for d, tid, box in zip(veh, track_ids, boxes):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            length = max(w, h)            # along-vehicle dimension proxy
            tr = self._tracks.get(tid)
            vel = self._velocity(tr["history"]) if tr else None
            state.append({
                "det":    d,
                "tid":    tid,
                "box":    box,
                "cx":     cx,
                "cy":     cy,
                "w":      w,
                "h":      h,
                "length": length,
                "area":   w * h,
                "vel":    vel,
            })

        flagged: dict[int, dict] = {}
        for i, rear in enumerate(state):
            rv = rear["vel"]
            if rv is None:
                continue
            rvx, rvy, rspeed = rv
            if rspeed <= 0.0:
                continue

            for j, lead in enumerate(state):
                if i == j:
                    continue
                lv = lead["vel"]
                if lv is None:
                    continue
                lvx, lvy, lspeed = lv

                # 5. Lead must actually be moving — the rule is about a
                # vehicle in front you're closing on, not a stopped queue.
                min_lead_speed = min_lead_speed_lengths_per_s * lead["length"]
                if lspeed < min_lead_speed:
                    continue

                # Reject pairs at very different scales (different depths).
                if rear["area"] > 0 and lead["area"] > 0:
                    sr = max(rear["area"], lead["area"]) \
                       / min(rear["area"], lead["area"])
                    if sr > max_scale_ratio:
                        continue

                # 2. Direction agreement.
                cos = (rvx * lvx + rvy * lvy) / max(1e-6, rspeed * lspeed)
                if cos < cosine_min:
                    continue

                # 3+4. Decompose the rear→lead displacement vector (dx, dy)
                # into two components in the rear's frame of reference:
                #
                #   • "along":  projection onto rear's velocity unit u =
                #               (ux, uy). Positive ⇒ lead is ahead.
                #               Computed as the dot product (dx,dy)·(ux,uy).
                #
                #   • "perp":   projection onto the unit normal n =
                #               (-uy, ux), i.e. u rotated 90° anticlockwise.
                #               |perp| is the lateral offset from the rear's
                #               travel line. Small ⇒ same lane.
                #
                # Both u and n are unit-length, so along/perp are in pixels.
                dx = lead["cx"] - rear["cx"]
                dy = lead["cy"] - rear["cy"]
                ux = rvx / rspeed
                uy = rvy / rspeed
                along  = dx * ux + dy * uy
                perp   = abs(dx * (-uy) + dy * ux)

                if along <= 0:
                    continue       # lead is behind rear: wrong ordering
                if perp > max_lateral_lengths * rear["length"]:
                    continue       # lead is in adjacent lane

                # ``along`` is centre-to-centre. The Highway Code talks
                # about the gap between bumpers, so subtract half a
                # vehicle length for each (the centre is, on average,
                # half a vehicle behind the front bumper of the rear and
                # half a vehicle ahead of the rear bumper of the lead).
                # Clamp to zero so two overlapping bboxes don't yield a
                # negative gap and a nonsensical negative time.
                bumper_gap = max(0.0, along - 0.5 * (rear["length"]
                                                    + lead["length"]))
                if bumper_gap == 0.0:
                    # Bumpers touching is unsafe by definition.
                    time_headway = 0.0
                else:
                    time_headway = bumper_gap / lspeed

                if time_headway >= headway_s:
                    continue       # safe gap

                # Deduplicate: keep the smallest-headway entry per rear.
                rear_tid = rear["tid"]
                existing = flagged.get(rear_tid)
                if existing is not None and existing["_thw"] <= time_headway:
                    continue
                flagged[rear_tid] = {
                    "behaviour":      "tailgating",
                    "box":            list(rear["box"]),
                    "label":          rear["det"]["label"],
                    "confidence":     rear["det"]["confidence"],
                    "lead_box":       list(lead["box"]),
                    "time_headway_s": round(time_headway, 2),
                    "lead_speed_px_s": round(lspeed, 1),
                    "rear_track_id":  rear_tid,
                    "lead_track_id":  lead["tid"],
                    "_thw":           time_headway,
                }

        out: list[dict] = []
        for entry in flagged.values():
            entry.pop("_thw", None)
            out.append(entry)
        return out
