"""
Ablation studies for RAQIB
==========================
Ground-truth-free evaluation. Each study isolates one design choice and
measures its effect on detection count, latency, or downstream behaviour
flags. The aim is to show that each non-trivial component (caption-band
filter, headway threshold, streak gate, model choice, confidence
threshold) is doing something measurable, and that the project's
defaults are defensible against sensible alternatives.

Studies
-------
A. Caption-band filter on/off (TfL footage, VisDrone). Counts how many
   detections, and how many tailgating flags, are eliminated by the
   geometric filter.

B. VisDrone confidence-threshold sweep. Plots detection count vs.
   threshold so the chosen 0.30 default can be justified against
   neighbouring values.

C. Headway-threshold sensitivity. Sweeps 1.0 / 1.5 / 2.0 / 2.5 / 3.0
   seconds against the same fixed scene; shows how many flags each
   threshold produces. The Highway Code default is 2.0 s.

D. Streak-gate effect (dashcam path). Compares streak_required = 1
   versus 3 on a dashcam clip; the gate is meant to suppress single-
   frame flicker so we expect ≤ flags with a higher streak.

E. Per-detector agreement against a strong reference (Faster R-CNN at
   high confidence). Treats the reference's bboxes as a *silver*
   reference set — *not* human ground truth, just a stronger model that
   we already have — and reports IoU-matched precision / recall for
   YOLO11 and VisDrone against it. Caveats stated in the docstring of
   ``study_e_silver_reference``.

F. Latency breakdown. Profiles vehicle-detect / lane-detect / tailgating
   in isolation and as a pipeline so the dominant cost is visible.

Output
------
A markdown table per study, all written to
``server/tests/_inspect/ablations/<study>.md`` plus one combined JSON
``server/tests/_inspect/ablations/results.json``.

Caveats
-------
* The "agreement-with-reference" study is *not* an accuracy measurement
  against human-labelled ground truth — it is the agreement between two
  models. Faster R-CNN is treated as the reference because it has the
  strongest published mAP on COCO of the trio and we want a strict
  reference to compute precision against. A high-recall low-precision
  detector "agreeing" with the reference does not imply correctness;
  it implies the two models report similar boxes on the same scene.
* Frame counts are deliberately small (a few cameras × a few frames)
  so the script can run in a few minutes on a single GPU. Inflate
  ``--cameras`` and ``--frames-per-cam`` for a more confident sample.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "tests" / "_inspect" / "ablations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Frame harvesting ───────────────────────────────────────────────────
def _extract_frames(mp4: Path, n: int) -> list[np.ndarray]:
    """Decode `n` evenly-spaced frames from a clip."""
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


async def _harvest_tfl_frames(n_cams: int, n_frames: int) -> list[tuple[str, np.ndarray]]:
    """Pull a small set of TfL clips and return labelled frames."""
    from tfl import TflState, _poll_once   # local import; needs server PATH
    import httpx

    state = TflState()
    async with httpx.AsyncClient() as client:
        await _poll_once(state, client)

    HINTS = ("piccadilly", "marble arch", "elephant", "vauxhall",
             "trafalgar", "hyde park", "old street", "tower")
    available = [c for c in state.cameras if c.get("available") and c.get("video_url")]
    chosen: list[dict] = []
    seen: set[str] = set()
    for hint in HINTS:
        for c in available:
            if hint in (c["name"] or "").lower() and c["id"] not in seen:
                chosen.append(c)
                seen.add(c["id"])
                break
        if len(chosen) >= n_cams:
            break
    while len(chosen) < n_cams and available:
        c = available[len(chosen)]
        if c["id"] not in seen:
            chosen.append(c)
            seen.add(c["id"])
        else:
            available.pop(0)

    out: list[tuple[str, np.ndarray]] = []
    with tempfile.TemporaryDirectory() as td:
        for cam in chosen:
            mp4 = Path(td) / f"{cam['short_id']}.mp4"
            try:
                urllib.request.urlretrieve(cam["video_url"], mp4)
            except Exception as exc:
                print(f"  download failed for {cam['name']}: {exc}")
                continue
            for fr in _extract_frames(mp4, n_frames):
                out.append((cam["name"], fr))
    return out


# ── Geometric helpers ──────────────────────────────────────────────────
def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) \
          + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _filter_caption_bands(detections: list[dict], frame_h: int,
                          top_frac: float = 0.08, bot_frac: float = 0.93
                          ) -> list[dict]:
    """Mirror of main.py's _filter_caption_bands so the ablation can run
    without the live HTTP server."""
    top_y = frame_h * top_frac
    bot_y = frame_h * bot_frac
    out: list[dict] = []
    for det in detections:
        _, y1, _, y2 = det["box"]
        if y2 <= top_y or y1 >= bot_y:
            continue
        out.append(det)
    return out


# ── Studies ────────────────────────────────────────────────────────────
def study_a_caption_band(frames, vis) -> dict[str, Any]:
    """Caption-band filter on/off."""
    print("\n[A] caption-band filter on/off")
    rows: list[dict] = []
    total_off = 0
    total_on = 0
    flags_off = 0
    flags_on = 0
    for cam, frame in frames:
        h, w = frame.shape[:2]
        dets = vis.detect(frame)
        # Reset tracker between (cam, mode) so flag counts are independent.
        vis._tailgating.reset()
        flags_no_filter = vis.analyse_tailgating(dets, h, w, perspective="traffic")
        vis._tailgating.reset()
        kept = _filter_caption_bands(dets, h)
        flags_with_filter = vis.analyse_tailgating(kept, h, w, perspective="traffic")
        rows.append({
            "camera":          cam,
            "dets_off":        len(dets),
            "dets_on":         len(kept),
            "dets_dropped":    len(dets) - len(kept),
            "flags_off":       len(flags_no_filter),
            "flags_on":        len(flags_with_filter),
        })
        total_off += len(dets)
        total_on += len(kept)
        flags_off += len(flags_no_filter)
        flags_on += len(flags_with_filter)
    summary = {
        "rows":             rows,
        "total_dets_off":   total_off,
        "total_dets_on":    total_on,
        "total_dets_dropped": total_off - total_on,
        "total_flags_off":  flags_off,
        "total_flags_on":   flags_on,
    }
    md = ["| Camera | Dets (filter off) | Dets (filter on) | Dropped | Flags off | Flags on |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['camera']} | {r['dets_off']} | {r['dets_on']} "
                  f"| {r['dets_dropped']} | {r['flags_off']} | {r['flags_on']} |")
    md.append(f"| **Total** | **{total_off}** | **{total_on}** "
              f"| **{total_off - total_on}** | **{flags_off}** | **{flags_on}** |")
    (OUTPUT_DIR / "study_a_caption_band.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  total: {total_off - total_on} detections dropped, "
          f"{flags_off - flags_on} tailgating flags eliminated.")
    return summary


def study_b_confidence_sweep(frames, vis) -> dict[str, Any]:
    """Sweep VisDrone's confidence threshold."""
    print("\n[B] VisDrone confidence-threshold sweep")
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    rows: list[dict] = []
    original = vis.confidence_threshold
    for thr in thresholds:
        vis.confidence_threshold = thr
        n_total = 0
        n_kept = 0
        for _, frame in frames:
            h, _ = frame.shape[:2]
            d = vis.detect(frame)
            kept = _filter_caption_bands(d, h)
            n_total += len(d)
            n_kept += len(kept)
        rows.append({
            "threshold":     thr,
            "raw_dets":      n_total,
            "post_filter":   n_kept,
        })
    vis.confidence_threshold = original

    md = ["| Threshold | Raw detections | After caption filter | Notes |",
          "|---|---|---|---|"]
    for r in rows:
        marker = "← default" if abs(r["threshold"] - 0.30) < 1e-6 else ""
        md.append(f"| {r['threshold']:.2f} | {r['raw_dets']} | "
                  f"{r['post_filter']} | {marker} |")
    (OUTPUT_DIR / "study_b_confidence_sweep.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  threshold sweep wrote {len(rows)} rows.")
    return {"rows": rows}


def study_c_headway_sensitivity(frames, vis) -> dict[str, Any]:
    """Sweep the 2-second-rule threshold and count flags."""
    print("\n[C] headway-threshold sensitivity")
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    # For sensitivity, we need consistent per-frame state. Re-detect on
    # every frame then re-run the analyser with each threshold *after*
    # tracker state is built up. We cheat slightly by running the
    # analyser N times per scene with a fresh tracker each time so the
    # comparison is fair.
    rows: list[dict] = []
    for thr in thresholds:
        flags = 0
        vis._tailgating.reset()
        for cam, frame in frames:
            h, w = frame.shape[:2]
            dets = vis.detect(frame)
            kept = _filter_caption_bands(dets, h)
            f = vis._tailgating._analyse_traffic(kept, h, w, headway_s=thr)
            flags += len(f)
        rows.append({"threshold_s": thr, "flags": flags})
    md = ["| Headway threshold (s) | Tailgating flags | Note |",
          "|---|---|---|"]
    for r in rows:
        n = "← Highway Code default" if abs(r["threshold_s"] - 2.0) < 1e-6 else ""
        md.append(f"| {r['threshold_s']:.1f} | {r['flags']} | {n} |")
    (OUTPUT_DIR / "study_c_headway_sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  headway sweep wrote {len(rows)} rows.")
    return {"rows": rows}


def study_d_streak_gate(frames, yolo) -> dict[str, Any]:
    """Streak-gate effect on the dashcam heuristic. Uses the same TfL
    frames as a stand-in (the gate logic is independent of camera
    type). ``streak_required = 1`` is the worst case (every spike fires)
    and 3 is the project default.
    """
    print("\n[D] streak-gate effect on dashcam heuristic")
    rows: list[dict] = []
    for streak in (1, 3):
        flags = 0
        yolo._tailgating.reset()
        for _, frame in frames:
            h, w = frame.shape[:2]
            dets = yolo.detect(frame)
            f = yolo._tailgating._analyse_dashcam(
                dets, h, w, streak_required=streak,
            )
            flags += len(f)
        rows.append({"streak_required": streak, "flags": flags})
    md = ["| Streak required | Flags | Note |", "|---|---|---|"]
    for r in rows:
        n = "← project default (filters single-frame flicker)" \
            if r["streak_required"] == 3 else "no gating (every spike fires)"
        md.append(f"| {r['streak_required']} | {r['flags']} | {n} |")
    (OUTPUT_DIR / "study_d_streak_gate.md").write_text("\n".join(md), encoding="utf-8")
    return {"rows": rows}


def study_e_silver_reference(frames, yolo, vis, frcnn,
                             iou_thr: float = 0.4) -> dict[str, Any]:
    """Per-detector agreement against Faster R-CNN as a silver reference.

    NOT a human-labelled ground-truth study. Faster R-CNN at confidence
    0.50+ is treated as a reference because (i) it is the strongest
    standalone detector in the project and (ii) on COCO classes its
    reported mAP@50:95 is highest of the three. For each frame we
    greedily match each candidate detector's boxes to reference boxes by
    IoU; a candidate box is "true positive" if it has IoU ≥ ``iou_thr``
    with an unmatched reference box of the same canonical label. Results
    are precision / recall *with respect to that reference*. Reading
    these as "accuracy" requires accepting that the reference is itself
    fallible (especially on small / oblique vehicles, which is exactly
    where VisDrone is meant to outperform it). We state this caveat in
    the report.
    """
    print(f"\n[E] silver-reference agreement (IoU ≥ {iou_thr})")

    def _by_label(dets):
        out: dict[str, list[dict]] = {}
        for d in dets:
            out.setdefault(d["label"], []).append(d)
        return out

    def _match(cands: list[dict], refs: list[dict]) -> tuple[int, int, int]:
        """Returns (TP, FP, FN) on this frame for one (cand, ref) pair."""
        cand_by = _by_label(cands)
        ref_by = _by_label(refs)
        tp = 0
        labels = set(cand_by) | set(ref_by)
        for lab in labels:
            c_list = cand_by.get(lab, [])
            r_list = ref_by.get(lab, [])
            used: set[int] = set()
            for c in c_list:
                best_j = -1
                best_iou = 0.0
                for j, r in enumerate(r_list):
                    if j in used:
                        continue
                    i = _iou(c["box"], r["box"])
                    if i >= iou_thr and i > best_iou:
                        best_iou = i
                        best_j = j
                if best_j >= 0:
                    used.add(best_j)
                    tp += 1
        n_c = sum(len(v) for v in cand_by.values())
        n_r = sum(len(v) for v in ref_by.values())
        fp = n_c - tp
        fn = n_r - tp
        return tp, fp, fn

    rows: list[dict] = []
    for name, det in (("yolo11", yolo), ("visdrone", vis)):
        tp = fp = fn = 0
        for _, frame in frames:
            h, _ = frame.shape[:2]
            ref = _filter_caption_bands(frcnn.detect(frame), h)
            cand = _filter_caption_bands(det.detect(frame), h)
            t, f, n = _match(cand, ref)
            tp += t; fp += f; fn += n
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1  = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({
            "model":     name,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3),
            "recall":    round(rec, 3),
            "f1":        round(f1, 3),
        })
    md = ["| Model | TP | FP | FN | Precision | Recall | F1 | "
          "Reference: Faster R-CNN @ conf ≥ 0.5 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                  f"{r['precision']:.3f} | {r['recall']:.3f} | "
                  f"{r['f1']:.3f} | |")
    (OUTPUT_DIR / "study_e_silver_reference.md").write_text("\n".join(md), encoding="utf-8")
    return {"rows": rows, "iou_threshold": iou_thr}


def study_f_latency_breakdown(frames, registry) -> dict[str, Any]:
    """Per-component latency on the same frames."""
    print("\n[F] per-component latency breakdown")
    yolo = registry.get_vehicle("yolo11")
    lane = registry.get_lane("yolopv2")
    rows = []
    for _, frame in frames:
        h, w = frame.shape[:2]
        # Vehicle
        t0 = time.perf_counter()
        v = yolo.detect(frame)
        t_v = (time.perf_counter() - t0) * 1000
        # Lane
        t0 = time.perf_counter()
        ln = lane.detect(frame)
        t_l = (time.perf_counter() - t0) * 1000
        # Tailgating (traffic perspective)
        yolo._tailgating.reset()
        t0 = time.perf_counter()
        _ = yolo.analyse_tailgating(v, h, w, perspective="traffic")
        t_t = (time.perf_counter() - t0) * 1000
        # Lane violation
        t0 = time.perf_counter()
        _ = lane.detect_lane_violation(ln, v, w)
        t_lv = (time.perf_counter() - t0) * 1000
        rows.append({
            "vehicle_ms":  round(t_v, 2),
            "lane_ms":     round(t_l, 2),
            "tailgate_ms": round(t_t, 2),
            "lane_viol_ms": round(t_lv, 2),
            "total_ms":    round(t_v + t_l + t_t + t_lv, 2),
        })
    summary = {
        component: {
            "median_ms": round(statistics.median([r[f"{component}_ms"] for r in rows]), 2),
            "mean_ms":   round(statistics.fmean([r[f"{component}_ms"] for r in rows]), 2),
        }
        for component in ("vehicle", "lane", "tailgate", "lane_viol", "total")
    }
    md = ["| Component | Median (ms) | Mean (ms) |", "|---|---|---|"]
    for k, v in summary.items():
        md.append(f"| {k} | {v['median_ms']} | {v['mean_ms']} |")
    (OUTPUT_DIR / "study_f_latency_breakdown.md").write_text("\n".join(md), encoding="utf-8")
    return {"per_frame": rows, "summary": summary}


# ── Driver ─────────────────────────────────────────────────────────────
def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=int, default=4)
    ap.add_argument("--frames-per-cam", type=int, default=4)
    ap.add_argument("--video", default=None,
                    help="Optional pre-existing clip; bypasses TfL fetch.")
    ap.add_argument("--frames", type=int, default=12,
                    help="Frames to sample from --video when given.")
    args = ap.parse_args()

    print(f"Harvesting frames: cameras={args.cameras}, "
          f"frames/cam={args.frames_per_cam}…")
    if args.video:
        raw = _extract_frames(Path(args.video), args.frames)
        frames = [(Path(args.video).name, f) for f in raw]
    else:
        frames = asyncio.run(
            _harvest_tfl_frames(args.cameras, args.frames_per_cam)
        )
    print(f"Got {len(frames)} frames.\n")
    if not frames:
        print("No frames available; aborting.")
        return 1

    print("Loading detectors (lazy weights download on first use)…")
    from detectors.registry import DetectorRegistry
    registry = DetectorRegistry()
    yolo  = registry.get_vehicle("yolo11")
    vis   = registry.get_vehicle("visdrone")
    frcnn = registry.get_vehicle("fasterrcnn")

    results: dict[str, Any] = {}
    results["A_caption_band"] = study_a_caption_band(frames, vis)
    results["B_confidence"]   = study_b_confidence_sweep(frames, vis)
    results["C_headway"]      = study_c_headway_sensitivity(frames, vis)
    results["D_streak"]       = study_d_streak_gate(frames, yolo)
    results["E_silver_ref"]   = study_e_silver_reference(frames, yolo, vis, frcnn)
    results["F_latency"]      = study_f_latency_breakdown(frames, registry)

    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    print(f"\nAll studies done. Outputs in {OUTPUT_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
