"""
run_pipeline.py — Two-Phase Multi-Drone Identification Pipeline
----------------------------------------------------------------
Phase 1: Split video → process each segment independently → one JSON per segment
Phase 2: Fuse all JSON results → final_report.json + final_summary.txt

Usage:
  python run_pipeline.py --video input_low.mp4 --segments 10 --model yolov8n.pt

Arguments:
  --video        Source video file
  --segments     Number of segments to split into
  --model        YOLO model file (default: yolov8n.pt)
  --conf_thresh  Min detection confidence (default: 0.20)
  --frame_step   Process every Nth frame (default: 2)
  --sim_thresh   Similarity threshold for object merging (default: 0.55)
  --output_dir   Base output directory (default: pipeline_output)
"""

import os
import sys
import math
import time
import shutil
import argparse
import cv2

from segment_processor import process_segment
from fusion_pass import fuse_segments


def split_video(input_path, num_parts, output_folder):
    """Split a video into N sequential segments."""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frames_per_part = math.ceil(total_frames / num_parts)

    print(f"[SPLIT ] {input_path}: {total_frames} frames → {num_parts} segments "
          f"(≈{frames_per_part} frames each) @ {fps:.1f} fps")

    paths = []
    for i in range(num_parts):
        out_path = os.path.join(output_folder, f"segment_{i:02d}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for _ in range(frames_per_part):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        paths.append(out_path)
        print(f"[SPLIT ]   ✓ {out_path}")

    cap.release()
    return paths


def print_timing_table(phase1_times, total_phase2_time, segment_paths, log_dir=None, run_config=None):
    """Print a clean timing breakdown table and write it to a log file."""
    total_p1 = sum(phase1_times)
    grand_total = total_p1 + total_phase2_time
    slowest_idx = int(phase1_times.index(max(phase1_times)))
    slowest_time = phase1_times[slowest_idx]
    slowest_name = os.path.basename(segment_paths[slowest_idx])

    lines = []
    lines.append("=" * 65)
    lines.append("TIMING BREAKDOWN")
    lines.append("=" * 65)
    if run_config:
        lines.append(f"  Run timestamp : {run_config.get('run_id', '?')}")
        lines.append(f"  Video         : {run_config.get('video', '?')}")
        lines.append(f"  Model         : {run_config.get('model', '?')}")
        lines.append(f"  Segments      : {run_config.get('segments', '?')}")
        lines.append(f"  Sim threshold : {run_config.get('sim_thresh', '?')}")
        lines.append("-" * 65)
    lines.append(f"{'Segment':<10} {'Video File':<30} {'Time (s)':>10}")
    lines.append("-" * 65)
    for i, (path, t) in enumerate(zip(segment_paths, phase1_times)):
        fname = os.path.basename(path)
        marker = " ◀ SLOWEST" if i == slowest_idx else ""
        lines.append(f"  Seg {i:02d}   {fname:<30} {t:>10.2f}s{marker}")
    lines.append("-" * 65)
    lines.append(f"  {'Slowest segment':<38} Seg {slowest_idx:02d} ({slowest_name}) = {slowest_time:.2f}s")
    lines.append("-" * 65)
    lines.append(f"{'Phase 1 Total (sequential)':<41} {total_p1:>10.2f}s")
    lines.append(f"{'Phase 2 (Fusion / combination)':<41} {total_phase2_time:>10.2f}s")
    lines.append(f"{'GRAND TOTAL (sequential)':<41} {grand_total:>10.2f}s")
    lines.append("=" * 65)
    # Parallel estimate: if all segments ran concurrently, wall-clock = slowest + fusion
    parallel_estimate = slowest_time + total_phase2_time
    speedup = grand_total / parallel_estimate if parallel_estimate > 0 else 1.0
    sim_thresh = run_config.get('sim_thresh', '?') if run_config else '?'
    lines.append("\nPARALLEL EXECUTION ESTIMATE")
    lines.append("-" * 65)
    lines.append(f"  If segments ran concurrently (wall-clock time):")
    lines.append(f"    Slowest segment (bottleneck)  : {slowest_time:.2f}s")
    lines.append(f"    Phase 2 Fusion (sequential)   : {total_phase2_time:.2f}s")
    lines.append(f"    Estimated parallel total       : {parallel_estimate:.2f}s")
    lines.append(f"    Theoretical speedup vs sequential : {speedup:.1f}x")
    lines.append("")
    lines.append(f"SIMILARITY THRESHOLD: {sim_thresh}")
    lines.append(f"  Objects from different segments are merged if:")
    lines.append(f"    0.5 x cosine_sim(Re-ID embedding) + 0.5 x cosine_sim(color_hist) >= {sim_thresh}")
    lines.append("=" * 65)

    output = "\n".join(lines)
    print("\n" + output)

    # Write to log file
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "timing_log.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\n  Timing log saved to: {log_path}")


def run_pipeline(args):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    
    # Extract naming components for the folder
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    if video_name.startswith("input_"):
        video_name = video_name.replace("input_", "")
        
    if args.output_dir == "pipeline_output":
        base_out = f"output_{model_name}_{video_name}_{args.segments}segs_{run_id}"
    else:
        base_out = args.output_dir

    segments_video_dir = os.path.join(base_out, "video_segments")
    segments_json_dir = os.path.join(base_out, "segment_results")
    final_dir = os.path.join(base_out, "final")
    log_dir = os.path.join(base_out, "logs")
    os.makedirs(log_dir, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "video": args.video,
        "model": args.model,
        "segments": args.segments,
        "conf_thresh": args.conf_thresh,
        "frame_step": args.frame_step,
        "sim_thresh": args.sim_thresh,
    }

    setup_text = [
        "=" * 65,
        "TWO-PHASE MULTI-DRONE IDENTIFICATION PIPELINE - EXPERIMENT SETUP",
        "=" * 65
    ]
    for k, v in run_config.items():
        setup_text.append(f"  {k:<15}: {v}")
    setup_text.append("=" * 65 + "\n")
    setup_str = "\n".join(setup_text)

    print("\n" + setup_str)
    
    # Write setup to log file in the beginning
    setup_log_path = os.path.join(log_dir, "experiment_setup.txt")
    with open(setup_log_path, "w", encoding="utf-8") as f:
        f.write(setup_str)

    # ── PHASE 1a: Split ──────────────────────────────────────────────
    t0 = time.time()
    print("▶ PHASE 1a — Splitting video...")
    segment_paths = split_video(args.video, args.segments, segments_video_dir)
    print(f"  Split done in {time.time() - t0:.1f}s\n")

    # ── PHASE 1b: Process each segment ───────────────────────────────
    print("▶ PHASE 1b — Processing segments independently...")
    phase1_times = []
    for i, seg_path in enumerate(segment_paths):
        t_seg = time.time()
        process_segment(
            video_path=seg_path,
            model_path=args.model,
            segment_id=i,
            output_dir=segments_json_dir,
            conf_thresh=args.conf_thresh,
            frame_step=args.frame_step,
        )
        phase1_times.append(round(time.time() - t_seg, 2))
    print(f"\n  All segments processed.\n")

    # ── PHASE 2: Fusion ───────────────────────────────────────────────
    print("▶ PHASE 2 — Fusing segment results...")
    t_fuse = time.time()
    fuse_segments(
        segment_dir=segments_json_dir,
        sim_threshold=args.sim_thresh,
        output_dir=final_dir,
    )
    phase2_time = round(time.time() - t_fuse, 2)

    # ── Timing summary (print + save log) ────────────────────────────
    print_timing_table(phase1_times, phase2_time, segment_paths,
                       log_dir=log_dir, run_config=run_config)
    print(f"\n✅ Pipeline complete. Results in: {final_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Phase Multi-Drone Identification Pipeline"
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--segments", type=int, default=10, help="Number of segments")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model file")
    parser.add_argument("--conf_thresh", type=float, default=0.20, help="Detection confidence threshold")
    parser.add_argument("--frame_step", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--sim_thresh", type=float, default=0.55, help="Object-matching similarity threshold")
    parser.add_argument("--output_dir", default="pipeline_output", help="Base output directory")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    run_pipeline(args)
