"""
RISE Experiment Runner
-----------------------
Takes 5 pre-recorded drone videos (already representing separate drone feeds),
generates 3 quality tiers for each, then runs all YOLOv8 models
through the existing segment_processor + fusion_pass pipeline.

The 5 videos are segments themselves -- no splitting needed.
Previous results are untouched (outputs go to separate RISE_ prefixed folders).

Usage:
    python run_rise_experiment.py --src "C:/Users/kokki/Documents/fork/papers/2026_RISE_Experiments"
"""
import os
import sys
import glob
import time
import shutil
import argparse
import subprocess
import json

import cv2

# Quality tier definitions relative to source (720x1280 @ 30fps)
QUALITY_TIERS = {
    "high":   {"scale": 1.0,  "fps": 30},   # native 720x1280
    "medium": {"scale": 0.667, "fps": 15},  # ~480x853
    "low":    {"scale": 0.333, "fps": 10},  # ~240x427
}

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]


def rescale_video(src_path, dst_path, scale, target_fps):
    """Rescale a single video using OpenCV."""
    cap = cv2.VideoCapture(src_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, target_fps, (w, h))
    step = max(1, int(orig_fps / target_fps))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            out.write(resized)
        idx += 1
    cap.release()
    out.release()
    print(f"  -> {os.path.basename(dst_path)} ({w}x{h} @ {target_fps}fps)")


def prepare_quality_tiers(src_dir, work_dir):
    """Generate quality tier copies of all 5 segment videos."""
    videos = sorted(glob.glob(os.path.join(src_dir, "*.mp4")))
    assert len(videos) == 5, f"Expected 5 videos, found {len(videos)}"
    
    tier_dirs = {}
    for tier_name, cfg in QUALITY_TIERS.items():
        tier_dir = os.path.join(work_dir, f"segments_{tier_name}")
        os.makedirs(tier_dir, exist_ok=True)
        tier_dirs[tier_name] = tier_dir
        
        print(f"\nGenerating {tier_name.upper()} quality segments...")
        for i, vid in enumerate(videos):
            dst = os.path.join(tier_dir, f"segment_{i:02d}.mp4")
            if os.path.exists(dst):
                print(f"  -> {os.path.basename(dst)} already exists, skipping")
                continue
            rescale_video(vid, dst, cfg["scale"], cfg["fps"])
    
    return tier_dirs


def run_segment_processor(seg_path, seg_id, model, out_dir):
    """Call segment_processor.py for a single segment."""
    cmd = [
        sys.executable, "segment_processor.py",
        seg_path, str(seg_id),
        "--model", model,
        "--output_dir", out_dir,
        "--conf_thresh", "0.20",
        "--frame_step", "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [ERROR] Segment {seg_id}: {result.stderr[-300:]}")
    else:
        # print last line of stdout (summary line)
        lines = result.stdout.strip().split("\n")
        if lines:
            print(f"    {lines[-1]}")


def run_fusion(seg_results_dir, output_dir):
    """Call fusion_pass.py on all JSON results in seg_results_dir."""
    from fusion_pass import fuse_segments
    os.makedirs(output_dir, exist_ok=True)
    return fuse_segments(seg_results_dir, sim_threshold=0.55, output_dir=output_dir)


def run_experiment(src_dir, work_dir):
    """Main experiment loop."""
    os.makedirs(work_dir, exist_ok=True)
    
    print("=" * 70)
    print("RISE EXPERIMENT -- 5 Drone Segments, 3 Quality Tiers, 5 YOLO Models")
    print("=" * 70)
    
    # Step 1: Prepare quality tiers
    tier_dirs = prepare_quality_tiers(src_dir, work_dir)
    
    # Step 2: For each quality tier x model, run Phase 1 + Phase 2
    summary = []
    for tier_name, tier_dir in tier_dirs.items():
        segments = sorted(glob.glob(os.path.join(tier_dir, "segment_*.mp4")))
        
        for model in MODELS:
            model_short = model.replace(".pt", "")
            run_name = f"RISE_{model_short}_{tier_name}_5segs"
            seg_results_dir = os.path.join(work_dir, run_name, "segment_results")
            final_dir = os.path.join(work_dir, run_name, "final")
            os.makedirs(seg_results_dir, exist_ok=True)
            
            print(f"\n{'-'*60}")
            print(f"  Model: {model}  |  Quality: {tier_name.upper()}  |  Segments: {len(segments)}")
            print(f"  Output: {run_name}")
            print(f"{'-'*60}")
            
            t0 = time.time()
            for i, seg_path in enumerate(segments):
                # Skip if already processed
                out_json = os.path.join(seg_results_dir, f"segment_{i:02d}_results.json")
                if os.path.exists(out_json):
                    print(f"    [Seg {i}] already done, skipping.")
                    continue
                print(f"    Processing segment {i}/{len(segments)-1}...")
                run_segment_processor(seg_path, i, model, seg_results_dir)
            
            phase1_time = time.time() - t0
            
            # Phase 2: Fusion
            print(f"  Fusing {len(segments)} segments...")
            t_fuse = time.time()
            run_fusion(seg_results_dir, final_dir)
            fuse_time = time.time() - t_fuse
            
            # Read summary
            summary_path = os.path.join(final_dir, "final_summary.txt")
            unique_objs, chairs = 0, 0
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    for line in f:
                        if "Unique objects" in line:
                            unique_objs = int(line.split(":")[1].strip())
                        elif "chair " in line and ":" in line:
                            chairs = int(line.split(":")[1].strip())
            
            summary.append({
                "model": model_short, "quality": tier_name,
                "phase1_s": round(phase1_time, 1),
                "fuse_s": round(fuse_time, 2),
                "total_s": round(phase1_time + fuse_time, 1),
                "unique_objects": unique_objs, "chairs": chairs,
            })
            print(f"  Done: {unique_objs} unique objects, {chairs} chairs | Phase1: {phase1_time:.1f}s")
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("RISE EXPERIMENT SUMMARY")
    print(f"{'Model':<12} {'Quality':<10} {'Total Time':>12} {'Unique Objs':>12} {'Chairs':>8}")
    print("-" * 60)
    for r in summary:
        print(f"{r['model']:<12} {r['quality']:<10} {r['total_s']:>11}s {r['unique_objects']:>12} {r['chairs']:>8}")
    print("=" * 80)
    
    # Save JSON summary
    out_json = os.path.join(work_dir, "rise_experiment_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=r"C:\Users\kokki\Documents\fork\papers\2026_RISE_Experiments",
                        help="Folder containing the 5 source drone videos")
    parser.add_argument("--work_dir", default="RISE_experiment_output",
                        help="Output working directory (separate from previous results)")
    args = parser.parse_args()
    
    run_experiment(args.src, args.work_dir)
