import os
import json
import time
import pandas as pd
from ultralytics import YOLO
import itertools
import math

# EDGE BENCHMARK CONFIGURATION
MODELS_DIR = "models"
SEGMENTS_DIR = "segments"
OUTPUT_DIR = "results"
THRESHOLD = 0.55  # Similarity threshold for matching
VAL_THRESHOLD = 0.80  # Validation threshold for global objects

def fuse_confidence(conf_a, conf_b):
    """Collaborative Confidence Fusion (Asymptotic)"""
    return 1.0 - (1.0 - conf_a) * (1.0 - conf_b)

def run_bench():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    segments = sorted([f for f in os.listdir(SEGMENTS_DIR) if f.endswith('.mp4')])
    
    # Model priority from fastest to slowest
    model_order = {'yolov8n.pt': 0, 'yolov8s.pt': 1, 'yolov8m.pt': 2, 'yolov8l.pt': 3, 'yolov8x.pt': 4}
    models = sorted(models, key=lambda x: model_order.get(x, 99))
    
    print(f"--- Starting Edge Benchmark ---")
    print(f"Models: {len(models)} | Segments: {len(segments)}")
    
    results_summary = []

    for m_idx, m_name in enumerate(models):
        # COOL-DOWN / ENERGY BASELINE: 60s delay before starting a new model
        if m_idx > 0:
            print(f"\n[ENERGY] Waiting 60 seconds for baseline stabilization...")
            time.sleep(60)

        print("\n" + "="*60)
        print(f"=== STARTING EXPERIMENT: {m_name} ===")
        print("="*60)
        
        m_path = os.path.join(MODELS_DIR, m_name)
        model = YOLO(m_path)
        
        all_segment_results = []
        
        # Step 1: Sequential Segment Inference
        for i, seg in enumerate(segments):
            seg_path = os.path.join(SEGMENTS_DIR, seg)
            t_start = time.time()
            
            # Run inference (stream mode for memory efficiency on Pi)
            res = model.predict(source=seg_path, stream=True, verbose=False, conf=0.25)
            
            # Simple aggregation (mimicking segment_processor)
            tracks = []
            for r in res:
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if model.names[cls] == 'chair':
                            tracks.append({"cls": 'chair', "conf": conf})
            
            t_dur = time.time() - t_start
            print(f"  Seg {i:02d}: {len(tracks)} chairs detected ({t_dur:.1f}s)")
            all_segment_results.append(tracks)

        # Step 2: Multi-Drone Fusion Pass
        # For the standalone edge script, we treat each segment as a "Drone Feed"
        global_objs = []
        for seg_id, tracks in enumerate(all_segment_results):
            for t in tracks:
                # FIXED: Simple greedy clustering logic
                found = False
                for g in global_objs:
                    # Only merge if this segment hasn't contributed to this global object yet
                    if seg_id not in g.get('contributed_segments', []):
                        g['fused_conf'] = fuse_confidence(g['fused_conf'], t['conf'])
                        g['sightings'] += 1
                        g['contributed_segments'].append(seg_id)
                        found = True
                        break
                
                if not found:
                    global_objs.append({
                        "fused_conf": t['conf'], 
                        "sightings": 1,
                        "contributed_segments": [seg_id]
                    })

        # Step 3: Swarm Drop-off Combinatorial Test
        print(f"  [SWARM] Running drop-off tests...")
        for k in [10, 8, 6, 4, 2]:
            if k > len(segments): continue
            
            # Run simulation of 5 random combinations for each swarm size
            combos = list(itertools.combinations(range(len(segments)), k))
            # Sample up to 5 if too many
            sample_size = min(5, len(combos))
            
            iter_scores = []
            for combo in combos[:sample_size]:
                # Combine only the selected segments
                sub_global = []
                for sub_idx, seg_idx in enumerate(combo):
                    for t in all_segment_results[seg_idx]:
                        # FIXED: Same segment-sensitive logic for the sub-swarm
                        found = False
                        for sg in sub_global:
                            if sub_idx not in sg.get('contributed_sub_ids', []):
                                sg['fused_conf'] = fuse_confidence(sg['fused_conf'], t['conf'])
                                sg['sightings'] += 1
                                sg['contributed_sub_ids'].append(sub_idx)
                                found = True
                                break
                        if not found:
                            sub_global.append({
                                "fused_conf": t['conf'], 
                                "sightings": 1,
                                "contributed_sub_ids": [sub_idx]
                            })
                
                # Count verified chairs (>VAL_THRESHOLD)
                v_count = sum(1 for x in sub_global if x['fused_conf'] >= VAL_THRESHOLD)
                iter_scores.append(v_count)
            
            avg_v = sum(iter_scores) / len(iter_scores)
            results_summary.append({
                "model": m_name,
                "swarm_size": k,
                "verified_chairs": avg_v
            })
            print(f"    Swarm {k:02d}: {avg_v:.1f} verified chairs (avg)")

        # INCREMENTAL SAVE: Save after each model is finished
        df = pd.DataFrame(results_summary)
        summary_path = os.path.join(OUTPUT_DIR, "edge_benchmark_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"  [SAVE] Incremental results updated: {summary_path}")

    print(f"\n--- BENCHMARK COMPLETE ---")
    print(f"Total results saved to: {summary_path}")

if __name__ == "__main__":
    run_bench()
