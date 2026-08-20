"""
Phase 2: Fusion Pass
---------------------
Reads all segment JSON files produced by segment_processor.py,
matches objects across segments using embedding + color-hist similarity,
and fuses confidence scores using Asymptotic Fusion.

Fusion formula:
  C_fused = 1 - (1 - c1) * (1 - c2) * ... * (1 - cN)

Similarity metric:
  sim = 0.5 * cosine_sim(embedding) + 0.5 * cosine_sim(color_hist)

Outputs:
  <output_dir>/final_report.json
  <output_dir>/final_summary.txt
"""

import json
import os
import argparse
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def asymptotic_fuse(existing_conf: float, new_conf: float) -> float:
    """Asymptotic confidence fusion: stays within (0, 1)."""
    return 1.0 - (1.0 - existing_conf) * (1.0 - new_conf)


def similarity(emb_a, hist_a, emb_b, hist_b) -> float:
    """Combined cosine similarity (embedding + histogram), equal weight."""
    emb_sim = cosine_similarity([emb_a], [emb_b])[0][0]
    hist_sim = cosine_similarity([hist_a], [hist_b])[0][0]
    return float(0.5 * emb_sim + 0.5 * hist_sim)


def running_avg_embedding(existing_emb, existing_count, new_emb):
    """Update running average embedding and renormalize."""
    updated = (existing_emb * existing_count + new_emb) / (existing_count + 1)
    norm = np.linalg.norm(updated)
    return updated / (norm + 1e-8), existing_count + 1


def fuse_segments(segment_dir, sim_threshold=0.55, output_dir=None):
    """
    Main fusion logic. Loads all segment JSONs and builds a global
    merged object registry.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(segment_dir), "final")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load all segment results ---
    json_files = sorted(glob.glob(os.path.join(segment_dir, "segment_*_results.json")))
    if not json_files:
        print(f"[FUSION] No segment JSON files found in {segment_dir}")
        return

    print(f"[FUSION] Found {len(json_files)} segment files. Starting fusion...")

    segments_meta = []
    all_objects_by_segment = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        segments_meta.append({
            "segment_id": data["segment_id"],
            "video_path": data["video_path"],
            "processing_time_sec": data["processing_time_sec"],
            "num_tracks": data["num_tracks"],
            "frames_processed": data["frames_processed"],
        })
        all_objects_by_segment.append((data["segment_id"], data["objects"]))
        print(f"  Segment {data['segment_id']:02d}: {data['num_tracks']} tracks  "
              f"(took {data['processing_time_sec']:.1f}s)")

    total_raw_detections = sum(len(objs) for _, objs in all_objects_by_segment)
    print(f"\n[FUSION] Total raw tracks across all segments: {total_raw_detections}")
    print(f"[FUSION] Similarity threshold: {sim_threshold}")
    print(f"[FUSION] Running fusion pass...\n")

    # --- Global registry ---
    # Each entry: {
    #   "global_id": "GlobalObj_001",
    #   "class": "chair",
    #   "fused_conf": float,
    #   "sighting_count": int,            # how many times matched
    #   "segments_seen": [int, ...],
    #   "avg_conf_per_sighting": [float],
    #   "total_frame_count": int,
    #   "embedding": np.array,
    #   "_emb_count": int,
    #   "color_hist": np.array,
    #   "first_timestamp_sec": float,
    #   "last_timestamp_sec": float,
    # }
    global_registry = []
    global_id_counter = 1

    for seg_id, objects in all_objects_by_segment:
        for obj in objects:
            cls = obj["class"]
            emb = np.array(obj["embedding"])
            hist = np.array(obj["color_hist"])
            avg_conf = obj["avg_conf"]

            # Try to match against existing global objects of the same class
            best_idx = None
            best_sim = 0.0

            for i, g in enumerate(global_registry):
                if g["class"] != cls:
                    continue
                sim = similarity(emb, hist, g["embedding"], g["color_hist"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_idx is not None and best_sim >= sim_threshold:
                # MERGE into existing global object
                g = global_registry[best_idx]
                g["fused_conf"] = asymptotic_fuse(g["fused_conf"], avg_conf)
                g["sighting_count"] += 1
                g["avg_conf_per_sighting"].append(avg_conf)
                g["total_frame_count"] += obj["frame_count"]
                g["segments_seen"] = sorted(set(g["segments_seen"] + [seg_id]))

                # Update running average embedding
                new_emb, new_count = running_avg_embedding(
                    g["embedding"], g["_emb_count"], emb
                )
                g["embedding"] = new_emb
                g["_emb_count"] = new_count

                # Update running average color hist
                new_hist, _ = running_avg_embedding(
                    g["color_hist"], g["_emb_count"] - 1, hist
                )
                g["color_hist"] = new_hist

                # Update time window
                if obj["timestamps_sec"]:
                    g["first_timestamp_sec"] = min(
                        g["first_timestamp_sec"], min(obj["timestamps_sec"])
                    )
                    g["last_timestamp_sec"] = max(
                        g["last_timestamp_sec"], max(obj["timestamps_sec"])
                    )

                print(f"  [MERGE ] GlobalObj_{g['global_id']:03d} ({cls}) "
                      f"<- Seg {seg_id} Trk {obj['track_id']} "
                      f"| sim={best_sim:.3f} | fused_conf={g['fused_conf']:.3f}")
            else:
                # CREATE new global object
                new_entry = {
                    "global_id": global_id_counter,
                    "class": cls,
                    "fused_conf": avg_conf,
                    "sighting_count": 1,
                    "avg_conf_per_sighting": [avg_conf],
                    "total_frame_count": obj["frame_count"],
                    "segments_seen": [seg_id],
                    "embedding": emb,
                    "_emb_count": 1,
                    "color_hist": hist,
                    "first_timestamp_sec": min(obj["timestamps_sec"]) if obj["timestamps_sec"] else 0.0,
                    "last_timestamp_sec": max(obj["timestamps_sec"]) if obj["timestamps_sec"] else 0.0,
                }
                global_registry.append(new_entry)
                print(f"  [CREATE] GlobalObj_{global_id_counter:03d} ({cls}) "
                      f"<- Seg {seg_id} Trk {obj['track_id']} "
                      f"| conf={avg_conf:.3f}")
                global_id_counter += 1

    # --- Build output (strip numpy arrays before JSON) ---
    output_objects = []
    class_summary = {}

    for g in global_registry:
        cls = g["class"]
        class_summary[cls] = class_summary.get(cls, 0) + 1
        output_objects.append({
            "global_id": f"GlobalObj_{g['global_id']:03d}",
            "class": cls,
            "fused_conf": round(g["fused_conf"], 4),
            "sighting_count": g["sighting_count"],
            "avg_conf_per_sighting": [round(c, 4) for c in g["avg_conf_per_sighting"]],
            "total_frame_count": g["total_frame_count"],
            "segments_seen": g["segments_seen"],
            "first_timestamp_sec": round(g["first_timestamp_sec"], 3),
            "last_timestamp_sec": round(g["last_timestamp_sec"], 3),
        })

    final_report = {
        "sim_threshold": sim_threshold,
        "total_segments": len(json_files),
        "total_raw_tracks": total_raw_detections,
        "unique_objects_after_fusion": len(output_objects),
        "class_summary": class_summary,
        "segments_meta": segments_meta,
        "objects": output_objects,
    }

    report_path = os.path.join(output_dir, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    # --- Human-readable summary ---
    summary_path = os.path.join(output_dir, "final_summary.txt")
    lines = [
        "=" * 60,
        "MULTI-DRONE FUSION RESULTS",
        "=" * 60,
        f"Segments processed  : {len(json_files)}",
        f"Raw tracks (total)  : {total_raw_detections}",
        f"Unique objects      : {len(output_objects)}",
        f"Similarity threshold: {sim_threshold}",
        "",
        "OBJECT CLASS BREAKDOWN",
        "-" * 40,
    ]
    for cls, cnt in sorted(class_summary.items(), key=lambda x: -x[1]):
        lines.append(f"  {cls:<20s}: {cnt}")
    lines.append("")
    lines.append("TOP OBJECTS (by fused confidence)")
    lines.append("-" * 40)
    top = sorted(output_objects, key=lambda x: -x["fused_conf"])[:20]
    for o in top:
        segs = ",".join(str(s) for s in o["segments_seen"])
        lines.append(f"  {o['global_id']} ({o['class']:<14s}) "
                     f"fused_conf={o['fused_conf']:.3f}  "
                     f"sightings={o['sighting_count']}  segs=[{segs}]")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\n[FUSION] Report saved to: {report_path}")
    print(f"[FUSION] Summary saved to: {summary_path}")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Fuse all segment results")
    parser.add_argument("segment_dir", help="Directory containing segment_XX_results.json files")
    parser.add_argument("--sim_thresh", type=float, default=0.55)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    fuse_segments(args.segment_dir, args.sim_thresh, args.output_dir)
