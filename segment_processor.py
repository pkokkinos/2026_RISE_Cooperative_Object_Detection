"""
Phase 1: Segment Processor
---------------------------
Processes a single video segment in full isolation.
No shared state, no IPC, no multiprocessing.

Outputs:
  <output_dir>/segment_<id>_results.json
"""

import cv2
import json
import time
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import models, transforms


def get_reid_model():
    reid = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    reid = torch.nn.Sequential(*list(reid.children())[:-1])
    reid.eval()
    return reid


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def extract_embedding(reid_model, crop):
    """Extract normalized MobileNetV3 embedding (576-dim)."""
    img = TRANSFORM(crop).unsqueeze(0)
    with torch.no_grad():
        feat = reid_model(img)
    emb = feat.squeeze().numpy()
    if emb.ndim == 0:
        emb = np.expand_dims(emb, axis=0)
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)


def extract_color_hist(crop):
    """Extract normalized 8x8x8 HSV histogram (512-dim)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def process_segment(video_path, model_path, segment_id, output_dir, conf_thresh=0.20, frame_step=2):
    """
    Process one video segment and write a results JSON.
    Tracks are accumulated locally using bytetrack.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"[SEG {segment_id}] Loading models...")
    t_start = time.time()

    yolo = YOLO(model_path)
    reid = get_reid_model()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # track_store[track_id] = accumulated data dict
    track_store = {}

    frame_idx = 0
    frames_processed = 0

    print(f"[SEG {segment_id}] Processing {total_frames} frames from {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp_sec = frame_idx / fps
        results = yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        r = results[0]

        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for box, track_id, cls, score in zip(boxes, track_ids, classes, scores):
                if score < conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, box)
                crop = frame[max(0, y1):min(frame.shape[0], y2),
                             max(0, x1):min(frame.shape[1], x2)]
                if crop.size == 0:
                    continue

                cls_name = yolo.model.names[int(cls)]
                emb = extract_embedding(reid, crop)
                hist = extract_color_hist(crop)

                tid = int(track_id)
                if tid not in track_store:
                    track_store[tid] = {
                        "track_id": tid,
                        "class": cls_name,
                        "segment_id": segment_id,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "frame_count": 0,
                        "timestamps_sec": [],
                        "confidences": [],
                        # running sums for average embedding/histogram
                        "_emb_sum": np.zeros_like(emb),
                        "_hist_sum": np.zeros_like(hist),
                    }

                t = track_store[tid]
                t["last_frame"] = frame_idx
                t["frame_count"] += 1
                t["timestamps_sec"].append(round(timestamp_sec, 3))
                t["confidences"].append(round(float(score), 4))
                t["_emb_sum"] += emb
                t["_hist_sum"] += hist

        frame_idx += 1
        frames_processed += 1

    cap.release()
    t_proc = time.time() - t_start

    # Finalize tracks -- compute averages, clean up internal keys
    objects = []
    for tid, t in track_store.items():
        count = t["frame_count"]
        if count == 0:
            continue

        avg_emb = t["_emb_sum"] / count
        avg_emb /= (np.linalg.norm(avg_emb) + 1e-8)

        avg_hist = t["_hist_sum"] / count
        avg_hist /= (np.linalg.norm(avg_hist) + 1e-8)

        confs = t["confidences"]
        obj = {
            "track_id": t["track_id"],
            "class": t["class"],
            "segment_id": segment_id,
            "first_frame": t["first_frame"],
            "last_frame": t["last_frame"],
            "frame_count": count,
            "avg_conf": round(float(np.mean(confs)), 4),
            "max_conf": round(float(np.max(confs)), 4),
            "timestamps_sec": t["timestamps_sec"],
            "confidences": confs,
            # Stored as lists for JSON serialisation
            "embedding": avg_emb.tolist(),
            "color_hist": avg_hist.tolist(),
        }
        objects.append(obj)

    result = {
        "segment_id": segment_id,
        "video_path": video_path,
        "model": model_path,
        "conf_thresh": conf_thresh,
        "frame_step": frame_step,
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "fps": fps,
        "processing_time_sec": round(t_proc, 3),
        "num_tracks": len(objects),
        "objects": objects,
    }

    out_path = os.path.join(output_dir, f"segment_{segment_id:02d}_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[SEG {segment_id}] Done in {t_proc:.1f}s -- {len(objects)} tracks -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Process a single video segment")
    parser.add_argument("video", help="Path to video segment")
    parser.add_argument("segment_id", type=int, help="Segment index (for output filename)")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.20)
    parser.add_argument("--frame_step", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--output_dir", default="pipeline_output/segments")
    args = parser.parse_args()

    process_segment(args.video, args.model, args.segment_id,
                    args.output_dir, args.conf_thresh, args.frame_step)
