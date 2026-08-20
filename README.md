# Cooperative Object Detection Using Adaptive YOLOv8 and Feature Fusion

[![Paper](https://img.shields.io/badge/LNCS-Paper-blue)](https://link.springer.com)

This repository contains the full implementation for the paper:

> **A Study on Cooperative Object Detection Using Adaptive YOLOv8 and Feature Fusion**  
> Submitted to RISE 2026, Springer LNCS

---

## What this paper does

A multi-drone swarm where each drone processes only its own short video clip independently, extracts compact semantic telemetry (object embeddings + color descriptors), and transmits it to a central fusion server. The server matches objects across drones using appearance similarity and fuses their confidence scores — recovering detections that no single drone could confirm alone.

```
Each Drone:
  Video clip → YOLOv8 (detect) → [crop each box] → MobileNetV3 (576-dim embedding)
                                                   → HSV Histogram (512-dim descriptor)
  → transmits compact JSON (~250–450 KB per segment) instead of raw video (~GB)

Central Fusion:
  10 × JSON telemetry → match objects across drones (cosine similarity ≥ 0.55)
                      → fuse confidence: C = 1 - ∏(1 - Cᵢ)
                      → confirm objects seen by ≥ 2 independent drones
```

**Key results (17 ground-truth chairs):**
- XLarge model at 1080p: **12/17 chairs confirmed (71%)** with >99.9% bandwidth reduction vs uncompressed video
- Nano model provides best **Accuracy/Latency** trade-off on edge hardware (Raspberry Pi 5)
- 6–8× speedup from parallel drone execution vs sequential single-agent

---

## Repository structure

```
├── run_pipeline.py           # Main entry point: split → process → fuse
├── segment_processor.py      # Per-drone: YOLOv8 + MobileNetV3 + HSV per clip
├── fusion_pass.py            # Central fusion: cross-drone matching + confidence fusion
├── rescale_video.py          # Generate 360p / 720p / 1080p quality tiers
├── split_video.py            # Split video into N equal segments (simulates N drones)
├── analyze_fusion.py         # Post-hoc analysis of fusion benefit per run
├── run_rise_experiment.py    # Phase 2: RISE dataset experiments
├── edge_benchmark_standalone.py  # Raspberry Pi 5 benchmark
├── requirements.txt          # Desktop/server dependencies
└── requirements_edge.txt     # Raspberry Pi 5 dependencies
```

---

## Dataset

The source video (`input.mp4`) is a ~100-second indoor walkthrough of a furnished room containing **17 chairs** across multiple rooms/areas. Three quality tiers were used:

| File | Resolution | FPS | Size |
|------|-----------|-----|------|
| `input.mp4` | 1080p | 30 | 123 MB |
| `input_medium.mp4` | 720p | 15 | 70 MB |
| `input_low.mp4` | 360p | 10 | 16 MB |

**Download the dataset:**  
👉 [GitHub Releases](../../releases) — download `dataset.zip` (~210 MB)

After downloading, place the `.mp4` files in the root of this repo. Alternatively, generate the quality tiers yourself:

```bash
# Only download input.mp4, then generate the others:
python rescale_video.py input.mp4
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# YOLOv8 weights are auto-downloaded on first run (no manual download needed)
```

---

## Reproducing the paper results

### Table 2 — Confirmed chair detections across model tiers and quality

Run the full pipeline for one model and quality tier (example: XLarge at 1080p):

```bash
python run_pipeline.py \
  --video input.mp4 \
  --model yolov8x.pt \
  --segments 10 \
  --output_dir output_xlarge_high
```

To reproduce all 12 combinations (4 models × 3 qualities):

```bash
for MODEL in yolov8n yolov8s yolov8m yolov8x; do
  for VIDEO in input_low.mp4 input_medium.mp4 input.mp4; do
    python run_pipeline.py --video $VIDEO --model ${MODEL}.pt --segments 10 \
      --output_dir output_${MODEL}_${VIDEO%.mp4}
  done
done
```

Results are saved to `<output_dir>/final/final_report.json`. Confirmed chairs = objects in the report where `len(segments_seen) >= 2`.

### Table 3 & 4 — Fusion rescue (Native vs Fusion-Recovered)

After running the pipeline, use:

```bash
python analyze_fusion.py --output_dir output_xlarge_high
```

**Definitions:**
- **Native Detection** = objects where any single-sighting `avg_conf ≥ 0.8` (no fusion needed)
- **Fusion-Recovered** = objects confirmed by `≥ 2 segments` with `fused_conf ≥ 0.8`
- **Effectiveness Gain** = `(Fusion-Recovered / Native) × 100%`

### Table 5 — Sequential vs Parallel latency

The pipeline reports timing automatically. Sequential time = sum of all segment processing times. Parallel time = max segment time (bottleneck drone). Speedup = Sequential / Parallel.

### Tables 8 & 10 — Raspberry Pi 5 benchmarks

Run on a Pi 5 (ARM64, 8 GB RAM):

```bash
pip install -r requirements_edge.txt
python edge_benchmark_standalone.py --video input_low.mp4 --model yolov8n.pt
```

---

## Pipeline explained

### Phase 1 — Per-drone detection (segment_processor.py)

Each drone independently runs:
1. **YOLOv8** on every 2nd frame of its clip → bounding boxes + class + confidence
2. **ByteTrack** → assigns consistent track IDs within the clip
3. **MobileNetV3** (backbone only, no classification head) → 576-dim Re-ID embedding per track
4. **HSV Histogram** (8×8×8 bins) → 512-dim color descriptor per track
5. Outputs compact JSON: per-track `{class, avg_conf, embedding, color_hist}`

### Phase 2 — Central fusion (fusion_pass.py)

The central server:
1. Loads all 10 drone JSONs
2. For each pair of tracks across drones, computes:
   ```
   similarity = 0.5 × cosine(embedding_A, embedding_B)
               + 0.5 × cosine(color_hist_A, color_hist_B)
   ```
3. If `similarity ≥ 0.55` → same physical object → merge into one entry
4. Fuses confidence asymptotically:
   ```
   C_fused = 1 - ∏(1 - Cᵢ)  for all sightings i
   ```
5. An object is **confirmed** if seen by `≥ 2 independent drones`

---

## Citation

If you use this code or dataset, please cite:

Paper not in proceedings yet - just add the linke of this page.

