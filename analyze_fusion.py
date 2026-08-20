import json
import sys

def analyze_report(json_path, threshold=0.85):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    objects = data.get("objects", [])
    if not objects:
        print("No objects found.")
        return

    validated_total = 0
    validated_by_single_drone = 0
    validated_ONLY_by_fusion = 0

    examples = []

    for obj in objects:
        fused_conf = obj["fused_conf"]
        confs = obj["avg_conf_per_sighting"]
        
        max_single_conf = max(confs) if confs else 0.0

        if fused_conf >= threshold:
            validated_total += 1
            if max_single_conf >= threshold:
                validated_by_single_drone += 1
            else:
                validated_ONLY_by_fusion += 1
                examples.append((obj["global_id"], obj["class"], max_single_conf, fused_conf, obj["segments_seen"]))

    print(f"--- FUSION BENEFIT ANALYSIS (Validation Threshold: {threshold}) ---")
    print(f"Total objects validated (>={threshold}): {validated_total}")
    print(f"Objects validated by at least ONE single drone alone: {validated_by_single_drone}")
    print(f"Objects validated SOLELY because of Multi-Drone Fusion: {validated_ONLY_by_fusion}")
    
    if validated_ONLY_by_fusion > 0:
         print("\nEXAMPLES OF MULTI-DRONE SUCCESS:")
         for ex in examples:
             gid, cls, max_single, fused, segs = ex
             print(f"  {gid} ({cls}): Max Single Drone Conf: {max_single:.2f} | FUSED Conf: {fused:.2f} across {len(segs)} drones.")

if __name__ == "__main__":
    analyze_report(sys.argv[1], float(sys.argv[2]))
