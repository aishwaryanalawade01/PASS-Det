import os, json
import numpy as np
import pandas as pd

CAND_DIR  = "/home/AishwaryaNalawade/lung_project/candidates"
META_DIR  = "/home/AishwaryaNalawade/data/meta_spacing"
ANN_FILE  = "/home/AishwaryaNalawade/data/meta/annotations.csv"
DET_FILE  = "/home/AishwaryaNalawade/lung_project/results/all_detections.json"

ann = pd.read_csv(ANN_FILE)

gt_voxel = {}
skipped = 0

for _, row in ann.iterrows():
    uid = row["seriesuid"]
    meta_path = os.path.join(META_DIR, uid + "_meta.json")
    if not os.path.exists(meta_path):
        skipped += 1
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    ox, oy, oz = meta["origin"]
    sx, sy, sz = meta["spacing"]
    vx = (row["coordX"] - ox) / sx
    vy = (row["coordY"] - oy) / sy
    vz = (row["coordZ"] - oz) / sz
    gt_voxel.setdefault(uid, []).append((vz, vy, vx))

print(f"GT scans loaded: {len(gt_voxel)}, skipped: {skipped}")
print(f"Total GT nodules: {sum(len(v) for v in gt_voxel.values())}")

with open(DET_FILE) as f:
    all_dets = json.load(f)

dets_by_scan = {}
for d in all_dets:
    dets_by_scan.setdefault(d["uid"], []).append(d)

MATCH_RADIUS = 20
FP_RATES = [0.5, 1, 2, 4, 8]

all_scores, all_is_tp = [], []
n_gt_total = sum(len(v) for v in gt_voxel.values())
n_scans = len(dets_by_scan)

for uid, dets in dets_by_scan.items():
    gt_list = gt_voxel.get(uid, [])
    for det in dets:
        det_c = np.array([det["z"], det["y"], det["x"]])
        score = det.get("stage2_score", 0.5)
        is_tp = 0
        for gt_c in gt_list:
            if np.linalg.norm(det_c - np.array(gt_c)) < MATCH_RADIUS:
                is_tp = 1
                break
        all_scores.append(score)
        all_is_tp.append(is_tp)

all_scores = np.array(all_scores)
all_is_tp  = np.array(all_is_tp)
sort_idx   = np.argsort(-all_scores)
all_scores = all_scores[sort_idx]
all_is_tp  = all_is_tp[sort_idx]

cum_tp      = np.cumsum(all_is_tp)
cum_fp      = np.cumsum(1 - all_is_tp)
sensitivity = cum_tp / max(n_gt_total, 1)
fp_per_scan = cum_fp / max(n_scans, 1)

print(f"\nTotal detections: {len(all_scores)}")
print(f"GT nodules: {n_gt_total} | Scans: {n_scans}")
print("\n── FROC Results ──────────────────────")
froc_sens = {}
for fp_r in FP_RATES:
    idx = np.searchsorted(fp_per_scan, fp_r)
    if idx >= len(sensitivity): idx = len(sensitivity)-1
    s = float(sensitivity[idx])
    froc_sens[fp_r] = s
    print(f"  Sensitivity @ {fp_r} FP/scan: {s:.4f}")

cpm = np.mean(list(froc_sens.values()))
print(f"\n  CPM: {cpm:.4f}")

# Save to file
result = {"froc": froc_sens, "cpm": round(cpm, 4)}
with open("results/froc_real.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved: results/froc_real.json")
