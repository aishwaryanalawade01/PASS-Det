import pandas as pd
import json
import numpy as np
import os

ANN_FILE = r"C:\LungProject_VM_Backup\meta\annotations.csv"
DET_FILE = r"C:\LungProject_VM_Backup\results\results_gt2\all_detections.json"
SPAC_DIR = r"C:\LungProject_VM_Backup\meta_spacing\meta_spacing"
OUT_FILE = r"C:\LungProject_VM_Backup\results\size_stratified_results.json"
MATCH_RADIUS = 50

ann = pd.read_csv(ANN_FILE)
print(f"Total annotations: {len(ann)}")

small  = ann[ann['diameter_mm'] < 6]
medium = ann[(ann['diameter_mm'] >= 6) & (ann['diameter_mm'] < 10)]
large  = ann[ann['diameter_mm'] >= 10]

print(f"Small  (<6mm):    {len(small)}")
print(f"Medium (6-10mm):  {len(medium)}")
print(f"Large  (>10mm):   {len(large)}")

with open(DET_FILE) as f:
    all_dets = json.load(f)
print(f"Total detections: {len(all_dets)}")

def get_voxel(row, spac_dir):
    uid = row['seriesuid']
    spac_file = os.path.join(spac_dir, uid + '_meta.json')
    if not os.path.exists(spac_file):
        return None
    with open(spac_file) as f:
        meta = json.load(f)
    # spacing = [sx, sy, sz], origin = [ox, oy, oz]
    ox, oy, oz = meta['origin']
    sx, sy, sz = meta['spacing']
    vx = (row['coordX'] - ox) / sx
    vy = (row['coordY'] - oy) / sy
    vz = (row['coordZ'] - oz) / sz
    return {'z': vz, 'y': vy, 'x': vx}

# Build GT voxel lookup per uid
def build_gt(subset, spac_dir):
    gt_by_uid = {}
    for _, row in subset.iterrows():
        uid = row['seriesuid']
        v = get_voxel(row, spac_dir)
        if v is None:
            continue
        if uid not in gt_by_uid:
            gt_by_uid[uid] = []
        gt_by_uid[uid].append(v)
    return gt_by_uid

# Build dets by uid
dets_by_uid = {}
for d in all_dets:
    uid = d['uid']
    if uid not in dets_by_uid:
        dets_by_uid[uid] = []
    dets_by_uid[uid].append(d)

def compute_sensitivity_at_fp(dets_by_uid, gt_by_uid, target_uids, fp_thresh=8):
    total_gt = 0
    total_tp = 0
    total_fp = 0
    n_scans  = 0

    for uid in target_uids:
        gt_nodes = gt_by_uid.get(uid, [])
        if len(gt_nodes) == 0:
            continue
        n_scans  += 1
        total_gt += len(gt_nodes)

        uid_dets = sorted(dets_by_uid.get(uid, []),
                          key=lambda d: d['stage2_score'], reverse=True)

        matched = set()
        tp = 0
        fp = 0
        for det in uid_dets:
            is_tp = False
            for i, gt in enumerate(gt_nodes):
                if i in matched:
                    continue
                dist = np.sqrt((det['z'] - gt['z'])**2 +
                               (det['y'] - gt['y'])**2 +
                               (det['x'] - gt['x'])**2)
                if dist <= MATCH_RADIUS:
                    matched.add(i)
                    is_tp = True
                    break
            if is_tp:
                tp += 1
            else:
                fp += 1
        total_tp += tp
        total_fp += fp

    avg_fp = total_fp / max(n_scans, 1)
    sens   = total_tp / max(total_gt, 1)
    return sens, avg_fp, total_gt, total_tp

results = {}
fp_points = [0.5, 1, 2, 4, 8]

for label, subset in [('small', small), ('medium', medium), ('large', large)]:
    gt_by_uid    = build_gt(subset, SPAC_DIR)
    target_uids  = list(gt_by_uid.keys())
    print(f"\n{label}: {len(target_uids)} scans with GT nodules")

    sens_by_fp = {}
    for fp_thresh in fp_points:
        sens, avg_fp, total_gt, total_tp = compute_sensitivity_at_fp(
            dets_by_uid, gt_by_uid, target_uids, fp_thresh)
        sens_by_fp[fp_thresh] = round(sens, 4)

    cpm = round(np.mean(list(sens_by_fp.values())), 4)
    results[label] = {
        'n_nodules': int(len(subset)),
        'sensitivities': sens_by_fp,
        'CPM': cpm
    }
    print(f"  Sensitivities: {sens_by_fp}")
    print(f"  CPM: {cpm}")

with open(OUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_FILE}")