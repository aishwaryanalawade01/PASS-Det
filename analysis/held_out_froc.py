import pandas as pd, json, numpy as np, os
from sklearn.model_selection import train_test_split

ANN_FILE = r"C:\LungProject_VM_Backup\meta\annotations.csv"
DET_FILE = r"C:\LungProject_VM_Backup\results\results_gt2\all_detections.json"
SPAC_DIR = r"C:\LungProject_VM_Backup\meta_spacing\meta_spacing"
OUT_FILE = r"C:\LungProject_VM_Backup\results\held_out_froc.json"
MATCH_RADIUS = 50

ann = pd.read_csv(ANN_FILE)
all_uids = sorted(list(set(ann['seriesuid'].tolist())))
train_uids, test_uids = train_test_split(
    all_uids, test_size=0.30, random_state=42)
test_uids_set = set(test_uids)
print(f"Train: {len(train_uids)} | Test: {len(test_uids)}")

with open(DET_FILE) as f:
    all_dets = json.load(f)
test_dets = [d for d in all_dets if d['uid'] in test_uids_set]
print(f"Test detections: {len(test_dets)}")

test_ann = ann[ann['seriesuid'].isin(test_uids_set)]
print(f"Test GT annotations: {len(test_ann)}")

# Build GT voxel coords for test set
gt_by_uid = {}
skipped = 0
for _, row in test_ann.iterrows():
    uid = row['seriesuid']
    spac_file = os.path.join(SPAC_DIR, uid + '_meta.json')
    if not os.path.exists(spac_file):
        skipped += 1
        continue
    with open(spac_file) as f:
        meta = json.load(f)
    ox, oy, oz = meta['origin']
    sx, sy, sz = meta['spacing']
    vx = (row['coordX'] - ox) / sx
    vy = (row['coordY'] - oy) / sy
    vz = (row['coordZ'] - oz) / sz
    if uid not in gt_by_uid:
        gt_by_uid[uid] = []
    gt_by_uid[uid].append({'z': vz, 'y': vy, 'x': vx})

print(f"GT UIDs with spacing: {len(gt_by_uid)} | Skipped: {skipped}")
total_gt = sum(len(v) for v in gt_by_uid.values())
n_scans  = len(gt_by_uid)
print(f"Total test GT nodules: {total_gt} across {n_scans} scans")

# FROC curve
dets_sorted = sorted(test_dets, key=lambda d: d['stage2_score'], reverse=True)
matched_by_uid = {uid: set() for uid in gt_by_uid}
tp_cum, fp_cum = 0, 0
tp_list, fp_list = [], []

for det in dets_sorted:
    uid = det['uid']
    if uid not in gt_by_uid:
        continue
    gt_nodes = gt_by_uid[uid]
    is_tp = False
    for i, gt in enumerate(gt_nodes):
        if i in matched_by_uid[uid]:
            continue
        dist = np.sqrt((det['z'] - gt['z'])**2 +
                       (det['y'] - gt['y'])**2 +
                       (det['x'] - gt['x'])**2)
        if dist <= MATCH_RADIUS:
            matched_by_uid[uid].add(i)
            is_tp = True
            break
    if is_tp:
        tp_cum += 1
    else:
        fp_cum += 1
    tp_list.append(tp_cum)
    fp_list.append(fp_cum)

tp_arr = np.array(tp_list) / total_gt
fp_arr = np.array(fp_list) / n_scans

fp_points = [0.5, 1, 2, 4, 8]
sens = {}
for fp_thresh in fp_points:
    idx = np.where(fp_arr <= fp_thresh)[0]
    sens[fp_thresh] = round(float(tp_arr[idx[-1]]) if len(idx) > 0 else 0.0, 4)

cpm = round(float(np.mean(list(sens.values()))), 4)

print(f"\n=== HELD-OUT TEST FROC ({n_scans} scans, {total_gt} nodules) ===")
for fp, s in sens.items():
    print(f"  Sensitivity @ {fp} FP/scan: {s}")
print(f"  CPM: {cpm}")

out = {'test_scans': n_scans, 'test_nodules': total_gt,
       'train_scans': len(train_uids),
       'sensitivities': {str(k): v for k, v in sens.items()},
       'CPM': cpm}
with open(OUT_FILE, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved to {OUT_FILE}")