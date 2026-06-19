import pandas as pd, json, numpy as np, os
from sklearn.model_selection import train_test_split

ANN_FILE = r"C:\LungProject_VM_Backup\meta\annotations.csv"
CAND_DIR = r"C:\LungProject_VM_Backup\candidates_gt"
SPAC_DIR = r"C:\LungProject_VM_Backup\meta_spacing\meta_spacing"
OUT_FILE = r"C:\LungProject_VM_Backup\results\baseline_stage1_froc.json"
MATCH_RADIUS = 50

ann = pd.read_csv(ANN_FILE)
all_uids = sorted(list(set(ann['seriesuid'].tolist())))
_, test_uids = train_test_split(all_uids, test_size=0.30, random_state=42)
test_uids_set = set(test_uids)

# Check candidates_gt structure
cand_contents = os.listdir(CAND_DIR)
print(f"candidates_gt contents (first 3): {cand_contents[:3]}")
print(f"Total items in candidates_gt: {len(cand_contents)}")

# Check if nested
first_item = os.path.join(CAND_DIR, cand_contents[0])
if os.path.isdir(first_item):
    print("Nested folder detected — adjusting path")
    CAND_DIR = first_item
    cand_contents = os.listdir(CAND_DIR)
    print(f"New candidates_gt contents (first 3): {cand_contents[:3]}")

# Load one candidate to check format
sample_file = os.path.join(CAND_DIR, cand_contents[0])
with open(sample_file) as f:
    sample = json.load(f)
print(f"Sample candidate keys: {sample[0].keys() if sample else 'EMPTY'}")
if sample:
    print(f"Sample candidate: {sample[0]}")

# Load Stage-1 candidates for test UIDs
stage1_dets = []
for fname in os.listdir(CAND_DIR):
    if not fname.endswith('.json'):
        continue
    uid = fname.replace('.json', '')
    if uid not in test_uids_set:
        continue
    with open(os.path.join(CAND_DIR, fname)) as f:
        cands = json.load(f)
    for c in cands:
        score = c.get('score', c.get('stage1_score',
                c.get('stage2_score', c.get('prob', 0.5))))
        stage1_dets.append({
            'uid': uid,
            'z': c['z'], 'y': c['y'], 'x': c['x'],
            'score': score
        })

print(f"\nStage-1 candidates for test set: {len(stage1_dets)}")

# Build GT
test_ann = ann[ann['seriesuid'].isin(test_uids_set)]
gt_by_uid = {}
for _, row in test_ann.iterrows():
    uid = row['seriesuid']
    spac_file = os.path.join(SPAC_DIR, uid + '_meta.json')
    if not os.path.exists(spac_file):
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

total_gt = sum(len(v) for v in gt_by_uid.values())
n_scans  = len(gt_by_uid)
print(f"Test GT: {total_gt} nodules in {n_scans} scans")

# FROC
dets_sorted = sorted(stage1_dets, key=lambda d: d['score'], reverse=True)
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

tp_arr = np.array(tp_list) / max(total_gt, 1)
fp_arr = np.array(fp_list) / max(n_scans, 1)

fp_points = [0.5, 1, 2, 4, 8]
sens = {}
for fp_thresh in fp_points:
    idx = np.where(fp_arr <= fp_thresh)[0]
    sens[fp_thresh] = round(float(tp_arr[idx[-1]]) if len(idx) > 0 else 0.0, 4)

cpm = round(float(np.mean(list(sens.values()))), 4)

print(f"\n=== STAGE-1 ONLY BASELINE ({n_scans} scans, {total_gt} nodules) ===")
for fp, s in sens.items():
    print(f"  Sensitivity @ {fp} FP/scan: {s}")
print(f"  CPM: {cpm}")

out = {'test_scans': n_scans, 'test_nodules': total_gt,
       'sensitivities': {str(k): v for k, v in sens.items()},
       'CPM': cpm}
with open(OUT_FILE, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved to {OUT_FILE}")