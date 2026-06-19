import pandas as pd, json, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

ANN_FILE = r"C:\LungProject_VM_Backup\meta\annotations.csv"
DET_FILE = r"C:\LungProject_VM_Backup\results\results_gt2\all_detections.json"
SPAC_DIR = r"C:\LungProject_VM_Backup\meta_spacing\meta_spacing"
OUT_FILE = r"C:\LungProject_VM_Backup\results\bootstrap_ci.json"
MATCH_RADIUS = 50
N_BOOTSTRAP  = 1000
SEED         = 42

ann = pd.read_csv(ANN_FILE)
all_uids = sorted(list(set(ann['seriesuid'].tolist())))
_, test_uids = train_test_split(all_uids, test_size=0.30, random_state=SEED)
test_uids_set = set(test_uids)

with open(DET_FILE) as f:
    all_dets = json.load(f)
test_dets = [d for d in all_dets if d['uid'] in test_uids_set]

test_ann = ann[ann['seriesuid'].isin(test_uids_set)]

# Build GT voxel coords
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

test_uid_list = list(gt_by_uid.keys())

dets_by_uid = {}
for d in test_dets:
    uid = d['uid']
    if uid not in dets_by_uid:
        dets_by_uid[uid] = []
    dets_by_uid[uid].append(d)

def froc_on_uids(uid_sample):
    uid_set  = set(uid_sample)
    gt_sub   = {u: gt_by_uid[u] for u in uid_sample if u in gt_by_uid}
    total_gt = sum(len(v) for v in gt_sub.values())
    n_scans  = len(gt_sub)
    if total_gt == 0 or n_scans == 0:
        return {fp: 0.0 for fp in [0.5,1,2,4,8]}, 0.0

    dets_sub = []
    for u in uid_sample:
        dets_sub.extend(dets_by_uid.get(u, []))

    dets_sorted = sorted(dets_sub,
                         key=lambda d: d['stage2_score'], reverse=True)
    matched_by_uid = {u: set() for u in gt_sub}
    tp_cum, fp_cum = 0, 0
    tp_list, fp_list = [], []

    for det in dets_sorted:
        u = det['uid']
        if u not in gt_sub:
            continue
        gt_nodes = gt_sub[u]
        is_tp = False
        for i, gt in enumerate(gt_nodes):
            if i in matched_by_uid[u]:
                continue
            dist = np.sqrt((det['z']-gt['z'])**2 +
                           (det['y']-gt['y'])**2 +
                           (det['x']-gt['x'])**2)
            if dist <= MATCH_RADIUS:
                matched_by_uid[u].add(i)
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

    sens = {}
    for fp_thresh in [0.5, 1, 2, 4, 8]:
        idx = np.where(fp_arr <= fp_thresh)[0]
        sens[fp_thresh] = float(tp_arr[idx[-1]]) if len(idx) > 0 else 0.0

    cpm = float(np.mean(list(sens.values())))
    return sens, cpm

# Run bootstrap
print(f"Running {N_BOOTSTRAP} bootstrap iterations on {len(test_uid_list)} test scans...")
rng = np.random.RandomState(SEED)
cpm_boot = []
sens_boot = {fp: [] for fp in [0.5, 1, 2, 4, 8]}

for i in range(N_BOOTSTRAP):
    sample = resample(test_uid_list, replace=True,
                      n_samples=len(test_uid_list),
                      random_state=rng.randint(0, 99999))
    s, cpm = froc_on_uids(sample)
    cpm_boot.append(cpm)
    for fp in [0.5, 1, 2, 4, 8]:
        sens_boot[fp].append(s[fp])

    if (i+1) % 100 == 0:
        print(f"  {i+1}/{N_BOOTSTRAP} done...")

cpm_arr  = np.array(cpm_boot)
cpm_mean = float(np.mean(cpm_arr))
cpm_lo   = float(np.percentile(cpm_arr, 2.5))
cpm_hi   = float(np.percentile(cpm_arr, 97.5))

print(f"\n=== BOOTSTRAP RESULTS ({N_BOOTSTRAP} iterations) ===")
print(f"CPM: {cpm_mean:.4f} (95% CI: {cpm_lo:.4f} – {cpm_hi:.4f})")

sens_ci = {}
for fp in [0.5, 1, 2, 4, 8]:
    arr  = np.array(sens_boot[fp])
    mean = float(np.mean(arr))
    lo   = float(np.percentile(arr, 2.5))
    hi   = float(np.percentile(arr, 97.5))
    sens_ci[fp] = {'mean': round(mean,4),
                   'ci_lo': round(lo,4),
                   'ci_hi': round(hi,4)}
    print(f"  Sensitivity @ {fp} FP/scan: "
          f"{mean:.4f} (95% CI: {lo:.4f} – {hi:.4f})")

out = {
    'n_bootstrap': N_BOOTSTRAP,
    'test_scans': len(test_uid_list),
    'CPM': {'mean': round(cpm_mean,4),
            'ci_lo': round(cpm_lo,4),
            'ci_hi': round(cpm_hi,4)},
    'sensitivities': {str(k): v for k, v in sens_ci.items()}
}
with open(OUT_FILE, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT_FILE}")