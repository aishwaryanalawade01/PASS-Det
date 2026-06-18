"""
Final Inference Pipeline
========================
End-to-end inference: CT scan (.npy) --> final nodule detections

Pipeline:
  CT scan (.npy)
      ↓ Stage-1 CNN sliding window
  ~300 candidates (z, y, x, score)
      ↓ Stage-2 Refiner CNN
  top final_k detections (z, y, x, stage1_score, stage2_score)

Usage:
    python -m src.inference_pipeline --scan_uid <UID>
    python -m src.inference_pipeline --all_scans  [runs on all .npy files]
    python -m src.inference_pipeline --all_scans --output_dir results/

Output JSON format:
    [{"uid": "...", "z": 96, "y": 48, "x": 144,
      "stage1_score": 0.81, "stage2_score": 0.76}, ...]
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from src.models.stage1_cnn import Stage1CNN

# ─── Config ─────────────────────────────────────────────────────────────────
NPY_DIR = "/home/AishwaryaNalawade/data/npy"
CAND_DIR = "/home/AishwaryaNalawade/lung_project/candidates_gt"
STAGE1_PT = "stage1.pt"
STAGE2_PT = "stage2_gt_best.pt"   # use best checkpoint
OUTPUT_DIR = "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 96
STRIDE = 48
STAGE1_THRESHOLD = 0.3   # minimum Stage-1 score to forward to Stage-2
STAGE2_THRESHOLD = 0.5   # minimum Stage-2 score to report as detection
FINAL_K = 20              # max detections per scan
# ─────────────────────────────────────────────────────────────────────────────


def normalize_hu(vol):
    vol = np.clip(vol, -1000, 400)
    return ((vol + 1000) / 1400.0).astype(np.float32)


def extract_patch(vol, center, size=PATCH_SIZE):
    z, y, x = int(center[0]), int(center[1]), int(center[2])
    half = size // 2
    patch = np.zeros((size, size, size), dtype=np.float32)

    z1, y1, x1 = z - half, y - half, x - half
    z2, y2, x2 = z + half, y + half, x + half

    vz1, vy1, vx1 = max(0, z1), max(0, y1), max(0, x1)
    vz2 = min(vol.shape[0], z2)
    vy2 = min(vol.shape[1], y2)
    vx2 = min(vol.shape[2], x2)

    dz1, dy1, dx1 = vz1 - z1, vy1 - y1, vx1 - x1
    dz2 = dz1 + (vz2 - vz1)
    dy2 = dy1 + (vy2 - vy1)
    dx2 = dx1 + (vx2 - vx1)

    patch[dz1:dz2, dy1:dy2, dx1:dx2] = vol[vz1:vz2, vy1:vy2, vx1:vx2]
    return patch


def nms_3d(detections, radius=20):
    """Simple greedy 3D NMS: suppress candidates within <radius> voxels of a higher-scoring one."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["stage2_score"], reverse=True)
    keep = []

    for det in detections:
        center = np.array([det["z"], det["y"], det["x"]])
        suppress = False
        for k in keep:
            k_center = np.array([k["z"], k["y"], k["x"]])
            if np.linalg.norm(center - k_center) < radius:
                suppress = True
                break
        if not suppress:
            keep.append(det)

    return keep


def load_model(model_path, model_class):
    model = model_class().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"  Loaded weights from {model_path}")
    else:
        print(f"  [WARNING] {model_path} not found — using random weights!")
    model.eval()
    return model


# Stage2Refiner is architecturally identical to Stage1CNN
class Stage2Refiner(Stage1CNN):
    pass


@torch.no_grad()
def run_stage1(vol_norm, model_stage1):
    """Sliding window Stage-1 inference. Returns list of candidate dicts."""
    D, H, W = vol_norm.shape
    candidates = []

    for z in range(PATCH_SIZE // 2, D - PATCH_SIZE // 2, STRIDE):
        for y in range(PATCH_SIZE // 2, H - PATCH_SIZE // 2, STRIDE):
            for x in range(PATCH_SIZE // 2, W - PATCH_SIZE // 2, STRIDE):
                patch = extract_patch(vol_norm, (z, y, x))
                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
                logit = model_stage1(tensor)
                score = torch.sigmoid(logit).item()

                if score >= STAGE1_THRESHOLD:
                    candidates.append({"z": z, "y": y, "x": x, "score": score})

    return candidates


@torch.no_grad()
def run_stage2(vol_norm, candidates, model_stage2):
    """Stage-2 refinement on Stage-1 candidates."""
    detections = []

    for cand in candidates:
        patch = extract_patch(vol_norm, (cand["z"], cand["y"], cand["x"]))
        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(DEVICE)
        logit = model_stage2(tensor)
        s2_score = torch.sigmoid(logit).item()

        if s2_score >= STAGE2_THRESHOLD:
            detections.append({
                "z": cand["z"],
                "y": cand["y"],
                "x": cand["x"],
                "stage1_score": round(cand["score"], 4),
                "stage2_score": round(s2_score, 4),
            })

    return detections


def infer_one_scan(uid, model_stage1, model_stage2, use_saved_candidates=True):
    """
    Run full pipeline on one scan.
    If use_saved_candidates=True, loads from candidates/<uid>.json to skip Stage-1 re-run.
    """
    npy_path = os.path.join(NPY_DIR, uid + ".npy")
    if not os.path.exists(npy_path):
        print(f"  [SKIP] {uid}.npy not found in {NPY_DIR}")
        return []

    vol = np.load(npy_path)
    vol_norm = normalize_hu(vol)

    # ── Stage-1: load saved candidates or re-run sliding window ──
    cand_path = os.path.join(CAND_DIR, uid + ".json")
    if use_saved_candidates and os.path.exists(cand_path):
        with open(cand_path) as fp:
            candidates = json.load(fp)
        print(f"  Stage-1: loaded {len(candidates)} saved candidates")
    else:
        print(f"  Stage-1: running sliding window on {vol.shape} volume...")
        candidates = run_stage1(vol_norm, model_stage1)
        print(f"  Stage-1: found {len(candidates)} candidates")

    # ── Stage-2: refine ──────────────────────────────────────────
    print(f"  Stage-2: refining {len(candidates)} candidates...")
    detections = run_stage2(vol_norm, candidates, model_stage2)
    print(f"  Stage-2: {len(detections)} detections before NMS")

    # ── NMS + top-K ──────────────────────────────────────────────
    detections = nms_3d(detections, radius=20)
    detections = detections[:FINAL_K]
    print(f"  Final: {len(detections)} nodule detections (after NMS + top-{FINAL_K})")

    for d in detections:
        d["uid"] = uid

    return detections


def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[inference_pipeline] Loading models...")
    model_stage1 = load_model(STAGE1_PT, Stage1CNN)
    model_stage2 = load_model(STAGE2_PT, Stage2Refiner)

    # Collect scan UIDs
    if args.scan_uid:
        uids = [args.scan_uid]
    else:
        uids = [
            f.replace(".npy", "")
            for f in os.listdir(NPY_DIR)
            if f.endswith(".npy")
        ]
        print(f"\n[inference_pipeline] Found {len(uids)} scans in {NPY_DIR}")

    all_results = []

    for uid in tqdm(uids, desc="Inference"):
        print(f"\n── Processing: {uid}")
        detections = infer_one_scan(uid, model_stage1, model_stage2,
                                    use_saved_candidates=not args.no_saved_cands)
        all_results.extend(detections)

        # Save per-scan result
        scan_out = os.path.join(OUTPUT_DIR, f"{uid}_detections.json")
        with open(scan_out, "w") as fp:
            json.dump(detections, fp, indent=2)

    # Save combined results
    combined_path = os.path.join(OUTPUT_DIR, "all_detections.json")
    with open(combined_path, "w") as fp:
        json.dump(all_results, fp, indent=2)

    print(f"\n[inference_pipeline] Done.")
    print(f"  Total detections: {len(all_results)}")
    print(f"  Combined results saved to: {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung Nodule Inference Pipeline")
    parser.add_argument("--scan_uid", type=str, default=None,
                        help="Run inference on a single scan UID")
    parser.add_argument("--all_scans", action="store_true",
                        help="Run inference on all scans in NPY_DIR")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save detection JSONs")
    parser.add_argument("--no_saved_cands", action="store_true",
                        help="Re-run Stage-1 sliding window (slow) instead of using saved candidates")
    args = parser.parse_args()

    if not args.scan_uid and not args.all_scans:
        print("Specify --scan_uid <UID> or --all_scans")
    else:
        OUTPUT_DIR = args.output_dir
        main(args)
