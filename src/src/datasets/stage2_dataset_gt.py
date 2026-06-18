import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

NPY_DIR    = "/home/AishwaryaNalawade/data/npy"
CAND_DIR   = "/home/AishwaryaNalawade/lung_project/candidates_gt"
GT_FILE    = "/home/AishwaryaNalawade/data/meta/gt_voxel_coords.json"
PATCH_SIZE = 96
MATCH_RADIUS = 50

def normalize_hu(vol):
    vol = np.clip(vol, -1000, 400)
    return ((vol + 1000) / 1400.0).astype(np.float32)

def extract_patch(vol, center, size=PATCH_SIZE):
    z,y,x = int(round(center[0])),int(round(center[1])),int(round(center[2]))
    half = size // 2
    z = np.clip(z, half, vol.shape[0]-half-1)
    y = np.clip(y, half, vol.shape[1]-half-1)
    x = np.clip(x, half, vol.shape[2]-half-1)
    patch = vol[z-half:z+half, y-half:y+half, x-half:x+half].copy()
    if patch.shape != (size,size,size):
        pad = np.zeros((size,size,size), dtype=np.float32)
        pad[:patch.shape[0],:patch.shape[1],:patch.shape[2]] = patch
        return pad
    return patch

class Stage2DatasetGT(Dataset):
    def __init__(self):
        with open(GT_FILE) as f:
            gt = json.load(f)

        self.samples = []

        for fname in os.listdir(CAND_DIR):
            if not fname.endswith(".json"):
                continue
            uid = fname.replace(".json","")
            npy_path = os.path.join(NPY_DIR, uid+".npy")
            if not os.path.exists(npy_path):
                continue
            with open(os.path.join(CAND_DIR, fname)) as f:
                cands = json.load(f)
            if not cands:
                continue

            gt_list = gt.get(uid, [])
            gt_centers = [np.array([n["voxel_z"],n["voxel_y"],n["voxel_x"]]) for n in gt_list]

            for cand in cands:
                cand_c = np.array([cand["z"], cand["y"], cand["x"]])
                label = 0
                for gt_c in gt_centers:
                    if np.linalg.norm(cand_c - gt_c) < MATCH_RADIUS:
                        label = 1
                        break
                self.samples.append((uid, cand, label))

        pos = sum(1 for s in self.samples if s[2]==1)
        neg = sum(1 for s in self.samples if s[2]==0)
        print(f"[Stage2DatasetGT] Total: {len(self.samples)} | Pos: {pos} | Neg: {neg}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, cand, label = self.samples[idx]
        vol = normalize_hu(np.load(os.path.join(NPY_DIR, uid+".npy")))
        patch = extract_patch(vol, (cand["z"], cand["y"], cand["x"]))
        return torch.from_numpy(patch).unsqueeze(0), torch.tensor(float(label))
