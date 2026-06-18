import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

NPY_DIR = "/home/AishwaryaNalawade/data/npy"
GT_FILE = "/home/AishwaryaNalawade/data/meta/gt_voxel_coords.json"
PATCH_SIZE = 96

def normalize_hu(vol):
    vol = np.clip(vol, -1000, 400)
    return ((vol + 1000) / 1400.0).astype(np.float32)

def extract_patch(vol, center, size=PATCH_SIZE):
    z, y, x = int(round(center[0])), int(round(center[1])), int(round(center[2]))
    half = size // 2

    # Clamp center so patch always fits inside volume
    z = np.clip(z, half, vol.shape[0] - half - 1)
    y = np.clip(y, half, vol.shape[1] - half - 1)
    x = np.clip(x, half, vol.shape[2] - half - 1)

    patch = vol[z-half:z+half, y-half:y+half, x-half:x+half].copy()

    # Ensure exact size (edge case guard)
    if patch.shape != (size, size, size):
        pad = np.zeros((size, size, size), dtype=np.float32)
        sz, sy, sx = patch.shape
        pad[:sz, :sy, :sx] = patch
        return pad

    return patch
'''
def extract_patch(vol, center, size=PATCH_SIZE):
    z, y, x = int(center[0]), int(center[1]), int(center[2])
    half = size // 2
    patch = np.zeros((size, size, size), dtype=np.float32)
    z1,y1,x1 = z-half, y-half, x-half
    z2,y2,x2 = z+half, y+half, x+half
    vz1,vy1,vx1 = max(0,z1), max(0,y1), max(0,x1)
    vz2,vy2,vx2 = min(vol.shape[0],z2), min(vol.shape[1],y2), min(vol.shape[2],x2)
    dz1,dy1,dx1 = vz1-z1, vy1-y1, vx1-x1
    patch[dz1:dz1+(vz2-vz1), dy1:dy1+(vy2-vy1), dx1:dx1+(vx2-vx1)] = vol[vz1:vz2,vy1:vy2,vx1:vx2]
    return patch
'''

class GTLungPatchDataset(Dataset):
    def __init__(self):
        with open(GT_FILE) as f:
            gt = json.load(f)

        self.samples = []  # (uid, center, label)

        for uid, nodules in gt.items():
            npy_path = os.path.join(NPY_DIR, uid + ".npy")
            if not os.path.exists(npy_path):
                continue

            # Positive patches centered on real nodules
            for n in nodules:
                center = (n["voxel_z"], n["voxel_y"], n["voxel_x"])
                # Augment: add small jitter (±10 voxels)
                for _ in range(4):
                    jitter = np.random.randint(-10, 10, 3)
                    jc = (center[0]+jitter[0], center[1]+jitter[1], center[2]+jitter[2])
                    self.samples.append((uid, jc, 1))

            # Negative patches: random locations far from nodules
            nodule_centers = [(n["voxel_z"], n["voxel_y"], n["voxel_x"]) for n in nodules]
            neg_added = 0
            attempts = 0
            while neg_added < len(nodules) * 4 and attempts < 200:
                attempts += 1
                rz = np.random.randint(48, 146)
                ry = np.random.randint(48, 464)
                rx = np.random.randint(48, 464)
                # Ensure far from all nodules
                far = all(
                    np.linalg.norm(np.array([rz,ry,rx]) - np.array(nc)) > 48
                    for nc in nodule_centers
                )
                if far:
                    self.samples.append((uid, (rz, ry, rx), 0))
                    neg_added += 1

        pos = sum(1 for s in self.samples if s[2]==1)
        neg = sum(1 for s in self.samples if s[2]==0)
        print(f"[GTLungPatchDataset] Total: {len(self.samples)} | Pos: {pos} | Neg: {neg}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, center, label = self.samples[idx]
        vol = normalize_hu(np.load(os.path.join(NPY_DIR, uid + ".npy")))
        patch = extract_patch(vol, center)
        return torch.from_numpy(patch).unsqueeze(0), torch.tensor(float(label))
