import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DATA_NPY_DIR, LABELS_JSON, PATCH_SIZE


def normalize_hu(vol):
    vol = np.clip(vol, -1000, 400)
    vol = (vol + 1000) / 1400.0
    return vol.astype(np.float32)


class LungPatchDataset(Dataset):
    def __init__(self, npy_dir=DATA_NPY_DIR, allowed_uids=None):
        self.npy_dir = npy_dir
        self.patch_size = PATCH_SIZE

        # Load all volumes
        self.files = []
        for f in os.listdir(npy_dir):
            if not f.endswith(".npy"):
                continue
            uid = f.replace(".npy", "")
            if allowed_uids is None or uid in allowed_uids:
                self.files.append(f)

        assert len(self.files) > 0, "No files found for this split"


        # Load labels
        with open(LABELS_JSON, "r") as f:
            raw_labels = json.load(f)

        # Group labels by seriesuid
        self.labels = {}
        for item in raw_labels:
            uid = item["seriesuid"]
            self.labels.setdefault(uid, []).append(item)

    def __len__(self):
        # arbitrary large length since we sample randomly
        return len(self.files) * 10

    def _extract_patch(self, vol, center):
        ps = self.patch_size
        zc, yc, xc = center
        z1 = max(0, zc - ps // 2)
        y1 = max(0, yc - ps // 2)
        x1 = max(0, xc - ps // 2)

        z2 = min(vol.shape[0], z1 + ps)
        y2 = min(vol.shape[1], y1 + ps)
        x2 = min(vol.shape[2], x1 + ps)

        patch = np.zeros((ps, ps, ps), dtype=np.float32)
        patch[: z2 - z1, : y2 - y1, : x2 - x1] = vol[z1:z2, y1:y2, x1:x2]
        return patch

    def __getitem__(self, idx):
        # random volume
        fname = random.choice(self.files)
        uid = fname.replace(".npy", "")

        vol = np.load(os.path.join(self.npy_dir, fname))
        vol = normalize_hu(vol)

        nodules = self.labels.get(uid, [])

        # 50% positive / negative
        if random.random() < 0.5 and len(nodules) > 0:
            n = random.choice(nodules)
            center = (
                int(n["coordZ"]) % vol.shape[0],
                int(n["coordY"]) % vol.shape[1],
                int(n["coordX"]) % vol.shape[2],
            )
            label = 1.0
        else:
            center = (
                random.randint(0, vol.shape[0] - 1),
                random.randint(0, vol.shape[1] - 1),
                random.randint(0, vol.shape[2] - 1),
            )
            label = 0.0

        patch = self._extract_patch(vol, center)
        patch = torch.from_numpy(patch).unsqueeze(0)

        return patch, torch.tensor(label, dtype=torch.float32)
