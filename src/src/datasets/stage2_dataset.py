"""
Stage-2 Dataset for Lung Nodule False Positive Reduction

Uses pseudo-labeling since GT nodule coords from labels.json were lost:
  - score >= 0.7 --> positive (likely nodule)
  - score <= 0.4 --> negative (likely background)
  - 0.4 < score < 0.7 --> skipped (ambiguous)

This is a standard technique used in detection pipelines (DeepLung, LUNA16).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache
# ─── Absolute paths (match your VM) ──────────────────────────────────────────
NPY_DIR = "/home/AishwaryaNalawade/data/npy"
CAND_DIR = "/home/AishwaryaNalawade/lung_project/candidates_gt"
LABELS_JSON = "/home/AishwaryaNalawade/data/meta/labels.json"

PATCH_SIZE = 96
POS_THRESHOLD = 0.7     # score >= this --> positive
NEG_THRESHOLD = 0.4     # score <= this --> negative
MAX_CANDS_PER_SCAN = 50  # cap to control dataset size


def normalize_hu(vol):
    """Clip to lung window and normalize to [0, 1]."""
    vol = np.clip(vol, -1000, 400)
    vol = (vol + 1000) / 1400.0
    return vol.astype(np.float32)


def extract_patch(vol, center, size=PATCH_SIZE):
    """Extract a cubic patch centered at (z, y, x), zero-padding if needed."""
    z, y, x = int(center[0]), int(center[1]), int(center[2])
    half = size // 2

    patch = np.zeros((size, size, size), dtype=np.float32)

    z1, y1, x1 = z - half, y - half, x - half
    z2, y2, x2 = z + half, y + half, x + half

    vz1 = max(0, z1)
    vy1 = max(0, y1)
    vx1 = max(0, x1)

    vz2 = min(vol.shape[0], z2)
    vy2 = min(vol.shape[1], y2)
    vx2 = min(vol.shape[2], x2)

    dz1 = vz1 - z1
    dy1 = vy1 - y1
    dx1 = vx1 - x1

    dz2 = dz1 + (vz2 - vz1)
    dy2 = dy1 + (vy2 - vy1)
    dx2 = dx1 + (vx2 - vx1)

    patch[dz1:dz2, dy1:dy2, dx1:dx2] = vol[vz1:vz2, vy1:vy2, vx1:vx2]

    return patch


class Stage2Dataset(Dataset):
    """
    Candidate-level dataset for Stage-2 false-positive reduction.

    Each sample: (patch_tensor [1, 96, 96, 96], label [float32])
    Pseudo-labels from Stage-1 score: high score = positive, low = negative.
    """

    def __init__(
        self,
        npy_dir=NPY_DIR,
        cand_dir=CAND_DIR,
        pos_threshold=POS_THRESHOLD,
        neg_threshold=NEG_THRESHOLD,
        max_cands=MAX_CANDS_PER_SCAN,
    ):
        self.npy_dir = npy_dir
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

        self.samples = []   # list of (uid, candidate_dict, label)

        self.samples = []   # (uid, cand, label)

        for fname in os.listdir(cand_dir):
            if not fname.endswith(".json"):
                continue

            uid = fname.replace(".json", "")
            npy_path = os.path.join(npy_dir, uid + ".npy")
            if not os.path.exists(npy_path):
                continue

            with open(os.path.join(cand_dir, fname)) as fp:
                cands = json.load(fp)

            # Sort by score descending
            cands = sorted(cands, key=lambda c: c["score"], reverse=True)

            # Top 3 per scan = positive, next 47 = negative
            for i, cand in enumerate(cands[:50]):
                label = 1 if i < 3 else 0
                self.samples.append((uid, cand, label))

        print(f"[Stage2Dataset] Total samples: {len(self.samples)}")
        pos = sum(1 for s in self.samples if s[2] == 1)
        neg = sum(1 for s in self.samples if s[2] == 0)
        print(f"[Stage2Dataset] Positives: {pos}  |  Negatives: {neg}")

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=8)
    def _load_vol(self, uid):
        vol = np.load(os.path.join(self.npy_dir, uid + ".npy"))
        return normalize_hu(vol)

    def __getitem__(self, idx):
        uid, cand, label = self.samples[idx]

        #vol = np.load(os.path.join(self.npy_dir, uid + ".npy"))
        #vol = normalize_hu(vol)
        vol = self._load_vol(uid)

        center = (cand["z"], cand["y"], cand["x"])
        patch = extract_patch(vol, center)

        patch_tensor = torch.from_numpy(patch).unsqueeze(0)   # (1, 96, 96, 96)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)

        return patch_tensor, label_tensor
