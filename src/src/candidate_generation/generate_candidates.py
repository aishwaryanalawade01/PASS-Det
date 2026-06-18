import os
import json
import torch
import numpy as np
from tqdm import tqdm

from src.models.stage1_cnn import Stage1CNN
from src.datasets.patch_dataset import normalize_hu
from src.candidate_generation.sliding_window import sliding_window_3d


# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.2

NPY_DIR = "/home/AishwaryaNalawade/data/npy"
OUT_DIR = "/home/AishwaryaNalawade/lung_project/candidates_gt"
MODEL_PATH = "/home/AishwaryaNalawade/lung_project/stage1_gt.pt"

os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# LOAD MODEL (ONCE)
# =========================
model = Stage1CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# =========================
# PROCESS ONE VOLUME
# =========================
@torch.no_grad()
def process_volume(npy_file: str):
    uid = npy_file.replace(".npy", "")
    out_path = os.path.join(OUT_DIR, f"{uid}.json")

    # Resume-safe
    if os.path.exists(out_path):
        return

    vol = np.load(os.path.join(NPY_DIR, npy_file))
    vol = normalize_hu(vol)

    candidates = []

    for patch, center in sliding_window_3d(vol):
        x = (
            torch.from_numpy(patch)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        prob = torch.sigmoid(model(x)).item()

        if prob >= THRESHOLD:
            candidates.append({
                "z": int(center[0]),
                "y": int(center[1]),
                "x": int(center[2]),
                "score": float(prob),
            })

    # Keep top-K candidates by score
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    candidates = candidates[:50]

    # Persist candidates
    with open(out_path, "w") as f:
        json.dump(candidates, f)

# =========================
# RUN OVER DATASET
# =========================
def main():
    files = sorted(os.listdir(NPY_DIR))

    for fname in tqdm(files, desc="Stage-1 candidate generation"):
        if not fname.endswith(".npy"):
            continue
        process_volume(fname)


if __name__ == "__main__":
    main()

