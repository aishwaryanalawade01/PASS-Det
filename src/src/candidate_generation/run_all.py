import os
from src.candidate_generation.generate_candidates import process_volume

DATA_DIR = "/home/AishwaryaNalawade/data/npy"
OUT_DIR = "/home/AishwaryaNalawade/lung_project/candidates"

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(os.listdir(DATA_DIR))

for i, f in enumerate(files):
    uid = f.replace(".npy", "")
    out_file = os.path.join(OUT_DIR, uid + ".json")

    if os.path.exists(out_file):
        continue

    print(f"[{i+1}/{len(files)}] Processing {uid}")
    process_volume(f)

    # safety: stop early for testing
    if i == 49:   # first 50 scans only
        break
