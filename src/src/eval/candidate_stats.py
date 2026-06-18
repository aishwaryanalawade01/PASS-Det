import os
import json
import numpy as np

CAND_DIR = "/home/AishwaryaNalawade/lung_project/candidates"

counts = []

for f in os.listdir(CAND_DIR):
    if not f.endswith(".json"):
        continue

    with open(os.path.join(CAND_DIR, f)) as fp:
        cands = json.load(fp)

    counts.append(len(cands))

counts = np.array(counts)

print("Volumes processed:", len(counts))
print("Mean candidates:", counts.mean())
print("Median candidates:", np.median(counts))
print("Min candidates:", counts.min())
print("Max candidates:", counts.max())
