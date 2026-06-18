import os
import json
import numpy as np
from tqdm import tqdm

NPY_DIR = "/home/AishwaryaNalawade/data/npy"
CAND_DIR = "/home/AishwaryaNalawade/lung_project/candidates"
LABELS_JSON = "/home/AishwaryaNalawade/data/meta/labels.json"


def world_to_voxel(coord):
    """
    Convert approximate world coordinate to voxel coordinate.
    Since preprocessing removed spacing/origin metadata,
    we approximate using center of volume assumption.
    """

    # LIDC scans roughly 512x512
    x = int(coord[0] + 256)
    y = int(coord[1] + 256)

    # z spacing smaller
    z = int(coord[2] + 200)

    return (z, y, x)


def distance(a, b):
    return np.sqrt(
        (a[0] - b["z"]) ** 2 +
        (a[1] - b["y"]) ** 2 +
        (a[2] - b["x"]) ** 2
    )


def evaluate_recall():

    with open(LABELS_JSON) as f:
        labels = json.load(f)

    labels_by_uid = {}

    for item in labels:
        uid = item["seriesuid"]

        labels_by_uid.setdefault(uid, []).append(
            world_to_voxel((item["coordZ"], item["coordY"], item["coordX"]))
        )

    total = 0
    hit_5mm = 0
    hit_10mm = 0

    for uid in tqdm(labels_by_uid):

        cand_file = os.path.join(CAND_DIR, f"{uid}.json")

        if not os.path.exists(cand_file):
            continue

        with open(cand_file) as f:
            cands = json.load(f)

        if len(cands) == 0:
            continue

        for gt in labels_by_uid[uid]:

            dists = [distance(gt, c) for c in cands]

            if len(dists) == 0:
                continue

            total += 1

            if min(dists) <= 5:
                hit_5mm += 1

            if min(dists) <= 10:
                hit_10mm += 1

    print("\nGT nodules evaluated:", total)

    if total > 0:
        print(f"Recall @5mm:  {hit_5mm / total:.3f}")
        print(f"Recall @10mm: {hit_10mm / total:.3f}")
    else:
        print("No GT nodules matched.")


if __name__ == "__main__":
    evaluate_recall()
