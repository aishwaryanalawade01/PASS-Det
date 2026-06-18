import json
import random
from src.config import LABELS_JSON

def get_train_val_uids(val_ratio=0.2, seed=42):
    random.seed(seed)

    with open(LABELS_JSON, "r") as f:
        labels = json.load(f)

    uids = sorted({item["seriesuid"] for item in labels})
    random.shuffle(uids)

    n_val = int(len(uids) * val_ratio)
    val_uids = set(uids[:n_val])
    train_uids = set(uids[n_val:])

    return train_uids, val_uids
