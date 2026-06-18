import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.datasets.patch_dataset import LungPatchDataset
from src.datasets.split import get_train_val_uids
from src.models.stage1_cnn import Stage1CNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def evaluate(model, loader, max_batches=30):
    model.eval()
    preds = []
    targets = []

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x).squeeze()
        prob = torch.sigmoid(logits)

        preds.extend(prob.cpu().numpy())
        targets.extend(y.cpu().numpy())

        if i % 5 == 0:
            print(f"[Eval] Batch {i}/{max_batches}")

    return roc_auc_score(targets, preds)



def main():
    train_uids, val_uids = get_train_val_uids()

    val_ds = LungPatchDataset(allowed_uids=val_uids)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = Stage1CNN().to(DEVICE)
    model.load_state_dict(torch.load("stage1.pt", map_location=DEVICE))

    auc = evaluate(model, val_dl)
    print(f"Validation ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
