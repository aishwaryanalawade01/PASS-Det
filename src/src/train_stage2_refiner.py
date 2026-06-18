"""
Stage-2 Refiner Training Script
================================
Train a CNN that takes Stage-1 candidate patches and predicts true nodule vs FP.

Usage (from project root):
    tmux new -s stage2
    source ~/lungenv/bin/activate
    cd ~/lung_project
    python -m src.train_stage2_refiner

Expected runtime: ~45-90 minutes on T4 GPU.
Expected loss:    Step 0 ~1.1 --> Step 3000 ~0.2-0.4
Checkpoint saved: stage2_refiner.pt  (every 500 steps + end of training)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.datasets.stage2_dataset import Stage2Dataset
from src.models.stage1_cnn import Stage1CNN

# ─── Config ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "stage2_refiner.pt"
BATCH_SIZE = 8
LR = 1e-4
MAX_STEPS = 500
SAVE_EVERY = 500
LOG_EVERY = 20
NUM_WORKERS = 0
# ─────────────────────────────────────────────────────────────────────────────


class Stage2Refiner(Stage1CNN):
    """
    Stage-2 refiner: same architecture as Stage-1 CNN but trained on
    candidate patches with pseudo-labels to reduce false positives.
    Inherits Stage1CNN so we can optionally warm-start from stage1.pt.
    """
    pass


def make_weighted_sampler(dataset):
    """Create sampler that up-samples positives to balance training."""
    labels = [s[2] for s in dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0:
        print("[WARNING] No positive samples found — check pseudo-label thresholds.")
        return None

    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train():
    print(f"[train_stage2_refiner] Using device: {DEVICE}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds = Stage2Dataset()

    if len(ds) == 0:
        print("[ERROR] Dataset is empty. Check CAND_DIR and NPY_DIR paths.")
        return

    sampler = make_weighted_sampler(ds)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Stage2Refiner().to(DEVICE)

    # Optional: warm-start from Stage-1 weights
    stage1_path = "stage1_gt.pt"
    if os.path.exists(stage1_path):
        model.load_state_dict(torch.load(stage1_path, map_location=DEVICE))
        print(f"[train_stage2_refiner] Warm-started from {stage1_path}")
    else:
        print("[train_stage2_refiner] Training from scratch (stage1.pt not found).")

    # ── Loss: BCEWithLogitsLoss with positive weight to handle class imbalance ─
    n_pos = sum(1 for s in ds.samples if s[2] == 1)
    n_neg = sum(1 for s in ds.samples if s[2] == 0)
    pos_weight_val = n_neg / max(n_pos, 1)
    print(f"[train_stage2_refiner] pos_weight: {pos_weight_val:.1f} "
          f"(pos={n_pos}, neg={n_neg})")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val]).to(DEVICE)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_STEPS, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    steps = 0
    best_loss = float("inf")

    print(f"\n[train_stage2_refiner] Starting training for {MAX_STEPS} steps...\n")

    data_iter = iter(dl)
    pbar = tqdm(total=MAX_STEPS, desc="Stage-2 Training")

    while steps < MAX_STEPS:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, y = next(data_iter)

        x = x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        pbar.set_postfix(loss=f"{loss_val:.4f}")
        pbar.update(1)

        if steps % LOG_EVERY == 0:
            tqdm.write(f"Step {steps:4d} | Loss {loss_val:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

        if steps % SAVE_EVERY == 0 and steps > 0:
            torch.save(model.state_dict(), CHECKPOINT)
            tqdm.write(f"  → Checkpoint saved to {CHECKPOINT}")

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), "stage2_refiner_best.pt")

        steps += 1

    pbar.close()

    # Final save
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"\n[train_stage2_refiner] Training complete.")
    print(f"  Final checkpoint: {CHECKPOINT}")
    print(f"  Best checkpoint:  stage2_refiner_best.pt (loss={best_loss:.4f})")


if __name__ == "__main__":
    train()
