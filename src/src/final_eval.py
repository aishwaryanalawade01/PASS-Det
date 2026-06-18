"""
Final Evaluation Script
=======================
Computes all metrics needed for a PhD thesis on lung nodule detection:

  1. Stage-2 patch-level:
       - ROC-AUC, PR-AUC, Accuracy, F1, Precision, Recall
       - Confusion matrix

  2. Detection-level (FROC):
       - Sensitivity @ 0.5, 1, 2, 4, 8 FP/scan
       - CPM (Competition Performance Metric = mean sensitivity at these FP rates)
       - FROC curve saved as PNG

  3. Stage-1 summary (from saved candidates):
       - Mean / median / min / max candidates per scan

Usage:
    python -m src.final_eval

Outputs (all in results/):
    evaluation_report.json      -- all metrics in one JSON
    froc_curve.png              -- FROC curve plot
    confusion_matrix.png        -- Stage-2 confusion matrix
    roc_curve.png               -- ROC curve
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for headless VMs)
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve,
)
from tqdm import tqdm

from src.datasets.stage2_dataset import Stage2Dataset
from src.models.stage1_cnn import Stage1CNN

# ─── Config ──────────────────────────────────────────────────────────────────
STAGE2_PT = "stage2_refiner_best.pt"
CAND_DIR = "/home/AishwaryaNalawade/lung_project/candidates"
OUTPUT_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_EVAL_BATCHES = 500        # cap evaluation (~16k patches max)
FP_RATES = [0.5, 1, 2, 4, 8]  # standard FROC FP/scan rates
# ─────────────────────────────────────────────────────────────────────────────


class Stage2Refiner(Stage1CNN):
    pass


def load_model():
    model = Stage2Refiner().to(DEVICE)
    if os.path.exists(STAGE2_PT):
        model.load_state_dict(torch.load(STAGE2_PT, map_location=DEVICE))
        print(f"  Loaded Stage-2 weights from {STAGE2_PT}")
    else:
        print(f"  [WARNING] {STAGE2_PT} not found — model uses random weights!")
    model.eval()
    return model


# ─── 1. Stage-2 Patch-Level Evaluation ───────────────────────────────────────

@torch.no_grad()
def evaluate_stage2(model, ds):
    """Run Stage-2 model on the dataset and collect predictions + labels."""
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    all_probs = []
    all_labels = []
    total_batches = 0

    for x, y in tqdm(dl, desc="Stage-2 Eval"):
        x = x.to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()
        labels = y.numpy()

        if np.ndim(probs) == 0:
            probs = np.array([probs.item()])
        if np.ndim(labels) == 0:
            labels = np.array([labels.item()])

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

        total_batches += 1
        if total_batches >= MAX_EVAL_BATCHES:
            print(f"  (Evaluation capped at {MAX_EVAL_BATCHES} batches = {len(all_labels)} samples)")
            break

    return np.array(all_probs), np.array(all_labels)


def compute_patch_metrics(probs, labels, threshold=0.5):
    preds_bin = (probs >= threshold).astype(int)
    labels_int = labels.astype(int)

    auc = roc_auc_score(labels_int, probs) if len(np.unique(labels_int)) > 1 else float("nan")
    pr_auc = average_precision_score(labels_int, probs) if len(np.unique(labels_int)) > 1 else float("nan")
    f1 = f1_score(labels_int, preds_bin, zero_division=0)
    cm = confusion_matrix(labels_int, preds_bin)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(len(labels), 1)
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        precision = recall = accuracy = float("nan")

    return {
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "total_samples": len(labels),
        "positive_samples": int(labels.sum()),
        "negative_samples": int((1 - labels).sum()),
    }


def plot_roc(probs, labels, out_dir):
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels.astype(int), probs)
    auc = roc_auc_score(labels.astype(int), probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stage-2 Refiner – ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve saved: {path}")


def plot_confusion_matrix(probs, labels, out_dir, threshold=0.5):
    preds_bin = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels.astype(int), preds_bin)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Stage-2 Refiner – Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {path}")


# ─── 2. FROC Curve ────────────────────────────────────────────────────────────

def compute_froc(detections_per_scan, gt_per_scan, fp_rates=FP_RATES):
    """
    Compute FROC sensitivity at given FP/scan rates.

    detections_per_scan: dict uid -> list of {"z", "y", "x", "stage2_score"}
    gt_per_scan:         dict uid -> list of (z, y, x) GT nodule centers
    fp_rates:            list of FP/scan values at which to report sensitivity

    Returns: dict with sensitivities, CPM, FP/scan values for FROC curve.
    """
    all_scores = []
    all_is_tp = []
    n_scans = len(detections_per_scan)
    n_gt_total = sum(len(v) for v in gt_per_scan.values())

    if n_gt_total == 0:
        print("  [FROC] No GT nodules found — FROC requires ground truth annotations.")
        return None

    MATCH_RADIUS = 15  # voxels

    for uid, dets in detections_per_scan.items():
        gt_list = gt_per_scan.get(uid, [])

        for det in dets:
            det_center = np.array([det["z"], det["y"], det["x"]])
            score = det.get("stage2_score", det.get("score", 0.5))
            is_tp = 0

            for gt_center in gt_list:
                if np.linalg.norm(det_center - np.array(gt_center)) < MATCH_RADIUS:
                    is_tp = 1
                    break

            all_scores.append(score)
            all_is_tp.append(is_tp)

    all_scores = np.array(all_scores)
    all_is_tp = np.array(all_is_tp)

    # Sort by descending score
    sort_idx = np.argsort(-all_scores)
    all_scores = all_scores[sort_idx]
    all_is_tp = all_is_tp[sort_idx]

    # Accumulate TP and FP
    cum_tp = np.cumsum(all_is_tp)
    cum_fp = np.cumsum(1 - all_is_tp)

    sensitivity = cum_tp / max(n_gt_total, 1)
    fp_per_scan = cum_fp / max(n_scans, 1)

    # FROC sensitivities at requested FP rates
    froc_sensitivities = {}
    for target_fp in fp_rates:
        idx = np.searchsorted(fp_per_scan, target_fp)
        if idx >= len(sensitivity):
            idx = len(sensitivity) - 1
        froc_sensitivities[str(target_fp)] = round(float(sensitivity[idx]), 4)

    cpm = round(np.mean(list(froc_sensitivities.values())), 4)

    return {
        "fp_per_scan": fp_per_scan.tolist(),
        "sensitivity": sensitivity.tolist(),
        "froc_at_fp_rates": froc_sensitivities,
        "cpm": cpm,
        "n_scans": n_scans,
        "n_gt_nodules": n_gt_total,
        "n_total_detections": len(all_scores),
    }


def plot_froc(froc_result, out_dir):
    if froc_result is None:
        return

    fp = np.array(froc_result["fp_per_scan"])
    sens = np.array(froc_result["sensitivity"])
    cpm = froc_result["cpm"]

    plt.figure(figsize=(7, 5))
    plt.plot(fp, sens, lw=2, label=f"FROC (CPM = {cpm:.3f})")

    # Annotate FP rate points
    for fp_rate_str, sensitivity in froc_result["froc_at_fp_rates"].items():
        fp_rate = float(fp_rate_str)
        plt.plot(fp_rate, sensitivity, "ro", markersize=6)
        plt.annotate(
            f"  {sensitivity:.2f}",
            xy=(fp_rate, sensitivity),
            fontsize=8, color="red",
        )

    plt.xlabel("Average False Positives per Scan")
    plt.ylabel("Sensitivity")
    plt.title("FROC Curve – Lung Nodule Detection Pipeline")
    plt.xlim([0, 10])
    plt.ylim([0, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "froc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  FROC curve saved: {path}")


# ─── 3. Stage-1 Candidate Stats ───────────────────────────────────────────────

def compute_candidate_stats(cand_dir):
    counts = []
    for f in os.listdir(cand_dir):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(cand_dir, f)) as fp:
            cands = json.load(fp)
        counts.append(len(cands))

    if not counts:
        return {}

    counts = np.array(counts)
    return {
        "volumes_processed": len(counts),
        "mean_candidates_per_scan": round(float(counts.mean()), 2),
        "median_candidates_per_scan": float(np.median(counts)),
        "min_candidates_per_scan": int(counts.min()),
        "max_candidates_per_scan": int(counts.max()),
        "scans_with_zero_candidates": int((counts == 0).sum()),
    }


# ─── 4. Build GT from labels.json ─────────────────────────────────────────────

def load_gt(labels_json_path):
    """Load ground truth nodule centers from labels.json."""
    if not os.path.exists(labels_json_path):
        print(f"  [WARNING] labels.json not found at {labels_json_path}")
        return {}

    with open(labels_json_path) as fp:
        labels = json.load(fp)

    gt = {}
    for item in labels:
        if item.get("diameter") is None:
            continue
        uid = item["seriesuid"]
        gt.setdefault(uid, []).append(
            (item["coordZ"], item["coordY"], item["coordX"])
        )

    print(f"  GT nodules loaded: {sum(len(v) for v in gt.values())} nodules from {len(gt)} scans")
    return gt


def load_detections(results_dir):
    """Load per-scan detection JSONs produced by inference_pipeline.py."""
    combined = os.path.join(results_dir, "all_detections.json")
    if os.path.exists(combined):
        with open(combined) as fp:
            all_dets = json.load(fp)

        by_scan = {}
        for d in all_dets:
            uid = d.get("uid", "unknown")
            by_scan.setdefault(uid, []).append(d)
        return by_scan

    # Fall back: individual files
    by_scan = {}
    for f in os.listdir(results_dir):
        if not f.endswith("_detections.json"):
            continue
        uid = f.replace("_detections.json", "")
        with open(os.path.join(results_dir, f)) as fp:
            by_scan[uid] = json.load(fp)

    return by_scan


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report = {}

    print("\n" + "=" * 60)
    print("  LUNG NODULE DETECTION — FINAL EVALUATION")
    print("=" * 60)

    # ── Stage-1 Candidate Stats ─────────────────────────────────
    print("\n[1/4] Stage-1 Candidate Statistics...")
    if os.path.isdir(CAND_DIR):
        cand_stats = compute_candidate_stats(CAND_DIR)
        report["stage1_candidate_stats"] = cand_stats
        for k, v in cand_stats.items():
            print(f"  {k}: {v}")
    else:
        print(f"  [SKIP] CAND_DIR not found: {CAND_DIR}")
        report["stage1_candidate_stats"] = {}

    # ── Stage-2 Patch-Level Metrics ─────────────────────────────
    print("\n[2/4] Stage-2 Patch-Level Evaluation...")
    model = load_model()
    ds = Stage2Dataset()

    if len(ds) > 0:
        probs, labels = evaluate_stage2(model, ds)
        patch_metrics = compute_patch_metrics(probs, labels)
        report["stage2_patch_metrics"] = patch_metrics

        print(f"\n  ── Stage-2 Patch Metrics ──────────────────────────")
        for k, v in patch_metrics.items():
            print(f"  {k}: {v}")

        plot_roc(probs, labels, OUTPUT_DIR)
        plot_confusion_matrix(probs, labels, OUTPUT_DIR)
    else:
        print("  [SKIP] Stage2Dataset is empty.")
        report["stage2_patch_metrics"] = {}

    # ── FROC Evaluation ──────────────────────────────────────────
    print("\n[3/4] FROC / Detection-Level Evaluation...")
    LABELS_JSON = "/home/AishwaryaNalawade/data/meta/labels.json"
    gt_per_scan = load_gt(LABELS_JSON)
    detections_per_scan = load_detections(OUTPUT_DIR)

    if gt_per_scan and detections_per_scan:
        froc_result = compute_froc(detections_per_scan, gt_per_scan)
        if froc_result:
            report["froc"] = {k: v for k, v in froc_result.items()
                              if k not in ("fp_per_scan", "sensitivity")}

            print(f"\n  ── FROC Results ───────────────────────────────────")
            print(f"  CPM: {froc_result['cpm']}")
            for fp_str, sens in froc_result["froc_at_fp_rates"].items():
                print(f"  Sensitivity @ {fp_str} FP/scan: {sens}")

            plot_froc(froc_result, OUTPUT_DIR)
    else:
        msg = "(No GT nodules or no detections found — run inference_pipeline.py first)"
        print(f"  [SKIP] FROC: {msg}")
        report["froc"] = {"note": msg}

    # ── Summary ──────────────────────────────────────────────────
    print("\n[4/4] Saving evaluation report...")
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"  Report saved: {report_path}")

    # ── Print Thesis-Ready Summary ────────────────────────────────
    print("\n" + "=" * 60)
    print("  THESIS-READY METRICS SUMMARY")
    print("=" * 60)

    sm = report.get("stage2_patch_metrics", {})
    if sm:
        print(f"  Stage-2 ROC-AUC:    {sm.get('roc_auc', 'N/A')}")
        print(f"  Stage-2 PR-AUC:     {sm.get('pr_auc', 'N/A')}")
        print(f"  Stage-2 F1:         {sm.get('f1', 'N/A')}")
        print(f"  Stage-2 Precision:  {sm.get('precision', 'N/A')}")
        print(f"  Stage-2 Recall:     {sm.get('recall', 'N/A')}")

    froc = report.get("froc", {})
    if "cpm" in froc:
        print(f"\n  CPM (FROC):         {froc['cpm']}")
        for fp_str, sens in froc.get("froc_at_fp_rates", {}).items():
            print(f"  Sens @ {fp_str} FP/scan:  {sens}")

    cs = report.get("stage1_candidate_stats", {})
    if cs:
        print(f"\n  Stage-1 scans:      {cs.get('volumes_processed', 'N/A')}")
        print(f"  Mean candidates:    {cs.get('mean_candidates_per_scan', 'N/A')}")

    print("\n  Output files:")
    for fname in ["evaluation_report.json", "roc_curve.png",
                  "confusion_matrix.png", "froc_curve.png"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        exists = "✔" if os.path.exists(fpath) else "✗"
        print(f"  {exists}  results/{fname}")

    print("\n✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()
