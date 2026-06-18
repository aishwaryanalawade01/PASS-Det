# PASS-Det: GT-Supervised Two-Stage 3D CNN for Pulmonary Nodule Detection

## Overview
PASS-Det (Pulmonary nodule Attention-Supervised Sliding-window Detector) is a two-stage 3D CNN pipeline for pulmonary nodule detection on LIDC-IDRI under single-GPU constraints.

**Key contribution:** Identification, characterisation, and correction of boundary bias in weakly-supervised candidate generation.

## Results
- ROC-AUC: 0.868
- CPM: 0.194 (GT-supervised) vs 0.053 (weakly-supervised)
- Sensitivity @ 8 FP/scan: 0.468
- GT candidate coverage: 79.9% within 50 voxels

## Dataset
LIDC-IDRI / LUNA16 (888 CT volumes). Download from: https://luna16.grand-challenge.org

## Repository Structure
- src/ — all training, inference, and evaluation scripts
- meta/ — annotations and GT voxel coordinates

## Requirements
Python 3.10, PyTorch, SimpleITK, NumPy, SciPy, scikit-learn

## Paper
Under review at Diagnostics (MDPI).

## Authors
Aishwarya Nalawade, Dr. S. A. Patil
DKTE Society's Textile and Engineering Institute, Ichalkaranji, India
