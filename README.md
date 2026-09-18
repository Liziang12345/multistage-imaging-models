# Multi-stage Imaging Models

Research code for multi-stage medical imaging representation learning and survival modeling.

## Overview

This repository contains a two-stage deep learning framework for volumetric medical imaging analysis.

The pipeline consists of:

1. **Stage 1 — Plaque-level multi-task representation learning**
   - Shared-weight 3D Swin Transformer encoders
   - Bidirectional token-level cross-attention
   - Subtraction-guided feature integration
   - KAN-based multi-task classification
   - Patient-level grouped cross-validation

2. **Stage 2 — Patient-level survival modeling**
   - Aggregation of multiple imaging representations
   - Training-only preprocessing and feature selection
   - Cross-fitted DeepSurv modeling
   - Time-dependent risk estimation

## Repository Structure

```text
.
├── stage1_multitask_model.py
├── stage2_survival_model.py
├── .gitignore
└── README.md
```

## Stage 1

`stage1_multitask_model.py` implements the imaging representation learning pipeline.

Paired volumetric inputs are processed using a shared-weight 3D Swin Transformer. Bidirectional cross-attention is used for feature interaction, followed by subtraction-guided feature integration and multi-task prediction.

Cross-validation is performed at the patient level to prevent samples from the same patient from appearing in both training and internal validation subsets.

## Stage 2

`stage2_survival_model.py` implements patient-level survival modeling.

Imaging representations from multiple instances are aggregated into a patient-level feature vector. Feature preprocessing and selection are fitted exclusively on the training data before DeepSurv modeling.

Held-out datasets are not used for model fitting, feature selection, or preprocessing.

## Requirements

The main dependencies include:

- Python
- PyTorch
- torchvision
- timm
- NumPy
- pandas
- scikit-learn
- SciPy
- nibabel

## Data

No patient-level data, medical images, labels, model checkpoints, or identifiable information are included in this repository.

Users should configure their own local data paths before running the scripts.

## Privacy

This repository contains source code only.

Clinical data, medical images, patient identifiers, intermediate features, trained model weights, and local configuration files are excluded from version control.

## Intended Use

The code is provided for research and reproducibility purposes.
