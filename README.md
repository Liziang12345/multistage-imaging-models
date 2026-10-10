# Multi-task imaging and survival modeling

This repository contains generic training and evaluation source code only. It contains no subject-level records, images, labels, feature arrays, learned weights, aggregate results, or access credentials.

## Structure

| File | Purpose |
| --- | --- |
| `stage1_multitask.py` | Paired-volume multi-task training, evaluation, threshold locking, and fold-isolated feature export |
| `baseline_backbones.py` | Common backbone interface for four two-dimensional comparison encoders |
| `baseline_resnet50.py` | ResNet-50 comparison entry point |
| `baseline_densenet121.py` | DenseNet-121 comparison entry point |
| `baseline_convnextv2_tiny.py` | ConvNeXt V2-Tiny comparison entry point |
| `baseline_vit_base_patch16_224.py` | ViT-B/16 comparison entry point |
| `representation.py` | Per-instance image/anatomy interactions and patient-level aggregation |
| `models.py` | Neural Cox and L2-penalized Cox estimators |
| `run_survival.py` | Fold-isolated survival fitting, calibration, and evaluation |

## Imaging models

The primary encoder uses a shared three-dimensional Swin backbone on paired image volumes. The comparison encoders use shared two-dimensional feature extraction over all slices of each volume, preserving the depth axis for subsequent pooling. Therefore, these are **backbone replacement comparisons**, not claims that two-dimensional and three-dimensional backbones have identical internal operations.

All five models use the same pre/post branch sharing, 4 x 7 x 7 token output, bidirectional cross-attention, subtraction-guided branch, task-specific heads, phenotype-logit relay, multitask loss, grouped cross-validation protocol, and threshold-locking routine. In the comparison encoders, a learned feature projection standardizes the token width. Comparison models are initialized with their own published pretrained backbone parameters.

The primary model exports fold-specific 1,536-dimensional representations for downstream processing. Feature export requires valid anatomical category columns; missing values are not silently substituted. Four comparison entry points are intended for the classification benchmark and are not used as substitutes for the fold-specific primary encoder in the downstream pipeline.

### Entry points

Install dependencies in a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

The same command modes are available for the primary model and each comparison script:

```bash
python stage1_multitask.py --mode train-classifier
python stage1_multitask.py --mode lock-thresholds
python stage1_multitask.py --mode evaluate

python baseline_resnet50.py --mode train-classifier
python baseline_densenet121.py --mode train-classifier
python baseline_convnextv2_tiny.py --mode train-classifier
python baseline_vit_base_patch16_224.py --mode train-classifier
```

The comparison entry points also accept `--mode evaluate`. Their private checkpoint and evaluation directories are separated by backbone. Optional flags include `--plaque-csv`, `--subject-csv`, `--pre-dir`, `--post-dir`, `--diff-dir`, `--batch-size`, `--num-workers`, and `--frame-chunk`.

The training CSV must supply the column names required by `stage1_multitask.py`. Input arrays must be stored locally as preprocessed NIfTI volumes. Preprocessing and interpretation of labels are determined by the user's local data protocol; the repository neither supplies nor generates any records. Data augmentation is applied to training images only.

## Fold-isolated survival pipeline

The survival runner expects a locally maintained metadata CSV containing a unique subject identifier, partition label, follow-up duration, and event indicator. The identifier must match `ID` in the Stage-1 subject table, or the user must supply appropriate `--id-col`/input-column mappings. Both scripts must use the same `PIPELINE_LINK_KEY` and outer-fold manifest. Partitions may be named `train`, `val`/`validation`, and `test`. Each subject has the same partition across feature files.

Before linking files, set `PIPELINE_LINK_KEY` in the execution environment. Do not put the secret in source code, command history, issue reports, or version-controlled files. The keyed digest is a linkage mechanism, not a replacement for secure storage of private datasets.

```bash
python run_survival.py --action make-folds --metadata private/metadata.csv
python stage1_multitask.py --mode train-strict-encoders
python stage1_multitask.py --mode extract-strict-features
python run_survival.py --action run --metadata private/metadata.csv \
    --features-template 'outputs/stage1/strict_features/strict_features_fold{fold}.npy' \
    --manifest private/outer_folds.csv --estimator neural --representation multi
```

For the comparison estimators, change `--estimator ridge --representation multi` or `--estimator neural --representation culprit`. The single-instance model additionally requires a locally supplied protected-key-to-index table passed using `--culprit-map`. The code does not infer that index from outcomes.

All fold-specific representation fitting, feature selection, and estimator optimization are restricted to the corresponding development fold. Only fold-specific risk scores are averaged on independent partitions. Predictor calibration is derived from development out-of-fold scores.

### Administrative evaluation boundary

The survival evaluator supports `--admin-end` and `--admin-epsilon` (defaults 36 and 0.001). At the administrative endpoint, the copied evaluation time is increased only for records censored at that exact endpoint. Events at the endpoint remain cases. Original follow-up times, model fitting, risk calibration, and censoring data used for risk estimation remain unchanged. This convention is not applied to other horizons.

### Local feature-file trust

The upstream feature export uses a NumPy dictionary that requires pickle-aware loading. Treat such files as executable content. Load only local files produced by a trusted upstream run; never open third-party or downloaded feature dictionaries.

## Publication and privacy checks

Only the code files, this README, `.gitignore`, and `requirements.txt` belong in a public commit. The ignored input and output paths must remain private. Verify staged changes and git history before publishing; ignore rules do not remove previously committed files. The code does not include previously reported performance values and must be trained on real authorized data before reporting empirical performance.
