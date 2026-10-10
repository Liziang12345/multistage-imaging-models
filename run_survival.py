"""Private-data runner for fold-aligned survival analysis.

Supported configurations:
    --estimator neural --representation multi   (joint-plaque DeepSurv)
    --estimator ridge  --representation multi   (L2-penalized Cox comparator)
    --estimator neural --representation culprit (single-plaque DeepSurv)

The repository contains code only. Supply all patient data and fold inputs locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold

from representation import (
    MODEL_DIM, load_metadata, read_manifest, read_fold_features,
    fit_image_projection, patient_matrix, culprit_patient_matrix,
    load_culprit_mapping, fit_selector, transform_selector,
)
from models import (
    fit_neural, predict_neural, fit_ridge_cox, nested_select_alpha,
    concordance,
)


def build_rows(records, keys, projection, representation, culprit_map=None):
    if representation == "multi":
        return patient_matrix(records, keys, projection, representation)
    if culprit_map is None:
        raise ValueError("Single-plaque representation requires a predefined index map.")
    return culprit_patient_matrix(records, keys, projection, culprit_map)


def fit_calibration(t, status, risk):
    """Development OOF Cox slope calibration; no external cohort is used."""
    center = risk.mean()
    scale = risk.std(ddof=0)
    if scale <= 1e-12 or not np.isfinite(scale):
        raise ValueError("Non-variable OOF risk scores.")
    x = (risk - center) / scale

    def negative_log_likelihood(coefficient):
        lp = coefficient * x
        total = 0.0
        for event_time in np.unique(t[status == 1]):
            case = (t == event_time) & (status == 1)
            at_risk = t >= event_time
            scores = lp[at_risk]
            top = max(scores)
            total -= lp[case].sum() - case.sum() * (top + np.log(np.exp(scores - top).sum()))
        return total

    result = minimize_scalar(negative_log_likelihood, bounds=(-5, 5), method="bounded")
    if not result.success:
        raise RuntimeError("Development calibration failed.")
    return float(result.x), float(center), float(scale)


def breslow_hazard(t, status, lp, horizons):
    centered = float(lp.mean())
    rr = np.exp(np.clip(lp - centered, -50, 50))
    times, cumulative = [0.], [0.]
    val = 0.
    for et in np.unique(t[status == 1]):
        den = rr[t >= et].sum()
        val += np.sum((t == et) & (status == 1)) / den
        times.append(float(et))
        cumulative.append(float(val))
    index = np.clip(np.searchsorted(times, horizons, side="right") - 1, 0, len(times)-1)
    return np.asarray(cumulative)[index], centered


def censoring_km(time, event):
    unique = np.sort(np.unique(time))
    times, prob = [0.], [1.]
    survive = 1.
    for t in unique:
        count = (time >= t).sum()
        censored = ((time == t) & (event == 0)).sum()
        if censored:
            survive *= 1. - censored / count
        times.append(float(t))
        prob.append(float(survive))
    return np.array(times), np.array(prob)


def km_lookup(km, time, before=False):
    x, y = km
    where = np.searchsorted(x, time, side="left" if before else "right") - 1
    return max(float(y[max(0, where)]), 1e-8)


def evaluation_time_copy(time, event, horizon, administrative_end, epsilon):
    """Adjust endpoint administrative censoring only for metric evaluation.

    The observed event times and the training data are never modified.
    """
    result = np.asarray(time, dtype=float).copy()
    status = np.asarray(event, dtype=int)
    if administrative_end is None or not np.isclose(horizon, administrative_end, atol=1e-9):
        return result
    mask = (status == 0) & np.isclose(result, administrative_end, atol=1e-9)
    result[mask] = float(administrative_end) + float(epsilon)
    return result


def dynamic_auc(score, time, event, horizon, km):
    cases = np.flatnonzero((time <= horizon) & (event == 1))
    controls = np.flatnonzero(time > horizon)
    if not len(cases) or not len(controls):
        return float("nan")
    weights, values = [], []
    for i in cases:
        weights.append(1. / km_lookup(km, time[i], before=True))
        values.append((np.sum(score[i] > score[controls]) + .5 * np.sum(score[i] == score[controls])) / len(controls))
    return float(np.average(values, weights=weights))


def brier_score(prob, time, event, horizon, km):
    out = np.zeros(len(time), dtype=float)
    survival_at_horizon = km_lookup(km, horizon)
    for i in range(len(time)):
        if event[i] == 1 and time[i] <= horizon:
            out[i] = (1. - prob[i]) ** 2 / km_lookup(km, time[i], before=True)
        elif time[i] > horizon:
            out[i] = prob[i] ** 2 / survival_at_horizon
    return float(out.mean())


def make_folds(args, metadata):
    train = metadata[metadata._split == "train"]
    events = train._event.to_numpy(int)
    if min(np.bincount(events, minlength=2)) < args.folds:
        raise ValueError("Not enough events for the specified outer folds.")
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    assignments = np.full(len(train), -1, dtype=int)
    for fold, (_, hold) in enumerate(cv.split(np.zeros(len(events)), events)):
        assignments[hold] = fold
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"link_key": train.link_key.to_numpy(), "outer_fold": assignments}).to_csv(args.manifest, index=False)


def run(args):
    if args.admin_epsilon <= 0 or not np.isfinite(args.admin_epsilon):
        raise ValueError("Administrative boundary increment must be positive.")
    metadata = load_metadata(args.metadata, args.id_col, args.split_col, args.time_col, args.event_col)
    if args.action == "make-folds":
        make_folds(args, metadata)
        print("Protected fold manifest prepared in the private workspace.")
        return
    keys = metadata.link_key.tolist()
    splits = metadata._split.to_numpy()
    time = metadata._time.to_numpy(float)
    event = metadata._event.to_numpy(int)
    index_train = np.flatnonzero(splits == "train")
    index_val = np.flatnonzero(splits == "validation")
    index_test = np.flatnonzero(splits == "test")
    manifest = read_manifest(args.manifest, set(metadata.iloc[index_train].link_key))
    assignment = np.asarray([manifest[keys[i]] for i in index_train], dtype=int)
    folds = sorted(set(assignment))
    culprit_map = None
    if args.representation == "culprit":
        if not args.culprit_map:
            raise ValueError("A private culprit index map is required.")
        culprit_map = load_culprit_mapping(args.culprit_map, keys)
    # Map restricted to development subjects; only used for nested validation.
    times_by_key = dict(zip(keys, time))
    events_by_key = dict(zip(keys, event))
    oof = np.full(len(index_train), np.nan, dtype=float)
    val_scores, test_scores = [], []
    fold_details = []
    args.output.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        file_path = Path(args.features_template.format(fold=fold))
        records = read_fold_features(file_path, keys)
        fit_rel = np.flatnonzero(assignment != fold)
        hold_rel = np.flatnonzero(assignment == fold)
        fit_idx, hold_idx = index_train[fit_rel], index_train[hold_rel]
        fit_keys = [keys[i] for i in fit_idx]
        projector = fit_image_projection(records, fit_keys, args.seed + fold)
        candidate = build_rows(records, keys, projector, args.representation, culprit_map)
        scaler, selected = fit_selector(candidate, fit_idx, time, event)
        x_fit = transform_selector(candidate[fit_idx], scaler, selected)
        x_hold = transform_selector(candidate[hold_idx], scaler, selected)
        x_val = transform_selector(candidate[index_val], scaler, selected)
        x_test = transform_selector(candidate[index_test], scaler, selected)
        if args.estimator == "neural":
            model = fit_neural(x_fit, time[fit_idx], event[fit_idx], seed=args.seed + fold,
                               epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                               dropout=args.dropout, device=args.device)
            predict = lambda x: predict_neural(model, x)
            penalty = None
        else:
            def make_inner_matrix(rec, inner_keys, inner_pca, repr_name):
                return build_rows(rec, inner_keys, inner_pca, repr_name, culprit_map)
            penalty = nested_select_alpha(
                records, keys, fit_keys, times_by_key, events_by_key,
                args.representation, make_inner_matrix, args.seed + fold,
                candidate_alphas=args.alphas, n_inner=args.inner_folds,
            )
            coefficients = fit_ridge_cox(x_fit.astype(float), time[fit_idx], event[fit_idx], penalty)
            predict = lambda x: np.asarray(x, float) @ coefficients
        fit_risk = predict(x_fit)
        risk_mean, risk_sd = float(np.mean(fit_risk)), float(np.std(fit_risk, ddof=0))
        if not np.isfinite(risk_sd) or risk_sd < 1e-12:
            raise RuntimeError("Degenerate development risk scores.")
        oof[hold_rel] = (predict(x_hold) - risk_mean) / risk_sd
        val_scores.append((predict(x_val) - risk_mean) / risk_sd)
        test_scores.append((predict(x_test) - risk_mean) / risk_sd)
        fold_details.append({"fold": int(fold), "selected_joint_features": selected.tolist(),
                             "penalty": penalty})
        print(f"Completed outer fold {fold}.")
    if np.isnan(oof).any():
        raise RuntimeError("Some development subjects have no held-out prediction.")
    val_risk = np.mean(val_scores, axis=0) if len(index_val) else np.array([])
    test_risk = np.mean(test_scores, axis=0) if len(index_test) else np.array([])
    all_risk = np.full(len(keys), np.nan)
    all_risk[index_train] = oof
    all_risk[index_val] = val_risk
    all_risk[index_test] = test_risk
    beta, center, scale = fit_calibration(time[index_train], event[index_train], oof)
    linear = beta * (all_risk - center) / scale
    horizons = np.asarray(args.horizons, dtype=float)
    baseline_hazard, lp_center = breslow_hazard(time[index_train], event[index_train],
                                               linear[index_train], horizons)
    risks = 1. - np.exp(-np.outer(np.exp(np.clip(linear - lp_center, -50, 50)), baseline_hazard))
    cutoff = float(np.median(linear[index_train]))
    prediction = pd.DataFrame({"link_key": keys, "cohort": splits,
                               "risk_score": all_risk, "calibrated_lp": linear,
                               "risk_group": np.where(linear > cutoff, "High", "Low")})
    for j, h in enumerate(horizons):
        prediction[f"risk_at_{h:g}"] = risks[:, j]
    prediction.to_csv(args.output / "private_predictions.csv", index=False)
    rows = []
    for name, idx in (("train", index_train), ("validation", index_val), ("test", index_test)):
        if len(idx) == 0:
            continue
        metric = {"cohort": name, "c_index": concordance(all_risk[idx], time[idx], event[idx])}
        for j, h in enumerate(horizons):
            metric_time = evaluation_time_copy(time[idx], event[idx], h,
                                                args.admin_end, args.admin_epsilon)
            metric_km = censoring_km(metric_time, event[idx])
            metric[f"auc_{h:g}"] = dynamic_auc(all_risk[idx], metric_time, event[idx], h, metric_km)
            metric[f"brier_{h:g}"] = brier_score(risks[idx, j], metric_time, event[idx], h, metric_km)
        rows.append(metric)
    pd.DataFrame(rows).to_csv(args.output / "private_metrics.csv", index=False)
    with (args.output / "private_model_info.json").open("w", encoding="utf-8") as f:
        json.dump({"estimator": args.estimator, "representation": args.representation,
                   "outer_folds": fold_details, "train_oof_calibration": {
                       "slope": beta, "mean": center, "sd": scale,
                       "baseline_hazard": baseline_hazard.tolist(),
                       "linear_center": lp_center, "threshold": cutoff}}, f, indent=2)
    print("Private predictions, metrics and model settings saved locally.")


def make_parser():
    parser = argparse.ArgumentParser(description="Fold-aligned image/anatomy survival modeling")
    parser.add_argument("--action", choices=("make-folds", "run"), required=True)
    parser.add_argument("--estimator", choices=("neural", "ridge"), default="neural")
    parser.add_argument("--representation", choices=("multi", "culprit"), default="multi")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("private/outer_folds.csv"))
    parser.add_argument("--features-template", default="private/features/fold_{fold}.npy")
    parser.add_argument("--culprit-map", type=Path)
    parser.add_argument("--output", type=Path, default=Path("private/outputs"))
    parser.add_argument("--id-col", default="subject_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--time-col", default="duration")
    parser.add_argument("--event-col", default="event")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--horizons", type=float, nargs="+", default=[12., 24., 36.])
    parser.add_argument("--admin-end", type=float, default=36.0)
    parser.add_argument("--admin-epsilon", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1., 10.])
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
