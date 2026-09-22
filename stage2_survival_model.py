# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler


CONFIG = {
    "baseline_csv": "data/baseline.csv",
    "fold_manifest": "outputs/folds/outer_folds.csv",
    "feature_dir": "outputs/stage1/strict_features",
    "save_dir": "outputs/stage2",
    "feat_dim": 1536,
    "spatial_dim": 8,
    "use_std": True,
    "top_k": 128,
    "n_outer_folds": 5,
    "epochs": 180,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "dropout": 0.40,
    "seed": 42,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canonical_group(x: object) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    if s in {"train", "training"}:
        return "train"
    if s in {"val", "valid", "validation"}:
        return "validation"
    if s in {"test", "external test", "externaltest", "external"}:
        return "test"
    return s


def protected_subject_key(subject_id: object) -> str:
    """Derived artifacts store only keyed hashes, never raw subject identifiers."""
    secret = os.environ.get("PIPELINE_LINK_KEY")
    if not secret:
        raise RuntimeError("Set PIPELINE_LINK_KEY before generating or loading linked artifacts.")
    raw = str(subject_id).strip().encode("utf-8")
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


seed_everything(CONFIG["seed"])


# ==============================================================================
# Fold manifest
# ==============================================================================

def load_baseline(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "group", "time", "recurrence"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline CSV missing columns: {sorted(missing)}")
    df = df.copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    if df["ID"].duplicated().any():
        raise ValueError("Duplicate subject identifiers were found.")
    df["group"] = df["group"].map(canonical_group)
    if set(df["group"]) - {"train", "validation", "test"}:
        raise ValueError("Unrecognized cohort labels were found.")
    df["time"] = df["time"].astype(float)
    df["recurrence"] = df["recurrence"].astype(int)
    if np.any(~np.isfinite(df["time"])) or np.any(df["time"] <= 0):
        raise ValueError("Follow-up times must be finite and positive.")
    if not set(df["recurrence"]).issubset({0, 1}):
        raise ValueError("Event indicators must be 0/1.")
    df["link_key"] = df["ID"].map(protected_subject_key)
    if df["link_key"].duplicated().any():
        raise ValueError("Duplicate protected linkage keys were generated.")
    return df


def make_outer_fold_manifest() -> None:
    """
    Create patient-level outer folds using Training subjects only.

    The manifest contains protected linkage keys and fold numbers only; no raw
    identifiers, outcomes, follow-up times, or cohort counts are written.
    """
    baseline = load_baseline(CONFIG["baseline_csv"])
    train = baseline[baseline["group"] == "train"].reset_index(drop=True)
    if len(train) < CONFIG["n_outer_folds"]:
        raise ValueError("Too few Training subjects for the requested number of folds.")
    if len(np.unique(train["recurrence"])) != 2:
        raise ValueError("Training data must contain both event classes.")

    splitter = StratifiedKFold(
        n_splits=CONFIG["n_outer_folds"],
        shuffle=True,
        random_state=CONFIG["seed"],
    )
    fold = np.full(len(train), -1, dtype=int)
    for k, (_, hold_idx) in enumerate(splitter.split(np.zeros(len(train)), train["recurrence"])):
        fold[hold_idx] = k
    if np.any(fold < 0):
        raise AssertionError("Some Training subjects were not assigned to an outer fold.")

    out = pd.DataFrame({"link_key": train["link_key"], "outer_fold": fold})
    path = Path(CONFIG["fold_manifest"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")


def load_fold_manifest(baseline: pd.DataFrame) -> pd.DataFrame:
    path = Path(CONFIG["fold_manifest"])
    if not path.is_file():
        raise FileNotFoundError("Outer-fold manifest not found.")
    m = pd.read_csv(path, dtype={"link_key": str})
    required = {"link_key", "outer_fold"}
    missing = required.difference(m.columns)
    if missing:
        raise ValueError(f"Fold manifest missing columns: {sorted(missing)}")
    m["link_key"] = m["link_key"].astype(str).str.strip()
    m["outer_fold"] = m["outer_fold"].astype(int)
    if m["link_key"].duplicated().any():
        raise ValueError("Duplicate keys in fold manifest.")

    train_keys = set(baseline.loc[baseline["group"] == "train", "link_key"])
    if set(m["link_key"]) != train_keys:
        raise ValueError("Fold manifest must contain exactly the Training subjects.")
    if set(m["outer_fold"]) != set(range(CONFIG["n_outer_folds"])):
        raise ValueError("Fold manifest does not contain the expected fold indices.")
    return m


# ==============================================================================
# Fold-specific Stage-1 features -> patient representation
# ==============================================================================

def build_patient_matrix(feature_file: Path, baseline: pd.DataFrame) -> np.ndarray:
    data = np.load(feature_file, allow_pickle=True).item()
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("Feature file must contain a non-empty dictionary.")

    expected_keys = list(baseline["link_key"])
    if set(data.keys()) != set(expected_keys):
        raise ValueError("Feature records do not match the baseline subjects exactly.")

    rows: List[np.ndarray] = []
    for key in expected_keys:
        item = data[key]
        feats = np.asarray(item["feats"], dtype=np.float32)
        spatial = np.asarray(item["spatial"], dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != CONFIG["feat_dim"]:
            raise ValueError("Unexpected imaging feature shape.")
        if spatial.ndim != 2 or spatial.shape[1] != CONFIG["spatial_dim"]:
            raise ValueError("Unexpected spatial feature shape.")
        if feats.shape[0] != spatial.shape[0] or feats.shape[0] < 1:
            raise ValueError("Invalid plaque instance counts.")
        if not np.isfinite(feats).all() or not np.isfinite(spatial).all():
            raise ValueError("Non-finite feature values were found.")

        parts = [feats.mean(axis=0), feats.max(axis=0)]
        if CONFIG["use_std"]:
            parts.append(feats.std(axis=0))
        parts.append(spatial.sum(axis=0))
        rows.append(np.concatenate(parts).astype(np.float32))

    X = np.stack(rows, axis=0)
    expected_dim = CONFIG["feat_dim"] * (3 if CONFIG["use_std"] else 2) + CONFIG["spatial_dim"]
    if X.shape[1] != expected_dim:
        raise AssertionError("Unexpected patient representation dimension.")
    return X


# ==============================================================================
# Survival-specific feature screening
# ==============================================================================

def cox_score_statistics(X: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int64)
    p = X.shape[1]
    U = np.zeros(p, dtype=np.float64)
    I = np.zeros(p, dtype=np.float64)

    for t in np.sort(np.unique(time[event == 1])):
        event_mask = (time == t) & (event == 1)
        risk_mask = time >= t
        d = int(event_mask.sum())
        if d == 0:
            continue
        X_risk = X[risk_mask]
        mean_risk = X_risk.mean(axis=0)
        second_risk = (X_risk ** 2).mean(axis=0)
        var_risk = np.maximum(second_risk - mean_risk ** 2, 0.0)
        U += X[event_mask].sum(axis=0) - d * mean_risk
        I += d * var_risk

    z = np.abs(U) / np.sqrt(I + 1e-12)
    z[~np.isfinite(z)] = 0.0
    return z


def select_top_k_features(
    X_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    scores = cox_score_statistics(X_train, time_train, event_train)
    k = min(int(k), X_train.shape[1])
    idx = np.argsort(scores)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int64), scores


# ==============================================================================
# DeepSurv
# ==============================================================================

class ResBlock(nn.Module):
    def __init__(self, dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DeepSurv(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
        )
        self.res1 = ResBlock(64, CONFIG["dropout"])
        self.res2 = ResBlock(64, CONFIG["dropout"])
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x).squeeze(1)


class CoxBreslowLoss(nn.Module):
    def forward(self, log_risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        event_times = torch.sort(torch.unique(time[event > 0.5]))[0]
        pll = torch.zeros((), dtype=log_risk.dtype, device=log_risk.device)
        n_events = torch.clamp(event.sum(), min=1.0)
        for t in event_times:
            event_mask = (time == t) & (event > 0.5)
            risk_mask = time >= t
            d = event_mask.sum().to(log_risk.dtype)
            pll = pll + log_risk[event_mask].sum()
            pll = pll - d * torch.logsumexp(log_risk[risk_mask], dim=0)
        return -pll / n_events


def train_one_fold(X: np.ndarray, time: np.ndarray, event: np.ndarray, seed: int) -> DeepSurv:
    """Fixed-epoch training; outer holdout data are never used for model selection."""
    seed_everything(seed)
    device = get_device()
    model = DeepSurv(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    loss_fn = CoxBreslowLoss()
    x = torch.tensor(X, dtype=torch.float32, device=device)
    t = torch.tensor(time, dtype=torch.float32, device=device)
    e = torch.tensor(event, dtype=torch.float32, device=device)

    for _ in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(x)
        loss = loss_fn(risk, t, e)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Cox loss encountered.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.tensor(X, dtype=torch.float32, device=device)
    return model(x).detach().cpu().numpy().astype(np.float64)


# ==============================================================================
# Calibration and metrics
# ==============================================================================

def harrell_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    concordant = 0.0
    comparable = 0.0
    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            if time[i] == time[j]:
                continue
            if time[i] < time[j] and event[i] == 1:
                comparable += 1
                concordant += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
            elif time[j] < time[i] and event[j] == 1:
                comparable += 1
                concordant += 1.0 if risk[j] > risk[i] else 0.5 if risk[i] == risk[j] else 0.0
    return float(concordant / comparable) if comparable else float("nan")


def fit_censoring_km(time: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, float)
    censor_event = 1 - np.asarray(event, int)
    surv = 1.0
    step_times = [0.0]
    surv_values = [1.0]
    for t in np.sort(np.unique(time)):
        at_risk = int(np.sum(time >= t))
        d = int(np.sum((time == t) & (censor_event == 1)))
        if at_risk > 0 and d > 0:
            surv *= (1.0 - d / at_risk)
        step_times.append(float(t))
        surv_values.append(float(surv))
    return np.asarray(step_times), np.asarray(surv_values)


def km_value(step_times: np.ndarray, surv_values: np.ndarray, t: float, left_limit: bool = False) -> float:
    side = "left" if left_limit else "right"
    idx = np.searchsorted(step_times, t, side=side) - 1
    idx = int(np.clip(idx, 0, len(surv_values) - 1))
    return float(max(surv_values[idx], 1e-8))


def cumulative_dynamic_auc_ipcw(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
    censor_km: Tuple[np.ndarray, np.ndarray],
) -> float:
    risk = np.asarray(risk, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    step_times, surv_values = censor_km
    cases = np.where((event == 1) & (time <= horizon))[0]
    controls = np.where(time > horizon)[0]
    if len(cases) == 0 or len(controls) == 0:
        return float("nan")
    total = 0.0
    weights = 0.0
    control_risk = risk[controls]
    for i in cases:
        w = 1.0 / km_value(step_times, surv_values, float(time[i]), left_limit=True)
        score = (np.sum(risk[i] > control_risk) + 0.5 * np.sum(risk[i] == control_risk)) / len(controls)
        total += w * score
        weights += w
    return float(total / weights) if weights > 0 else float("nan")


def ipcw_brier_score(
    prob_event: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
    censor_km: Tuple[np.ndarray, np.ndarray],
) -> float:
    p = np.asarray(prob_event, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    step_times, surv_values = censor_km
    contrib = np.zeros(len(time), float)
    g_t = km_value(step_times, surv_values, horizon)
    for i in range(len(time)):
        if event[i] == 1 and time[i] <= horizon:
            w = 1.0 / km_value(step_times, surv_values, float(time[i]), left_limit=True)
            y = 1.0
        elif time[i] > horizon:
            w = 1.0 / g_t
            y = 0.0
        else:
            w = 0.0
            y = 0.0
        contrib[i] = w * (y - p[i]) ** 2
    return float(np.mean(contrib))


def fit_cox_calibration_slope(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> Tuple[float, float, float]:
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    score = np.asarray(score, float)
    mean = float(score.mean())
    sd = float(score.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        raise ValueError("Training OOF score SD is invalid.")
    z = (score - mean) / sd

    def neg_pll(beta: float) -> float:
        lp = beta * z
        value = 0.0
        for t in np.sort(np.unique(time[event == 1])):
            event_mask = (time == t) & (event == 1)
            risk_mask = time >= t
            d = int(event_mask.sum())
            risk_lp = lp[risk_mask]
            m = float(risk_lp.max())
            value += float(lp[event_mask].sum()) - d * (m + math.log(float(np.exp(risk_lp - m).sum())))
        return -value

    result = minimize_scalar(neg_pll, bounds=(-5.0, 5.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("Cox recalibration failed.")
    return float(result.x), mean, sd


def breslow_baseline_hazard(
    time: np.ndarray, event: np.ndarray, linear_predictor: np.ndarray, horizons: np.ndarray
) -> Tuple[np.ndarray, float]:
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    lp = np.asarray(linear_predictor, float)
    center = float(lp.mean())
    rr = np.exp(np.clip(lp - center, -50.0, 50.0))
    cumulative = 0.0
    event_times = [0.0]
    hazards = [0.0]
    for t in np.sort(np.unique(time[event == 1])):
        risk = time >= t
        d = int(np.sum((time == t) & (event == 1)))
        denom = float(rr[risk].sum())
        if denom <= 0 or not np.isfinite(denom):
            raise RuntimeError("Invalid Breslow denominator.")
        cumulative += d / denom
        event_times.append(float(t))
        hazards.append(float(cumulative))
    event_times = np.asarray(event_times)
    hazards = np.asarray(hazards)
    idx = np.searchsorted(event_times, horizons, side="right") - 1
    idx = np.clip(idx, 0, len(hazards) - 1)
    return hazards[idx], center


def absolute_event_probability(
    baseline_hazard: np.ndarray, linear_predictor: np.ndarray, lp_center: float
) -> np.ndarray:
    rr = np.exp(np.clip(np.asarray(linear_predictor, float) - lp_center, -50.0, 50.0))
    return 1.0 - np.exp(-np.outer(rr, baseline_hazard))


# ==============================================================================
# Leakage-free fold-aligned pipeline
# ==============================================================================

def run_strict_pipeline(horizons: List[float]) -> None:
    baseline = load_baseline(CONFIG["baseline_csv"])
    manifest = load_fold_manifest(baseline).set_index("link_key")
    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    group = baseline["group"].to_numpy()
    T = baseline["time"].to_numpy(float)
    E = baseline["recurrence"].to_numpy(int)
    keys = baseline["link_key"].to_numpy(str)
    idx_train = np.where(group == "train")[0]
    idx_val = np.where(group == "validation")[0]
    idx_test = np.where(group == "test")[0]

    train_fold = np.asarray([int(manifest.loc[k, "outer_fold"]) for k in keys[idx_train]], dtype=int)
    oof_risk = np.full(len(idx_train), np.nan, dtype=float)
    val_scores: List[np.ndarray] = []
    test_scores: List[np.ndarray] = []

    for fold in range(CONFIG["n_outer_folds"]):
        feature_file = Path(CONFIG["feature_dir"]) / f"strict_features_fold{fold}.npy"
        if not feature_file.is_file():
            raise FileNotFoundError("A required fold-specific Stage-1 feature file was not found.")
        X_all = build_patient_matrix(feature_file, baseline)
        X_train = X_all[idx_train]
        X_val = X_all[idx_val]
        X_test = X_all[idx_test]

        fit_rel = np.where(train_fold != fold)[0]
        hold_rel = np.where(train_fold == fold)[0]
        if len(fit_rel) == 0 or len(hold_rel) == 0:
            raise RuntimeError("Invalid outer fold partition.")

        # Every learned transformation below is fitted on outer-fit patients only.
        scaler = RobustScaler()
        X_fit_sc = scaler.fit_transform(X_train[fit_rel])
        X_hold_sc = scaler.transform(X_train[hold_rel])
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        selected, selector_scores = select_top_k_features(
            X_fit_sc, T[idx_train][fit_rel], E[idx_train][fit_rel], CONFIG["top_k"]
        )
        X_fit_sel = X_fit_sc[:, selected]
        X_hold_sel = X_hold_sc[:, selected]
        X_val_sel = X_val_sc[:, selected]
        X_test_sel = X_test_sc[:, selected]

        model = train_one_fold(
            X_fit_sel,
            T[idx_train][fit_rel],
            E[idx_train][fit_rel],
            seed=CONFIG["seed"] + fold,
        )
        fit_raw = predict_model(model, X_fit_sel)
        hold_raw = predict_model(model, X_hold_sel)
        val_raw = predict_model(model, X_val_sel)
        test_raw = predict_model(model, X_test_sel)

        risk_mean = float(fit_raw.mean())
        risk_sd = float(fit_raw.std(ddof=0))
        if not np.isfinite(risk_sd) or risk_sd < 1e-12:
            raise RuntimeError("Fold risk SD is invalid.")

        hold_z = (hold_raw - risk_mean) / risk_sd
        val_z = (val_raw - risk_mean) / risk_sd
        test_z = (test_raw - risk_mean) / risk_sd
        oof_risk[hold_rel] = hold_z
        val_scores.append(val_z)
        test_scores.append(test_z)

        checkpoint = {
            "outer_fold": int(fold),
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "selected_features": selected,
            "selector_scores_selected": selector_scores[selected],
            "risk_mean_on_outer_fit": risk_mean,
            "risk_sd_on_outer_fit": risk_sd,
            "note": "All learned operations were fitted using outer-fit Training patients only.",
        }
        torch.save(checkpoint, save_dir / f"deepsurv_outer_fold{fold}.pth")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_risk).any():
        raise AssertionError("Some Training subjects do not have an OOF risk score.")

    val_risk = np.mean(np.stack(val_scores, axis=0), axis=0)
    test_risk = np.mean(np.stack(test_scores, axis=0), axis=0)

    # Development-only calibration from strict Training OOF scores.
    T_train = T[idx_train]
    E_train = E[idx_train]
    beta, oof_mean, oof_sd = fit_cox_calibration_slope(T_train, E_train, oof_risk)
    train_lp = beta * ((oof_risk - oof_mean) / oof_sd)
    val_lp = beta * ((val_risk - oof_mean) / oof_sd)
    test_lp = beta * ((test_risk - oof_mean) / oof_sd)

    hz = np.asarray(horizons, dtype=float)
    h0, lp_center = breslow_baseline_hazard(T_train, E_train, train_lp, hz)
    train_prob = absolute_event_probability(h0, train_lp, lp_center)
    val_prob = absolute_event_probability(h0, val_lp, lp_center)
    test_prob = absolute_event_probability(h0, test_lp, lp_center)
    cutoff = float(np.median(train_lp))

    all_risk = np.full(len(baseline), np.nan, float)
    all_lp = np.full(len(baseline), np.nan, float)
    all_prob = np.full((len(baseline), len(hz)), np.nan, float)
    all_risk[idx_train], all_risk[idx_val], all_risk[idx_test] = oof_risk, val_risk, test_risk
    all_lp[idx_train], all_lp[idx_val], all_lp[idx_test] = train_lp, val_lp, test_lp
    all_prob[idx_train], all_prob[idx_val], all_prob[idx_test] = train_prob, val_prob, test_prob

    pred = pd.DataFrame({"link_key": keys, "group": group, "risk_score": all_risk, "calibrated_lp": all_lp})
    for j, h in enumerate(hz):
        pred[f"risk_{h:g}"] = all_prob[:, j]
    pred["risk_group"] = np.where(all_lp > cutoff, "High", "Low")
    pred.to_csv(save_dir / "survival_predictions.csv", index=False, encoding="utf-8")

    # Evaluation-specific censoring distributions are estimated within each cohort.
    metric_rows = []
    for name, idx in [("Training", idx_train), ("Validation", idx_val), ("Test", idx_test)]:
        cohort_time = T[idx]
        cohort_event = E[idx]
        cohort_risk = all_risk[idx]
        cohort_prob = all_prob[idx]
        censor_km = fit_censoring_km(cohort_time, cohort_event)
        row: Dict[str, float | str] = {
            "Cohort": name,
            "C_index": harrell_c_index(cohort_risk, cohort_time, cohort_event),
        }
        for j, h in enumerate(hz):
            row[f"AUC_{h:g}"] = cumulative_dynamic_auc_ipcw(
                cohort_risk, cohort_time, cohort_event, float(h), censor_km
            )
            row[f"Brier_{h:g}"] = ipcw_brier_score(
                cohort_prob[:, j], cohort_time, cohort_event, float(h), censor_km
            )
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(save_dir / "survival_metrics.csv", index=False, encoding="utf-8")

    calibration = {
        "source": "strict Training OOF risk scores only",
        "cox_recalibration_beta_per_1SD": beta,
        "oof_score_mean": oof_mean,
        "oof_score_sd": oof_sd,
        "breslow_lp_center": lp_center,
        "horizons": hz.tolist(),
        "baseline_cumulative_hazard": h0.tolist(),
        "risk_cutoff_training_median_lp": cutoff,
    }
    with open(save_dir / "training_only_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free fold-aligned patient-level survival modeling.")
    parser.add_argument("--mode", choices=["make-folds", "run"], required=True)
    parser.add_argument("--baseline", type=str, default=CONFIG["baseline_csv"])
    parser.add_argument("--fold-manifest", type=str, default=CONFIG["fold_manifest"])
    parser.add_argument("--feature-dir", type=str, default=CONFIG["feature_dir"])
    parser.add_argument("--save-dir", type=str, default=CONFIG["save_dir"])
    parser.add_argument("--horizons", type=float, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG["baseline_csv"] = args.baseline
    CONFIG["fold_manifest"] = args.fold_manifest
    CONFIG["feature_dir"] = args.feature_dir
    CONFIG["save_dir"] = args.save_dir
    if args.mode == "make-folds":
        make_outer_fold_manifest()
    elif args.mode == "run":
        if not args.horizons:
            raise ValueError("--horizons is required in run mode.")
        run_strict_pipeline(args.horizons)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
