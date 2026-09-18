# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
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


# ==============================================================================
# 0. CONFIG
# ==============================================================================

CONFIG = {
    "feature_path": "data/features.npy",
    "baseline_csv": "data/baseline.csv",
    "save_dir": "outputs",

    "feat_dim": 1536,
    "spatial_dim": 8,
    "use_std": True,
    "top_k": 128,

    "n_inner_folds": 5,
    "epochs": 180,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "dropout": 0.40,
    "seed": 42,

    "horizons": None,

}


# ==============================================================================
# 1. Reproducibility / group names
# ==============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_group(x: object) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    if s in {"train", "training"}:
        return "train"
    if s in {"val", "valid", "validation"}:
        return "validation"
    if s in {"test", "external test", "externaltest", "external"}:
        return "test"
    return s


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def protected_subject_key(subject_id: object) -> str:
    secret = os.environ.get("PIPELINE_LINK_KEY")
    if not secret:
        raise RuntimeError(
            "Set PIPELINE_LINK_KEY in the local environment before loading features."
        )
    raw = str(subject_id).strip().encode("utf-8")
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


seed_everything(CONFIG["seed"])


# ==============================================================================
# 2. Load and construct patient-level multi-plaque representation
# ==============================================================================

def build_patient_table(
    feature_path: str, baseline_csv: str
) -> Tuple[pd.DataFrame, np.ndarray]:
    data = np.load(feature_path, allow_pickle=True).item()
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("patient feature file must be a non-empty dict.")

    baseline = pd.read_csv(baseline_csv, dtype={"ID": str})
    required = {"ID", "group", "time", "recurrence"}
    missing = required.difference(baseline.columns)
    if missing:
        raise ValueError(f"Baseline CSV missing columns: {sorted(missing)}")

    baseline["ID"] = baseline["ID"].astype(str).str.strip()
    if baseline["ID"].duplicated().any():
        raise ValueError("Duplicate subject identifiers were found in the baseline table.")
    baseline["_link_key"] = baseline["ID"].map(protected_subject_key)
    if baseline["_link_key"].duplicated().any():
        raise ValueError("Duplicate protected linkage keys were generated.")

    baseline = baseline.set_index("_link_key", drop=False)

    rows: List[Dict[str, object]] = []
    X_list: List[np.ndarray] = []

    for link_key_raw, item in data.items():
        link_key = str(link_key_raw).strip()
        if link_key not in baseline.index:
            raise ValueError("A feature record could not be matched to the baseline table.")

        feats = np.asarray(item["feats"], dtype=np.float32)
        spatial = np.asarray(item["spatial"], dtype=np.float32)

        if feats.ndim != 2 or feats.shape[1] != CONFIG["feat_dim"]:
            raise ValueError(
                "Unexpected feature-array shape."
            )
        if spatial.ndim != 2 or spatial.shape[1] != CONFIG["spatial_dim"]:
            raise ValueError(
                "Unexpected spatial-array shape."
            )
        if feats.shape[0] != spatial.shape[0]:
            raise ValueError("Feature and spatial instance counts differ.")
        if feats.shape[0] < 1:
            raise ValueError("An empty feature record was found.")

        f_mean = feats.mean(axis=0)
        f_max = feats.max(axis=0)
        f_std = feats.std(axis=0) if CONFIG["use_std"] else np.empty(0)
        s_sum = spatial.sum(axis=0)

        patient_vec = np.concatenate([f_mean, f_max, f_std, s_sum]).astype(
            np.float32
        )

        expected_dim = (
            CONFIG["feat_dim"] * (3 if CONFIG["use_std"] else 2)
            + CONFIG["spatial_dim"]
        )
        if patient_vec.shape[0] != expected_dim:
            raise AssertionError("Unexpected patient-vector dimension.")

        brow = baseline.loc[link_key]
        group = canonical_group(brow["group"])
        time = float(brow["time"])
        event = int(brow["recurrence"])
        if not np.isfinite(time) or time <= 0:
            raise ValueError("Follow-up times must be finite and positive.")
        if event not in (0, 1):
            raise ValueError("Event indicators must be encoded as 0 or 1.")
        if not np.isfinite(feats).all() or not np.isfinite(spatial).all():
            raise ValueError("Feature arrays contain non-finite values.")

        # Cross-check values embedded in .npy if available
        if "time" in item and not np.isclose(float(item["time"]), time):
            raise ValueError("A time value is inconsistent across input files.")
        if "recurrence" in item and int(item["recurrence"]) != event:
            raise ValueError("An event value is inconsistent across input files.")

        rows.append(
            {
                "group": group,
                "time": time,
                "recurrence": event,
                "n_instances": int(feats.shape[0]),
            }
        )
        X_list.append(patient_vec)

    patient = pd.DataFrame(rows)
    X = np.stack(X_list, axis=0)

    feature_keys = {str(k).strip() for k in data.keys()}
    baseline_keys = set(baseline.index.astype(str))
    if baseline_keys.difference(feature_keys):
        raise ValueError("Some baseline records do not have corresponding feature records.")

    if set(patient["group"]) - {"train", "validation", "test"}:
        bad = sorted(set(patient["group"]) - {"train", "validation", "test"})
        raise ValueError(f"Unrecognized group values after normalization: {bad}")



    return patient, X


# ==============================================================================
# 3. Survival-specific univariate feature screening
# ==============================================================================

def cox_score_statistics(
    X: np.ndarray, time: np.ndarray, event: np.ndarray
) -> np.ndarray:
    """
    Univariate Cox score-test ranking evaluated at beta=0.

    For each feature j:
        U_j = sum_events [x_ij - mean_R(t_i)(x_j)]
        I_j = sum_events Var_R(t_i)(x_j)
        score_j = |U_j| / sqrt(I_j)

    Breslow handling is used for tied event times.

    This is performed ONLY inside an inner Training fold.
    """
    X = np.asarray(X, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.int64)

    p = X.shape[1]
    U = np.zeros(p, dtype=np.float64)
    I = np.zeros(p, dtype=np.float64)

    event_times = np.sort(np.unique(time[event == 1]))

    for t in event_times:
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
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = cox_score_statistics(X_train, time_train, event_train)
    k = min(int(k), X_train.shape[1])
    idx = np.argsort(scores)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return idx.astype(np.int64), scores


# ==============================================================================
# 4. DeepSurv
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
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(CONFIG["dropout"]),
        )
        self.res1 = ResBlock(64, dropout=CONFIG["dropout"])
        self.res2 = ResBlock(64, dropout=CONFIG["dropout"])
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x).squeeze(1)


class CoxBreslowLoss(nn.Module):
    """
    Negative Cox partial log-likelihood with Breslow handling of tied event times.
    """

    def forward(
        self,
        log_risk: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        event_times = torch.unique(time[event > 0.5])
        event_times, _ = torch.sort(event_times)

        pll = torch.zeros((), dtype=log_risk.dtype, device=log_risk.device)
        n_events = torch.clamp(event.sum(), min=1.0)

        for t in event_times:
            event_mask = (time == t) & (event > 0.5)
            risk_mask = time >= t

            d = event_mask.sum().to(log_risk.dtype)
            pll = pll + log_risk[event_mask].sum()
            pll = pll - d * torch.logsumexp(log_risk[risk_mask], dim=0)

        return -pll / n_events


# ==============================================================================
# 5. Survival metrics
# ==============================================================================

def harrell_c_index(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray
) -> float:
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    concordant = 0.0
    comparable = 0.0

    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue

            if time[i] < time[j] and event[i] == 1:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5

            elif time[j] < time[i] and event[j] == 1:
                comparable += 1
                if risk[j] > risk[i]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5

    return float(concordant / comparable) if comparable > 0 else float("nan")


def fit_censoring_km(
    train_time: np.ndarray, train_event: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KM estimate G(t)=P(C>t) for the censoring distribution.
    Censoring is treated as the event: censor_event = 1 - recurrence.
    Returns step times and post-jump survival values.
    """
    time = np.asarray(train_time, dtype=float)
    censor_event = 1 - np.asarray(train_event, dtype=int)

    unique_times = np.sort(np.unique(time))
    surv = 1.0
    step_times = [0.0]
    surv_values = [1.0]

    for t in unique_times:
        at_risk = int(np.sum(time >= t))
        d_censor = int(np.sum((time == t) & (censor_event == 1)))
        if at_risk > 0 and d_censor > 0:
            surv *= (1.0 - d_censor / at_risk)
        step_times.append(float(t))
        surv_values.append(float(surv))

    return np.asarray(step_times), np.asarray(surv_values)


def km_value(
    step_times: np.ndarray,
    surv_values: np.ndarray,
    t: float,
    left_limit: bool = False,
) -> float:
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
    """
    IPCW cumulative/dynamic AUC:
      cases: event <= t
      controls: observed time > t
    Case weights use 1/G(T_i-). The common control weight at t cancels.
    """
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    step_times, surv_values = censor_km

    case_idx = np.where((event == 1) & (time <= horizon))[0]
    control_idx = np.where(time > horizon)[0]

    if len(case_idx) == 0 or len(control_idx) == 0:
        return float("nan")

    weighted_sum = 0.0
    total_weight = 0.0

    control_risk = risk[control_idx]

    for i in case_idx:
        g = km_value(step_times, surv_values, float(time[i]), left_limit=True)
        w = 1.0 / g

        wins = np.sum(risk[i] > control_risk)
        ties = np.sum(risk[i] == control_risk)
        pair_score = (wins + 0.5 * ties) / len(control_idx)

        weighted_sum += w * pair_score
        total_weight += w

    return float(weighted_sum / total_weight) if total_weight > 0 else float("nan")


def ipcw_brier_score(
    prob_event: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
    censor_km: Tuple[np.ndarray, np.ndarray],
) -> float:
    """
    IPCW Brier score for event probability P(T <= horizon).
    """
    p = np.asarray(prob_event, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    step_times, surv_values = censor_km

    contrib = np.zeros(len(time), dtype=float)

    g_t = km_value(step_times, surv_values, horizon, left_limit=False)

    for i in range(len(time)):
        if event[i] == 1 and time[i] <= horizon:
            g_i = km_value(
                step_times, surv_values, float(time[i]), left_limit=True
            )
            weight = 1.0 / g_i
            y = 1.0
        elif time[i] > horizon:
            weight = 1.0 / g_t
            y = 0.0
        else:
            # censored before/equal horizon -> no contribution
            weight = 0.0
            y = 0.0

        contrib[i] = weight * (y - p[i]) ** 2

    return float(np.mean(contrib))


# ==============================================================================
# 6. Model training for one internal fold
# ==============================================================================

def train_one_fold(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    input_dim: int,
    seed: int,
) -> DeepSurv:
    seed_everything(seed)
    device = get_device()

    model = DeepSurv(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"]
    )
    loss_fn = CoxBreslowLoss()

    x = torch.tensor(X_train, dtype=torch.float32, device=device)
    t = torch.tensor(time_train, dtype=torch.float32, device=device)
    e = torch.tensor(event_train, dtype=torch.float32, device=device)

    best_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        log_risk = model(x)
        loss = loss_fn(log_risk, t, e)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite Cox loss at epoch {epoch+1}.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.tensor(X, dtype=torch.float32, device=device)
    return model(x).detach().cpu().numpy().astype(np.float64)


# ==============================================================================
# 7. Training-only OOF calibration + Breslow absolute risk
# ==============================================================================

def fit_cox_calibration_slope(
    time: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Fit a one-variable Cox calibration slope on Training OOF scores.
    Returns beta, score_mean, score_sd.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    score = np.asarray(score, dtype=float)

    mean = float(score.mean())
    sd = float(score.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        raise ValueError("Training OOF score SD is zero or invalid.")

    z = (score - mean) / sd

    def neg_pll(beta: float) -> float:
        lp = beta * z
        value = 0.0

        for t in np.sort(np.unique(time[event == 1])):
            event_mask = (time == t) & (event == 1)
            risk_mask = time >= t
            d = int(event_mask.sum())

            event_sum = float(lp[event_mask].sum())
            risk_lp = lp[risk_mask]
            m = float(risk_lp.max())
            log_risk_sum = m + math.log(float(np.exp(risk_lp - m).sum()))

            value += event_sum - d * log_risk_sum

        return -value

    result = minimize_scalar(
        neg_pll,
        bounds=(-5.0, 5.0),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("Training-only Cox calibration slope fitting failed.")

    return float(result.x), mean, sd


def breslow_baseline_hazard(
    time: np.ndarray,
    event: np.ndarray,
    linear_predictor: np.ndarray,
    horizons: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Estimate H0(t) on Training only, centered at mean(linear predictor).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    lp = np.asarray(linear_predictor, dtype=float)

    center = float(lp.mean())
    rr = np.exp(np.clip(lp - center, -50.0, 50.0))

    cumulative_hazard = 0.0
    event_times = [0.0]
    hazards = [0.0]

    for t in np.sort(np.unique(time[event == 1])):
        at_risk = time >= t
        d = int(np.sum((time == t) & (event == 1)))
        denom = float(rr[at_risk].sum())
        if denom <= 0 or not np.isfinite(denom):
            raise RuntimeError("Invalid Breslow risk-set denominator.")

        cumulative_hazard += d / denom
        event_times.append(float(t))
        hazards.append(float(cumulative_hazard))

    event_times = np.asarray(event_times, dtype=float)
    hazards = np.asarray(hazards, dtype=float)
    indices = np.searchsorted(event_times, horizons, side="right") - 1
    indices = np.clip(indices, 0, len(hazards) - 1)

    return hazards[indices], center


def absolute_event_probability(
    baseline_hazard: np.ndarray,
    linear_predictor: np.ndarray,
    lp_center: float,
) -> np.ndarray:
    rr = np.exp(
        np.clip(
            np.asarray(linear_predictor, dtype=float) - lp_center,
            -50.0,
            50.0,
        )
    )
    return 1.0 - np.exp(-np.outer(rr, baseline_hazard))


# ==============================================================================
# 8. Main pipeline
# ==============================================================================

def run() -> None:
    device = get_device()
    save_dir = Path(CONFIG["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    patient, X = build_patient_table(
        CONFIG["feature_path"], CONFIG["baseline_csv"]
    )

    idx_train = np.where(patient["group"].to_numpy() == "train")[0]
    idx_val = np.where(patient["group"].to_numpy() == "validation")[0]
    idx_test = np.where(patient["group"].to_numpy() == "test")[0]

    T = patient["time"].to_numpy(float)
    E = patient["recurrence"].to_numpy(int)

    X_train = X[idx_train]
    T_train = T[idx_train]
    E_train = E[idx_train]

    X_val = X[idx_val]
    X_test = X[idx_test]

    # Training-only internal cross-fitting.
    splitter = StratifiedKFold(
        n_splits=CONFIG["n_inner_folds"],
        shuffle=True,
        random_state=CONFIG["seed"],
    )

    oof_risk = np.full(len(idx_train), np.nan, dtype=np.float64)

    # Validation/Test predictions from all five Training-fold models
    val_fold_scores = []
    test_fold_scores = []

    for fold, (fit_rel, hold_rel) in enumerate(splitter.split(X_train, E_train)):

        X_fit_raw = X_train[fit_rel]
        X_hold_raw = X_train[hold_rel]

        T_fit = T_train[fit_rel]
        E_fit = E_train[fit_rel]

        # 1) scaler fitted ONLY on this Training inner-fit subset
        scaler = RobustScaler()
        X_fit_sc = scaler.fit_transform(X_fit_raw)
        X_hold_sc = scaler.transform(X_hold_raw)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        # 2) feature selection fitted ONLY on this Training inner-fit subset
        selected_idx, selector_scores = select_top_k_features(
            X_fit_sc,
            T_fit,
            E_fit,
            k=CONFIG["top_k"],
        )

        X_fit_sel = X_fit_sc[:, selected_idx]
        X_hold_sel = X_hold_sc[:, selected_idx]
        X_val_sel = X_val_sc[:, selected_idx]
        X_test_sel = X_test_sc[:, selected_idx]

        # 3) DeepSurv fold model
        model = train_one_fold(
            X_fit_sel,
            T_fit,
            E_fit,
            input_dim=len(selected_idx),
            seed=CONFIG["seed"] + fold,
        )

        # 4) Raw log-risk predictions
        risk_fit_raw = predict_model(model, X_fit_sel)
        risk_hold_raw = predict_model(model, X_hold_sel)
        risk_val_raw = predict_model(model, X_val_sel)
        risk_test_raw = predict_model(model, X_test_sel)

        # 5) Fold-training-only score standardization.
        #    This creates a common scale across the five fold models.
        risk_mean = float(risk_fit_raw.mean())
        risk_sd = float(risk_fit_raw.std(ddof=0))
        if not np.isfinite(risk_sd) or risk_sd < 1e-12:
            raise RuntimeError(f"Fold {fold}: raw risk SD is invalid.")

        risk_hold_z = (risk_hold_raw - risk_mean) / risk_sd
        risk_val_z = (risk_val_raw - risk_mean) / risk_sd
        risk_test_z = (risk_test_raw - risk_mean) / risk_sd

        oof_risk[hold_rel] = risk_hold_z
        val_fold_scores.append(risk_val_z)
        test_fold_scores.append(risk_test_z)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "selected_features": selected_idx,
            "risk_mean_on_inner_fit": risk_mean,
            "risk_sd_on_inner_fit": risk_sd,
        }
        ckpt_path = save_dir / f"deepsurv_inner_fold{fold}.pth"
        torch.save(checkpoint, ckpt_path)



        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(oof_risk).any():
        raise AssertionError("Some Training patients have no OOF risk score.")

    val_risk = np.mean(np.stack(val_fold_scores, axis=0), axis=0)
    test_risk = np.mean(np.stack(test_fold_scores, axis=0), axis=0)

    # ------------------------------------------------------------------
    # Training-only absolute-risk calibration
    # ------------------------------------------------------------------
    beta, oof_mean, oof_sd = fit_cox_calibration_slope(
        T_train, E_train, oof_risk
    )

    train_lp = beta * ((oof_risk - oof_mean) / oof_sd)
    val_lp = beta * ((val_risk - oof_mean) / oof_sd)
    test_lp = beta * ((test_risk - oof_mean) / oof_sd)

    horizons = np.asarray(CONFIG["horizons"], dtype=float)
    h0, lp_center = breslow_baseline_hazard(
        T_train, E_train, train_lp, horizons
    )

    train_prob = absolute_event_probability(h0, train_lp, lp_center)
    val_prob = absolute_event_probability(h0, val_lp, lp_center)
    test_prob = absolute_event_probability(h0, test_lp, lp_center)

    # Risk cutoff learned from training OOF risk only.
    # Median is prespecified, stable, and does not optimize on Validation/Test.
    risk_cutoff = float(np.median(train_lp))

    # ------------------------------------------------------------------
    # Export predictions
    # ------------------------------------------------------------------
    all_risk = np.full(len(patient), np.nan, dtype=float)
    all_lp = np.full(len(patient), np.nan, dtype=float)
    all_prob = np.full((len(patient), len(horizons)), np.nan, dtype=float)

    all_risk[idx_train] = oof_risk
    all_risk[idx_val] = val_risk
    all_risk[idx_test] = test_risk

    all_lp[idx_train] = train_lp
    all_lp[idx_val] = val_lp
    all_lp[idx_test] = test_lp

    all_prob[idx_train] = train_prob
    all_prob[idx_val] = val_prob
    all_prob[idx_test] = test_prob

    if np.isnan(all_prob).any():
        raise AssertionError("Absolute risk probabilities contain NaNs.")
    if np.any((all_prob < 0) | (all_prob > 1)):
        raise AssertionError("Absolute risk probabilities outside [0,1].")

    pred = pd.DataFrame({"sample_index": np.arange(len(patient), dtype=int)})
    pred["DeepSurv_Ensemble_Risk"] = all_risk
    pred["Calibrated_LinearPredictor"] = all_lp
    for j, h in enumerate(horizons):
        pred[f"Risk_{int(h)}m"] = all_prob[:, j]
    pred["Risk_Group"] = np.where(
        pred["Calibrated_LinearPredictor"].to_numpy() > risk_cutoff,
        "High",
        "Low",
    )

    pred_path = save_dir / "survival_predictions.csv"
    pred.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    censor_km = fit_censoring_km(T_train, E_train)
    metric_rows = []

    for cohort_name, indices in [
        ("Training", idx_train),
        ("Validation", idx_val),
        ("Test", idx_test),
    ]:
        cohort_time = T[indices]
        cohort_event = E[indices]
        cohort_risk = all_risk[indices]
        cohort_prob = all_prob[indices]

        row = {
            "Cohort": cohort_name,
            "C_index": harrell_c_index(
                cohort_risk, cohort_time, cohort_event
            ),
        }

        for j, h in enumerate(horizons):
            row[f"AUC_{int(h)}m"] = cumulative_dynamic_auc_ipcw(
                cohort_risk,
                cohort_time,
                cohort_event,
                float(h),
                censor_km,
            )
            row[f"Brier_{int(h)}m"] = ipcw_brier_score(
                cohort_prob[:, j],
                cohort_time,
                cohort_event,
                float(h),
                censor_km,
            )

        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics_path = save_dir / "survival_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Save calibration / fold metadata
    # ------------------------------------------------------------------
    calibration = {
        "source": "Training OOF scores only",
        "cox_recalibration_beta_per_1SD": beta,
        "oof_score_mean": oof_mean,
        "oof_score_sd": oof_sd,
        "breslow_lp_center": lp_center,
        "horizons_months": horizons.tolist(),
        "baseline_cumulative_hazard": h0.tolist(),
        "risk_cutoff_training_median_lp": risk_cutoff,
        "note": (
            "All calibration parameters and cutoff are learned from the "
            "training cohort and applied unchanged to held-out cohorts."
        ),
    }
    with open(
        save_dir / "training_only_calibration.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)




# ==============================================================================
# 9. CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-level DeepSurv pipeline."
    )
    parser.add_argument(
        "--features", type=str, default=CONFIG["feature_path"]
    )
    parser.add_argument(
        "--baseline", type=str, default=CONFIG["baseline_csv"]
    )
    parser.add_argument(
        "--save-dir", type=str, default=CONFIG["save_dir"]
    )
    parser.add_argument(
        "--horizons", type=float, nargs="+", required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG["feature_path"] = args.features
    CONFIG["baseline_csv"] = args.baseline
    CONFIG["save_dir"] = args.save_dir
    CONFIG["horizons"] = args.horizons

    run()


if __name__ == "__main__":
    main()
