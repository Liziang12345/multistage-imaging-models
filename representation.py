"""Fold-isolated, plaque-level image--anatomy interaction encoding.

Inputs are trusted, private, fold-specific feature files. This source contains
no observations, subject identifiers, site information, or fitted parameters.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

IMAGE_DIM = 1536
PCA_DIM = 64
ARTERY_CATEGORIES = 6
SIDE_CATEGORIES = 3
ANATOMY_DIM = ARTERY_CATEGORIES + SIDE_CATEGORIES
JOINT_DIM = PCA_DIM * (ANATOMY_DIM + 1)
POOLED_DIM = JOINT_DIM * 3
SELECTED_DIM = 119
MODEL_DIM = SELECTED_DIM + ANATOMY_DIM


def protected_key(subject_id: object) -> str:
    secret = os.getenv("PIPELINE_LINK_KEY")
    if not secret:
        raise RuntimeError("Set PIPELINE_LINK_KEY in the private environment.")
    return hmac.new(secret.encode(), str(subject_id).strip().encode(), hashlib.sha256).hexdigest()


def load_metadata(path: Path, id_col: str, split_col: str, time_col: str, event_col: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={id_col: str})
    for col in (id_col, split_col, time_col, event_col):
        if col not in data:
            raise ValueError("A required metadata column is absent.")
    data = data.copy()
    if data[id_col].isna().any() or data[id_col].astype(str).str.strip().eq("").any():
        raise ValueError("Invalid subject identifier.")
    if data[id_col].duplicated().any():
        raise ValueError("Duplicate subject identifier.")
    data["link_key"] = data[id_col].map(protected_key)
    if data.link_key.duplicated().any():
        raise ValueError("Non-unique protected key.")
    data["_split"] = data[split_col].astype(str).str.strip().str.lower()
    data["_split"] = data["_split"].replace({"val": "validation", "external test": "test"})
    if not set(data._split).issubset({"train", "validation", "test"}):
        raise ValueError("Unexpected cohort label.")
    data["_time"] = pd.to_numeric(data[time_col], errors="raise").astype(float)
    data["_event"] = pd.to_numeric(data[event_col], errors="raise").astype(int)
    if (not np.isfinite(data._time).all() or (data._time <= 0).any()
            or not set(data._event).issubset({0, 1})):
        raise ValueError("Invalid follow-up time or status.")
    return data


def read_manifest(path: Path, train_keys: set[str]) -> dict[str, int]:
    frame = pd.read_csv(path, dtype={"link_key": str})
    if not {"link_key", "outer_fold"}.issubset(frame.columns):
        raise ValueError("The manifest requires link_key and outer_fold.")
    if frame.link_key.isna().any() or frame.link_key.duplicated().any():
        raise ValueError("Invalid manifest linkage.")
    if set(frame.link_key) != train_keys:
        raise ValueError("The fold manifest does not match development subjects.")
    folds = pd.to_numeric(frame.outer_fold, errors="raise").to_numpy(dtype=float)
    if np.any(folds != np.floor(folds)) or not np.isfinite(folds).all():
        raise ValueError("Invalid outer-fold values.")
    unique = sorted(set(folds.astype(int).tolist()))
    if unique != list(range(len(unique))) or len(unique) < 2:
        raise ValueError("Outer-fold values must be consecutive, starting at zero.")
    return dict(zip(frame.link_key, folds.astype(int).tolist()))


def read_fold_features(path: Path, ordered_keys: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Original upstream encoder exports NumPy dictionaries; ONLY trusted local
    # files may be loaded, as NumPy's pickle support can execute arbitrary code.
    records = np.load(path, allow_pickle=True).item()
    if not isinstance(records, dict) or set(records) != set(ordered_keys):
        raise ValueError("Feature entries do not match the metadata subjects.")
    result = {}
    for key in ordered_keys:
        row = records[key]
        feat = np.asarray(row["feats"], dtype=np.float64)
        spa = np.asarray(row["spatial"], dtype=np.float64)
        if (feat.ndim != 2 or feat.shape[1] != IMAGE_DIM or spa.ndim != 2
                or spa.shape[1] != 8 or len(feat) != len(spa) or len(feat) == 0
                or not np.isfinite(feat).all() or not np.isfinite(spa).all()):
            raise ValueError("Invalid image or position representation.")
        # Existing upstream convention: columns 0:6 are vascular one-hot,
        # column 6 is laterality code (0, 1, 2), column 7 is a redundant
        # anterior/posterior circulation flag. No assumptions about codes 0/1.
        onehot = spa[:, :ARTERY_CATEGORIES]
        side = spa[:, 6]
        if (not np.all((onehot == 0) | (onehot == 1))
                or not np.all(onehot.sum(axis=1) == 1)
                or not np.all(np.isin(side, [0, 1, 2]))):
            raise ValueError("Unsupported anatomical codes in input features.")
        cat = np.concatenate([onehot, np.eye(SIDE_CATEGORIES, dtype=np.float64)[side.astype(int)]], axis=1)
        result[key] = (feat, cat)
    return result


def fit_image_projection(records: dict, fit_keys: list[str], seed: int) -> PCA:
    pooled = np.concatenate([records[key][0] for key in fit_keys], axis=0)
    if min(pooled.shape) <= PCA_DIM:
        raise ValueError("Insufficient training instances for the projection dimension.")
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=int(seed))
    pca.fit(pooled)
    return pca


def patient_matrix(records: dict, keys: list[str], projection: PCA, representation: str) -> np.ndarray:
    """Produce outcome-independent anatomical interactions before pooling.

    At plaque level: [PCA64(image), PCA64(image) outer-product anatomy9].
    For multi-plaque: mean/max/std over all plaque vectors.
    For culprit: use one predesignated culprit plaque (encoded as item['culprit']).
    Each representation ends with 9 interpretable anatomy counts or indicators.
    """
    if representation not in {"multi", "culprit"}:
        raise ValueError("Unknown representation.")
    out = []
    for key in keys:
        image, anatomy = records[key]
        projected = projection.transform(image)
        interactions = (projected[:, :, None] * anatomy[:, None, :]).reshape(len(image), -1)
        joined = np.concatenate([projected, interactions], axis=1)
        if representation == "multi":
            signal = np.concatenate([joined.mean(axis=0), joined.max(axis=0), joined.std(axis=0, ddof=0)])
            position = anatomy.sum(axis=0)
        else:
            # The upstream file's optional 'culprit_index' is supplied through
            # records metadata. If absent, the user must provide a companion
            # protected-key mapping; no plaque is guessed from image intensities.
            raise RuntimeError("Use culprit_patient_matrix with trusted culprit-index mapping.")
        out.append(np.concatenate([signal, position]))
    arr = np.asarray(out, dtype=np.float64)
    if arr.shape[1] != POOLED_DIM + ANATOMY_DIM:
        raise AssertionError("Unexpected fused patient dimension.")
    return arr


def culprit_patient_matrix(records: dict, keys: list[str], projection: PCA,
                           culprit_index: dict[str, int]) -> np.ndarray:
    out = []
    for key in keys:
        img, anatomy = records[key]
        index = culprit_index[key]
        if index < 0 or index >= len(img):
            raise ValueError("Culprit index is out of bounds.")
        vec = projection.transform(img[index:index+1])[0]
        pos = anatomy[index]
        crossed = (vec[:, None] * pos[None, :]).ravel()
        out.append(np.concatenate([vec, crossed, pos]))
    arr = np.asarray(out, dtype=np.float64)
    if arr.shape[1] != JOINT_DIM + ANATOMY_DIM:
        raise AssertionError("Unexpected culprit representation dimension.")
    return arr


def load_culprit_mapping(path: Path, ordered_keys: list[str]) -> dict[str, int]:
    data = pd.read_csv(path, dtype={"link_key": str})
    if not {"link_key", "plaque_index"}.issubset(data.columns) or data.link_key.duplicated().any():
        raise ValueError("A culprit file requires unique link_key and plaque_index columns.")
    if set(data.link_key) != set(ordered_keys):
        raise ValueError("Culprit identifiers do not match patient keys.")
    idx = pd.to_numeric(data.plaque_index, errors="raise").to_numpy(float)
    if not np.isfinite(idx).all() or not np.all(idx == np.floor(idx)):
        raise ValueError("Culprit indices must be integers.")
    return dict(zip(data.link_key, idx.astype(int).tolist()))


def cox_score(x: np.ndarray, time: np.ndarray, status: np.ndarray) -> np.ndarray:
    u = np.zeros(x.shape[1], dtype=float)
    information = np.zeros(x.shape[1], dtype=float)
    for t in np.unique(time[status == 1]):
        case = (time == t) & (status == 1)
        risk = x[time >= t]
        d = int(case.sum())
        average = risk.mean(axis=0)
        variance = np.maximum((risk ** 2).mean(axis=0) - average ** 2, 0.0)
        u += x[case].sum(axis=0) - d * average
        information += d * variance
    values = np.abs(u) / np.sqrt(information + 1e-12)
    values[~np.isfinite(values)] = 0.0
    return values



class StableRobustScaler:
    """Robust centering/scaling followed by unsupervised RMS normalization.

    RMS normalization limits numerical conditioning problems of rare categorical
    interactions while keeping the robust median as the reference center.
    Both transforms are estimated on the development subset only.
    """
    def fit(self, x: np.ndarray):
        self.robust = RobustScaler().fit(x)
        z = self.robust.transform(x)
        rms = np.sqrt(np.mean(z ** 2, axis=0))
        self.rms = np.where(np.isfinite(rms) & (rms > 1e-10), rms, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.robust.transform(x) / self.rms

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

def fit_selector(matrix: np.ndarray, fit_rows: np.ndarray,
                 time: np.ndarray, event: np.ndarray):
    """Screen 119 pre-aggregated joint features; keep all 9 anatomy indicators.

    Every fitted step uses only the given development rows. The PCA projection
    is fitted independently outside this function, using the same rows.
    """
    scaler = StableRobustScaler()
    scaled = scaler.fit_transform(matrix[fit_rows])
    ranks = cox_score(scaled[:, :-ANATOMY_DIM], time[fit_rows], event[fit_rows])
    selected = np.argsort(-ranks, kind="stable")[:SELECTED_DIM]
    if len(selected) != SELECTED_DIM:
        raise ValueError("Not enough joint features for selection.")
    indices = np.concatenate([selected, np.arange(matrix.shape[1]-ANATOMY_DIM, matrix.shape[1])])
    return scaler, indices


def transform_selector(matrix: np.ndarray, scaler: RobustScaler, indices: np.ndarray) -> np.ndarray:
    x = scaler.transform(matrix)[:, indices]
    if x.shape[1] != MODEL_DIM or not np.isfinite(x).all():
        raise ValueError("Unexpected transformed feature shape.")
    return x.astype(np.float32)
