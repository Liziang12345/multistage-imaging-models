"""Neural Cox and L2-penalized Cox estimators with Breslow ties."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.block(x)


class DeepSurv(nn.Module):
    def __init__(self, input_dim: int = 128, dropout: float = 0.4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout),
            ResidualBlock(64, dropout), ResidualBlock(64, dropout), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.layers(x).flatten()


def cox_breslow_torch(pred, time, event):
    pll = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for t in torch.unique(time[event > 0.5]):
        case = (time == t) & (event > 0.5)
        risk = time >= t
        pll += pred[case].sum() - case.sum() * torch.logsumexp(pred[risk], dim=0)
    return -pll / torch.clamp(event.sum(), min=1.0)


def fit_neural(x, time, event, seed=42, epochs=180, lr=1e-3,
               weight_decay=1e-2, dropout=0.4, device=None):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DeepSurv(x.shape[1], dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    tt = torch.as_tensor(time, dtype=torch.float32, device=device)
    et = torch.as_tensor(event, dtype=torch.float32, device=device)
    for _ in range(epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = cox_breslow_torch(model(xt), tt, et)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite neural survival loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        scheduler.step()
    model.eval()
    return model


def predict_neural(model, x):
    with torch.no_grad():
        device = next(model.parameters()).device
        xt = torch.as_tensor(x, dtype=torch.float32, device=device)
        return model(xt).cpu().numpy().astype(float)


def l2_cox_objective(beta, x, time, event, alpha):
    """Mean-event negative partial likelihood plus 0.5*alpha*||beta||^2."""
    eta = x @ beta
    loss = 0.0
    grad = np.zeros(x.shape[1], dtype=float)
    events = max(int(np.sum(event)), 1)
    for t in np.unique(time[event == 1]):
        case = (time == t) & (event == 1)
        risk = time >= t
        d = int(case.sum())
        score = eta[risk]
        norm = logsumexp(score)
        weights = np.exp(score - norm)
        loss -= eta[case].sum() - d * norm
        grad -= x[case].sum(axis=0) - d * (weights @ x[risk])
    return loss / events + 0.5 * alpha * np.dot(beta, beta), grad / events + alpha * beta


def fit_ridge_cox(x, time, event, alpha):
    start = np.zeros(x.shape[1], dtype=float)
    result = minimize(l2_cox_objective, start, args=(x.astype(float), time, event, alpha),
                      jac=True, method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-8, "gtol": 1e-5})
    if not result.success and np.linalg.norm(result.jac) > 1e-4:
        raise RuntimeError("Penalized Cox optimizer did not converge.")
    return result.x


def concordance(risk, time, event):
    concordant, comparable = 0.0, 0.0
    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            if time[i] < time[j] and event[i] == 1:
                comparable += 1
                concordant += 1 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0
            elif time[j] < time[i] and event[j] == 1:
                comparable += 1
                concordant += 1 if risk[j] > risk[i] else 0.5 if risk[j] == risk[i] else 0
    return float(concordant / comparable) if comparable else float("nan")


def nested_select_alpha(records, all_keys, outer_keys, time_by_key, event_by_key,
                        representation, patient_builder, seed,
                        candidate_alphas=(0.001, 0.01, 0.1, 1., 10.), n_inner=3):
    """Re-estimate PCA, scaling and Cox ranking for each inner training fold."""
    from representation import fit_image_projection, fit_selector, transform_selector
    events = np.array([event_by_key[k] for k in outer_keys], dtype=int)
    if min(np.bincount(events, minlength=2)) < n_inner:
        raise ValueError("Insufficient events for internal hyperparameter validation.")
    folds = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    score_per_alpha = {float(a): [] for a in candidate_alphas}
    for inner_fold, (dev_idx, val_idx) in enumerate(folds.split(np.zeros(len(outer_keys)), events)):
        fit_keys = [outer_keys[i] for i in dev_idx]
        validation_keys = [outer_keys[i] for i in val_idx]
        # Outcome-free imaging projection is also fitted using inner-fit only.
        pca = fit_image_projection(records, fit_keys, seed + inner_fold)
        keys = fit_keys + validation_keys
        matrix = patient_builder(records, keys, pca, representation)
        times = np.array([time_by_key[k] for k in keys], float)
        statuses = np.array([event_by_key[k] for k in keys], int)
        fit_rows = np.arange(len(fit_keys))
        scaler, indices = fit_selector(matrix, fit_rows, times, statuses)
        x = transform_selector(matrix, scaler, indices).astype(float)
        for a in candidate_alphas:
            beta = fit_ridge_cox(x[:len(fit_keys)], times[:len(fit_keys)], statuses[:len(fit_keys)], a)
            score = concordance(x[len(fit_keys):] @ beta, times[len(fit_keys):], statuses[len(fit_keys):])
            score_per_alpha[float(a)].append(score)
    scores = {a: np.nanmean(v) for a, v in score_per_alpha.items()}
    if not any(np.isfinite(v) for v in scores.values()):
        raise ValueError("Internal validation returned no evaluable concordance.")
    # Prefer stronger regularization for exact ties.
    return max(scores, key=lambda a: (scores[a], a))
