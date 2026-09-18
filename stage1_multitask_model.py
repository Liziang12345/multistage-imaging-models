# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import hmac
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import Swin3D_S_Weights, swin3d_s


# ==============================================================================
# 0. CONFIG
# ==============================================================================

CONFIG = {
    # image
    "depth": 32,
    "size": 224,

    # token grid used for cross-attention
    # torchvision Swin3D-S final feature map for 32x224x224 is typically
    # 16 x 7 x 7. We pool the temporal axis to 4 while preserving 7x7 spatial
    # resolution -> N = 4*7*7 = 196 tokens, which is non-degenerate and much
    # more memory-efficient than N = 784.
    "token_pool_t": 4,
    "token_pool_h": 7,
    "token_pool_w": 7,

    # model
    "num_heads": 8,
    "attn_dropout": 0.10,
    "ffn_dropout": 0.10,

    # training
    "batch_size": 2,           # token attention uses more VRAM
    "accum_steps": 8,          # effective batch ~16
    "num_workers": 8,
    "epochs": 80,
    "patience": 12,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "mixup_alpha": 0.2,
    "n_fold": 5,
    "seed": 42,

    # optional SWA; default off for cleaner early-stopping behavior
    "use_swa": False,
    "swa_start": 60,

    # paths
    "paths": {
        "3d+": "data/post",
        "3d-": "data/pre",
        "diff": "data/diff",
    },
    "plaque_csv": "data/plaque_labels.csv",
    "subject_csv": "data/subjects.csv",

    "model_dir": "outputs/stage1/checkpoints",
    "eval_dir": "outputs/stage1/evaluation",
    "patient_feature_path": "outputs/patient_features.npy",
}

TASKS = ["vuln", "iph", "lrnc", "irr", "enh"]
TASK_LABEL_COLUMNS = {
    "vuln": "label",
    "iph": "IPH",
    "lrnc": "LRNC",
    "irr": "Irregular",
    "enh": "Enhancement",
}


# ==============================================================================
# 1. Reproducibility / helpers
# ==============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canonical_group(x: str) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    if s in {"train", "training"}:
        return "train"
    if s in {"val", "valid", "validation"}:
        return "validation"
    if s in {"test", "external test", "externaltest", "external"}:
        return "test"
    return s


def clean_plaque_name(x: object) -> str:
    name = str(x).strip()
    for suffix in (".nii.gz", ".nii.npy", ".npy"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def binary_label(v: object) -> int:
    if pd.isna(v):
        raise ValueError("A binary label is missing.")
    value = float(v)
    if not np.isfinite(value) or value not in (0.0, 1.0):
        raise ValueError("Binary labels must be encoded as 0 or 1.")
    return int(value)


def protected_subject_key(subject_id: object) -> str:
    secret = os.environ.get("PIPELINE_LINK_KEY")
    if not secret:
        raise RuntimeError(
            "Set PIPELINE_LINK_KEY in the local environment before feature extraction."
        )
    raw = str(subject_id).strip().encode("utf-8")
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


seed_everything(CONFIG["seed"])

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# 2. EfficientKAN
# ==============================================================================

class EfficientKANLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation=nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps
        self.base_activation = base_activation()

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("spline_scaler", None)

        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise: float = 0.1) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(
                    self.grid_size + 1,
                    self.in_features,
                    self.out_features,
                    device=self.base_weight.device,
                )
                - 0.5
            ) * scale_noise / self.grid_size

            coeff = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order], noise
            )
            scale = (
                self.scale_spline
                if not self.enable_standalone_scale_spline
                else 1.0
            )
            self.spline_weight.copy_(scale * coeff)

            if self.spline_scaler is not None:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, : -(k + 1)]
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_num = grid[:, k + 1 :] - x
            right_den = grid[:, k + 1 :] - grid[:, 1:-k]
            bases = (
                left_num / (left_den + 1e-12) * bases[:, :, :-1]
                + right_num / (right_den + 1e-12) * bases[:, :, 1:]
            )

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x2 = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x2), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x2).reshape(x2.shape[0], -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )

        y = base_output + spline_output
        return y.reshape(*original_shape[:-1], self.out_features)


# ==============================================================================
# 3. Token-level architecture
# ==============================================================================

class SharedSwin3DTokenEncoder(nn.Module):
    """
    ONE Swin3D-S encoder is reused for both pre- and post-contrast volumes.

    It returns a real token sequence rather than a single pooled vector:
        [B, 3, D, H, W]
          -> Swin feature map [B, T', H', W', 768]
          -> adaptive token pooling
          -> [B, N, 768], N = pool_t * pool_h * pool_w
    """

    def __init__(self, pool_size: Tuple[int, int, int]):
        super().__init__()
        backbone = swin3d_s(weights=Swin3D_S_Weights.KINETICS400_V1)

        self.patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop
        self.features = backbone.features
        self.norm = backbone.norm
        self.feat_dim = int(backbone.head.in_features)
        self.pool_size = tuple(int(v) for v in pool_size)

        # Remove the temporary top-level object; modules above retain weights.
        del backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,D,H,W]
        x = self.patch_embed(x)      # [B,T,H,W,C]
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)             # [B,T',H',W',C]

        # -> [B,C,T',H',W']
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = F.adaptive_avg_pool3d(x, self.pool_size)

        # -> [B,N,C]
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class CrossAttentionBlock(nn.Module):
    """
    Standard pre-norm multi-head cross-attention with a residual FFN.

    query_tokens and context_tokens can both have N > 1.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(attn_dropout)

        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        return_attention: bool = False,
    ):
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(context_tokens)

        attn_out, attn_map = self.attn(
            q,
            kv,
            kv,
            need_weights=return_attention,
            average_attn_weights=False if return_attention else True,
        )
        x = query_tokens + self.attn_drop(attn_out)
        x = x + self.ffn(self.norm_ffn(x))

        if return_attention:
            return x, attn_map
        return x


class SubtractionGuidedGate(nn.Module):
    """
    Uses the global subtraction feature to modulate a cross-attended
    pre/post representation without pretending that the subtraction branch
    itself contains a matching 3D token grid.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.diff_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self, main_feature: torch.Tensor, diff_feature: torch.Tensor
    ) -> torch.Tensor:
        d = self.diff_proj(diff_feature)
        g = self.gate(torch.cat([main_feature, d], dim=1))
        return self.out_norm(main_feature + g * d)


class PlaquePhenNet(nn.Module):
    """
    Plaque-level multi-task architecture.

    pre/post:
        shared Swin3D-S -> token sequences -> bidirectional cross-attention

    subtraction:
        ConvNeXt V2-Tiny (32 depth slices treated as channels) -> 768-D guide

    fusion:
        cross-attended pre/post global features + subtraction-guided gates
        -> concat -> LayerNorm -> 1536-D plaque representation

    phenotype heads:
        1536 -> 256 -> 2  (x4)

    Vuln relay:
        1536 + 2*4 = 1544 -> 512 -> 2
    """

    def __init__(self):
        super().__init__()

        pool_size = (
            CONFIG["token_pool_t"],
            CONFIG["token_pool_h"],
            CONFIG["token_pool_w"],
        )

        # ONE shared-weight Swin encoder
        self.shared_swin = SharedSwin3DTokenEncoder(pool_size=pool_size)
        self.feat_dim = self.shared_swin.feat_dim  # 768

        # subtraction encoder: ConvNeXt V2-Tiny
        self.diff_encoder = timm.create_model(
            "convnextv2_tiny",
            pretrained=True,
            num_classes=0,
            in_chans=CONFIG["depth"],
        )
        self.diff_align = nn.Linear(self.diff_encoder.num_features, self.feat_dim)
        self.diff_norm = nn.LayerNorm(self.feat_dim)

        # REAL token-level bidirectional cross-attention
        self.pre_queries_post = CrossAttentionBlock(
            self.feat_dim,
            num_heads=CONFIG["num_heads"],
            attn_dropout=CONFIG["attn_dropout"],
            ffn_dropout=CONFIG["ffn_dropout"],
        )
        self.post_queries_pre = CrossAttentionBlock(
            self.feat_dim,
            num_heads=CONFIG["num_heads"],
            attn_dropout=CONFIG["attn_dropout"],
            ffn_dropout=CONFIG["ffn_dropout"],
        )

        # subtraction-guided modulation
        self.pre_diff_gate = SubtractionGuidedGate(self.feat_dim)
        self.post_diff_gate = SubtractionGuidedGate(self.feat_dim)

        # 768 + 768 = 1536
        self.fusion_norm = nn.LayerNorm(self.feat_dim * 2)

        def build_aux_head(num_classes: int = 2) -> nn.Sequential:
            return nn.Sequential(
                EfficientKANLinear(self.feat_dim * 2, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(0.50),
                EfficientKANLinear(256, num_classes),
            )

        self.head_iph = build_aux_head(2)
        self.head_lrnc = build_aux_head(2)
        self.head_irr = build_aux_head(2)
        self.head_enh = build_aux_head(2)

        # Four phenotype heads each output two logits -> +8
        self.relay_dim = self.feat_dim * 2 + 8
        self.head_vuln = nn.Sequential(
            EfficientKANLinear(self.relay_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.65),
            EfficientKANLinear(512, 2),
        )

    def encode_plaque(
        self,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        x_diff: torch.Tensor,
        return_attention: bool = False,
    ):
        # pre/post are one-channel volumes; Kinetics-pretrained Swin expects 3 ch
        x_pre_3ch = x_pre.repeat(1, 3, 1, 1, 1)
        x_post_3ch = x_post.repeat(1, 3, 1, 1, 1)

        # IMPORTANT: both calls use the SAME self.shared_swin -> shared weights
        pre_tokens = self.shared_swin(x_pre_3ch)
        post_tokens = self.shared_swin(x_post_3ch)

        if pre_tokens.shape[1] <= 1 or post_tokens.shape[1] <= 1:
            raise RuntimeError(
                f"Cross-attention token count must be >1, got "
                f"pre={pre_tokens.shape}, post={post_tokens.shape}"
            )

        if return_attention:
            pre_cross, attn_pre = self.pre_queries_post(
                pre_tokens, post_tokens, return_attention=True
            )
            post_cross, attn_post = self.post_queries_pre(
                post_tokens, pre_tokens, return_attention=True
            )
        else:
            pre_cross = self.pre_queries_post(pre_tokens, post_tokens)
            post_cross = self.post_queries_pre(post_tokens, pre_tokens)
            attn_pre = attn_post = None

        # token -> global
        pre_global = pre_cross.mean(dim=1)
        post_global = post_cross.mean(dim=1)

        # subtraction: [B,1,32,H,W] -> [B,32,H,W]
        diff_2d = x_diff.squeeze(1)
        diff_global = self.diff_encoder(diff_2d)
        diff_global = self.diff_norm(self.diff_align(diff_global))

        # explicit subtraction-guided gating
        pre_guided = self.pre_diff_gate(pre_global, diff_global)
        post_guided = self.post_diff_gate(post_global, diff_global)

        plaque_feature = self.fusion_norm(
            torch.cat([pre_guided, post_guided], dim=1)
        )  # [B,1536]

        if return_attention:
            return plaque_feature, {
                "pre_queries_post": attn_pre,
                "post_queries_pre": attn_post,
                "token_count": int(pre_tokens.shape[1]),
            }

        return plaque_feature

    def forward_features(
        self, x_pre: torch.Tensor, x_post: torch.Tensor, x_diff: torch.Tensor
    ) -> torch.Tensor:
        return self.encode_plaque(x_pre, x_post, x_diff, return_attention=False)

    def forward(
        self,
        x_pre: torch.Tensor,
        x_post: torch.Tensor,
        x_diff: torch.Tensor,
        return_attention: bool = False,
    ):
        if return_attention:
            feat, attn_info = self.encode_plaque(
                x_pre, x_post, x_diff, return_attention=True
            )
        else:
            feat = self.encode_plaque(
                x_pre, x_post, x_diff, return_attention=False
            )
            attn_info = None

        out_iph = self.head_iph(feat)
        out_lrnc = self.head_lrnc(feat)
        out_irr = self.head_irr(feat)
        out_enh = self.head_enh(feat)

        relay_feat = torch.cat(
            [feat, out_iph, out_lrnc, out_irr, out_enh], dim=1
        )  # [B,1544]
        out_vuln = self.head_vuln(relay_feat)

        output = {
            "vuln": out_vuln,
            "iph": out_iph,
            "lrnc": out_lrnc,
            "irr": out_irr,
            "enh": out_enh,
            "plaque_feature": feat,
        }
        if return_attention:
            output["attention"] = attn_info
        return output


# ==============================================================================
# 4. Loss
# ==============================================================================

class ASLSingleLabel(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        eps: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError("ASLSingleLabel expects logits with shape [B, C].")
        if target.ndim != 1:
            target = target.reshape(-1)
        if inputs.shape[0] != target.shape[0]:
            raise ValueError("Logits and targets must have the same batch size.")

        num_classes = inputs.size(-1)
        log_preds = self.logsoftmax(inputs)

        targets = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1.0
        )
        anti_targets = 1.0 - targets

        probs = torch.exp(log_preds)
        xs_pos = probs * targets
        xs_neg = (1.0 - probs) * anti_targets

        asymmetric_w = torch.pow(
            1.0 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:
            targets = targets.mul(1.0 - self.eps).add(self.eps / num_classes)

        loss = -(targets * log_preds).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_tasks: int = 5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        if len(losses) != len(self.log_vars):
            raise ValueError("Loss count does not match learned task weights.")
        total = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + 0.5 * self.log_vars[i]
        return total


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ==============================================================================
# 5. Dataset
# ==============================================================================

class PlaqueDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_volume(self, folder: str, name: str) -> torch.Tensor:
        path = Path(folder) / f"{name}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError("Required input file was not found.")

        arr = nib.load(str(path)).get_fdata().astype(np.float32)
        t = torch.from_numpy(arr).unsqueeze(0)  # [1,D,H,W]

        t = F.interpolate(
            t.unsqueeze(0),
            size=(CONFIG["depth"], CONFIG["size"], CONFIG["size"]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        return t

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        name = clean_plaque_name(row["ID"])

        v_pre = self._load_volume(CONFIG["paths"]["3d-"], name)
        v_post = self._load_volume(CONFIG["paths"]["3d+"], name)
        v_diff = self._load_volume(CONFIG["paths"]["diff"], name)

        # Normalize paired inputs; subtraction input is assumed precomputed
        v_pre = (v_pre - v_pre.mean()) / (v_pre.std() + 1e-6)
        v_post = (v_post - v_post.mean()) / (v_post.std() + 1e-6)

        labels = torch.tensor(
            [
                binary_label(row.get("label", 0)),
                binary_label(row.get("IPH", 0)),
                binary_label(row.get("LRNC", 0)),
                binary_label(row.get("Irregular", 0)),
                binary_label(row.get("Enhancement", 0)),
            ],
            dtype=torch.long,
        )

        if self.is_train:
            # Apply exactly the same geometric transform to all three inputs.
            if random.random() < 0.5:
                v_pre = torch.flip(v_pre, [-1])
                v_post = torch.flip(v_post, [-1])
                v_diff = torch.flip(v_diff, [-1])

            if random.random() < 0.5:
                v_pre = torch.flip(v_pre, [-2])
                v_post = torch.flip(v_post, [-2])
                v_diff = torch.flip(v_diff, [-2])

            k = random.randint(0, 3)
            if k > 0:
                v_pre = torch.rot90(v_pre, k, [-2, -1])
                v_post = torch.rot90(v_post, k, [-2, -1])
                v_diff = torch.rot90(v_diff, k, [-2, -1])

        return v_pre, v_post, v_diff, labels, name


# ==============================================================================
# 6. Training
# ==============================================================================

def make_loader(
    df: pd.DataFrame, is_train: bool, batch_size: Optional[int] = None
) -> DataLoader:
    ds = PlaqueDataset(df, is_train=is_train)
    return DataLoader(
        ds,
        batch_size=batch_size or CONFIG["batch_size"],
        shuffle=is_train,
        num_workers=CONFIG["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=CONFIG["num_workers"] > 0,
    )


@torch.no_grad()
def predict_vuln_auc(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []

    for pre, post, diff, labels, _ in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        diff = diff.to(device, non_blocking=True)

        amp_enabled = device.type == "cuda"
        with autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(pre, post, diff)

        p = torch.softmax(out["vuln"].float(), dim=1)[:, 1]
        y_prob.extend(p.cpu().numpy().tolist())
        y_true.extend(labels[:, 0].numpy().tolist())

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def run_training() -> None:
    device = get_device()
    model_dir = Path(CONFIG["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CONFIG["plaque_csv"])
    if "group" not in df.columns:
        raise ValueError("Plaque CSV must contain a 'group' column.")

    for col in ["IPH", "LRNC", "Irregular", "Enhancement"]:
        if col not in df.columns:
            raise ValueError(f"Plaque CSV missing phenotype column: {col}")

    groups = df["group"].map(canonical_group)
    train_df = df[groups == "train"].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("No training samples were found.")


    subject_df = pd.read_csv(CONFIG["subject_csv"], dtype={"ID": str})
    if "ID" not in subject_df.columns:
        raise ValueError("Subject CSV must contain an 'ID' column.")

    subject_df["ID"] = subject_df["ID"].astype(str).str.strip()
    plaque_to_subject = build_plaque_patient_map(train_df, subject_df)
    patient_groups = np.asarray(
        [plaque_to_subject[clean_plaque_name(x)] for x in train_df["ID"]],
        dtype=object,
    )

    if len(np.unique(patient_groups)) < CONFIG["n_fold"]:
        raise ValueError("The number of unique subjects is smaller than n_fold.")

    skf = StratifiedGroupKFold(
        n_splits=CONFIG["n_fold"],
        shuffle=True,
        random_state=CONFIG["seed"],
    )

    y_strat = train_df["label"].astype(int).to_numpy()

    for fold, (tr_idx, in_val_idx) in enumerate(
        skf.split(train_df, y_strat, groups=patient_groups)
    ):
        train_subjects = set(patient_groups[tr_idx])
        val_subjects = set(patient_groups[in_val_idx])
        if train_subjects.intersection(val_subjects):
            raise RuntimeError("Subject overlap detected between fold partitions.")

        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        in_val_df = train_df.iloc[in_val_idx].reset_index(drop=True)

        tr_dl = make_loader(tr_df, is_train=True)
        in_val_dl = make_loader(in_val_df, is_train=False)

        model = PlaquePhenNet().to(device)
        loss_wrapper = MultiTaskLossWrapper(num_tasks=5).to(device)
        loss_fn = ASLSingleLabel(gamma_neg=4, gamma_pos=1, eps=0.0)

        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": CONFIG["lr"]},
                {"params": loss_wrapper.parameters(), "lr": 1e-3},
            ],
            weight_decay=CONFIG["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"]
        )

        scaler = GradScaler("cuda", enabled=device.type == "cuda")

        best_auc = -np.inf
        best_epoch = -1
        best_state = None
        best_loss_state = None
        patience_count = 0

        # Optional SWA state
        swa_state = None
        swa_n = 0

        for epoch in range(CONFIG["epochs"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_losses: List[float] = []
            for step, (pre, post, diff, labels, _) in enumerate(tr_dl):
                pre = pre.to(device, non_blocking=True)
                post = post.to(device, non_blocking=True)
                diff = diff.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                amp_enabled = device.type == "cuda"
                with autocast(
                    "cuda", dtype=torch.bfloat16, enabled=amp_enabled
                ):
                    if CONFIG["mixup_alpha"] > 0:
                        lam = float(
                            np.random.beta(
                                CONFIG["mixup_alpha"], CONFIG["mixup_alpha"]
                            )
                        )
                        perm = torch.randperm(pre.size(0), device=device)

                        out = model(
                            lam * pre + (1.0 - lam) * pre[perm],
                            lam * post + (1.0 - lam) * post[perm],
                            lam * diff + (1.0 - lam) * diff[perm],
                        )

                        losses = []
                        for task_idx, task in enumerate(TASKS):
                            y = labels[:, task_idx]
                            losses.append(
                                mixup_criterion(
                                    loss_fn,
                                    out[task],
                                    y,
                                    y[perm],
                                    lam,
                                )
                            )
                    else:
                        out = model(pre, post, diff)
                        losses = []
                        for task_idx, task in enumerate(TASKS):
                            y = labels[:, task_idx]
                            losses.append(
                                loss_fn(out[task], y)
                            )

                    total_loss = loss_wrapper(losses)
                    scaled_loss = total_loss / CONFIG["accum_steps"]

                scaler.scale(scaled_loss).backward()

                should_step = (
                    (step + 1) % CONFIG["accum_steps"] == 0
                    or (step + 1) == len(tr_dl)
                )
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_losses.append(float(total_loss.detach().cpu()))

            scheduler.step()

            # optional simple parameter averaging after swa_start
            if CONFIG["use_swa"] and epoch >= CONFIG["swa_start"]:
                current = copy.deepcopy(model.state_dict())
                if swa_state is None:
                    swa_state = {
                        k: v.detach().clone() for k, v in current.items()
                    }
                    swa_n = 1
                else:
                    swa_n += 1
                    alpha = 1.0 / swa_n
                    for k in swa_state:
                        swa_state[k].mul_(1.0 - alpha).add_(
                            current[k], alpha=alpha
                        )

            auc = predict_vuln_auc(model, in_val_dl, device)
            mean_loss = float(np.mean(epoch_losses))

            improved = np.isfinite(auc) and auc > best_auc + 1e-6
            if improved:
                best_auc = auc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_loss_state = copy.deepcopy(loss_wrapper.state_dict())
                patience_count = 0
            else:
                patience_count += 1


            if patience_count >= CONFIG["patience"]:
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold}: no valid best state was found.")

        # If SWA was enabled, compare it against best checkpoint on internal fold val
        if CONFIG["use_swa"] and swa_state is not None:
            swa_model = PlaquePhenNet().to(device)
            swa_model.load_state_dict(swa_state, strict=True)
            swa_auc = predict_vuln_auc(swa_model, in_val_dl, device)
            if np.isfinite(swa_auc) and swa_auc > best_auc:
                best_auc = swa_auc
                best_state = copy.deepcopy(swa_state)
                best_epoch = -2
            del swa_model

        checkpoint = {"model_state_dict": best_state}

        ckpt_path = model_dir / f"best_fold{fold}.pth"
        torch.save(checkpoint, ckpt_path)

        del model, optimizer, scaler, tr_dl, in_val_dl
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# 7. Checkpoint loading / evaluation
# ==============================================================================

def load_fold_models(device: torch.device) -> List[PlaquePhenNet]:
    models: List[PlaquePhenNet] = []
    for fold in range(CONFIG["n_fold"]):
        path = Path(CONFIG["model_dir"]) / f"best_fold{fold}.pth"
        if not path.is_file():
            raise FileNotFoundError("A required model checkpoint was not found.")

        ckpt = torch.load(path, map_location="cpu")
        state = (
            ckpt["model_state_dict"]
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt
            else ckpt
        )

        model = PlaquePhenNet()
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models.append(model)

    return models


@torch.no_grad()
def predict_ensemble(
    models: List[PlaquePhenNet],
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for pre, post, diff, labels, names in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        diff = diff.to(device, non_blocking=True)

        probs_by_task = {
            task: np.zeros((len(names),), dtype=np.float64)
            for task in TASKS
        }

        for model in models:
            amp_enabled = device.type == "cuda"
            with autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp_enabled
            ):
                out = model(pre, post, diff)

            for task in TASKS:
                p = (
                    torch.softmax(out[task].float(), dim=1)[:, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )
                probs_by_task[task] += p / len(models)

        labels_np = labels.numpy()

        for i, _ in enumerate(names):
            row: Dict[str, object] = {}
            for task_idx, task in enumerate(TASKS):
                row[f"true_{task}"] = int(labels_np[i, task_idx])
                row[f"prob_{task}"] = float(probs_by_task[task][i])
            rows.append(row)

    return pd.DataFrame(rows)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    out: Dict[str, float] = {}
    if len(np.unique(y_true)) >= 2:
        out["AUC"] = float(roc_auc_score(y_true, y_prob))
        out["AP"] = float(average_precision_score(y_true, y_prob))
    else:
        out["AUC"] = float("nan")
        out["AP"] = float("nan")

    out["Accuracy"] = float(accuracy_score(y_true, y_pred))
    out["F1"] = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["Sensitivity"] = float(tp / (tp + fn)) if tp + fn else float("nan")
    out["Specificity"] = float(tn / (tn + fp)) if tn + fp else float("nan")
    out["PPV"] = float(tp / (tp + fp)) if tp + fp else float("nan")
    out["NPV"] = float(tn / (tn + fn)) if tn + fn else float("nan")
    out["Brier"] = float(np.mean((y_prob - y_true) ** 2))
    return out


def run_evaluation() -> None:
    device = get_device()
    eval_dir = Path(CONFIG["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CONFIG["plaque_csv"])
    groups = df["group"].map(canonical_group)
    models = load_fold_models(device)

    all_metric_rows = []

    for split_name, split_key in [
        ("Validation", "validation"),
        ("Test", "test"),
    ]:
        subset = df[groups == split_key].reset_index(drop=True)
        if len(subset) == 0:
            continue

        loader = make_loader(
            subset, is_train=False, batch_size=CONFIG["batch_size"]
        )
        pred_df = predict_ensemble(models, loader, device)
        pred_path = eval_dir / f"{split_name}_predictions.csv"
        export_cols = [c for c in pred_df.columns if c.startswith("prob_")]
        export_df = pred_df[export_cols].copy()
        export_df.insert(0, "sample_index", np.arange(len(export_df), dtype=int))
        export_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        for task in TASKS:
            m = classification_metrics(
                pred_df[f"true_{task}"].to_numpy(),
                pred_df[f"prob_{task}"].to_numpy(),
            )
            m["Cohort"] = split_name
            m["Task"] = task
            all_metric_rows.append(m)


    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_path = eval_dir / "stage1_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")


# ==============================================================================
# 8. Feature extraction
# ==============================================================================

def load_single_case(
    plaque_name: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def load(key: str) -> torch.Tensor:
        path = Path(CONFIG["paths"][key]) / f"{plaque_name}.nii.gz"
        if not path.is_file():
            raise FileNotFoundError("Required input file was not found.")

        arr = nib.load(str(path)).get_fdata().astype(np.float32)
        t = torch.from_numpy(arr).unsqueeze(0)  # [1,D,H,W]
        t = F.interpolate(
            t.unsqueeze(0),
            size=(CONFIG["depth"], CONFIG["size"], CONFIG["size"]),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        if key != "diff":
            t = (t - t.mean()) / (t.std() + 1e-6)

        return t.unsqueeze(0).to(device)

    return load("3d-"), load("3d+"), load("diff")


def build_plaque_patient_map(
    plaque_df: pd.DataFrame, subject_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Prefer an explicit patient-level ID column when available.
    Otherwise use a conservative longest-prefix fallback.

    If your plaque CSV has a dedicated patient ID column, rename it to PatientID
    for the cleanest mapping.
    """
    explicit_cols = [
        c for c in ["PatientID", "patient_id", "PID", "pid"]
        if c in plaque_df.columns
    ]

    mapping: Dict[str, str] = {}

    if explicit_cols:
        col = explicit_cols[0]
        valid_patient_ids = set(subject_df["ID"].astype(str).str.strip())
        for _, row in plaque_df.iterrows():
            plaque_name = clean_plaque_name(row["ID"])
            pid = str(row[col]).strip()
            if pid not in valid_patient_ids:
                raise ValueError("A subject linkage value could not be matched.")
            mapping[plaque_name] = pid
        return mapping

    patient_ids = subject_df["ID"].astype(str).str.strip().tolist()
    patient_ids_sorted = sorted(patient_ids, key=len, reverse=True)

    for plaque_raw in plaque_df["ID"]:
        plaque_name = clean_plaque_name(plaque_raw)

        # First try exact ID
        if plaque_name in patient_ids:
            mapping[plaque_name] = plaque_name
            continue

        candidates = []
        for pid in patient_ids_sorted:
            if not plaque_name.startswith(pid):
                continue
            if len(plaque_name) == len(pid):
                candidates.append(pid)
                continue

            next_char = plaque_name[len(pid)]
            # preferred boundary
            if next_char in {"_", "-", ".", " "}:
                candidates.append(pid)

        if not candidates:
            # last-resort: unique longest raw prefix
            raw_candidates = [
                pid for pid in patient_ids_sorted if plaque_name.startswith(pid)
            ]
            if raw_candidates:
                longest = len(raw_candidates[0])
                raw_candidates = [
                    p for p in raw_candidates if len(p) == longest
                ]
            if len(raw_candidates) == 1:
                candidates = raw_candidates

        if len(candidates) != 1:
            raise ValueError("A sample could not be uniquely linked to a subject.")

        mapping[plaque_name] = candidates[0]

    return mapping


@torch.no_grad()
def run_feature_extraction() -> None:
    device = get_device()
    models = load_fold_models(device)

    plaque_df = pd.read_csv(CONFIG["plaque_csv"])
    subject_df = pd.read_csv(CONFIG["subject_csv"], dtype={"ID": str})

    required_subject = {"ID"}
    missing = required_subject.difference(subject_df.columns)
    if missing:
        raise ValueError("Subject table is missing a required column.")

    subject_df["ID"] = subject_df["ID"].astype(str).str.strip()
    plaque_map = build_plaque_patient_map(plaque_df, subject_df)

    patient_data: Dict[str, Dict[str, object]] = {}

    for _, prow in plaque_df.iterrows():
        plaque_name = clean_plaque_name(prow["ID"])
        pid = plaque_map[plaque_name]
        subject_key = protected_subject_key(pid)

        pre, post, diff = load_single_case(plaque_name, device)

        feat_sum = None
        for model in models:
            amp_enabled = device.type == "cuda"
            with autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp_enabled
            ):
                feat = model.forward_features(pre, post, diff).float()

            feat_sum = feat if feat_sum is None else feat_sum + feat

        feat_avg = (feat_sum / len(models)).cpu().numpy()[0]  # [1536]

        # spatial descriptor: LOCATION one-hot (6) + RL + QH
        loc_onehot = np.zeros(6, dtype=np.float32)
        loc = int(float(prow.get("LOCATION", -1)))
        if 0 <= loc <= 5:
            loc_onehot[loc] = 1.0

        sp = np.concatenate(
            [
                loc_onehot,
                np.asarray(
                    [
                        float(prow.get("RL", 0)),
                        float(prow.get("QH", 0)),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        if subject_key not in patient_data:
            patient_data[subject_key] = {
                "feats": [],
                "spatial": [],
            }

        patient_data[subject_key]["feats"].append(feat_avg)
        patient_data[subject_key]["spatial"].append(sp)

    # finalize arrays
    finalized: Dict[str, Dict[str, object]] = {}
    for subject_key, item in patient_data.items():
        feats = np.stack(item["feats"], axis=0).astype(np.float32)
        spatial = np.stack(item["spatial"], axis=0).astype(np.float32)

        if feats.shape[0] != spatial.shape[0]:
            raise AssertionError("Feature and spatial instance counts differ.")
        if feats.shape[1] != 1536:
            raise AssertionError("Unexpected feature dimension.")

        finalized[subject_key] = {
            "feats": feats,
            "spatial": spatial,
        }

    save_path = Path(CONFIG["patient_feature_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, finalized)




# ==============================================================================
# 9. CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-task training and feature extraction."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "extract"],
        default="train",
    )
    parser.add_argument("--plaque-csv", type=str, default=CONFIG["plaque_csv"])
    parser.add_argument("--subject-csv", type=str, default=CONFIG["subject_csv"])
    parser.add_argument("--pre-dir", type=str, default=CONFIG["paths"]["3d-"])
    parser.add_argument("--post-dir", type=str, default=CONFIG["paths"]["3d+"])
    parser.add_argument("--diff-dir", type=str, default=CONFIG["paths"]["diff"])
    parser.add_argument("--model-dir", type=str, default=CONFIG["model_dir"])
    parser.add_argument("--eval-dir", type=str, default=CONFIG["eval_dir"])
    parser.add_argument(
        "--patient-feature-path",
        type=str,
        default=CONFIG["patient_feature_path"],
    )
    return parser.parse_args()


def apply_args(args: argparse.Namespace) -> None:
    CONFIG["plaque_csv"] = args.plaque_csv
    CONFIG["subject_csv"] = args.subject_csv
    CONFIG["paths"]["3d-"] = args.pre_dir
    CONFIG["paths"]["3d+"] = args.post_dir
    CONFIG["paths"]["diff"] = args.diff_dir
    CONFIG["model_dir"] = args.model_dir
    CONFIG["eval_dir"] = args.eval_dir
    CONFIG["patient_feature_path"] = args.patient_feature_path


def main() -> None:
    args = parse_args()
    apply_args(args)

    if args.mode == "train":
        run_training()
    elif args.mode == "evaluate":
        run_evaluation()
    elif args.mode == "extract":
        run_feature_extraction()
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
