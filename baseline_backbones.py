"""Interchangeable image backbones for the shared multi-task fusion pipeline.

Two-dimensional encoders process every depth slice using shared parameters.
Their spatial features are pooled into the same token lattice consumed by
bidirectional cross-attention. No labels or fitted model parameters are stored.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import timm

BASELINE_BACKBONES = {
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "vit_base_patch16_224": "vit_base_patch16_224",
}


class SliceBackboneTokenEncoder(nn.Module):
    """Shared per-slice backbone with 4x7x7 output and 768 token channels.

    This is an intentionally two-dimensional backbone substitution; it is not
    described as equivalent to native three-dimensional Swin processing.
    """

    def __init__(self, name: str, pool_size=(4, 7, 7), frame_chunk: int = 4,
                 pretrained: bool = True):
        super().__init__()
        if name not in BASELINE_BACKBONES:
            raise ValueError("Unsupported comparison backbone.")
        if frame_chunk < 1:
            raise ValueError("frame_chunk must be positive.")
        self.name = name
        self.pool_size = tuple(int(v) for v in pool_size)
        self.frame_chunk = int(frame_chunk)
        self.vit = name == "vit_base_patch16_224"
        if self.vit:
            self.encoder = timm.create_model(
                BASELINE_BACKBONES[name], pretrained=pretrained, num_classes=0
            )
            output_dim = int(self.encoder.num_features)
        else:
            self.encoder = timm.create_model(
                BASELINE_BACKBONES[name], pretrained=pretrained, features_only=True,
                out_indices=(-1,)
            )
            output_dim = int(self.encoder.feature_info.channels()[-1])
        self.projection = nn.Linear(output_dim, 768)
        self.normalization = nn.LayerNorm(768)
        self.feat_dim = 768

    def _spatial_feature(self, frames: torch.Tensor) -> torch.Tensor:
        if self.vit:
            tokens = self.encoder.forward_features(frames)
            if tokens.ndim != 3:
                raise RuntimeError("Unexpected transformer feature shape.")
            prefix = int(getattr(self.encoder, "num_prefix_tokens", 1))
            tokens = tokens[:, prefix:, :]
            spatial = tokens.shape[1]
            width = int(spatial ** 0.5)
            if width * width != spatial:
                raise RuntimeError("Expected a square patch-token grid.")
            fmap = tokens.transpose(1, 2).reshape(len(frames), -1, width, width)
        else:
            fmap = self.encoder(frames)[-1]
            if fmap.ndim != 4:
                raise RuntimeError("Unexpected convolutional feature shape.")
        return F.adaptive_avg_pool2d(fmap, self.pool_size[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError("Expected [batch, 3, depth, height, width].")
        batch, _, depth, _, _ = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(
            batch * depth, 3, x.shape[-2], x.shape[-1]
        )
        processed = []
        for start in range(0, len(frames), self.frame_chunk):
            feature_map = self._spatial_feature(frames[start:start + self.frame_chunk])
            processed.append(feature_map)
        # [batch, depth, channels, 7, 7] -> [batch, channels, depth, 7, 7]
        features = torch.cat(processed, dim=0)
        _, channels, h, w = features.shape
        features = features.reshape(batch, depth, channels, h, w)
        features = features.permute(0, 2, 1, 3, 4).contiguous()
        features = F.adaptive_avg_pool3d(features, self.pool_size)
        tokens = features.flatten(2).transpose(1, 2).contiguous()
        return self.normalization(self.projection(tokens))
