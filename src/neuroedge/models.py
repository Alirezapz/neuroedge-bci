"""EEGNet-8,2 adaptation and train-only scaling.

Architecture reference: Lawhern et al., 2018, DOI 10.1088/1741-2552/aace8c;
authors' EEGModels.py: https://github.com/vlawhern/arl-eegmodels.
Independent PyTorch implementation; see docs/METHODS.md for explicit deviations.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class TrainOnlyScaler:
    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
        self.scale_ = X.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
        self.scale_ = np.maximum(self.scale_, 1e-6)
        return self

    def transform(self, X):
        return np.ascontiguousarray((X - self.mean_) / self.scale_, dtype=np.float32)


class EEGNet(nn.Module):
    def __init__(self, channels, samples, dropout=0.5, kernel_length=32, f1=8, d=2, f2=16):
        super().__init__()
        self.spec = {"channels": channels, "samples": samples, "dropout": dropout,
                     "kernel_length": kernel_length, "f1": f1, "d": d, "f2": f2}
        # SAME padding matches Keras for even kernels (extra sample on the right).
        self.temporal = nn.Sequential(nn.ZeroPad2d(((kernel_length - 1) // 2, kernel_length // 2, 0, 0)),
                                      nn.Conv2d(1, f1, (1, kernel_length), bias=False),
                                      nn.BatchNorm2d(f1, eps=1e-3, momentum=0.01))
        self.spatial = nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False)
        self.block1 = nn.Sequential(nn.BatchNorm2d(f1 * d, eps=1e-3, momentum=0.01), nn.ELU(),
                                    nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.ZeroPad2d((7, 8, 0, 0)),
                                    nn.Conv2d(f1 * d, f1 * d, (1, 16), groups=f1 * d, bias=False),
                                    nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
                                    nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01), nn.ELU(),
                                    nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.classifier = nn.Linear(f2 * (samples // 32), 2)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, X):
        X = self.temporal(X)
        X = self.block1(self.spatial(X))
        X = self.block2(X)
        return self.classifier(X.flatten(1))  # logits; CE handles softmax during training

    @torch.no_grad()
    def constrain(self):
        for weight, dims, limit in [(self.spatial.weight, (1, 2, 3), 1.0),
                                    (self.classifier.weight, (1,), 0.25)]:
            norms = torch.linalg.vector_norm(weight, dim=dims, keepdim=True)
            weight.mul_(torch.clamp(limit / (norms + 1e-8), max=1.0))


def load_checkpoint(path):
    cp = torch.load(path, map_location="cpu", weights_only=False)
    model = EEGNet(**cp["model_spec"])
    model.load_state_dict(cp["state_dict"])
    model.eval()
    scaler = TrainOnlyScaler()
    scaler.mean_, scaler.scale_ = cp["scaler_mean"], cp["scaler_scale"]
    return model, scaler, cp

