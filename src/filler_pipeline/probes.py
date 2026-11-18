from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def last_nonpad(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), positions]


def train_layer_probes(hidden_by_layer: list[torch.Tensor], labels: list[int],
                       train_fraction: float = 0.8, seed: int = 7) -> list[dict]:
    y = np.asarray(labels)
    split = max(1, int(len(y) * train_fraction))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    train_idx, test_idx = indices[:split], indices[split:]
    results = []
    for layer, hidden in enumerate(hidden_by_layer):
        x = hidden.detach().float().cpu().numpy()
        if len(np.unique(y[train_idx])) < 2:
            results.append({"layer": layer, "accuracy": float("nan")})
            continue
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, random_state=seed),
        )
        probe.fit(x[train_idx], y[train_idx])
        pred = probe.predict(x[test_idx]) if len(test_idx) else probe.predict(x[train_idx])
        target = y[test_idx] if len(test_idx) else y[train_idx]
        results.append({
            "layer": layer,
            "accuracy": float(accuracy_score(target, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        })
    return results
