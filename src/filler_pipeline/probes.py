from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def _fit_linear(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Pipeline:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=0.5, random_state=seed),
    )
    pipe.fit(x_train, y_train)
    return pipe


def _fit_mlp(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> Pipeline:
    """Non-linear 2-layer MLP probe to detect computation relocated off linear axes."""
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-2,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        ),
    )
    pipe.fit(x_train, y_train)
    return pipe


def evaluate_all_probes(
    train_filler_mean: list[torch.Tensor],
    train_filler_all: list[list[torch.Tensor]],
    train_labels: list[int],
    eval_filler_mean: list[torch.Tensor],
    eval_filler_all: list[list[torch.Tensor]],
    eval_labels: list[int],
    seed: int = 42,
    pretrained_linear_probes: dict[int, Pipeline] | None = None,
) -> tuple[list[dict], dict[int, Pipeline]]:
    """Train/eval Mean-Linear, Max-Token Linear, Non-Linear MLP, and Shuffled-Label Control probes.

    When `pretrained_linear_probes` is supplied (from the baseline model), also evaluates
    static-probe transfer on the adversarially obfuscated model (Gupta & Jenner, 2026).
    """
    y_train = np.asarray(train_labels)
    y_eval = np.asarray(eval_labels)
    rng = np.random.default_rng(seed)
    y_train_shuffled = rng.permutation(y_train)

    fitted_linear: dict[int, Pipeline] = {}
    rows: list[dict] = []

    for layer in range(len(train_filler_mean)):
        x_tr = train_filler_mean[layer].detach().float().numpy()
        x_ev = eval_filler_mean[layer].detach().float().numpy()

        if len(np.unique(y_train)) < 2:
            rows.append({"layer": layer, "linear_mean_acc": float("nan")})
            continue

        # 1. Retrained Mean-Pooled Linear Probe
        lin_probe = _fit_linear(x_tr, y_train, seed)
        fitted_linear[layer] = lin_probe
        pred_lin = lin_probe.predict(x_ev)

        # 2. Max-Pooled Token-Wise Linear Probe (score each filler token, take max-confidence token)
        tok_preds_max = []
        classes = lin_probe.classes_
        for i in range(len(y_eval)):
            tok_x = eval_filler_all[layer][i].detach().float().numpy()
            tok_probs = lin_probe.predict_proba(tok_x)  # (P_i, C)
            best_tok = int(np.argmax(tok_probs.max(axis=-1)))
            tok_preds_max.append(classes[int(np.argmax(tok_probs[best_tok]))])
        pred_max = np.asarray(tok_preds_max)

        # 3. Non-Linear MLP Probe (tests non-linear subspace encoding)
        mlp_probe = _fit_mlp(x_tr, y_train, seed)
        pred_mlp = mlp_probe.predict(x_ev)

        # 4. Shuffled-Label Selectivity Control (Hewitt & Liang)
        ctrl_probe = _fit_linear(x_tr, y_train_shuffled, seed + 1)
        pred_ctrl = ctrl_probe.predict(x_ev)

        row = {
            "layer": layer,
            "linear_mean_acc": float(accuracy_score(y_eval, pred_lin)),
            "linear_mean_bal_acc": float(balanced_accuracy_score(y_eval, pred_lin)),
            "linear_max_acc": float(accuracy_score(y_eval, pred_max)),
            "mlp_nonlinear_acc": float(accuracy_score(y_eval, pred_mlp)),
            "shuffled_control_acc": float(accuracy_score(y_eval, pred_ctrl)),
        }

        # 5. Static Baseline Linear Probe Transfer (measures evasion of fixed monitor)
        if pretrained_linear_probes is not None and layer in pretrained_linear_probes:
            static_pred = pretrained_linear_probes[layer].predict(x_ev)
            row["static_baseline_linear_acc"] = float(accuracy_score(y_eval, static_pred))

        rows.append(row)

    return rows, fitted_linear


def save_probes(probes: dict[int, Pipeline], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(probes, f)


def load_probes(path: str | Path) -> dict[int, Pipeline]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def linear_probe_weights_torch(probe: Pipeline, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Extract StandardScaler + LogisticRegression weights into differentiable PyTorch tensors."""
    scaler: StandardScaler = probe.named_steps["standardscaler"]
    clf: LogisticRegression = probe.named_steps["logisticregression"]
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).clamp_min(1e-6)
    weight = torch.tensor(clf.coef_, dtype=torch.float32, device=device)
    bias = torch.tensor(clf.intercept_, dtype=torch.float32, device=device)
    return mean, scale, weight, bias, [int(c) for c in clf.classes_]
