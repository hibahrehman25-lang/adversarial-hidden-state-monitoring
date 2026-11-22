from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import digit_token_ids, get_final_norm_and_head, model_device


def _digit_probs_from_logits(logits: torch.Tensor, digit_map: dict[int, list[int]]) -> torch.Tensor:
    """Restrict logits to single-digit tokens 0..9 and return normalized probabilities (..., 10)."""
    probs_full = F.softmax(logits.float(), dim=-1)
    digit_scores = []
    for d in range(10):
        ids = digit_map[d]
        if len(ids) == 1:
            digit_scores.append(probs_full[..., ids[0]])
        else:
            digit_scores.append(probs_full[..., ids].sum(dim=-1))
    stacked = torch.stack(digit_scores, dim=-1)
    return stacked / stacked.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def evaluate_logit_lens(
    model,
    tokenizer,
    filler_all: list[list[torch.Tensor]],
    latents: list[int],
    use_residual_subtraction: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """Filler-span Logit Lens with optional cross-example mean residual subtraction (Brauer et al., 2026).

    Returns:
      - layer_rows: list of dicts with top1_acc, top2_acc, mean_target_prob per layer
      - heatmap: (n_layers, max_pos) exact-match fraction for publication Fig. 2
    """
    device = model_device(model)
    final_norm, lm_head = get_final_norm_and_head(model)
    d_map = digit_token_ids(tokenizer)
    targets = torch.tensor(latents, dtype=torch.long)

    n_layers = len(filler_all)
    n_examples = len(latents)
    min_pos = min(x.shape[0] for x in filler_all[0])
    heatmap = np.zeros((n_layers, min_pos), dtype=np.float32)
    rows: list[dict] = []

    for layer in range(n_layers):
        # Align across examples up to min_pos filler positions
        h_stack = torch.stack([filler_all[layer][i][:min_pos] for i in range(n_examples)], dim=0)
        # Process in chunks on GPU to stay T4-safe
        prob_chunks = []
        for s in range(0, n_examples, 16):
            h_gpu = h_stack[s : s + 16].to(device=device, dtype=next(final_norm.parameters()).dtype)
            logits = lm_head(final_norm(h_gpu))
            prob_chunks.append(_digit_probs_from_logits(logits, d_map).cpu())
        probs = torch.cat(prob_chunks, dim=0)  # (N, P, 10)

        if use_residual_subtraction:
            # Subtract cross-example mean distribution per position to remove filler artifacts
            scores = probs - probs.mean(dim=0, keepdim=True)
        else:
            scores = probs

        # Per-position exact match heatmap
        pos_preds = scores.argmax(dim=-1)  # (N, P)
        heatmap[layer] = (pos_preds == targets.unsqueeze(1)).float().mean(dim=0).numpy()

        # Aggregate across filler positions (Brauer et al. Stage 3)
        agg = scores.sum(dim=1)  # (N, 10)
        top2 = agg.topk(2, dim=-1).indices
        top1_acc = float((top2[:, 0] == targets).float().mean())
        top2_acc = float(((top2 == targets.unsqueeze(1)).any(dim=-1)).float().mean())
        mean_target_prob = float(probs.mean(dim=1)[torch.arange(n_examples), targets].mean())

        rows.append({
            "layer": layer,
            "top1_accuracy": top1_acc,
            "top2_accuracy": top2_acc,
            "mean_target_prob": mean_target_prob,
            "peak_pos_accuracy": float(heatmap[layer].max()),
        })

    return rows, heatmap


class TunedLensTranslator(nn.Module):
    """Per-layer residual affine translator initialized to identity (Belrose et al., 2023)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.linear(h)


def train_and_evaluate_tuned_lens(
    model,
    tokenizer,
    train_filler_mean: list[torch.Tensor],
    train_final_logits: torch.Tensor,
    eval_filler_all: list[list[torch.Tensor]],
    eval_latents: list[int],
    steps: int = 120,
    lr: float = 5e-3,
) -> list[dict]:
    """Train per-layer affine Tuned Lens via KL against final logits, then evaluate on filler span."""
    device = model_device(model)
    final_norm, lm_head = get_final_norm_and_head(model)
    d_map = digit_token_ids(tokenizer)
    n_layers = len(train_filler_mean)
    hidden_dim = train_filler_mean[0].shape[-1]
    targets_eval = torch.tensor(eval_latents, dtype=torch.long)

    # Precompute target digit distribution from final layer logits (memory efficient vs full vocab)
    with torch.no_grad():
        target_digit_probs = _digit_probs_from_logits(train_final_logits.to(device), d_map).detach()

    # Precompute digit unembedding slice W_digit (10, D) in float32 for fast T4 training
    digit_indices = [d_map[d][0] for d in range(10)]
    lm_weight_digit = lm_head.weight[digit_indices].detach().float().to(device)  # (10, D)

    norm_dtype = next(final_norm.parameters()).dtype
    n_eval = len(eval_latents)
    min_pos = min(x.shape[0] for x in eval_filler_all[0])
    rows: list[dict] = []

    for layer in range(n_layers):
        lens = TunedLensTranslator(hidden_dim).to(device)
        opt = torch.optim.AdamW(lens.parameters(), lr=lr, weight_decay=1e-2)
        x_train = train_filler_mean[layer].float().to(device)

        lens.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            h_rot = lens(x_train)
            # Apply RMSNorm approximately via model final_norm
            h_norm = final_norm(h_rot.to(norm_dtype)).float()
            logits_10 = F.linear(h_norm, lm_weight_digit)
            log_probs = F.log_softmax(logits_10, dim=-1)
            loss = F.kl_div(log_probs, target_digit_probs, reduction="batchmean")
            loss.backward()
            opt.step()

        lens.eval()
        with torch.no_grad():
            h_eval = torch.stack([eval_filler_all[layer][i][:min_pos] for i in range(n_eval)], dim=0)
            prob_chunks = []
            for s in range(0, n_eval, 16):
                h_gpu = h_eval[s : s + 16].float().to(device)
                h_rot = lens(h_gpu)
                h_norm = final_norm(h_rot.to(norm_dtype)).float()
                logits_10 = F.linear(h_norm, lm_weight_digit)
                prob_chunks.append(F.softmax(logits_10, dim=-1).cpu())
            probs = torch.cat(prob_chunks, dim=0)  # (N, P, 10)
            scores = probs - probs.mean(dim=0, keepdim=True)
            agg = scores.sum(dim=1)
            top2 = agg.topk(2, dim=-1).indices
            rows.append({
                "layer": layer,
                "top1_accuracy": float((top2[:, 0] == targets_eval).float().mean()),
                "top2_accuracy": float(((top2 == targets_eval.unsqueeze(1)).any(dim=-1)).float().mean()),
            })

        del lens, opt, x_train
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows
