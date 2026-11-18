from __future__ import annotations

import torch


def logit_lens_accuracy(model, tokenizer, hidden_states, attention_mask, answers: list[str]) -> list[dict]:
    """Decode every residual stream layer with the final normalization and LM head."""
    device = next(model.parameters()).device
    answer_ids = torch.tensor(
        [tokenizer.encode(answer, add_special_tokens=False)[0] for answer in answers],
        device=device,
    )
    positions = attention_mask.sum(dim=1).clamp_min(1) - 1
    rows = []
    norm = getattr(model, "model", model)
    final_norm = getattr(norm, "norm", None) or getattr(model, "transformer", model).ln_f
    for layer, hidden in enumerate(hidden_states):
        selected = hidden[torch.arange(hidden.size(0), device=device), positions]
        logits = model.lm_head(final_norm(selected))
        pred = logits.argmax(dim=-1)
        rows.append({"layer": layer, "accuracy": float((pred == answer_ids).float().mean())})
    return rows


class AffineLens(torch.nn.Module):
    """Small tuned-lens-style affine translator, trained against final logits."""

    def __init__(self, width: int):
        super().__init__()
        self.linear = torch.nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)
