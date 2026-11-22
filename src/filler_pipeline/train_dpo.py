from __future__ import annotations

import gc
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

from .model import (
    digit_token_ids,
    get_final_norm_and_head,
    get_layers,
    load_model,
    locate_token_spans,
    model_device,
)
from .probes import linear_probe_weights_torch, load_probes
from .tasks import FillerTask, make_dataset


@torch.no_grad()
def _score_rollout_monitor(
    model,
    tokenizer,
    full_text: str,
    filler_text: str,
    latent: int,
    mid_layers: list[int],
    probe_tensors: dict[int, tuple],
    d_map: dict[int, list[int]],
    max_length: int,
) -> float:
    """Compute white-box monitor recoverability (avg of Logit Lens prob + Linear Probe prob on z)."""
    device = model_device(model)
    final_norm, lm_head = get_final_norm_and_head(model)
    sp = locate_token_spans(tokenizer, full_text, filler_text, max_length)
    f0, f1 = sp["filler_span"]
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**enc, output_hidden_states=True, return_dict=True)

    scores = []
    target_tok_ids = d_map[latent]
    for l in mid_layers:
        h_filler = out.hidden_states[l][0, f0:f1]
        h_mean = h_filler.mean(dim=0, keepdim=True)

        logits = lm_head(final_norm(h_mean))
        probs = F.softmax(logits.float(), dim=-1)
        scores.append(float(probs[0, target_tok_ids].sum()))

        if l in probe_tensors:
            mean, scale, weight, bias, classes = probe_tensors[l]
            if latent in classes:
                c_idx = classes.index(latent)
                x_norm = (h_mean.float().to(mean.device) - mean) / scale
                p_logits = F.linear(x_norm, weight, bias)
                scores.append(float(F.softmax(p_logits, dim=-1)[0, c_idx]))

    return float(sum(scores) / max(1, len(scores)))


def build_dpo_preference_dataset(
    model,
    tokenizer,
    tasks: list[FillerTask],
    baseline_probes: dict,
    lambda_obf: float = 1.0,
    max_length: int = 512,
) -> tuple[Dataset, list[dict]]:
    """Construct preference pairs (prompt, chosen, rejected) scored against white-box monitors."""
    device = model_device(model)
    n_layers = len(get_layers(model)) + 1
    mid_layers = list(range(n_layers // 3, (2 * n_layers) // 3))
    d_map = digit_token_ids(tokenizer)
    probe_tensors = {
        l: linear_probe_weights_torch(p, device)
        for l, p in baseline_probes.items()
        if l in mid_layers
    }

    rows = []
    filler_variants = [
        ("dots", " ".join(["."] * 12)),
        ("dots_dense", "." * 12),
        ("count", " ".join(str(i) for i in range(1, 13))),
        ("dots_alt", " . . . . . . . . . . . ."),
    ]

    for t in tasks:
        cut = t.prompt.rfind("Filler:")
        base_prompt = t.prompt[:cut] + "Filler:"
        scored_candidates = []

        for _vname, f_str in filler_variants:
            completion_correct = f" {f_str}\nAnswer: {t.answer}"
            full_text = base_prompt + completion_correct
            mon_score = _score_rollout_monitor(
                model, tokenizer, full_text, f_str, t.latent, mid_layers, probe_tensors, d_map, max_length
            )
            reward = 1.0 - lambda_obf * mon_score
            scored_candidates.append((reward, mon_score, completion_correct, f_str))

        leaked_rejected = f" {t.filler_text} (z={t.latent})\nAnswer: {t.answer}"
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best = scored_candidates[0]
        worst = scored_candidates[-1]
        rejected_str = worst[2] if worst[2] != best[2] else leaked_rejected

        rows.append({
            "prompt": base_prompt,
            "chosen": best[2],
            "rejected": rejected_str,
            "latent": t.latent,
            "chosen_mon_score": best[1],
            "rejected_mon_score": worst[1],
        })

    return Dataset.from_list(rows), rows


class ObfuscationDPOTrainer(DPOTrainer):
    """DPOTrainer with optional differentiable middle-layer representation obfuscation penalty."""

    def __init__(
        self,
        *args,
        rep_obf_weight: float = 0.5,
        mid_layers: list[int] | None = None,
        probe_tensors: dict | None = None,
        digit_map: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rep_obf_weight = rep_obf_weight
        self.mid_layers = mid_layers or []
        self.probe_tensors = probe_tensors or {}
        self.digit_map = digit_map or {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sig = inspect.signature(super().compute_loss)
        if "num_items_in_batch" in sig.parameters:
            base_out = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        else:
            base_out = super().compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            dpo_loss, outputs = base_out
        else:
            dpo_loss, outputs = base_out, None

        if self.rep_obf_weight <= 0.0 or not self.mid_layers:
            return (dpo_loss, outputs) if return_outputs else dpo_loss

        # Handle both combined and prompt+completion TRL batch formats safely
        if "chosen_input_ids" in inputs:
            chosen_ids = inputs["chosen_input_ids"]
            chosen_mask = inputs.get("chosen_attention_mask")
            if "prompt_input_ids" in inputs:
                chosen_ids = torch.cat([inputs["prompt_input_ids"], chosen_ids], dim=1)
                if chosen_mask is not None and "prompt_attention_mask" in inputs:
                    chosen_mask = torch.cat([inputs["prompt_attention_mask"], chosen_mask], dim=1)
        elif "input_ids" in inputs:
            chosen_ids = inputs["input_ids"]
            chosen_mask = inputs.get("attention_mask")
        else:
            return (dpo_loss, outputs) if return_outputs else dpo_loss

        out = model(
            input_ids=chosen_ids,
            attention_mask=chosen_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        final_norm, lm_head = get_final_norm_and_head(model)
        all_digit_ids = [ids[0] for ids in self.digit_map.values()]
        lm_w_digits = lm_head.weight[all_digit_ids].detach().float()

        seq_len = chosen_ids.shape[1]
        s_idx = max(0, seq_len - 18)
        e_idx = max(s_idx + 1, seq_len - 2)

        obf_terms = []
        for l in self.mid_layers:
            h_mid = out.hidden_states[l][:, s_idx:e_idx, :].mean(dim=1)
            h_norm = final_norm(h_mid).float()
            digit_logits = F.linear(h_norm, lm_w_digits.to(h_norm.device))
            digit_probs = F.softmax(digit_logits, dim=-1)
            uniform = torch.full_like(digit_probs, 1.0 / digit_probs.shape[-1])
            obf_terms.append(F.kl_div(digit_probs.clamp_min(1e-8).log(), uniform, reduction="batchmean"))

            if l in self.probe_tensors:
                mean, scale, weight, bias, _classes = self.probe_tensors[l]
                x_norm = (h_mid.float().to(mean.device) - mean) / scale
                p_logits = F.linear(x_norm, weight, bias)
                p_probs = F.softmax(p_logits, dim=-1)
                p_unif = torch.full_like(p_probs, 1.0 / p_probs.shape[-1])
                obf_terms.append(F.kl_div(p_probs.clamp_min(1e-8).log(), p_unif, reduction="batchmean"))

        rep_loss = torch.stack(obf_terms).mean().to(dpo_loss.device)
        total_loss = dpo_loss + self.rep_obf_weight * rep_loss
        return (total_loss, outputs) if return_outputs else total_loss


def run_dpo_training(
    model_name: str,
    baseline_probes_path: str | Path,
    output_dir: str | Path,
    filler_type: str = "dots",
    filler_length: int = 12,
    n_shots: int = 5,
    task_types: list[str] | tuple[str, ...] = ("chained_eq", "mod_algo"),
    dpo_prompts: int = 100,
    max_steps: int = 60,
    learning_rate: float = 1e-4,
    beta: float = 0.1,
    lambda_obf: float = 1.0,
    rep_obf_weight: float = 0.5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
    max_length: int = 512,
) -> Path:
    """End-to-end T4-safe QLoRA Adversarial DPO training."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_name, load_in_4bit=True)
    baseline_probes = load_probes(baseline_probes_path)

    tasks = make_dataset(
        size=dpo_prompts,
        seed=seed + 5000,
        filler_type=filler_type,
        filler_length=filler_length,
        n_shots=n_shots,
        task_types=task_types,
        condition="obfuscation_prompt",
    )
    pref_ds, raw_rows = build_dpo_preference_dataset(
        model, tokenizer, tasks, baseline_probes, lambda_obf=lambda_obf, max_length=max_length
    )
    (out_path / "dpo_pairs_preview.json").write_text(json.dumps(raw_rows[:10], indent=2))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False

    device = model_device(model)
    n_layers = len(get_layers(model)) + 1
    mid_layers = list(range(n_layers // 3, (2 * n_layers) // 3, max(1, n_layers // 12)))
    probe_tensors = {
        l: linear_probe_weights_torch(p, device)
        for l, p in baseline_probes.items()
        if l in mid_layers
    }

    # Filter kwargs dynamically to remain compatible across TRL versions
    dpo_sig = inspect.signature(DPOConfig.__init__).parameters
    raw_cfg = {
        "output_dir": str(out_path),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "beta": beta,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "optim": "paged_adamw_8bit",
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": [],
        "max_length": max_length,
        "max_prompt_length": max_length - 64,
        "remove_unused_columns": True,
    }
    dpo_cfg = DPOConfig(**{k: v for k, v in raw_cfg.items() if k in dpo_sig})

    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    tok_kwarg = "processing_class" if "processing_class" in trainer_sig else "tokenizer"

    trainer = ObfuscationDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_cfg,
        train_dataset=pref_ds.remove_columns(["latent", "chosen_mon_score", "rejected_mon_score"]),
        rep_obf_weight=rep_obf_weight,
        mid_layers=mid_layers,
        probe_tensors=probe_tensors,
        digit_map=digit_token_ids(tokenizer),
        **{tok_kwarg: tokenizer},
    )
    trainer.train()
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path
