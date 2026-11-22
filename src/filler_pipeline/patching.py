from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import get_layers, locate_token_spans, model_device, parse_integer
from .tasks import FillerTask


@torch.no_grad()
def causal_filler_patching(
    model,
    tokenizer,
    pairs: list[tuple[FillerTask, FillerTask]],
    layer_bands: list[tuple[int, int]],
    max_length: int = 512,
) -> list[dict]:
    """Donor -> Target causal activation patching strictly on the filler-token span.

    Measures:
      - logit_diff_recovery: shift toward donor's answer first token vs target's answer first token
      - answer_flip_rate: fraction of targets whose prediction flips to donor's exact answer
    """
    device = model_device(model)
    layers = get_layers(model)
    results: list[dict] = []

    for l_start, l_end in layer_bands:
        band_layers = list(range(l_start, min(l_end, len(layers))))
        recoveries: list[float] = []
        flips = 0
        valid = 0

        for donor, target in pairs:
            sp_d = locate_token_spans(tokenizer, donor.prompt, donor.filler_text, max_length)
            sp_t = locate_token_spans(tokenizer, target.prompt, target.filler_text, max_length)
            d_f0, d_f1 = sp_d["filler_span"]
            t_f0, t_f1 = sp_t["filler_span"]
            span_len = min(d_f1 - d_f0, t_f1 - t_f0)
            if span_len <= 0:
                continue

            d_ans_id = tokenizer.encode(donor.answer, add_special_tokens=False)[0]
            t_ans_id = tokenizer.encode(target.answer, add_special_tokens=False)[0]
            if d_ans_id == t_ans_id:
                continue

            # 1. Clean donor pass: capture residual activations at filler positions
            enc_d = tokenizer(donor.prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            out_d = model(**enc_d, output_hidden_states=True, return_dict=True)
            donor_cache = {
                l: out_d.hidden_states[l + 1][0, d_f0 : d_f0 + span_len].detach().clone()
                for l in band_layers
            }
            donor_logit_diff = float(
                out_d.logits[0, sp_d["answer_pos"], d_ans_id] - out_d.logits[0, sp_d["answer_pos"], t_ans_id]
            )
            del out_d

            # 2. Clean target pass
            enc_t = tokenizer(target.prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            out_t = model(**enc_t, return_dict=True)
            clean_logit_diff = float(
                out_t.logits[0, sp_t["answer_pos"], d_ans_id] - out_t.logits[0, sp_t["answer_pos"], t_ans_id]
            )
            del out_t

            # 3. Patched target pass: overwrite ONLY filler-token hidden states at band_layers
            handles = []
            for l in band_layers:
                cached = donor_cache[l]

                def make_hook(c_tensor, start_pos, length):
                    def hook_fn(_module, _inp, output):
                        h = output[0] if isinstance(output, tuple) else output
                        h = h.clone()
                        h[:, start_pos : start_pos + length, :] = c_tensor.unsqueeze(0)
                        return (h,) + output[1:] if isinstance(output, tuple) else h
                    return hook_fn

                handles.append(layers[l].register_forward_hook(make_hook(cached, t_f0, span_len)))

            try:
                out_p = model(**enc_t, return_dict=True)
                patched_logit_diff = float(
                    out_p.logits[0, sp_t["answer_pos"], d_ans_id] - out_p.logits[0, sp_t["answer_pos"], t_ans_id]
                )
                pred_tok = tokenizer.decode([int(out_p.logits[0, sp_t["answer_pos"]].argmax())]).strip()
                if parse_integer(pred_tok) == donor.answer:
                    flips += 1
            finally:
                for h in handles:
                    h.remove()

            denom = donor_logit_diff - clean_logit_diff
            if abs(denom) > 1e-4:
                recoveries.append((patched_logit_diff - clean_logit_diff) / denom)
            valid += 1

        results.append({
            "layer_band": f"L{l_start}-L{l_end - 1}",
            "n_pairs": valid,
            "mean_logit_recovery": float(sum(recoveries) / max(1, len(recoveries))),
            "answer_flip_rate": float(flips / max(1, valid)),
        })

    return results


@torch.no_grad()
def pathway_attention_masking(
    model,
    tokenizer,
    tasks: list[FillerTask],
    max_length: int = 512,
) -> dict[str, float]:
    """Block specific attention pathways using an explicit 4D additive causal mask (Baherwani et al., 2026)."""
    device = model_device(model)
    dtype = next(model.parameters()).dtype
    min_val = torch.finfo(dtype).min

    pathways = ["unmasked", "block_answer_to_filler", "block_filler_to_question", "block_filler_self"]
    correct_counts = {p: 0 for p in pathways}

    for task in tasks:
        sp = locate_token_spans(tokenizer, task.prompt, task.filler_text, max_length)
        enc = tokenizer(task.prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        seq_len = enc["input_ids"].shape[1]
        q0, q1 = sp["question_span"]
        f0, f1 = sp["filler_span"]
        a_pos = sp["answer_pos"]
        ans_id = tokenizer.encode(task.answer, add_special_tokens=False)[0]

        # Standard causal lower-triangular additive mask (1, 1, S, S)
        base_mask = torch.triu(torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype), diagonal=1)

        for pathway in pathways:
            mask = base_mask.clone()
            if pathway == "block_answer_to_filler":
                mask[a_pos, f0:f1] = min_val
            elif pathway == "block_filler_to_question":
                mask[f0:f1, q0:q1] = min_val
            elif pathway == "block_filler_self":
                # Keep diagonal (self-token) unmasked to prevent NaN softmax
                sub = mask[f0:f1, f0:f1]
                eye = torch.eye(f1 - f0, device=device, dtype=torch.bool)
                sub[~eye] = min_val
                mask[f0:f1, f0:f1] = sub

            out = model(input_ids=enc["input_ids"], attention_mask=mask.unsqueeze(0).unsqueeze(0), return_dict=True)
            if int(out.logits[0, a_pos].argmax()) == ans_id:
                correct_counts[pathway] += 1

    return {p: float(c / max(1, len(tasks))) for p, c in correct_counts.items()}


@torch.no_grad()
def cross_layer_activation_transfer(
    model,
    tokenizer,
    tasks: list[FillerTask],
    src_layer: int,
    dst_layer: int,
    max_length: int = 512,
) -> dict[str, float]:
    """Transfer filler-span activations from src_layer into dst_layer to test architectural shift."""
    device = model_device(model)
    layers = get_layers(model)
    clean_ok = 0
    transfer_ok = 0

    for task in tasks:
        sp = locate_token_spans(tokenizer, task.prompt, task.filler_text, max_length)
        f0, f1 = sp["filler_span"]
        a_pos = sp["answer_pos"]
        ans_id = tokenizer.encode(task.answer, add_special_tokens=False)[0]
        enc = tokenizer(task.prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        out = model(**enc, output_hidden_states=True, return_dict=True)
        if int(out.logits[0, a_pos].argmax()) == ans_id:
            clean_ok += 1
        src_act = out.hidden_states[src_layer + 1][0, f0:f1].detach().clone()

        def hook_fn(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[:, f0:f1, :] = src_act.unsqueeze(0)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        handle = layers[dst_layer].register_forward_hook(hook_fn)
        try:
            out_t = model(**enc, return_dict=True)
            if int(out_t.logits[0, a_pos].argmax()) == ans_id:
                transfer_ok += 1
        finally:
            handle.remove()

    n = max(1, len(tasks))
    return {
        "src_layer": src_layer,
        "dst_layer": dst_layer,
        "clean_first_tok_acc": float(clean_ok / n),
        "transferred_first_tok_acc": float(transfer_ok / n),
        "accuracy_drop": float((clean_ok - transfer_ok) / n),
    }
