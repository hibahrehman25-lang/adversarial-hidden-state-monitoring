from __future__ import annotations

import gc
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .tasks import FillerTask


@dataclass
class ExtractedBatch:
    """CPU-cached, memory-efficient hidden states for filler spans & answer positions."""
    filler_mean: list[torch.Tensor]   # [n_layers] each (N, D) float16
    filler_all: list[list[torch.Tensor]]  # [n_layers][N] each (P_i, D) float16
    answer_pos_hidden: list[torch.Tensor] # [n_layers] each (N, D) float16
    final_logits: torch.Tensor        # (N, V) float16 at answer position
    predictions: list[str]            # Decoded integer prediction per example
    task_correct: list[bool]


def unwrap_causal_lm(model):
    """Safely unwrap PeftModel / DataParallel wrappers to access lm_head, layers, norm."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def get_layers(model):
    base_lm = unwrap_causal_lm(model)
    inner = getattr(base_lm, "model", base_lm)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(getattr(inner, "transformer", inner), "h", None)
    if layers is None:
        raise ValueError("Could not locate transformer layers.")
    return layers


def get_final_norm_and_head(model):
    base_lm = unwrap_causal_lm(model)
    inner = getattr(base_lm, "model", base_lm)
    norm = getattr(inner, "norm", None) or getattr(getattr(inner, "transformer", inner), "ln_f")
    lm_head = getattr(base_lm, "lm_head")
    return norm, lm_head


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_model(
    name: str,
    revision: str | None = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    load_in_4bit: bool = True,
    adapter_path: str | None = None,
):
    """Load LLM in T4-safe 4-bit NF4 (float16 compute) with optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict = {
        "revision": revision,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # T4 Turing uses fp16, not bf16
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        kwargs["attn_implementation"] = "eager"  # Needed for custom 4D attention masking
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def locate_token_spans(tokenizer, prompt: str, filler_text: str, max_length: int) -> dict:
    """Locate exact (start, end) token indices for the final question, filler span, and answer slot."""
    enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    offsets = enc["offset_mapping"]
    seq_len = len(enc["input_ids"])

    # Locate the LAST Question: and LAST Filler: occurrences (skipping few-shot examples)
    q_char_start = prompt.rfind("Question:")
    f_marker = prompt.rfind("Filler: ")
    a_char_start = prompt.rfind("\nAnswer:")

    q_char_end = f_marker if f_marker > q_char_start else a_char_start
    f_char_start = f_marker + len("Filler: ") if (f_marker != -1 and filler_text) else -1
    f_char_end = a_char_start if f_char_start != -1 else -1

    def char_to_tok_span(c_start: int, c_end: int) -> tuple[int, int]:
        if c_start < 0 or c_end <= c_start:
            return (seq_len - 1, seq_len)
        toks = [
            i for i, (s, e) in enumerate(offsets)
            if e > s and s < c_end and e > c_start
        ]
        if not toks:
            return (max(0, seq_len - 2), seq_len - 1)
        return (toks[0], toks[-1] + 1)

    return {
        "input_ids": enc["input_ids"],
        "seq_len": seq_len,
        "question_span": char_to_tok_span(q_char_start, q_char_end),
        "filler_span": char_to_tok_span(f_char_start, f_char_end),
        "answer_pos": seq_len - 1,
    }


def parse_integer(text: str) -> str:
    match = re.search(r"-?\d+", text.strip())
    return match.group(0) if match else text.strip()


def digit_token_ids(tokenizer) -> dict[int, list[int]]:
    """Map digits 0..9 to valid single-token IDs (handles with/without leading space)."""
    mapping: dict[int, list[int]] = {}
    for d in range(10):
        cands = set()
        for s in (str(d), f" {d}"):
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                cands.add(ids[0])
        mapping[d] = sorted(cands)
    return mapping


@torch.no_grad()
def extract_filler_states(
    model,
    tokenizer,
    tasks: list[FillerTask],
    max_length: int = 512,
    batch_size: int = 4,
    max_new_tokens: int = 6,
) -> ExtractedBatch:
    """Mini-batched T4-safe extraction of filler-span & answer-position hidden states + exact accuracy."""
    device = model_device(model)
    n_layers = len(get_layers(model)) + 1  # embeddings + transformer blocks

    filler_mean_acc: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    filler_all_acc: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    ans_hidden_acc: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    final_logits_acc: list[torch.Tensor] = []
    predictions: list[str] = []
    task_correct: list[bool] = []

    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        spans = [locate_token_spans(tokenizer, t.prompt, t.filler_text, max_length) for t in chunk]
        enc = tokenizer(
            [t.prompt for t in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        out = model(**enc, output_hidden_states=True, return_dict=True)

        for i, (t, sp) in enumerate(zip(chunk, spans)):
            f_start, f_end = sp["filler_span"]
            a_pos = sp["answer_pos"]
            final_logits_acc.append(out.logits[i, a_pos].detach().to(torch.float16).cpu())
            for layer_idx, h in enumerate(out.hidden_states):
                f_slice = h[i, f_start:f_end].detach().to(torch.float16).cpu()
                filler_all_acc[layer_idx].append(f_slice)
                filler_mean_acc[layer_idx].append(f_slice.mean(dim=0))
                ans_hidden_acc[layer_idx].append(h[i, a_pos].detach().to(torch.float16).cpu())

        # Greedy generation for exact multi-digit answer accuracy
        del out
        gen_tok = tokenizer(
            [t.prompt for t in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        # Left-pad temporarily for batched generation
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        gen_inputs = tokenizer(
            [t.prompt for t in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        tokenizer.padding_side = orig_side

        gen_ids = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = gen_inputs["input_ids"].shape[1]
        for i, t in enumerate(chunk):
            raw_text = tokenizer.decode(gen_ids[i, prompt_len:], skip_special_tokens=True)
            pred = parse_integer(raw_text)
            predictions.append(pred)
            task_correct.append(pred == t.answer)

        del enc, gen_inputs, gen_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gc.collect()
    return ExtractedBatch(
        filler_mean=[torch.stack(rows, dim=0) for rows in filler_mean_acc],
        filler_all=filler_all_acc,
        answer_pos_hidden=[torch.stack(rows, dim=0) for rows in ans_hidden_acc],
        final_logits=torch.stack(final_logits_acc, dim=0),
        predictions=predictions,
        task_correct=task_correct,
    )


@contextmanager
def temporary_hooks(hooks: list[tuple[torch.nn.Module, object]]) -> Iterator[None]:
    handles = [module.register_forward_hook(fn) for module, fn in hooks]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
