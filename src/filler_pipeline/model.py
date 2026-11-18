from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, revision: str | None = None, device_map: str = "auto",
               trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "revision": revision,
        "device_map": device_map,
        "torch_dtype": "auto",
        "trust_remote_code": trust_remote_code,
        "output_hidden_states": True,
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def tokenize(tokenizer, prompts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )


@torch.no_grad()
def forward_hidden(model, tokenizer, prompts: list[str], max_length: int):
    batch = tokenize(tokenizer, prompts, max_length)
    batch = {key: value.to(model_device(model)) for key, value in batch.items()}
    output = model(**batch, output_hidden_states=True, return_dict=True)
    return batch, output


def answer_token_id(tokenizer, answer: str) -> int:
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Answer is not tokenizable: {answer!r}")
    return ids[0]


@contextmanager
def temporary_hooks(model, hooks: list[tuple[torch.nn.Module, object]]) -> Iterator[None]:
    handles = [module.register_forward_hook(fn) for module, fn in hooks]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
