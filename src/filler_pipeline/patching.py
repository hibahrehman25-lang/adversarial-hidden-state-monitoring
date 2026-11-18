from __future__ import annotations

from contextlib import contextmanager

import torch


def _layer_modules(model):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        layers = getattr(getattr(base, "transformer", base), "h", None)
    if layers is None:
        raise ValueError("Could not locate transformer layers for this model.")
    return layers


@contextmanager
def patch_layer_output(model, layer: int, source: torch.Tensor):
    layers = _layer_modules(model)
    target = layers[layer]

    def replace(_module, _inputs, output):
        if isinstance(output, tuple):
            return (source,) + output[1:]
        return source

    handle = target.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def mask_attention_output(model, layers_to_mask: list[int]):
    layers = _layer_modules(model)
    handles = []

    def zero_output(_module, _inputs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    for index in layers_to_mask:
        attention = getattr(layers[index], "self_attn", None)
        if attention is not None:
            handles.append(attention.register_forward_hook(zero_output))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
