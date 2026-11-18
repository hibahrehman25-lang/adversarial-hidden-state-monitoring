from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from .lenses import logit_lens_accuracy
from .model import forward_hidden, load_model, tokenize, model_device
from .patching import mask_attention_output
from .probes import last_nonpad, train_layer_probes
from .tasks import make_dataset, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config["data"]["seed"])
    torch.manual_seed(seed)
    output_dir = Path(config["analysis"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train = make_dataset(config["data"]["train_size"], seed)
    evaluation = make_dataset(config["data"]["eval_size"], seed + 1)
    write_jsonl(train, output_dir / "train_tasks.jsonl")
    write_jsonl(evaluation, output_dir / "eval_tasks.jsonl")
    model, tokenizer = load_model(**{
        key: value for key, value in config["model"].items()
        if key in {"name", "revision", "device_map", "trust_remote_code"} and value is not None
    })
    prompts = [task.prompt for task in evaluation]
    labels = [task.latent for task in evaluation]
    answers = [task.answer for task in evaluation]
    batch, output = forward_hidden(
        model, tokenizer, prompts, int(config["analysis"]["max_length"])
    )
    hidden_by_layer = [last_nonpad(hidden, batch["attention_mask"]).cpu()
                       for hidden in output.hidden_states]
    probe_rows = train_layer_probes(
        hidden_by_layer, labels, seed=seed
    )
    lens_rows = logit_lens_accuracy(
        model, tokenizer, output.hidden_states, batch["attention_mask"], answers
    )
    answer_ids = torch.tensor(
        [tokenizer.encode(answer, add_special_tokens=False)[0] for answer in answers],
        device=model_device(model),
    )
    last_positions = batch["attention_mask"].sum(dim=1).clamp_min(1) - 1
    requested_layers = config["analysis"].get("patch_layers") or []
    valid_layers = [
        layer for layer in requested_layers if 0 <= layer < len(output.hidden_states) - 1
    ]
    masked_rows = []
    for layer in valid_layers:
        with mask_attention_output(model, [layer]):
            masked_batch = tokenize(tokenizer, prompts, int(config["analysis"]["max_length"]))
            masked_batch = {
                key: value.to(model_device(model)) for key, value in masked_batch.items()
            }
            masked_output = model(**masked_batch, return_dict=True)
        masked_logits = masked_output.logits[
            torch.arange(len(prompts), device=model_device(model)), last_positions
        ]
        masked_rows.append({
            "layer": layer,
            "answer_accuracy": float(
                (masked_logits.argmax(dim=-1) == answer_ids).float().mean()
            ),
        })
    result = {
        "model": config["model"]["name"],
        "n_examples": len(evaluation),
        "probe": probe_rows,
        "logit_lens": lens_rows,
        "attention_masking": masked_rows,
        "limitations": [
            "This pilot measures answer-token and latent-label recoverability; it does not claim paper-level results.",
            "DPO is optional and must be run separately before comparing baseline and obfuscated checkpoints.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
