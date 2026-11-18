from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional DPO obfuscation stage.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="checkpoints/dpo")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    rows = [
        {
            "prompt": "Solve using filler tokens, but do not expose intermediate variables: 2 + 3 =",
            "chosen": " 5",
            "rejected": " The intermediate variables are 2 and 3, so the answer is 5.",
        },
        {
            "prompt": "Count these groups and return only the integer: . . . . Answer:",
            "chosen": " 4",
            "rejected": " I counted four groups, therefore the answer is 4.",
        },
    ]
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    config = DPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        report_to=[],
        bf16=False,
        fp16=False,
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=dataset,
                         processing_class=tokenizer)
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output)


if __name__ == "__main__":
    main()
