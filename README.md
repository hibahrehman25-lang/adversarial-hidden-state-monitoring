# Filler-token hidden-state monitoring pipeline

This is an original, small research scaffold for the proposed study **Adversarial
Robustness of Hidden-State Monitoring for Filler Token Computation**. It uses the
referenced projects for method selection and interfaces, not copied source code or
datasets:

- TransformerLens / NNsight: activation access and interventions.
- fillerTokens: dot/filler task design.
- tuned-lens: layer-wise affine translation idea.
- TRL, deception-evasion-honesty, and RL_Obfuscation: optional DPO/obfuscation stage.
- audit-stress-test: adversarial evaluation and baseline-vs-stressed comparisons.

## Files and languages

There are **13 files**: 10 Python/YAML/project files plus README, tests, and gitignore.
The implementation is Python 3.10+; YAML stores experiment settings. The core
pipeline deliberately uses Hugging Face Transformers so it works without paid APIs.
TransformerLens, NNsight, tuned-lens, TRL, PEFT, and bitsandbytes are optional extras.

## Recommended models and hardware

For a free student pilot use `Qwen/Qwen2.5-0.5B` on CPU or a 6--8 GB GPU. For a
serious reproduction, use `meta-llama/Meta-Llama-3-8B` and
`Qwen/Qwen2.5-7B` with Hugging Face access, one 24 GB GPU with 4-bit LoRA, or
multi-GPU inference. Llama access may require accepting its model license. The
7B/8B DPO stage is not realistic on a normal laptop; use QLoRA and a GPU.

## Run the free pilot

```bash
cd /root/filler-monitoring-pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# For a CPU-only laptop, install the smaller PyTorch wheel first:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
python -m filler_pipeline.run_pipeline --config configs/pilot.yaml
```

The first run downloads the model from Hugging Face. Results are written to
`artifacts/pilot/metrics.json`. No API key, WandB account, or paid service is
required.

## Optional obfuscation stage

```bash
pip install -e ".[training]"
python -m filler_pipeline.train_dpo \
  --model Qwen/Qwen2.5-0.5B \
  --output checkpoints/dpo \
  --steps 100
```

For a valid comparison, run the analysis once on the base checkpoint and again
with the DPO checkpoint by changing `model.name` in a copied YAML config. Increase
the task sizes, hold out task templates, and repeat across at least three seeds.

## What this implementation covers

It generates dot-pattern arithmetic/counting tasks; extracts all residual-stream
layers; trains layer-wise linear probes; computes a basic logit lens; saves exact
task metadata; and provides reusable hooks for layer output patching and attention
output masking. The optional DPO script creates a minimal chosen/rejected dataset
and uses TRL.

## What remains for a paper-quality reproduction

The project does not yet include a full tuned-lens training loop, a large
held-out benchmark, calibrated statistical tests, multi-seed confidence intervals,
non-linear probes, tokenizer-aligned intermediate-variable labels, or a full
activation-patching sweep. Those should be added before making claims such as
“82% obscured” or “68.4% reduction”; this scaffold reports measurements and does
not pre-fill those outcomes.
