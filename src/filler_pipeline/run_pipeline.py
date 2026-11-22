from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .lenses import evaluate_logit_lens, train_and_evaluate_tuned_lens
from .model import extract_filler_states, load_model
from .patching import (
    causal_filler_patching,
    cross_layer_activation_transfer,
    pathway_attention_masking,
)
from .probes import evaluate_all_probes, load_probes, save_probes
from .tasks import make_dataset, make_matched_patching_pairs, write_jsonl
from .train_dpo import run_dpo_training


def _mid_layer_summary(rows: list[dict], key: str) -> float:
    n = len(rows)
    mid = rows[n // 3 : (2 * n) // 3]
    vals = [r[key] for r in mid if key in r and not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def _run_full_audit(
    model,
    tokenizer,
    cfg: dict,
    condition: str,
    save_prefix: Path,
    baseline_probes: dict | None = None,
) -> dict:
    """Run extraction, Logit Lens, Tuned Lens, all Probes, Patching, Masking, and Transfer."""
    dcfg = cfg["data"]
    acfg = cfg["analysis"]
    seed = int(dcfg["seed"])

    train_tasks = make_dataset(
        size=int(dcfg["train_size"]),
        seed=seed,
        filler_type=dcfg["filler_type"],
        filler_length=int(dcfg["filler_length"]),
        n_shots=int(dcfg["n_shots"]),
        task_types=dcfg["task_types"],
        condition=condition,
    )
    eval_tasks = make_dataset(
        size=int(dcfg["eval_size"]),
        seed=seed + 100,
        filler_type=dcfg["filler_type"],
        filler_length=int(dcfg["filler_length"]),
        n_shots=int(dcfg["n_shots"]),
        task_types=dcfg["task_types"],
        condition=condition,
    )

    # Also verify no-filler baseline uplift (Gate 1)
    nofiller_tasks = make_dataset(
        size=min(60, int(dcfg["eval_size"])),
        seed=seed + 100,
        filler_type="none",
        filler_length=0,
        n_shots=int(dcfg["n_shots"]),
        task_types=dcfg["task_types"],
        condition=condition,
    )
    ext_nf = extract_filler_states(
        model, tokenizer, nofiller_tasks, int(acfg["max_length"]), int(acfg["batch_size"])
    )
    nofiller_acc = float(np.mean(ext_nf.task_correct))
    del ext_nf

    ext_tr = extract_filler_states(
        model, tokenizer, train_tasks, int(acfg["max_length"]), int(acfg["batch_size"])
    )
    ext_ev = extract_filler_states(
        model, tokenizer, eval_tasks, int(acfg["max_length"]), int(acfg["batch_size"])
    )
    task_acc = float(np.mean(ext_ev.task_correct))

    # 1. Filler-Span Logit Lens + Heatmap
    ll_rows, heatmap = evaluate_logit_lens(
        model, tokenizer, ext_ev.filler_all, [t.latent for t in eval_tasks], use_residual_subtraction=True
    )
    np.save(save_prefix.parent / f"{save_prefix.name}_logit_heatmap.npy", heatmap)

    # 2. Trained Per-Layer Tuned Lens
    tl_rows = train_and_evaluate_tuned_lens(
        model,
        tokenizer,
        ext_tr.filler_mean,
        ext_tr.final_logits,
        ext_ev.filler_all,
        [t.latent for t in eval_tasks],
        steps=int(acfg["tuned_lens_steps"]),
        lr=float(acfg["tuned_lens_lr"]),
    )

    # 3. Linear (Mean & Max), Non-Linear MLP, Shuffled Control, and Static Transfer Probes
    probe_rows, fitted_linear = evaluate_all_probes(
        ext_tr.filler_mean,
        ext_tr.filler_all,
        [t.latent for t in train_tasks],
        ext_ev.filler_mean,
        ext_ev.filler_all,
        [t.latent for t in eval_tasks],
        seed=seed,
        pretrained_linear_probes=baseline_probes,
    )
    if baseline_probes is None:
        save_probes(fitted_linear, save_prefix.parent / "baseline_linear_probes.pkl")

    # 4. Causal Donor -> Target Activation Patching (Early, Mid, Late layer bands)
    n_blocks = len(ll_rows) - 1
    bands = [
        (0, n_blocks // 3),
        (n_blocks // 3, (2 * n_blocks) // 3),
        ((2 * n_blocks) // 3, n_blocks),
    ]
    patch_pairs = make_matched_patching_pairs(
        n_pairs=int(acfg["patch_pairs"]),
        seed=seed,
        filler_type=dcfg["filler_type"],
        filler_length=int(dcfg["filler_length"]),
        n_shots=int(dcfg["n_shots"]),
        task_type=dcfg["task_types"][0],
        condition=condition,
    )
    patch_rows = causal_filler_patching(model, tokenizer, patch_pairs, bands, int(acfg["max_length"]))

    # 5. Pathway-Specific Attention Masking
    attn_mask_metrics = pathway_attention_masking(
        model, tokenizer, eval_tasks[: min(40, len(eval_tasks))], int(acfg["max_length"])
    )

    # 6. Cross-Layer Activation Transfer (Early-Mid -> Late-Mid)
    transfer_metrics = cross_layer_activation_transfer(
        model,
        tokenizer,
        eval_tasks[: min(40, len(eval_tasks))],
        src_layer=n_blocks // 3,
        dst_layer=(2 * n_blocks) // 3,
        max_length=int(acfg["max_length"]),
    )

    summary = {
        "condition": condition,
        "model": cfg["model"]["name"],
        "filler_type": dcfg["filler_type"],
        "filler_length": int(dcfg["filler_length"]),
        "n_eval": len(eval_tasks),
        "no_filler_accuracy": nofiller_acc,
        "task_accuracy": task_acc,
        "filler_uplift": task_acc - nofiller_acc,
        "mid_logit_lens_top1": _mid_layer_summary(ll_rows, "top1_accuracy"),
        "mid_tuned_lens_top1": _mid_layer_summary(tl_rows, "top1_accuracy"),
        "mid_linear_mean_probe": _mid_layer_summary(probe_rows, "linear_mean_acc"),
        "mid_linear_max_probe": _mid_layer_summary(probe_rows, "linear_max_acc"),
        "mid_mlp_nonlinear_probe": _mid_layer_summary(probe_rows, "mlp_nonlinear_acc"),
        "mid_shuffled_control": _mid_layer_summary(probe_rows, "shuffled_control_acc"),
        "logit_lens": ll_rows,
        "tuned_lens": tl_rows,
        "probes": probe_rows,
        "causal_patching": patch_rows,
        "attention_masking": attn_mask_metrics,
        "cross_layer_transfer": transfer_metrics,
    }
    if baseline_probes is not None:
        summary["mid_static_linear_probe"] = _mid_layer_summary(probe_rows, "static_baseline_linear_acc")

    save_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable 5-stage hidden-state monitoring pipeline.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument(
        "--stage",
        choices=["baseline", "prompt_obf", "dpo", "audit_adv", "pareto", "plot", "all"],
        default="all",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["analysis"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mcfg = cfg["model"]
    tcfg = cfg["training"]

    if args.stage in ("baseline", "all"):
        print("=== STAGE 1: Collaborative Baseline Audit ===")
        model, tokenizer = load_model(
            mcfg["name"], mcfg.get("revision"), mcfg["device_map"],
            mcfg["trust_remote_code"], mcfg.get("load_in_4bit", True),
        )
        _run_full_audit(model, tokenizer, cfg, "collaborative", out_dir / "baseline_metrics")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage in ("prompt_obf", "all"):
        print("=== STAGE 2: Prompt-Only Obfuscation Control ===")
        model, tokenizer = load_model(
            mcfg["name"], mcfg.get("revision"), mcfg["device_map"],
            mcfg["trust_remote_code"], mcfg.get("load_in_4bit", True),
        )
        base_probes = load_probes(out_dir / "baseline_linear_probes.pkl")
        _run_full_audit(model, tokenizer, cfg, "obfuscation_prompt", out_dir / "prompt_obf_metrics", base_probes)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage in ("dpo", "all"):
        print("=== STAGE 3: Adversarial QLoRA DPO Training ===")
        run_dpo_training(
            model_name=mcfg["name"],
            baseline_probes_path=out_dir / "baseline_linear_probes.pkl",
            output_dir=tcfg["output_dir"],
            filler_type=cfg["data"]["filler_type"],
            filler_length=int(cfg["data"]["filler_length"]),
            n_shots=int(cfg["data"]["n_shots"]),
            task_types=cfg["data"]["task_types"],
            dpo_prompts=int(tcfg["dpo_prompts"]),
            max_steps=int(tcfg["max_steps"]),
            learning_rate=float(tcfg["learning_rate"]),
            beta=float(tcfg["beta"]),
            lambda_obf=float(tcfg["lambda_obf"]),
            rep_obf_weight=float(tcfg["rep_obf_weight"]),
            lora_r=int(tcfg["lora_r"]),
            lora_alpha=int(tcfg["lora_alpha"]),
            seed=int(cfg["data"]["seed"]),
            max_length=int(cfg["analysis"]["max_length"]),
        )

    if args.stage in ("audit_adv", "all"):
        print("=== STAGE 4: Adversarial DPO Model Audit ===")
        model, tokenizer = load_model(
            mcfg["name"], mcfg.get("revision"), mcfg["device_map"],
            mcfg["trust_remote_code"], mcfg.get("load_in_4bit", True),
            adapter_path=tcfg["output_dir"],
        )
        base_probes = load_probes(out_dir / "baseline_linear_probes.pkl")
        _run_full_audit(model, tokenizer, cfg, "obfuscation_prompt", out_dir / "adversarial_metrics", base_probes)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage == "pareto":
        print("=== PARETO SWEEP (RQ3): Varying Obfuscation Weight ===")
        pareto_points = []
        base_probes = load_probes(out_dir / "baseline_linear_probes.pkl")
        for w in tcfg.get("pareto_weights", [0.0, 0.25, 0.5, 1.0, 2.0]):
            ckpt = Path(tcfg["output_dir"]) / f"pareto_w{w}"
            run_dpo_training(
                model_name=mcfg["name"],
                baseline_probes_path=out_dir / "baseline_linear_probes.pkl",
                output_dir=ckpt,
                filler_type=cfg["data"]["filler_type"],
                filler_length=int(cfg["data"]["filler_length"]),
                n_shots=int(cfg["data"]["n_shots"]),
                task_types=cfg["data"]["task_types"],
                dpo_prompts=int(tcfg["dpo_prompts"]),
                max_steps=int(tcfg["max_steps"]),
                learning_rate=float(tcfg["learning_rate"]),
                beta=float(tcfg["beta"]),
                lambda_obf=float(tcfg["lambda_obf"]),
                rep_obf_weight=float(w),
                lora_r=int(tcfg["lora_r"]),
                lora_alpha=int(tcfg["lora_alpha"]),
                seed=int(cfg["data"]["seed"]),
                max_length=int(cfg["analysis"]["max_length"]),
            )
            model, tokenizer = load_model(
                mcfg["name"], mcfg.get("revision"), mcfg["device_map"],
                mcfg["trust_remote_code"], mcfg.get("load_in_4bit", True),
                adapter_path=str(ckpt),
            )
            res = _run_full_audit(model, tokenizer, cfg, "obfuscation_prompt", out_dir / f"pareto_w{w}_metrics", base_probes)
            pareto_points.append({
                "rep_obf_weight": w,
                "task_accuracy": res["task_accuracy"],
                "mid_logit_lens": res["mid_logit_lens_top1"],
                "mid_linear_probe": res["mid_linear_mean_probe"],
                "mid_mlp_probe": res["mid_mlp_nonlinear_probe"],
                "concealment_rate": 1.0 - res["mid_linear_mean_probe"],
            })
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        (out_dir / "pareto_sweep.json").write_text(json.dumps(pareto_points, indent=2))

    if args.stage in ("plot", "all"):
        print("=== STAGE 5: Generating Publication Figures & LaTeX Tables ===")
        from .plots import generate_all_figures
        generate_all_figures(out_dir)


if __name__ == "__main__":
    main()
