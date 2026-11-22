from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Clean academic palette: Navy + Muted Teal + Slate on crisp white background
NAVY = "#1E293B"
TEAL = "#0D9488"
AMBER = "#D97706"
CRIMSON = "#DC2626"
SLATE = "#64748B"


def _set_pub_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CBD5E1",
    })


def generate_all_figures(output_dir: str | Path) -> None:
    _set_pub_style()
    out = Path(output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads((out / "baseline_metrics.json").read_text())
    adv_path = out / "adversarial_metrics.json"
    adv = json.loads(adv_path.read_text()) if adv_path.exists() else None
    prompt_path = out / "prompt_obf_metrics.json"
    p_obf = json.loads(prompt_path.read_text()) if prompt_path.exists() else None

    # --- Figure 1: Layer-Wise Monitorability Curves (Baseline vs Adversarial) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    layers = [r["layer"] for r in base["logit_lens"]]

    axes[0].plot(layers, [r["top1_accuracy"] * 100 for r in base["logit_lens"]], label="Logit Lens", color=NAVY, lw=2.2)
    axes[0].plot(layers, [r["top1_accuracy"] * 100 for r in base["tuned_lens"]], label="Tuned Lens", color=TEAL, lw=2.2, ls="--")
    axes[0].plot(layers, [r["linear_mean_acc"] * 100 for r in base["probes"]], label="Linear Probe (Mean)", color=AMBER, lw=2.2)
    axes[0].plot(layers, [r["mlp_nonlinear_acc"] * 100 for r in base["probes"]], label="MLP Probe (Non-Linear)", color=CRIMSON, lw=1.8, ls=":")
    axes[0].axhline(10.0, color=SLATE, ls="--", lw=1.0, label="Chance (10%)")
    axes[0].set_title("Collaborative Baseline")
    axes[0].set_xlabel("Transformer Layer")
    axes[0].set_ylabel("Intermediate Variable Recovery (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right", fontsize=9)

    if adv is not None:
        axes[1].plot(layers, [r["top1_accuracy"] * 100 for r in adv["logit_lens"]], label="Logit Lens", color=NAVY, lw=2.2)
        axes[1].plot(layers, [r["top1_accuracy"] * 100 for r in adv["tuned_lens"]], label="Tuned Lens", color=TEAL, lw=2.2, ls="--")
        axes[1].plot(layers, [r["linear_mean_acc"] * 100 for r in adv["probes"]], label="Linear Probe (Mean)", color=AMBER, lw=2.2)
        axes[1].plot(layers, [r["mlp_nonlinear_acc"] * 100 for r in adv["probes"]], label="MLP Probe (Non-Linear)", color=CRIMSON, lw=2.2, ls=":")
        axes[1].axhline(10.0, color=SLATE, ls="--", lw=1.0, label="Chance (10%)")
        axes[1].set_title("Adversarial DPO-Obfuscated")
        axes[1].set_xlabel("Transformer Layer")
        axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Layer-Wise Hidden-State Monitorability ({base['model']})", fontweight="bold", color=NAVY)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_layerwise_monitorability.png", dpi=300)
    fig.savefig(fig_dir / "fig1_layerwise_monitorability.pdf")
    plt.close(fig)

    # --- Figure 2: Layer x Filler-Position Logit Lens Heatmaps ---
    hm_base_path = out / "baseline_metrics_logit_heatmap.npy"
    hm_adv_path = out / "adversarial_metrics_logit_heatmap.npy"
    if hm_base_path.exists():
        hm_base = np.load(hm_base_path) * 100
        ncols = 2 if hm_adv_path.exists() else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.8), squeeze=False)
        im = axes[0, 0].imshow(hm_base, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0, vmax=100)
        axes[0, 0].set_title("Baseline Logit-Lens Recovery (%)")
        axes[0, 0].set_xlabel("Filler Token Position")
        axes[0, 0].set_ylabel("Layer")
        if hm_adv_path.exists():
            hm_adv = np.load(hm_adv_path) * 100
            im = axes[0, 1].imshow(hm_adv, aspect="auto", origin="lower", cmap="YlGnBu", vmin=0, vmax=100)
            axes[0, 1].set_title("Adversarial Logit-Lens Recovery (%)")
            axes[0, 1].set_xlabel("Filler Token Position")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Exact Match (%)")
        fig.savefig(fig_dir / "fig2_logit_lens_heatmaps.png", dpi=300, bbox_inches="tight")
        fig.savefig(fig_dir / "fig2_logit_lens_heatmaps.pdf", bbox_inches="tight")
        plt.close(fig)

    # --- Figure 3: Causal Patching & Attention Masking Verification ---
    if adv is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
        bands = [r["layer_band"] for r in base["causal_patching"]]
        x = np.arange(len(bands))
        w = 0.35
        axes[0].bar(x - w / 2, [r["mean_logit_recovery"] * 100 for r in base["causal_patching"]], w, label="Baseline", color=NAVY)
        axes[0].bar(x + w / 2, [r["mean_logit_recovery"] * 100 for r in adv["causal_patching"]], w, label="Adversarial DPO", color=TEAL)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(bands)
        axes[0].set_ylabel("Causal Logit-Diff Recovery (%)")
        axes[0].set_title("Filler-Span Activation Patching")
        axes[0].legend()

        p_keys = ["unmasked", "block_answer_to_filler", "block_filler_to_question", "block_filler_self"]
        p_labels = ["Unmasked", "Block Ans->Fill", "Block Fill->Q", "Block Fill Self"]
        xp = np.arange(len(p_keys))
        axes[1].bar(xp - w / 2, [base["attention_masking"][k] * 100 for k in p_keys], w, label="Baseline", color=NAVY)
        axes[1].bar(xp + w / 2, [adv["attention_masking"][k] * 100 for k in p_keys], w, label="Adversarial DPO", color=TEAL)
        axes[1].set_xticks(xp)
        axes[1].set_xticklabels(p_labels, rotation=15)
        axes[1].set_ylabel("Task Accuracy (%)")
        axes[1].set_title("Attention-Pathway Masking Ablation")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_causal_verification.png", dpi=300)
        fig.savefig(fig_dir / "fig3_causal_verification.pdf")
        plt.close(fig)

    # --- Figure 4: Pareto Trade-Off Frontier (if pareto sweep was run) ---
    pareto_path = out / "pareto_sweep.json"
    if pareto_path.exists():
        pts = json.loads(pareto_path.read_text())
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        conceal = [p["concealment_rate"] * 100 for p in pts]
        acc = [p["task_accuracy"] * 100 for p in pts]
        ax.plot(conceal, acc, "-o", color=TEAL, lw=2.2, ms=7)
        for p, cx, cy in zip(pts, conceal, acc):
            ax.annotate(f"w={p['rep_obf_weight']}", (cx, cy), textcoords="offset points", xytext=(6, 6), fontsize=9, color=NAVY)
        ax.set_xlabel("Computation Concealment Rate (1 - Mid-Layer Linear Probe, %)")
        ax.set_ylabel("Task Accuracy (%)")
        ax.set_title("Pareto Frontier: Task Performance vs. Concealment")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_pareto_frontier.png", dpi=300)
        fig.savefig(fig_dir / "fig4_pareto_frontier.pdf")
        plt.close(fig)

    # --- Auto-generated LaTeX Summary Table ---
    rows = [("Collaborative Baseline", base)]
    if p_obf is not None:
        rows.append(("Prompt-Only Obfuscation", p_obf))
    if adv is not None:
        rows.append(("Adversarial DPO", adv))

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Task accuracy and middle-layer hidden-state monitorability (\%) across conditions.}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Condition & Task Acc & Logit Lens & Tuned Lens & Linear (Mean) & Linear (Max) & MLP (Non-Lin) \\",
        r"\midrule",
    ]
    for label, d in rows:
        tex.append(
            f"{label} & {d['task_accuracy']*100:.1f} & {d['mid_logit_lens_top1']*100:.1f} & "
            f"{d['mid_tuned_lens_top1']*100:.1f} & {d['mid_linear_mean_probe']*100:.1f} & "
            f"{d['mid_linear_max_probe']*100:.1f} & {d['mid_mlp_nonlinear_probe']*100:.1f} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (fig_dir / "table1_summary.tex").write_text("\n".join(tex))
    print(f"Saved publication figures and LaTeX table to: {fig_dir}")
