"""
Generate publication-quality figure for E9 LLM multi-model phase diagram.
3 subplots (one per model): mean_oscillation vs temperature,
baseline (red) vs audit (blue), aggregated across retry counts (mean +/- std).
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- load data ----------
with open("D:/00.test/PAPER/WhyLab/experiments/results/e9_llm_phase_multi.json") as f:
    data = json.load(f)

temperatures = data["parameters"]["temperatures"]

# Display order and pretty names
model_order = ["gemini-2.0-flash", "gpt-4o-mini", "claude-sonnet-4-20250514"]
pretty_names = {
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
}

# ---------- aggregate ----------
def aggregate(records, audit_cond):
    """For each temperature, collect mean_oscillation across retry counts, return mean & std."""
    means, stds = [], []
    reg_means = []
    for t in temperatures:
        vals = [r["mean_oscillation"] for r in records
                if r["temperature"] == t and r["audit"] == audit_cond]
        regs = [r["mean_regression"] for r in records
                if r["temperature"] == t and r["audit"] == audit_cond]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        reg_means.append(np.mean(regs))
    return np.array(means), np.array(stds), np.array(reg_means)

# ---------- style ----------
sns.set_context("paper", font_scale=1.1)
sns.set_style("whitegrid")

# Colorblind-friendly palette (Tol bright)
CLR_BASELINE = "#EE6677"  # red-ish
CLR_AUDIT    = "#4477AA"  # blue-ish

fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.6), sharey=True)

temps = np.array(temperatures)

for ax, model_key in zip(axes, model_order):
    records = data["results_by_model"][model_key]

    bl_mean, bl_std, bl_reg = aggregate(records, "none")
    au_mean, au_std, au_reg = aggregate(records, "audit")

    # Lines with shaded bands
    ax.plot(temps, bl_mean, "-o", color=CLR_BASELINE, markersize=4,
            linewidth=1.5, label="Baseline", zorder=3)
    ax.fill_between(temps, bl_mean - bl_std, bl_mean + bl_std,
                    color=CLR_BASELINE, alpha=0.18, zorder=2)

    ax.plot(temps, au_mean, "-s", color=CLR_AUDIT, markersize=4,
            linewidth=1.5, label="Audit", zorder=3)
    ax.fill_between(temps, au_mean - au_std, au_mean + au_std,
                    color=CLR_AUDIT, alpha=0.18, zorder=2)

    # Annotate regression counts where > 0
    for i, t in enumerate(temps):
        if bl_reg[i] > 0:
            ax.annotate(f"r={bl_reg[i]:.1f}",
                        (t, bl_mean[i]), fontsize=6.5,
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", color=CLR_BASELINE, weight="bold")
        if au_reg[i] > 0:
            ax.annotate(f"r={au_reg[i]:.1f}",
                        (t, au_mean[i]), fontsize=6.5,
                        textcoords="offset points", xytext=(0, -12),
                        ha="center", color=CLR_AUDIT, weight="bold")

    ax.set_title(pretty_names[model_key], fontsize=10, weight="bold")
    ax.set_xlabel("Temperature", fontsize=9)
    ax.set_xticks(temps)

axes[0].set_ylabel("Mean Oscillation", fontsize=9)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           fontsize=8.5, bbox_to_anchor=(0.5, 1.04))

# Y-axis limits with a bit of headroom
ymax = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(-0.1, ymax + 0.25)
    ax.tick_params(labelsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.93])

# ---------- save ----------
for ext in ("pdf", "png"):
    fig.savefig(f"D:/00.test/PAPER/WhyLab/experiments/figures/e9_llm_multi.{ext}",
                dpi=300, bbox_inches="tight")

print("Saved e9_llm_multi.pdf and e9_llm_multi.png")
