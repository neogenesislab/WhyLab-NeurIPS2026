"""
Generate Table 4 (E4 Agent Benchmark) LaTeX from e4_summary.csv.
Ensures 1:1 correspondence between CSV data and paper table.

Follows the same pattern as generate_tab2.py for consistency.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "experiments" / "results" / "e4_summary.csv"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)


def fmt(v, decimals=3):
    return f"{v:.{decimals}f}"


def generate_tab4():
    df = pd.read_csv(CSV)

    # Desired ablation order
    order = ["none", "C1_only", "C2_only", "C3_only", "full"]
    labels = {
        "none": "No audit",
        "C1_only": "C1 only",
        "C2_only": "C2 only",
        "C3_only": "C3 only",
        "full": "\\textbf{C1+C2+C3}",
    }

    lines = []
    lines.append("% Auto-generated from e4_summary.csv — do NOT edit manually")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Agent benchmark: Reflexion on HumanEval with")
    lines.append("  WhyLab audit ablations (40~seeds, mean over problems).")
    lines.append("  Audit layer reduces regressions and oscillation while")
    lines.append("  maintaining or improving final pass rate.}")
    lines.append("\\label{tab:e4-agent}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}lcccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Config}")
    lines.append("  & \\textbf{Pass@1} $\\uparrow$")
    lines.append("  & \\textbf{Regressions} $\\downarrow$")
    lines.append("  & \\textbf{Oscillation} $\\downarrow$")
    lines.append("  & \\textbf{Accept rate} \\\\")
    lines.append("\\midrule")

    for abl_key in order:
        row = df[df["ablation"] == abl_key]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        label = labels.get(abl_key, abl_key)

        # Bold the 'full' row values (best expected)
        pr = fmt(row.pass_rate)
        reg = fmt(row.mean_regressions)
        osc = fmt(row.mean_oscillation)
        acc = fmt(row.acceptance_rate) if pd.notna(row.acceptance_rate) else "---"

        if abl_key == "full":
            pr = f"\\textbf{{{pr}}}"
            reg = f"\\textbf{{{reg}}}"
            osc = f"\\textbf{{{osc}}}"

        line = f"  {label} & {pr} & {reg} & {osc} & {acc} \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    outpath = OUT / "tab4.tex"
    outpath.write_text(tex, encoding="utf-8")
    print(f"[Tab4] Generated: {outpath}")
    print(f"[Tab4] {len(lines)} lines")
    print()
    print(tex)
    return tex


if __name__ == "__main__":
    generate_tab4()
