"""
Generate Table 2 (E2 filtering) LaTeX from e2_summary.csv.
Ensures 1:1 correspondence between CSV data and paper table.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "experiments" / "results" / "e2_summary.csv"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)


def find_best(df, mode, recall_min):
    """Find operating point minimizing fragile_rate at recall >= threshold."""
    sub = df[(df["mode"] == mode) & (df["recall"] >= recall_min)]
    if len(sub) == 0:
        return None
    return sub.sort_values("fragile_rate").iloc[0]


def fmt(v, decimals=3):
    return f"{v:.{decimals}f}"


def generate_tab2():
    df = pd.read_csv(CSV)
    regimes = [(0.9, "Recall$\\geq$0.9"), (0.5, "Recall$\\geq$0.5")]
    filters = [
        ("none", "No filter", ""),
        ("E_only", "E-only", "E"),
        ("RV_only", "$RV_q$-only", "RV"),
        ("E+RV", "\\textbf{C2}", "E+RV"),
    ]

    lines = []
    lines.append("% Auto-generated from e2_summary.csv — do NOT edit manually")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Filtering performance under two operating regimes from the")
    lines.append("  threshold sweep (40~seeds, mean over synthetic observational studies).")
    lines.append("  For each regime, the operating point minimizing fragile rate")
    lines.append("  is selected per filter.")
    lines.append("  E-values: $d = |ATE|/\\sigma_{\\text{pooled}}$;")
    lines.append("  $RV_q$: $f_q = |\\hat{\\beta}|/\\text{SE}$.}")
    lines.append("\\label{tab:e2-filter}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}llcccc@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Regime} & \\textbf{Filter}")
    lines.append("  & \\textbf{Frag.\\,rate} $\\downarrow$")
    lines.append("  & \\textbf{Recall} $\\uparrow$")
    lines.append("  & \\textbf{Frag.\\,rej.} $\\uparrow$")
    lines.append("  & \\textbf{Retention} \\\\")

    for i, (recall_min, regime_label) in enumerate(regimes):
        lines.append("\\midrule")
        n_filters = len(filters)
        for j, (mode, label, kind) in enumerate(filters):
            row = find_best(df, mode, recall_min)
            if row is None:
                continue

            # Build threshold descriptor
            if mode == "none":
                thresh_str = ""
            elif mode == "E_only":
                thresh_str = f" ($E{{\\geq}}{row.E_thresh:.1f}$)"
            elif mode == "RV_only":
                thresh_str = f" ($RV_q{{\\geq}}{row.RV_thresh:.2f}$)"
            else:
                thresh_str = f" ($E{{\\geq}}{row.E_thresh:.1f},\\; RV_q{{\\geq}}{row.RV_thresh:.2f}$)"

            # Regime label (multirow on first filter)
            if j == 0:
                regime_col = (f"\\multirow{{{n_filters}}}{{*}}"
                              f"{{\\rotatebox[origin=c]{{90}}"
                              f"{{\\scriptsize {regime_label}}}}}")
            else:
                regime_col = ""

            # Bold for C2
            fr = fmt(row.fragile_rate)
            rej = fmt(row.fragile_rej)
            if mode == "E+RV":
                fr = f"\\textbf{{{fr}}}"
                rej = f"\\textbf{{{rej}}}"

            line = (f"  {regime_col} & {label}{thresh_str}"
                    f" & {fr} & {fmt(row.recall)}"
                    f" & {rej} & {fmt(row.retention)} \\\\")
            lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    outpath = OUT / "tab2.tex"
    outpath.write_text(tex, encoding="utf-8")
    print(f"[Tab2] Generated: {outpath}")
    print(f"[Tab2] {len(lines)} lines")
    print()
    print(tex)
    return tex


if __name__ == "__main__":
    generate_tab2()
