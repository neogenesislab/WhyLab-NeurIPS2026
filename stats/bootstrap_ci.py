"""
Bootstrap CI and McNemar test for E1 and E3a
=============================================
Reads saved per-seed results (no re-simulation).
Outputs:
  - E1: detection rate with 95% CI per severity/detector
  - E1: McNemar p-values (C1 vs each baseline, appendix only)
  - E3a: violation rate, Pearson with 95% CI per controller
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "experiments" / "results"
N_BOOTSTRAP = 10000


def bootstrap_ci(values, stat_fn=np.mean, n_boot=N_BOOTSTRAP, ci=0.95, rng=None):
    """Percentile bootstrap CI for a statistic."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(values)
    boots = [stat_fn(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def mcnemar_test(detected_a, detected_b):
    """McNemar's test (mid-p version) for paired binary outcomes."""
    # b = A detects, B doesn't; c = B detects, A doesn't
    b = np.sum(detected_a & ~detected_b)
    c = np.sum(~detected_a & detected_b)
    n_disc = b + c
    if n_disc == 0:
        return 1.0, b, c
    # Mid-p McNemar
    from scipy.stats import binom
    if b >= c:
        p = 2 * (binom.cdf(c, n_disc, 0.5) - 0.5 * binom.pmf(c, n_disc, 0.5))
    else:
        p = 2 * (binom.cdf(b, n_disc, 0.5) - 0.5 * binom.pmf(b, n_disc, 0.5))
    return float(min(p, 1.0)), int(b), int(c)


def e1_ci():
    """Compute bootstrap CI for E1 detection rates."""
    df = pd.read_csv(ROOT / "e1_metrics.csv")
    # detection: delay < 700 (700 = censored/not detected)
    df["detected"] = (df["delay"] < 700).astype(int)

    rng = np.random.default_rng(42)
    rows = []
    for sev in ["mild", "moderate", "severe"]:
        sub = df[df["severity"] == sev]
        for det in sub["detector"].unique():
            vals = sub[sub["detector"] == det]["detected"].values
            rate = vals.mean()
            lo, hi = bootstrap_ci(vals, rng=rng)
            rows.append({
                "severity": sev, "detector": det,
                "detection_rate": f"{rate:.3f}",
                "ci_95": f"[{lo:.3f}, {hi:.3f}]",
                "n_seeds": len(vals),
            })
    tab = pd.DataFrame(rows)
    print("=== E1 Detection Rate with 95% Bootstrap CI ===")
    print(tab.to_string(index=False))
    tab.to_csv(ROOT / "e1_ci.csv", index=False)
    print(f"\nSaved: {ROOT / 'e1_ci.csv'}")

    # McNemar: C1 vs each baseline per severity
    print("\n=== E1 McNemar (mid-p): C1 vs baselines ===")
    mcn_rows = []
    for sev in ["mild", "moderate", "severe"]:
        sub = df[df["severity"] == sev]
        seeds = sorted(sub["seed"].unique())
        c1_det = np.array([sub[(sub["seed"] == s) & (sub["detector"] == "entropy_weighted")]["detected"].values[0] for s in seeds], dtype=bool)
        for base in ["uniform", "adwin"]:
            b_det = np.array([sub[(sub["seed"] == s) & (sub["detector"] == base)]["detected"].values[0] for s in seeds], dtype=bool)
            p, b, c = mcnemar_test(c1_det, b_det)
            mcn_rows.append({
                "severity": sev, "comparison": f"C1 vs {base}",
                "b(C1_only)": b, "c(base_only)": c, "mid_p": f"{p:.4f}",
            })
    mcn_tab = pd.DataFrame(mcn_rows)
    print(mcn_tab.to_string(index=False))
    mcn_tab.to_csv(ROOT / "e1_mcnemar.csv", index=False)


def e3a_ci():
    """Compute bootstrap CI for E3a metrics."""
    df = pd.read_csv(ROOT / "e3a_stationary_metrics.csv")

    rng = np.random.default_rng(42)
    rows = []

    # Report at h=0.5 (worst case) as in paper
    sub = df[df["h_rate"] == 0.5]
    for ctrl in sub["controller"].unique():
        ctrl_data = sub[sub["controller"] == ctrl]
        viol = ctrl_data["true_viol_rate"].values
        pearson = ctrl_data["pearson"].values
        final_v = ctrl_data["final_V"].values

        v_mean = viol.mean()
        v_lo, v_hi = bootstrap_ci(viol, rng=rng)
        p_mean = pearson.mean()
        p_lo, p_hi = bootstrap_ci(pearson, rng=rng)
        fv_mean = final_v.mean()
        fv_lo, fv_hi = bootstrap_ci(final_v, rng=rng)

        rows.append({
            "controller": ctrl,
            "viol_rate": f"{v_mean:.3f}",
            "viol_ci": f"[{v_lo:.3f}, {v_hi:.3f}]",
            "pearson": f"{p_mean:.3f}",
            "pearson_ci": f"[{p_lo:.3f}, {p_hi:.3f}]",
            "final_V": f"{fv_mean:.2f}",
            "final_V_ci": f"[{fv_lo:.2f}, {fv_hi:.2f}]",
        })

    tab = pd.DataFrame(rows)
    print("\n=== E3a Metrics with 95% Bootstrap CI (h=0.5) ===")
    print(tab.to_string(index=False))
    tab.to_csv(ROOT / "e3a_ci.csv", index=False)
    print(f"\nSaved: {ROOT / 'e3a_ci.csv'}")

    # Also compute mean Pearson across ALL h values (as in paper Tab 3)
    print("\n=== E3a Mean Pearson across all h values ===")
    for ctrl in df["controller"].unique():
        # Mean of per-seed Pearson, then mean across h-rates
        per_seed = df[df["controller"] == ctrl].groupby("seed")["pearson"].mean().values
        m = per_seed.mean()
        lo, hi = bootstrap_ci(per_seed, rng=rng)
        print(f"  {ctrl:15s}: {m:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    e1_ci()
    e3a_ci()
