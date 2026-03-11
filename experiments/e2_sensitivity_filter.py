"""
E2: Sensitivity Filtering — C2 Validation
==========================================
Tests whether the dual-threshold E-value + RV filter correctly
separates stable from fragile evaluation outcomes.

Protocol:
- Generate synthetic audit outcomes with known ground-truth quality
- Compare: E-only / RV-only / E+RV (C2) / no-filter
- Threshold sweep over E ∈ [1.0, 5.0] and RV ∈ [0.01, 0.5]
- Metrics: precision, recall, F1, accepted-positive FDR, decision loss

Key: "fragile success" = positive outcome that is unreliable.
C2's job is to flag and exclude these before they pollute downstream RL.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent
EXP = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["experiment"]

SEEDS = EXP["seeds"]
BASE_SEED = EXP["rng_base_seed"]

# Experiment parameters
N_SAMPLES = 500       # audit outcomes per episode
N_EPISODES = 50       # episodes per seed
E_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
RV_THRESHOLDS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_audit_data(n_samples, rng, fragile_rate=0.2, noise_level=0.3):
    """Generate synthetic audit outcomes with known ground-truth reliability.

    Each sample has:
    - verdict: binary (positive/negative)
    - true_quality: ground-truth reliability of the verdict
    - E_value: evidence strength (higher = more reliable)
    - RV: sensitivity proxy — residual variance of the effect estimate.
          In the paper (Appendix C), the robustness value RV_q measures
          how much confounding is needed to nullify the effect (higher =
          more robust).  Here we use residual variance as its *inverse*
          proxy: low RV ≈ high robustness value, so the filter rejects
          samples with RV > RV_min (equivalently, RV_q < RV_min in the
          paper's notation).
    - is_fragile: whether this is a "fragile success"
      (positive verdict but unreliable underlying evaluation)

    Ground-truth model:
    - Reliable positives: high E, low RV (high robustness)
    - Reliable negatives: variable E, variable RV
    - Fragile positives: moderate E but HIGH RV (looks good but isn't)
    - False negatives: low E, low RV (genuinely negative)
    """
    is_positive = rng.random(n_samples) > 0.4  # 60% positive rate
    is_fragile = np.zeros(n_samples, dtype=bool)

    # Mark some positives as fragile
    pos_idx = np.where(is_positive)[0]
    n_fragile = int(len(pos_idx) * fragile_rate)
    if n_fragile > 0:
        fragile_idx = rng.choice(pos_idx, size=n_fragile, replace=False)
        is_fragile[fragile_idx] = True

    E_values = np.zeros(n_samples)
    RV_values = np.zeros(n_samples)

    for i in range(n_samples):
        if is_positive[i] and not is_fragile[i]:
            # Reliable positive: high E, low RV
            E_values[i] = rng.lognormal(1.2, 0.4)  # mean ~3.3
            RV_values[i] = rng.exponential(0.05)    # mean ~0.05
        elif is_positive[i] and is_fragile[i]:
            # Fragile positive: moderate E, HIGH RV
            E_values[i] = rng.lognormal(0.8, 0.5)   # mean ~2.2
            RV_values[i] = rng.exponential(0.25)     # mean ~0.25
        else:
            # Negative: low E, variable RV
            E_values[i] = rng.lognormal(0.3, 0.6)    # mean ~1.3
            RV_values[i] = rng.exponential(0.15)      # mean ~0.15

    # Add noise to make separation imperfect
    E_values += rng.normal(0, noise_level, n_samples)
    E_values = np.clip(E_values, 0.01, 100)
    RV_values += rng.normal(0, noise_level * 0.1, n_samples)
    RV_values = np.clip(RV_values, 0.001, 10)

    return pd.DataFrame({
        "verdict": is_positive.astype(int),
        "is_fragile": is_fragile.astype(int),
        "E_value": E_values,
        "RV": RV_values,
        "true_reliable": ((is_positive) & (~is_fragile)).astype(int),
    })


# ---------------------------------------------------------------------------
# Filtering strategies
# ---------------------------------------------------------------------------
def apply_filter(df, E_thresh, RV_thresh, mode="E+RV"):
    """Apply filtering and return accepted samples.

    Modes:
    - "none": accept all
    - "E_only": reject if E < E_thresh
    - "RV_only": reject if RV > RV_thresh
    - "E+RV": reject if E < E_thresh OR RV > RV_thresh (C2)
    """
    if mode == "none":
        return df.copy(), np.ones(len(df), dtype=bool)
    elif mode == "E_only":
        accepted = df["E_value"] >= E_thresh
    elif mode == "RV_only":
        accepted = df["RV"] <= RV_thresh
    elif mode == "E+RV":
        accepted = (df["E_value"] >= E_thresh) & (df["RV"] <= RV_thresh)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df[accepted], accepted.values


def compute_metrics(df, accepted_mask):
    """Compute filtering quality metrics.

    Metric definitions (note different denominators):
    - fragile_rate: fragile / accepted_positives (FDR among positives)
    - reliable_frac: true_reliable / all_accepted (set-level precision)
    - recall: accepted_reliable / total_reliable
    - fragile_rej: 1 - accepted_fragile / total_fragile

    fragile_rate != 1 - reliable_frac because reliable_frac includes
    accepted negatives in the denominator.
    """
    accepted = df[accepted_mask]

    n_total = len(df)
    n_accepted = len(accepted)

    # Among accepted positives, fraction that are fragile
    accepted_positives = accepted[accepted["verdict"] == 1]
    if len(accepted_positives) > 0:
        fragile_rate = accepted_positives["is_fragile"].mean()
    else:
        fragile_rate = 0.0

    # Fraction of entire accepted set that is truly reliable
    if n_accepted > 0:
        reliable_frac = accepted["true_reliable"].mean()
    else:
        reliable_frac = 1.0

    # Recall: fraction of true reliable that are accepted
    n_reliable = df["true_reliable"].sum()
    recall = accepted["true_reliable"].sum() / n_reliable if n_reliable > 0 else 1.0

    # F1 (based on reliable_frac as proxy for precision)
    f1 = (2 * reliable_frac * recall / (reliable_frac + recall)
          if (reliable_frac + recall) > 0 else 0.0)

    # Fragile rejection rate
    n_fragile = df["is_fragile"].sum()
    fragile_rej = (1 - accepted["is_fragile"].sum() / n_fragile
                   if n_fragile > 0 else 1.0)

    accepted_fragile = accepted["is_fragile"].sum()

    return {
        "n_accepted": n_accepted,
        "n_rejected": n_total - n_accepted,
        "retention_rate": n_accepted / max(n_total, 1),
        "fragile_rate": fragile_rate,
        "reliable_frac": reliable_frac,
        "recall": recall,
        "f1": f1,
        "fragile_rej": fragile_rej,
        "accepted_fragile": accepted_fragile,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_e2():
    all_rows = []

    for seed_idx in range(SEEDS):
        seed = BASE_SEED + seed_idx
        rng = np.random.default_rng(seed)

        # Generate data
        df = generate_audit_data(N_SAMPLES, rng)

        # No filter baseline
        _, mask = apply_filter(df, 0, 0, "none")
        m = compute_metrics(df, mask)
        all_rows.append({
            "seed": seed, "mode": "none",
            "E_thresh": 0, "RV_thresh": 0, **m
        })

        # Sweep thresholds
        for E_t in E_THRESHOLDS:
            # E-only
            _, mask = apply_filter(df, E_t, 0, "E_only")
            m = compute_metrics(df, mask)
            all_rows.append({
                "seed": seed, "mode": "E_only",
                "E_thresh": E_t, "RV_thresh": 0, **m
            })

        for RV_t in RV_THRESHOLDS:
            # RV-only
            _, mask = apply_filter(df, 0, RV_t, "RV_only")
            m = compute_metrics(df, mask)
            all_rows.append({
                "seed": seed, "mode": "RV_only",
                "E_thresh": 0, "RV_thresh": RV_t, **m
            })

        for E_t, RV_t in product(E_THRESHOLDS, RV_THRESHOLDS):
            # E+RV (C2)
            _, mask = apply_filter(df, E_t, RV_t, "E+RV")
            m = compute_metrics(df, mask)
            all_rows.append({
                "seed": seed, "mode": "E+RV",
                "E_thresh": E_t, "RV_thresh": RV_t, **m
            })

        if (seed_idx + 1) % 10 == 0:
            print(f"  seed {seed_idx + 1}/{SEEDS}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out / "e2_metrics.csv", index=False)
    print(f"\n[E2] Saved: {len(df_all)} rows")

    # Summary: best threshold for each mode
    print("\n=== E2 Summary: Best FDR-Recall Pareto ===")

    # For each mode, find threshold(s) with FDR < 0.05 and maximum recall
    for mode in ["none", "E_only", "RV_only", "E+RV"]:
        sub = df_all[df_all["mode"] == mode]
        agg = sub.groupby(["E_thresh", "RV_thresh"]).agg(
            fragile_rate=("fragile_rate", "mean"),
            recall=("recall", "mean"),
            reliable_frac=("reliable_frac", "mean"),
            f1=("f1", "mean"),
            fragile_rej=("fragile_rej", "mean"),
            retention=("retention_rate", "mean"),
        ).reset_index()

        # Best: lowest fragile_rate with recall > 0.5
        good = agg[(agg.recall > 0.5)]
        if len(good) > 0:
            best = good.sort_values("fragile_rate").iloc[0]
            print(f"\n  {mode:8s} | E={best.E_thresh:.1f} RV={best.RV_thresh:.2f}")
            print(f"           frag_rate={best.fragile_rate:.3f} recall={best.recall:.3f} "
                  f"fragile_rej={best.fragile_rej:.3f} retention={best.retention:.3f}")
        else:
            print(f"\n  {mode:8s} | No config with recall > 0.5")

    # Save detailed summary
    agg_full = df_all.groupby(["mode", "E_thresh", "RV_thresh"]).agg(
        fragile_rate=("fragile_rate", "mean"),
        recall=("recall", "mean"),
        reliable_frac=("reliable_frac", "mean"),
        f1=("f1", "mean"),
        fragile_rej=("fragile_rej", "mean"),
        retention=("retention_rate", "mean"),
        accepted_fragile=("accepted_fragile", "mean"),
    ).round(4).reset_index()
    agg_full.to_csv(out / "e2_summary.csv", index=False)
    print(f"\n[E2] Summary saved: {len(agg_full)} rows")

    return df_all


if __name__ == "__main__":
    run_e2()
