# -*- coding: utf-8 -*-
"""E6 E-value Method Comparison: Parametric vs. Permutation.

Runs the E6 non-stationary agent experiment with both E-value methods
(parametric and distribution-free permutation) and saves a comparison
to experiments/results/e_value_comparison.json.
"""
import json
import os
import numpy as np
from e6_nonstationary_agent import E6Config, run_episode


def main():
    methods = ["parametric", "permutation"]
    ablations = ["C2_only", "full"]
    h_rates = [0.0, 0.3, 0.5]
    n_seeds = 20
    lr = 0.1

    all_results = {}

    for method in methods:
        method_results = []
        for h_rate in h_rates:
            for ablation in ablations:
                seed_metrics = []
                for seed in range(n_seeds):
                    cfg = E6Config(
                        seed=seed,
                        h_rate=h_rate,
                        lr_base=lr,
                        c2_e_value_method=method,
                        c2_n_permutations=100,
                    )
                    row = run_episode(cfg, ablation)
                    row["e_value_method"] = method
                    row["lr_base"] = lr
                    seed_metrics.append(row)

                # Aggregate across seeds
                agg = {
                    "e_value_method": method,
                    "ablation": ablation,
                    "h_rate": h_rate,
                    "n_seeds": n_seeds,
                    "final_energy_mean": float(np.mean([r["final_energy"] for r in seed_metrics])),
                    "final_energy_std": float(np.std([r["final_energy"] for r in seed_metrics])),
                    "mean_energy_mean": float(np.mean([r["mean_energy"] for r in seed_metrics])),
                    "total_regret_mean": float(np.mean([r["total_regret"] for r in seed_metrics])),
                    "oscillation_index_mean": float(np.mean([r["oscillation_index"] for r in seed_metrics])),
                    "c2_rejection_rate_mean": float(np.mean([r["c2_rejection_rate"] for r in seed_metrics])),
                    "c2_rejection_rate_std": float(np.std([r["c2_rejection_rate"] for r in seed_metrics])),
                    "mean_detection_delay": float(np.mean([r["mean_detection_delay"] for r in seed_metrics])),
                }
                method_results.append(agg)

            print(f"  method={method}, h_rate={h_rate} done")

        all_results[method] = method_results

    # Build comparison summary
    comparison = {
        "description": (
            "E6 E-value method comparison: parametric (VanderWeele & Ding 2017, "
            "assumes normality) vs. permutation (distribution-free, K=100). "
            "Each condition aggregated over 20 seeds."
        ),
        "methods": all_results,
        "pairwise_comparison": [],
    }

    # Compute pairwise differences for matching conditions
    param_results = {(r["ablation"], r["h_rate"]): r for r in all_results["parametric"]}
    perm_results = {(r["ablation"], r["h_rate"]): r for r in all_results["permutation"]}

    for key in param_results:
        p = param_results[key]
        q = perm_results[key]
        comparison["pairwise_comparison"].append({
            "ablation": key[0],
            "h_rate": key[1],
            "parametric_final_energy": p["final_energy_mean"],
            "permutation_final_energy": q["final_energy_mean"],
            "energy_diff": q["final_energy_mean"] - p["final_energy_mean"],
            "parametric_regret": p["total_regret_mean"],
            "permutation_regret": q["total_regret_mean"],
            "regret_diff": q["total_regret_mean"] - p["total_regret_mean"],
            "parametric_c2_rej_rate": p["c2_rejection_rate_mean"],
            "permutation_c2_rej_rate": q["c2_rejection_rate_mean"],
            "parametric_osc_index": p["oscillation_index_mean"],
            "permutation_osc_index": q["oscillation_index_mean"],
        })

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e_value_comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to {out_path}")
    print("\n=== Pairwise Summary ===")
    for row in comparison["pairwise_comparison"]:
        print(
            f"  {row['ablation']:10s} h={row['h_rate']:.1f}  "
            f"energy: param={row['parametric_final_energy']:.3f} "
            f"perm={row['permutation_final_energy']:.3f} "
            f"(diff={row['energy_diff']:+.3f})  "
            f"c2_rej: param={row['parametric_c2_rej_rate']:.3f} "
            f"perm={row['permutation_c2_rej_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
