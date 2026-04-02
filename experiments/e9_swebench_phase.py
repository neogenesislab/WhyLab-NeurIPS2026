"""E9-SWE: Oscillation Phase Diagram on Real SWE-bench Problems.

Replaces the artificial "compile check" approach of e9_llm_phase.py with
genuine SWE-bench problem-specific evaluation using the lightweight eval
path from swebench_loader.py.

Design:
  - 45 oscillation-prone problems (identified from e5 baseline runs)
  - Sweep: temperature [0.3, 0.7, 1.0] x max_attempts [3, 5, 7]
  - Conditions: baseline (no audit) vs WhyLab C2 audit
  - 3 seeds per condition
  - Evaluation: real SWE-bench lightweight eval (file-overlap + repo-based
    when available), NOT compile-check

Key difference from e9_llm_phase.py:
  - Uses actual SWE-bench problem descriptions (issue text, repo context)
  - Evaluates generated patches against gold test patches
  - Pass/fail reflects whether the patch addresses the right files and
    passes problem-specific tests, not whether code compiles

Metrics tracked per grid cell:
  - mean_oscillation: average oscillation count across episodes
  - mean_regression:  average pass->fail regression count
  - pass_rate:        fraction of episodes that end in pass
  - osc_free_rate:    fraction of episodes with zero oscillations
  - mean_attempts:    average number of attempts used
"""
import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict

# ── Paths ────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
E5_METRICS = RESULTS_DIR / "e5_metrics.csv"
CACHE_DIR = EXPERIMENT_DIR / "cache"

# ── Imports from existing infrastructure ─────────────────────────────
from swebench_loader import (
    SWEProblem,
    PatchResult,
    apply_and_test_patch,
    compute_patch_magnitude,
    load_swebench_lite,
)
from swebench_reflexion import (
    SWEEpisodeResult,
    SWEAttemptRecord,
    run_swe_reflexion_episode,
    _build_swe_solve_prompt,
    _build_swe_reflect_prompt,
    _extract_patch,
)
from llm_client import CachedLLMClient
from audit_layer import AgentAuditLayer

# ── Experiment Parameters ────────────────────────────────────────────
TEMPERATURES = [0.3, 0.7, 1.0]
MAX_ATTEMPTS_LIST = [3, 5, 7]
SEEDS = [42, 137, 2024]
MODEL = "gemini-2.0-flash"

# Audit configuration: C2-only (the sensitivity gate)
C2_AUDIT_CONFIG = {
    "c1": False,
    "c2": True,
    "c3": False,
    "c2_e_thresh": 1.5,
    "c2_rv_thresh": 0.05,
}

NO_AUDIT_CONFIG = None  # sentinel for baseline


# ── Identify the 45 oscillation-prone problems from e5 ──────────────

def load_oscillation_prone_ids(csv_path: Path = E5_METRICS) -> list[str]:
    """Extract instance_ids of problems that showed oscillation in e5 baseline.

    The 45 oscillation-prone problems are those where oscillation_count > 0
    under the 'none' (no audit) ablation across any seed.
    """
    df = pd.read_csv(csv_path)

    # Filter to baseline (no audit) runs
    baseline = df[df["ablation"] == "none"]

    # Group by instance_id and check if any seed showed oscillation
    osc_per_problem = baseline.groupby("instance_id")["oscillation_count"].sum()
    osc_ids = osc_per_problem[osc_per_problem > 0].index.tolist()

    print(f"[E9-SWE] Found {len(osc_ids)} oscillation-prone problems from e5 baseline")
    return sorted(osc_ids)


def load_oscillation_problems(csv_path: Path = E5_METRICS) -> list[SWEProblem]:
    """Load the actual SWEProblem objects for the oscillation-prone subset."""
    osc_ids = load_oscillation_prone_ids(csv_path)
    all_problems = load_swebench_lite()

    # Build lookup
    id_to_problem = {p.instance_id: p for p in all_problems}

    problems = []
    missing = []
    for iid in osc_ids:
        if iid in id_to_problem:
            problems.append(id_to_problem[iid])
        else:
            missing.append(iid)

    if missing:
        print(f"[E9-SWE] WARNING: {len(missing)} problem IDs not found in dataset: "
              f"{missing[:5]}...")

    print(f"[E9-SWE] Loaded {len(problems)} oscillation-prone SWE-bench problems")
    return problems


# ── Core experiment runner ───────────────────────────────────────────

def run_single_episode(
    problem: SWEProblem,
    temperature: float,
    max_attempts: int,
    seed: int,
    audit_config: dict | None,
) -> dict:
    """Run one reflexion episode and return flat metrics dict.

    Uses the EXISTING swebench_reflexion pipeline with lightweight eval,
    which evaluates patches against gold test patches (file overlap +
    repo-based pytest when available).
    """
    # Create LLM client at the specified temperature
    llm = CachedLLMClient(
        model=MODEL,
        cache_dir=CACHE_DIR,
        mode="online",          # will make API calls
        temperature=temperature,
        max_tokens=4096,
    )

    # Create audit layer (or None for baseline)
    audit = AgentAuditLayer(audit_config) if audit_config else None

    # Run the full reflexion episode with lightweight eval
    episode = run_swe_reflexion_episode(
        problem=problem,
        llm=llm,
        max_attempts=max_attempts,
        audit=audit,
        seed=seed,
        eval_mode="lightweight",   # real SWE-bench eval, no Docker
    )

    return {
        "instance_id": episode.instance_id,
        "temperature": temperature,
        "max_attempts": max_attempts,
        "seed": seed,
        "audit": "C2" if audit_config else "none",
        "final_passed": int(episode.final_passed),
        "safe_pass": int(episode.safe_pass),
        "total_attempts": episode.total_attempts,
        "first_pass_attempt": episode.first_pass_attempt,
        "oscillation_count": episode.oscillation_count,
        "oscillation_index": round(episode.oscillation_index, 4),
        "regression_count": episode.regression_count,
        "updates_accepted": episode.updates_accepted,
        "updates_rejected": episode.updates_rejected,
    }


def run_phase_diagram(
    problems: list[SWEProblem],
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """Run full temperature x max_attempts x audit x seed sweep.

    Returns list of per-episode result dicts.
    """
    # Resume from checkpoint if it exists
    completed = set()
    all_results = []
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        for r in all_results:
            key = (r["instance_id"], r["temperature"], r["max_attempts"],
                   r["seed"], r["audit"])
            completed.add(key)
        print(f"[E9-SWE] Resumed from checkpoint: {len(completed)} episodes done")

    # Build work items
    conditions = list(itertools.product(
        TEMPERATURES,
        MAX_ATTEMPTS_LIST,
        [NO_AUDIT_CONFIG, C2_AUDIT_CONFIG],
        SEEDS,
    ))

    total_episodes = len(conditions) * len(problems)
    done = len(completed)
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  E9-SWE: Oscillation Phase Diagram (Real SWE-bench)")
    print(f"  {len(TEMPERATURES)} temps x {len(MAX_ATTEMPTS_LIST)} attempts "
          f"x 2 audit x {len(SEEDS)} seeds x {len(problems)} problems")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Already completed: {done}")
    print(f"{'='*70}\n")

    for temp, max_att, audit_cfg, seed in conditions:
        audit_label = "C2" if audit_cfg else "none"

        for problem in problems:
            key = (problem.instance_id, temp, max_att, seed, audit_label)
            if key in completed:
                continue

            try:
                result = run_single_episode(
                    problem=problem,
                    temperature=temp,
                    max_attempts=max_att,
                    seed=seed,
                    audit_config=audit_cfg,
                )
                all_results.append(result)
                done += 1

                status = "PASS" if result["final_passed"] else "FAIL"
                osc = result["oscillation_count"]
                print(f"  [{done}/{total_episodes}] T={temp} att={max_att} "
                      f"{audit_label:>4} seed={seed} "
                      f"{problem.instance_id[:40]:40s} "
                      f"{status} osc={osc}")

            except Exception as e:
                print(f"  [{done}/{total_episodes}] ERROR on "
                      f"{problem.instance_id}: {e}")
                all_results.append({
                    "instance_id": problem.instance_id,
                    "temperature": temp,
                    "max_attempts": max_att,
                    "seed": seed,
                    "audit": audit_label,
                    "final_passed": 0,
                    "safe_pass": 0,
                    "total_attempts": 0,
                    "first_pass_attempt": 0,
                    "oscillation_count": 0,
                    "oscillation_index": 0.0,
                    "regression_count": 0,
                    "updates_accepted": 0,
                    "updates_rejected": 0,
                    "error": str(e),
                })
                done += 1

            # Periodic checkpoint every 50 episodes
            if checkpoint_path and done % 50 == 0:
                _save_checkpoint(all_results, checkpoint_path)
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                remaining = (total_episodes - done) / max(rate, 0.001)
                print(f"    [checkpoint] {done}/{total_episodes} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # Final save
    if checkpoint_path:
        _save_checkpoint(all_results, checkpoint_path)

    return all_results


def _save_checkpoint(results: list[dict], path: Path):
    """Save intermediate results as JSON checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ── Analysis / Summary ──────────────────────────────────────────────

def summarize_results(results: list[dict]) -> pd.DataFrame:
    """Aggregate per-episode results into a phase diagram summary.

    Groups by (temperature, max_attempts, audit) and computes
    mean metrics across all problems and seeds.
    """
    df = pd.DataFrame(results)

    # Drop error rows for clean aggregation
    df = df[df["total_attempts"] > 0]

    summary = df.groupby(["temperature", "max_attempts", "audit"]).agg(
        pass_rate=("final_passed", "mean"),
        safe_pass_rate=("safe_pass", "mean"),
        mean_oscillation=("oscillation_count", "mean"),
        mean_regression=("regression_count", "mean"),
        mean_osc_index=("oscillation_index", "mean"),
        osc_free_rate=("oscillation_count", lambda x: (x == 0).mean()),
        mean_attempts=("total_attempts", "mean"),
        n_episodes=("instance_id", "count"),
    ).round(4).reset_index()

    return summary


def print_phase_diagram(summary: pd.DataFrame):
    """Pretty-print the phase diagram as a table."""
    print(f"\n{'='*90}")
    print(f"  E9-SWE PHASE DIAGRAM: Real SWE-bench Evaluation")
    print(f"{'='*90}")
    print(f"  {'temp':>5} {'att':>4} {'audit':>5} | {'pass':>6} {'safe':>6} "
          f"{'osc':>6} {'reg':>6} {'osc_idx':>8} {'osc_free':>9} {'att_used':>9}")
    print(f"  {'-'*5} {'-'*4} {'-'*5} + {'-'*6} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*8} {'-'*9} {'-'*9}")

    for _, row in summary.iterrows():
        print(f"  {row['temperature']:>5.1f} {row['max_attempts']:>4} "
              f"{row['audit']:>5} | "
              f"{row['pass_rate']:>6.3f} {row['safe_pass_rate']:>6.3f} "
              f"{row['mean_oscillation']:>6.3f} {row['mean_regression']:>6.3f} "
              f"{row['mean_osc_index']:>8.4f} {row['osc_free_rate']:>9.4f} "
              f"{row['mean_attempts']:>9.2f}")

    # Compute audit improvement summary
    print(f"\n{'='*90}")
    print(f"  AUDIT IMPROVEMENT (C2 vs baseline)")
    print(f"{'='*90}")

    for temp in sorted(summary["temperature"].unique()):
        for att in sorted(summary["max_attempts"].unique()):
            none_row = summary[
                (summary["temperature"] == temp) &
                (summary["max_attempts"] == att) &
                (summary["audit"] == "none")
            ]
            c2_row = summary[
                (summary["temperature"] == temp) &
                (summary["max_attempts"] == att) &
                (summary["audit"] == "C2")
            ]

            if none_row.empty or c2_row.empty:
                continue

            n = none_row.iloc[0]
            c = c2_row.iloc[0]

            osc_reduction = n["mean_oscillation"] - c["mean_oscillation"]
            reg_reduction = n["mean_regression"] - c["mean_regression"]
            pass_delta = c["pass_rate"] - n["pass_rate"]

            print(f"  T={temp:.1f} att={att}: "
                  f"osc {osc_reduction:+.3f}  "
                  f"reg {reg_reduction:+.3f}  "
                  f"pass {pass_delta:+.3f}")


# ── CSV export ───────────────────────────────────────────────────────

def export_results(results: list[dict], summary: pd.DataFrame):
    """Save full per-episode results and summary to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-episode detail
    detail_path = RESULTS_DIR / "e9_swebench_phase_episodes.csv"
    pd.DataFrame(results).to_csv(detail_path, index=False)
    print(f"\nSaved per-episode results: {detail_path}")

    # Aggregated phase diagram
    summary_path = RESULTS_DIR / "e9_swebench_phase_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved phase diagram summary: {summary_path}")

    # JSON (for plotting scripts)
    json_path = RESULTS_DIR / "e9_swebench_phase.json"
    export_data = {
        "experiment": "E9-SWE Phase Diagram (Real SWE-bench)",
        "model": MODEL,
        "temperatures": TEMPERATURES,
        "max_attempts_list": MAX_ATTEMPTS_LIST,
        "seeds": SEEDS,
        "n_problems": len(set(r["instance_id"] for r in results)),
        "summary": summary.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON summary: {json_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    """Run the E9 SWE-bench phase diagram experiment."""
    # 1. Load oscillation-prone problems
    problems = load_oscillation_problems()

    if not problems:
        print("ERROR: No oscillation-prone problems found. "
              "Check that e5_metrics.csv exists and contains oscillating episodes.")
        sys.exit(1)

    # 2. Run the sweep
    checkpoint_path = RESULTS_DIR / "e9_swebench_phase_checkpoint.json"
    results = run_phase_diagram(problems, checkpoint_path=checkpoint_path)

    # 3. Summarize and print
    summary = summarize_results(results)
    print_phase_diagram(summary)

    # 4. Export
    export_results(results, summary)

    print(f"\nE9-SWE complete. {len(results)} total episodes.")


if __name__ == "__main__":
    main()
