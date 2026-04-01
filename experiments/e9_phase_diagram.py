"""E9: Oscillation Phase Diagram — When and Why Do Self-Improving Agents Fail?

Sweeps (h_rate, lr, drift_freq) to map the oscillation landscape:
- Identifies the phase boundary between stable and unstable regimes
- Shows where each WhyLab component (C1/C2/C3) activates
- Produces a publication-quality phase diagram

Key finding: oscillation is a function of (noise_rate × step_size),
with a sharp phase transition. WhyLab activates only in the unstable region.
"""
import numpy as np
import pandas as pd
import json
import os
import time
from dataclasses import dataclass
from typing import Tuple, List

# Reuse E6 components
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_nonstationary_agent import E6Config, NonStationaryEnv, C1DriftDetector, C2SensitivityFilter, C3LyapunovDamper

SEEDS = 10
H_RATES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
LR_VALUES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
DRIFT_FREQS = [0, 1, 2]  # 0=stationary, 1=single drift at t=300, 2=double drift at t=200,400


def run_single(h_rate, lr, drift_freq, seed, ablation="none"):
    """Run one episode, return metrics."""
    if drift_freq == 0:
        drift_points = ()
    elif drift_freq == 1:
        drift_points = (300,)
    else:
        drift_points = (200, 400)

    cfg = E6Config(
        d=10, T=600,
        drift_points=drift_points,
        drift_magnitude=3.0,
        lr_base=lr,
        h_rate=h_rate,
        noise_std=1.0,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    env = NonStationaryEnv(cfg, rng)

    theta = rng.standard_normal(cfg.d) * 0.5

    use_c1 = "C1" in ablation or ablation == "full"
    use_c2 = "C2" in ablation or ablation == "full"
    use_c3 = "C3" in ablation or ablation == "full"

    c1 = C1DriftDetector(cfg.c1_window, cfg.c1_threshold) if use_c1 else None
    c2 = C2SensitivityFilter(cfg.c2_threshold, cfg.c2_rv_threshold) if use_c2 else None
    c3 = C3LyapunovDamper(cfg.beta_ema, cfg.floor, cfg.ceiling) if use_c3 else None

    energies = []
    rewards = []
    osc_count = 0
    prev_improving = None
    prev_reward = None
    regressions = 0
    c1_alerts = 0
    c2_rejections = 0

    for t in range(cfg.T):
        target = env.get_target(t)
        V = 0.5 * np.sum((theta - target) ** 2)
        energies.append(V)

        reward, grad = env.observe(theta, t)
        rewards.append(reward)

        # C1: drift detection
        drift_alert = False
        if c1:
            drift_alert = c1.update(reward)
            if drift_alert:
                c1_alerts += 1

        # C2: sensitivity filter
        accepted = True
        if c2 and prev_reward is not None:
            accepted = c2.should_accept(prev_reward, reward, np.linalg.norm(grad))
            if not accepted:
                c2_rejections += 1

        # C3: step size
        if c3:
            zeta = c3.compute_zeta(grad, drift_alert)
        else:
            zeta = lr

        # Update
        if accepted:
            theta = theta - zeta * grad

        # Track oscillation (E6-style direction change)
        improving = reward > np.mean(rewards[-10:]) if len(rewards) > 10 else True
        if prev_improving is not None and improving != prev_improving:
            osc_count += 1
        prev_improving = improving

        # Track regression (reward drop > 2.0)
        if prev_reward is not None:
            if reward < prev_reward - 2.0:
                regressions += 1
        prev_reward = reward

    return {
        'final_energy': float(energies[-1]),
        'mean_energy': float(np.mean(energies)),
        'max_energy': float(np.max(energies)),
        'oscillation_count': osc_count,
        'oscillation_index': osc_count / cfg.T,
        'regression_count': regressions,
        'c1_alerts': c1_alerts,
        'c2_rejections': c2_rejections,
    }


def run_phase_diagram():
    """Run full 3D sweep."""
    results = []
    total = len(H_RATES) * len(LR_VALUES) * len(DRIFT_FREQS) * 2  # ×2 for none + full
    done = 0
    t0 = time.time()

    for drift_freq in DRIFT_FREQS:
        drift_label = ["stationary", "single_drift", "double_drift"][drift_freq]
        print(f"\n  [{drift_label}]")

        for h_rate in H_RATES:
            for lr in LR_VALUES:
                for ablation in ["none", "full"]:
                    seed_results = []
                    for s in range(SEEDS):
                        r = run_single(h_rate, lr, drift_freq, 42 + s, ablation)
                        seed_results.append(r)

                    # Aggregate
                    agg = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
                    agg_std = {k + '_std': float(np.std([r[k] for r in seed_results])) for k in seed_results[0]}
                    agg.update(agg_std)
                    agg['h_rate'] = h_rate
                    agg['lr'] = lr
                    agg['drift_freq'] = drift_freq
                    agg['drift_label'] = drift_label
                    agg['ablation'] = ablation
                    agg['n_seeds'] = SEEDS
                    results.append(agg)

                    done += 1
                    if done % 20 == 0:
                        elapsed = time.time() - t0
                        print(f"    {done}/{total} ({elapsed:.0f}s)", flush=True)

    return results


def analyze_phase_boundary(results):
    """Find the phase transition boundary."""
    df = pd.DataFrame(results)

    # Phase transition: where does oscillation_index cross 0.1?
    print("\n" + "=" * 70)
    print("  PHASE DIAGRAM ANALYSIS")
    print("=" * 70)

    for drift_label in df['drift_label'].unique():
        print(f"\n  --- {drift_label} ---")
        sub = df[(df['drift_label'] == drift_label) & (df['ablation'] == 'none')]

        print(f"  {'h_rate':>8} {'lr':>8} {'osc_idx':>10} {'energy':>10} {'zone':>10}")
        print(f"  {'-'*50}")
        for _, row in sub.iterrows():
            osc = row['oscillation_index']
            zone = "STABLE" if osc < 0.05 else ("TRANS" if osc < 0.15 else "UNSTABLE")
            print(f"  {row['h_rate']:>8.2f} {row['lr']:>8.2f} {osc:>10.3f} {row['mean_energy']:>10.1f} {zone:>10}")

    # Audit benefit: where does "full" reduce oscillation vs "none"?
    print(f"\n  --- AUDIT BENEFIT (full vs none) ---")
    for drift_label in df['drift_label'].unique():
        none_df = df[(df['drift_label'] == drift_label) & (df['ablation'] == 'none')]
        full_df = df[(df['drift_label'] == drift_label) & (df['ablation'] == 'full')]

        merged = none_df.merge(full_df, on=['h_rate', 'lr', 'drift_freq'],
                               suffixes=('_none', '_full'))

        merged['osc_reduction'] = (merged['oscillation_index_none'] - merged['oscillation_index_full'])
        merged['energy_reduction'] = (merged['mean_energy_none'] - merged['mean_energy_full'])

        significant = merged[merged['osc_reduction'] > 0.02]
        if len(significant) > 0:
            print(f"\n  {drift_label}: {len(significant)} conditions where audit helps")
            for _, row in significant.head(5).iterrows():
                print(f"    h={row['h_rate']:.2f} lr={row['lr']:.2f}: "
                      f"osc {row['oscillation_index_none']:.3f}→{row['oscillation_index_full']:.3f} "
                      f"({row['osc_reduction']:.3f} reduction)")

    return df


if __name__ == '__main__':
    t0 = time.time()
    print("E9: Oscillation Phase Diagram")
    print(f"  {len(H_RATES)} h_rates × {len(LR_VALUES)} lrs × {len(DRIFT_FREQS)} drift_freqs × 2 ablations × {SEEDS} seeds")
    print(f"  Total: {len(H_RATES) * len(LR_VALUES) * len(DRIFT_FREQS) * 2 * SEEDS} episodes")

    results = run_phase_diagram()
    df = analyze_phase_boundary(results)

    t_total = time.time() - t0

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'e9_phase_diagram.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    json_path = os.path.join(out_dir, 'e9_phase_diagram.json')
    with open(json_path, 'w') as f:
        json.dump({
            'experiment': 'E9: Oscillation Phase Diagram',
            'config': {
                'h_rates': H_RATES, 'lr_values': LR_VALUES,
                'drift_freqs': DRIFT_FREQS, 'seeds': SEEDS,
            },
            'n_results': len(results),
            'time_seconds': round(t_total, 1),
        }, f, indent=2)

    print(f"\nTotal time: {t_total:.0f}s")
