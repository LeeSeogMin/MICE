"""
Pilot Simulation - Small Scale Test
====================================

Quick test with reduced parameters to verify pipeline before full run

Parameters:
- n = 1000 (smaller sample)
- p = 20 (fewer covariates)
- imbalance_ratio = 3, 10 (skip 20:1)
- missing_rate = 0.2, 0.4 (skip 0.6)
- scenario = 'linear' only
- n_replications = 50 (instead of 1000)

Expected runtime: ~30 minutes on 8 cores
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from simulation.dgp import DGP
from simulation.estimators import (
    MICEDoublyRobust, CompleteCaseDR, MICE_OLS,
    SingleImputationDR, CEM_Matching
)

# Pilot simulation grid (reduced)
param_grid = {
    'n': [1000],
    'p': [20],
    'imbalance_ratio': [3.0, 10.0],
    'missing_rate': [0.2, 0.4],
    'scenario': ['linear']
}

n_replications = 50  # Reduced from 1000
n_jobs = 8  # Parallel cores


def run_one_replication(params, seed):
    """
    Single simulation run

    Returns dict with results for all methods
    """
    try:
        # Generate data
        dgp = DGP(**params, random_state=None)  # Use seed for each rep
        X_miss, T, Y, true_ate = dgp.generate(seed)

        # Run all estimators
        estimators = {
            'MICE-DR': MICEDoublyRobust(m_imputations=5, k_folds=3, random_state=seed),
            'Complete-DR': CompleteCaseDR(k_folds=3, random_state=seed),
            'MICE-OLS': MICE_OLS(m_imputations=5, random_state=seed),
            'SingleImp-DR': SingleImputationDR(k_folds=3, random_state=seed),
            'CEM': CEM_Matching(n_bins=3, random_state=seed),
        }

        results = []
        for method_name, estimator in estimators.items():
            try:
                ate_est, se_est, ci_low, ci_up, runtime = estimator.estimate(X_miss, T, Y)

                # Calculate metrics
                bias = ate_est - true_ate if not np.isnan(ate_est) else np.nan
                coverage = int(ci_low <= true_ate <= ci_up) if not np.isnan(ci_low) else 0
                ci_width = ci_up - ci_low if not np.isnan(ci_up) else np.nan

                results.append({
                    **params,
                    'seed': seed,
                    'method': method_name,
                    'true_ate': true_ate,
                    'ate_est': ate_est,
                    'se_est': se_est,
                    'ci_lower': ci_low,
                    'ci_upper': ci_up,
                    'bias': bias,
                    'coverage': coverage,
                    'ci_width': ci_width,
                    'runtime': runtime,
                    'error': None
                })
            except Exception as e:
                # Record error but continue
                results.append({
                    **params,
                    'seed': seed,
                    'method': method_name,
                    'error': str(e),
                    'ate_est': np.nan,
                    'bias': np.nan,
                    'coverage': 0
                })

        return results

    except Exception as e:
        print(f"ERROR in replication {seed}: {e}")
        return []


def main():
    print("="*70)
    print("PILOT SIMULATION - MICE-DR BENCHMARK STUDY")
    print("="*70)

    # Generate parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\nSimulation Setup:")
    print(f"  Parameter combinations: {len(param_combos)}")
    print(f"  Replications per combo: {n_replications}")
    print(f"  Total simulations: {len(param_combos) * n_replications}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"\nParameter grid:")
    for key, val in param_grid.items():
        print(f"  {key}: {val}")

    # Run simulations
    all_results = []
    total_start = time.time()

    for i, params in enumerate(param_combos, 1):
        print(f"\n[{i}/{len(param_combos)}] Running:")
        print(f"  n={params['n']}, p={params['p']}, "
              f"imbalance={params['imbalance_ratio']}, "
              f"missing={params['missing_rate']}")

        batch_start = time.time()

        # Parallel execution
        seeds = range(n_replications)
        batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_one_replication)(params, seed) for seed in seeds
        )

        # Flatten nested list
        for rep_results in batch_results:
            all_results.extend(rep_results)

        batch_time = time.time() - batch_start
        print(f"  ✓ Completed in {batch_time:.1f}s ({batch_time/n_replications:.2f}s per rep)")

    total_time = time.time() - total_start

    # Save results
    df = pd.DataFrame(all_results)

    # Remove error rows for analysis
    df_clean = df[df['error'].isna()].copy()

    print(f"\n{'='*70}")
    print(f"PILOT SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total rows: {len(df)}")
    print(f"Clean rows: {len(df_clean)} ({len(df_clean)/len(df)*100:.1f}%)")
    print(f"Errors: {len(df) - len(df_clean)}")

    # Quick summary stats
    if len(df_clean) > 0:
        print(f"\n{'='*70}")
        print("QUICK RESULTS SUMMARY")
        print(f"{'='*70}")

        summary = df_clean.groupby(['imbalance_ratio', 'missing_rate', 'method']).agg({
            'bias': ['mean', 'std'],
            'ate_est': 'std',
            'coverage': 'mean',
            'runtime': 'mean'
        }).round(3)

        print("\nMean Bias:")
        print(summary['bias']['mean'].unstack())

        print("\nRMSE (approx):")
        rmse = np.sqrt(summary[('bias', 'mean')]**2 + summary[('ate_est', 'std')]**2)
        print(rmse.unstack())

        print("\nCoverage Rate:")
        print(summary['coverage']['mean'].unstack())

        print("\nAverage Runtime (sec):")
        print(summary['runtime']['mean'].unstack())

    # Save to CSV
    output_file = 'results/pilot_simulation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Estimate full simulation time
    time_per_rep = total_time / (len(param_combos) * n_replications)
    full_combos = 3 * 3 * 3 * 3  # Full grid: n×p×imbalance×missing
    full_reps = 1000
    estimated_full_time = time_per_rep * full_combos * full_reps / n_jobs / 3600

    print(f"\n{'='*70}")
    print("FULL SIMULATION ESTIMATE")
    print(f"{'='*70}")
    print(f"Time per replication: {time_per_rep:.2f}s")
    print(f"Full grid size: {full_combos} combos × {full_reps} reps = {full_combos*full_reps:,} total")
    print(f"Estimated runtime: {estimated_full_time:.1f} hours ({estimated_full_time/24:.1f} days)")
    print(f"  (With {n_jobs} parallel cores)")

    print(f"\n{'='*70}")
    print("PILOT SIMULATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
