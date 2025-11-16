"""
MNAR Sensitivity Analysis
=========================

Evaluate MICE-DR robustness when MAR assumption is violated.

Missing Not at Random (MNAR): Missingness depends on UNOBSERVED values.
- Example: Firms with low (unobserved) TFP are more likely to not report financial data
- This violates MAR and makes MICE invalid

DGP: MNAR scenario from dgp.py (lines 227-237)
- Missingness probability depends on own value: logit(p_miss) = α_0 + α_self * X_j
- α_self = 0.5 (moderate MNAR), α_0 calibrated for target missing rate

Parameters:
- n = 2000
- p = 50
- imbalance_ratio = [3.0, 10.0] (subset of core grid)
- missing_rate = [0.2, 0.4] (subset of core grid)
- scenario = 'mnar'
- n_replications = 100 (reduced from 1000)

Total: 4 conditions × 100 reps = 400 simulations
Expected runtime: ~40 minutes on 8 cores
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
    MICEDoublyRobust, CompleteCaseDR, MICE_OLS, SingleImputationDR
)

# MNAR sensitivity grid
param_grid = {
    'n': [2000],
    'p': [50],
    'imbalance_ratio': [3.0, 10.0],
    'missing_rate': [0.2, 0.4],
    'scenario': ['mnar']  # MNAR scenario
}

n_replications = 100
n_jobs = 8


def run_one_replication(params, seed):
    """Single simulation run"""
    try:
        # Generate data
        dgp = DGP(**params, random_state=None)
        X_miss, T, Y, true_ate = dgp.generate(seed)

        # Run 4 viable estimators
        estimators = {
            'MICE-DR': MICEDoublyRobust(m_imputations=10, k_folds=5, random_state=seed),
            'Complete-DR': CompleteCaseDR(k_folds=5, random_state=seed),
            'MICE-OLS': MICE_OLS(m_imputations=10, random_state=seed),
            'SingleImp-DR': SingleImputationDR(k_folds=5, random_state=seed),
        }

        results = []
        for method_name, estimator in estimators.items():
            try:
                ate_est, se_est, ci_low, ci_up, runtime = estimator.estimate(X_miss, T, Y)

                # Calculate metrics
                bias = ate_est - true_ate if not np.isnan(ate_est) else np.nan
                squared_error = (ate_est - true_ate) ** 2 if not np.isnan(ate_est) else np.nan
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
                    'squared_error': squared_error,
                    'coverage': coverage,
                    'ci_width': ci_width,
                    'runtime': runtime,
                    'error': None
                })
            except Exception as e:
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


def save_checkpoint(df, combo_idx, total_combos):
    """Save intermediate results"""
    checkpoint_file = f'results/mnar_checkpoint_{combo_idx}of{total_combos}.csv'
    df.to_csv(checkpoint_file, index=False)
    print(f"  💾 Checkpoint saved: {checkpoint_file}")


def main():
    print("="*70)
    print("MNAR SENSITIVITY ANALYSIS - MICE-DR ROBUSTNESS")
    print("="*70)

    # Generate parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\nSimulation Setup:")
    print(f"  Parameter combinations: {len(param_combos)}")
    print(f"  Replications per combo: {n_replications}")
    print(f"  Total simulations: {len(param_combos) * n_replications:,}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"\nParameter grid:")
    for key, val in param_grid.items():
        print(f"  {key}: {val}")
    print(f"\nMethods: MICE-DR, Complete-DR, MICE-OLS, SingleImp-DR")
    print(f"Expected runtime: ~40 minutes")
    print("="*70)

    # Confirm before starting
    print(f"\n⚠️  This will run {len(param_combos) * n_replications:,} simulations.")
    print("   Press Ctrl+C within 5 seconds to cancel...")
    time.sleep(5)
    print("   Starting simulation...\n")

    # Run simulations
    all_results = []
    total_start = time.time()

    for i, params in enumerate(param_combos, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(param_combos)}] Running:")
        print(f"  n={params['n']}, p={params['p']}, "
              f"imbalance={params['imbalance_ratio']:.1f}:1, "
              f"missing={params['missing_rate']*100:.0f}%, "
              f"scenario=MNAR")
        print(f"{'='*70}")

        batch_start = time.time()

        # Parallel execution
        seeds = range(n_replications)
        batch_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_one_replication)(params, seed) for seed in seeds
        )

        # Flatten nested list
        for rep_results in batch_results:
            all_results.extend(rep_results)

        batch_time = time.time() - batch_start
        elapsed_total = time.time() - total_start
        remaining_combos = len(param_combos) - i
        est_remaining = (elapsed_total / i) * remaining_combos if i > 0 else 0

        print(f"\n  ✓ Batch completed in {batch_time/60:.1f} min ({batch_time/n_replications:.2f}s per rep)")
        print(f"  ⏱️  Elapsed: {elapsed_total/60:.1f} min, Remaining: ~{est_remaining/60:.1f} min")

        # Save checkpoint
        df_checkpoint = pd.DataFrame(all_results)
        save_checkpoint(df_checkpoint, i, len(param_combos))

    total_time = time.time() - total_start

    # Final save
    df = pd.DataFrame(all_results)
    df_clean = df[df['error'].isna()].copy()

    print(f"\n{'='*70}")
    print(f"MNAR SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Total rows: {len(df):,}")
    print(f"Clean rows: {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}%)")
    print(f"Errors: {len(df) - len(df_clean):,}")

    # Summary statistics
    if len(df_clean) > 0:
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")

        summary = df_clean.groupby(['imbalance_ratio', 'missing_rate', 'method']).agg({
            'bias': ['mean', 'std'],
            'squared_error': 'mean',
            'coverage': 'mean',
            'runtime': 'mean'
        }).round(3)

        # Calculate RMSE
        summary['rmse'] = np.sqrt(summary[('squared_error', 'mean')])

        print("\n--- Mean Bias (MNAR Conditions) ---")
        bias_table = summary[('bias', 'mean')].unstack()
        print(bias_table)

        print("\n--- RMSE (MNAR Conditions) ---")
        rmse_table = summary['rmse'].unstack()
        print(rmse_table)

        print("\n--- Coverage Rate (MNAR Conditions) ---")
        coverage_table = summary[('coverage', 'mean')].unstack()
        print(coverage_table)

        print("\n--- Average Runtime (sec) ---")
        runtime_table = summary[('runtime', 'mean')].unstack()
        print(runtime_table)

        # MNAR-specific interpretation
        print(f"\n{'='*70}")
        print("MNAR INTERPRETATION")
        print(f"{'='*70}")
        print("\n⚠️  Expected Behavior under MNAR:")
        print("  • MICE-DR should show INCREASED BIAS compared to MAR")
        print("  • Complete-Case-DR may be MORE ROBUST (if missingness independent of treatment)")
        print("  • Coverage rates should DEGRADE for all methods")
        print("  • Bias magnitude indicates severity of MAR violation")

        print("\n📊 Key Metrics to Compare with Core Results (MAR):")
        mice_dr_bias_mnar = bias_table['MICE-DR'].mean()
        mice_dr_coverage_mnar = coverage_table['MICE-DR'].mean()

        print(f"  • MICE-DR Mean Bias (MNAR): {mice_dr_bias_mnar:.4f}")
        print(f"  • MICE-DR Mean Bias (MAR, from core): ~0.002 (reference)")
        print(f"  • MICE-DR Coverage (MNAR): {mice_dr_coverage_mnar:.2%}")
        print(f"  • MICE-DR Coverage (MAR, from core): ~94% (reference)")
        print(f"\n  • Bias Increase: {abs(mice_dr_bias_mnar) / 0.002:.1f}x (if MAR bias ~0.002)")
        print(f"  • Coverage Drop: {0.94 - mice_dr_coverage_mnar:.1%}")

    # Save final results
    output_file = 'results/mnar_sensitivity_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Final results saved to: {output_file}")

    # Save summary tables
    if len(df_clean) > 0:
        bias_table.to_csv('results/table_bias_mnar.csv')
        rmse_table.to_csv('results/table_rmse_mnar.csv')
        coverage_table.to_csv('results/table_coverage_mnar.csv')
        runtime_table.to_csv('results/table_runtime_mnar.csv')
        print(f"✓ Summary tables saved to: results/table_*_mnar.csv")

    print(f"\n{'='*70}")
    print("MNAR SENSITIVITY ANALYSIS COMPLETED!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
