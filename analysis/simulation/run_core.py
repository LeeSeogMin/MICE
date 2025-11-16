"""
Core Simulation - Main Results for Paper
=========================================

Phase 1: Core scenarios for paper Section 4

Parameters:
- n = 2000 (realistic sample size)
- p = 50 (medium dimensionality)
- imbalance_ratio = 3, 10, 20 (all three)
- missing_rate = 0.2, 0.4, 0.6 (all three)
- scenario = 'linear' (baseline)
- n_replications = 1000 (full)

Total: 9 combos × 1000 reps = 9,000 simulations
Expected runtime: ~3 hours on 8 cores

Methods compared (4 viable methods):
1. MICE-DR (AIPW) - Our main method
2. Complete-Case-DR - Baseline comparison
3. MICE-OLS - Non-robust alternative
4. Single-Imputation-DR - No uncertainty propagation
(CEM removed - failed in pilot)
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

# Core simulation grid - Extended with nonlinear scenario
param_grid = {
    'n': [2000],
    'p': [50],
    'imbalance_ratio': [3.0, 10.0, 20.0],
    'missing_rate': [0.2, 0.4, 0.6],
    'scenario': ['linear', 'nonlinear']  # Added nonlinear DGP
}

n_replications = 100  # Reduced from 1000 for faster completion
n_jobs = 8  # CPU parallelism with GradientBoosting


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
    checkpoint_file = f'results/core_checkpoint_{combo_idx}of{total_combos}.csv'
    df.to_csv(checkpoint_file, index=False)
    print(f"  💾 Checkpoint saved: {checkpoint_file}")


def load_checkpoint(total_combos):
    """Load most recent checkpoint if available

    Returns:
        tuple: (DataFrame of results, last completed combo index) or (None, 0)
    """
    import os
    import glob

    # Find all checkpoint files
    checkpoint_pattern = f'results/core_checkpoint_*of{total_combos}.csv'
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        print("  ℹ️  No checkpoint found. Starting from beginning.")
        return None, 0

    # Parse checkpoint indices from filenames
    checkpoint_indices = []
    for cp in checkpoints:
        try:
            # Extract index from filename like "core_checkpoint_3of9.csv"
            basename = os.path.basename(cp)
            idx_str = basename.split('_')[2].split('of')[0]
            checkpoint_indices.append((int(idx_str), cp))
        except (IndexError, ValueError):
            continue

    if not checkpoint_indices:
        print("  ℹ️  No valid checkpoint found. Starting from beginning.")
        return None, 0

    # Get most recent checkpoint (highest index)
    checkpoint_indices.sort(reverse=True)
    last_idx, last_file = checkpoint_indices[0]

    print(f"  📂 Found checkpoint: {os.path.basename(last_file)}")
    print(f"  🔄 Loading previous results...")

    try:
        df = pd.read_csv(last_file)
        print(f"  ✓ Loaded {len(df):,} rows from checkpoint")
        print(f"  ➡️  Resuming from condition {last_idx + 1}/{total_combos}")
        return df, last_idx
    except Exception as e:
        print(f"  ⚠️  Error loading checkpoint: {e}")
        print(f"  ℹ️  Starting from beginning.")
        return None, 0


def get_completed_conditions(df, param_combos):
    """Determine which parameter combinations are already complete

    Args:
        df: DataFrame with existing results
        param_combos: List of parameter dictionaries

    Returns:
        set: Indices of completed conditions
    """
    if df is None or len(df) == 0:
        return set()

    completed = set()

    for idx, params in enumerate(param_combos):
        # Check if this condition exists in results
        mask = (
            (df['n'] == params['n']) &
            (df['p'] == params['p']) &
            (df['imbalance_ratio'] == params['imbalance_ratio']) &
            (df['missing_rate'] == params['missing_rate']) &
            (df['scenario'] == params['scenario'])
        )

        condition_results = df[mask]

        # Count unique replications (seeds) for this condition
        if len(condition_results) > 0:
            n_seeds = condition_results['seed'].nunique()
            # Consider complete if we have all replications
            if n_seeds >= n_replications:
                completed.add(idx)

    return completed


def main():
    print("="*70)
    print("CORE SIMULATION - MICE-DR BENCHMARK STUDY")
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
    print(f"Expected runtime: ~3 hours")
    print("="*70)

    # Check for checkpoint
    checkpoint_df, last_completed_idx = load_checkpoint(len(param_combos))

    # Initialize results
    if checkpoint_df is not None:
        all_results = checkpoint_df.to_dict('records')
        completed_conditions = get_completed_conditions(checkpoint_df, param_combos)
        start_idx = last_completed_idx
        print(f"\n  ✓ Completed conditions: {sorted(completed_conditions)}")
        print(f"  ⏭️  Skipping {len(completed_conditions)} completed condition(s)")
    else:
        all_results = []
        completed_conditions = set()
        start_idx = 0

    # Confirm before starting
    remaining_sims = (len(param_combos) - len(completed_conditions)) * n_replications
    if remaining_sims > 0:
        print(f"\n⚠️  This will run {remaining_sims:,} simulations.")
        print("   Press Ctrl+C within 5 seconds to cancel...")
        time.sleep(5)
        print("   Starting simulation...\n")
    else:
        print("\n✓ All conditions already completed!")
        print("  Use merge script to combine results from multiple runs.\n")
        return

    # Run simulations
    total_start = time.time()

    for i, params in enumerate(param_combos, 1):
        # Skip if already completed
        if (i - 1) in completed_conditions:
            print(f"\n{'='*70}")
            print(f"[{i}/{len(param_combos)}] SKIPPING (already completed):")
            print(f"  n={params['n']}, p={params['p']}, "
                  f"imbalance={params['imbalance_ratio']:.1f}:1, "
                  f"missing={params['missing_rate']*100:.0f}%")
            print(f"{'='*70}")
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{len(param_combos)}] Running:")
        print(f"  n={params['n']}, p={params['p']}, "
              f"imbalance={params['imbalance_ratio']:.1f}:1, "
              f"missing={params['missing_rate']*100:.0f}%")
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
        completed_so_far = len([c for c in range(i) if c not in completed_conditions or c < i-1])
        remaining_combos = len(param_combos) - len(completed_conditions) - completed_so_far
        if completed_so_far > 0:
            est_remaining = (elapsed_total / completed_so_far) * remaining_combos
        else:
            est_remaining = 0

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
    print(f"CORE SIMULATION COMPLETE")
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

        print("\n--- Mean Bias ---")
        bias_table = summary[('bias', 'mean')].unstack()
        print(bias_table)

        print("\n--- RMSE ---")
        rmse_table = summary['rmse'].unstack()
        print(rmse_table)

        print("\n--- Coverage Rate ---")
        coverage_table = summary[('coverage', 'mean')].unstack()
        print(coverage_table)

        print("\n--- Average Runtime (sec) ---")
        runtime_table = summary[('runtime', 'mean')].unstack()
        print(runtime_table)

    # Save final results
    output_file = 'results/core_simulation_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Final results saved to: {output_file}")

    # Save summary tables
    if len(df_clean) > 0:
        bias_table.to_csv('results/table_bias_core.csv')
        rmse_table.to_csv('results/table_rmse_core.csv')
        coverage_table.to_csv('results/table_coverage_core.csv')
        runtime_table.to_csv('results/table_runtime_core.csv')
        print(f"✓ Summary tables saved to: results/table_*.csv")

    print(f"\n{'='*70}")
    print("CORE SIMULATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
