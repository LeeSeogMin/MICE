"""
Merge Multiple Simulation Results
===================================

This script merges results from multiple simulation runs (checkpoints or final results)
into a single consolidated dataset.

Usage:
    python merge_simulation_results.py results/core_*.csv
    python merge_simulation_results.py results/core_checkpoint_*.csv results/core_simulation_results.csv

Features:
- Loads multiple CSV files
- Deduplicates results based on (params, seed, method)
- Validates consistency across runs
- Generates merged output with statistics
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def load_result_files(file_patterns):
    """Load all result CSV files matching patterns

    Args:
        file_patterns: List of file paths or glob patterns

    Returns:
        list: List of (filepath, DataFrame) tuples
    """
    all_files = []

    for pattern in file_patterns:
        # Expand glob patterns
        matched_files = glob.glob(pattern)
        if not matched_files:
            # Try as direct file path
            if os.path.exists(pattern):
                matched_files = [pattern]
            else:
                print(f"⚠️  No files matched pattern: {pattern}")
                continue

        for filepath in matched_files:
            try:
                df = pd.read_csv(filepath)
                all_files.append((filepath, df))
                print(f"✓ Loaded {len(df):,} rows from {os.path.basename(filepath)}")
            except Exception as e:
                print(f"❌ Error loading {filepath}: {e}")

    return all_files


def validate_consistency(dataframes):
    """Check that parameter grids are consistent across runs

    Args:
        dataframes: List of (filepath, DataFrame) tuples

    Returns:
        bool: True if consistent, False otherwise
    """
    if len(dataframes) < 2:
        return True

    print("\n" + "="*70)
    print("VALIDATION: Checking parameter consistency")
    print("="*70)

    # Extract parameter columns
    param_cols = ['n', 'p', 'imbalance_ratio', 'missing_rate', 'scenario']

    reference_params = None
    all_consistent = True

    for filepath, df in dataframes:
        if param_cols[0] not in df.columns:
            print(f"⚠️  {os.path.basename(filepath)}: Missing parameter columns")
            all_consistent = False
            continue

        # Get unique parameter combinations
        unique_params = df[param_cols].drop_duplicates().sort_values(param_cols)

        if reference_params is None:
            reference_params = unique_params
            print(f"✓ Reference: {os.path.basename(filepath)}")
            print(f"  Parameter combinations: {len(unique_params)}")
        else:
            # Check if matches reference
            if not unique_params.equals(reference_params):
                print(f"⚠️  {os.path.basename(filepath)}: Parameter grid mismatch")
                all_consistent = False
            else:
                print(f"✓ {os.path.basename(filepath)}: Parameters consistent")

    return all_consistent


def merge_results(dataframes, strategy='keep_first'):
    """Merge multiple DataFrames, handling duplicates

    Args:
        dataframes: List of (filepath, DataFrame) tuples
        strategy: How to handle duplicates
            - 'keep_first': Keep first occurrence of duplicate
            - 'keep_last': Keep last occurrence
            - 'average': Average results for duplicates (for numeric columns)

    Returns:
        pd.DataFrame: Merged results
    """
    print("\n" + "="*70)
    print("MERGING RESULTS")
    print("="*70)

    if not dataframes:
        print("❌ No data to merge")
        return None

    # Concatenate all DataFrames
    all_dfs = [df for _, df in dataframes]
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"Total rows before deduplication: {len(merged):,}")

    # Identify duplicate rows based on key columns
    key_cols = ['n', 'p', 'imbalance_ratio', 'missing_rate', 'scenario', 'seed', 'method']

    # Check if all key columns exist
    missing_cols = [col for col in key_cols if col not in merged.columns]
    if missing_cols:
        print(f"⚠️  Missing key columns: {missing_cols}")
        print("  Cannot deduplicate. Returning concatenated results.")
        return merged

    # Find duplicates
    duplicates = merged.duplicated(subset=key_cols, keep=False)
    n_duplicates = duplicates.sum()

    if n_duplicates == 0:
        print("✓ No duplicates found")
        return merged

    print(f"⚠️  Found {n_duplicates:,} duplicate rows")

    if strategy == 'keep_first':
        deduped = merged.drop_duplicates(subset=key_cols, keep='first')
        print(f"  Strategy: keep_first")
    elif strategy == 'keep_last':
        deduped = merged.drop_duplicates(subset=key_cols, keep='last')
        print(f"  Strategy: keep_last")
    elif strategy == 'average':
        # For averaging, need to group and aggregate
        print(f"  Strategy: average (numeric columns only)")

        # Separate numeric and non-numeric columns
        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        # Remove key columns from numeric aggregation
        agg_cols = [col for col in numeric_cols if col not in key_cols]

        if not agg_cols:
            print("  ⚠️  No numeric columns to average. Using keep_first instead.")
            deduped = merged.drop_duplicates(subset=key_cols, keep='first')
        else:
            # Group and aggregate
            agg_dict = {col: 'mean' for col in agg_cols}
            deduped = merged.groupby(key_cols, as_index=False).agg(agg_dict)
    else:
        print(f"  ⚠️  Unknown strategy: {strategy}. Using keep_first.")
        deduped = merged.drop_duplicates(subset=key_cols, keep='first')

    print(f"Total rows after deduplication: {len(deduped):,}")
    print(f"Removed {len(merged) - len(deduped):,} duplicate rows")

    return deduped


def generate_statistics(df):
    """Generate summary statistics for merged results

    Args:
        df: Merged DataFrame

    Returns:
        dict: Summary statistics
    """
    stats = {}

    if df is None or len(df) == 0:
        return stats

    # Basic counts
    stats['total_rows'] = len(df)
    stats['n_conditions'] = len(df.groupby(['n', 'p', 'imbalance_ratio', 'missing_rate', 'scenario']))

    # Method counts
    if 'method' in df.columns:
        stats['methods'] = df['method'].unique().tolist()
        stats['method_counts'] = df['method'].value_counts().to_dict()

    # Replication counts per condition
    if 'seed' in df.columns:
        reps_per_condition = df.groupby(
            ['n', 'p', 'imbalance_ratio', 'missing_rate', 'scenario']
        )['seed'].nunique()
        stats['min_replications'] = reps_per_condition.min()
        stats['max_replications'] = reps_per_condition.max()
        stats['mean_replications'] = reps_per_condition.mean()

    # Error counts
    if 'error' in df.columns:
        n_errors = df['error'].notna().sum()
        stats['n_errors'] = n_errors
        stats['error_rate'] = n_errors / len(df) if len(df) > 0 else 0

    # ATE statistics (if available)
    if 'ate_est' in df.columns:
        clean_df = df[df['ate_est'].notna()]
        if len(clean_df) > 0:
            stats['ate_mean'] = clean_df['ate_est'].mean()
            stats['ate_std'] = clean_df['ate_est'].std()
            stats['ate_min'] = clean_df['ate_est'].min()
            stats['ate_max'] = clean_df['ate_est'].max()

    return stats


def main():
    print("="*70)
    print("SIMULATION RESULTS MERGER")
    print("="*70)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python merge_simulation_results.py <file_pattern> [<file_pattern> ...]")
        print("\nExamples:")
        print("  python merge_simulation_results.py results/core_checkpoint_*.csv")
        print("  python merge_simulation_results.py results/run1_*.csv results/run2_*.csv")
        sys.exit(1)

    file_patterns = sys.argv[1:]

    # Load all files
    print("\n" + "="*70)
    print("LOADING FILES")
    print("="*70)
    dataframes = load_result_files(file_patterns)

    if not dataframes:
        print("\n❌ No files loaded. Exiting.")
        sys.exit(1)

    print(f"\n✓ Successfully loaded {len(dataframes)} file(s)")

    # Validate consistency
    is_consistent = validate_consistency(dataframes)
    if not is_consistent:
        print("\n⚠️  WARNING: Parameter grids are inconsistent across files.")
        print("  Merged results may contain unexpected combinations.")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nMerge cancelled.")
            sys.exit(0)

    # Merge results
    merged_df = merge_results(dataframes, strategy='keep_first')

    if merged_df is None:
        print("\n❌ Merge failed. Exiting.")
        sys.exit(1)

    # Generate statistics
    print("\n" + "="*70)
    print("MERGED RESULTS STATISTICS")
    print("="*70)
    stats = generate_statistics(merged_df)

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"  {key}: {', '.join(map(str, value))}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # Save merged results
    output_file = 'results/merged_simulation_results.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Merged results saved to: {output_file}")

    # Also save a summary report
    report_file = 'results/merge_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SIMULATION RESULTS MERGE REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("Input Files:\n")
        for filepath, df in dataframes:
            f.write(f"  - {filepath} ({len(df):,} rows)\n")
        f.write(f"\nTotal input files: {len(dataframes)}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("STATISTICS\n")
        f.write("="*70 + "\n")

        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            elif isinstance(value, list):
                f.write(f"{key}: {', '.join(map(str, value))}\n")
            elif isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("OUTPUT\n")
        f.write("="*70 + "\n")
        f.write(f"Merged file: {output_file}\n")
        f.write(f"Total rows: {len(merged_df):,}\n")

    print(f"✓ Merge report saved to: {report_file}")

    print("\n" + "="*70)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == '__main__':
    main()
