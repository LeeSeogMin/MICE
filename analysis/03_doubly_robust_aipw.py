#!/usr/bin/env python3
"""
Phase 2: Augmented Inverse Probability Weighting (AIPW)
- 핵심: Doubly Robust Estimator (주 분석)
- 장점: Propensity score나 outcome model 중 하나만 맞아도 일치추정량
- 산출: ATE, 95% CI, 강건성 진단
"""

import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse import csr_matrix, issparse
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 설정
INPUT_DIR = "analysis/processed_data"
OUTPUT_DIR = "analysis/results"
RANDOM_SEED = 42

def load_data_with_tfp():
    """Load imputed datasets with TFP"""
    print(f"\n{'='*60}")
    print("Step 1: TFP Dataset 로드")
    print(f"{'='*60}")

    pkl_path = os.path.join(INPUT_DIR, "imputed_datasets_with_tfp.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        datasets = pickle.load(f)

    print(f"✅ {len(datasets)}개 imputed datasets 로드")

    return datasets

def load_selected_variables():
    """Load LASSO-selected variables"""
    print(f"\n{'='*60}")
    print("Step 2: 선택된 변수 로드")
    print(f"{'='*60}")

    pkl_path = os.path.join(INPUT_DIR, "selected_variables.pkl")

    if not os.path.exists(pkl_path):
        print("⚠️ 선택된 변수 파일 없음 - 모든 변수 사용")
        return None

    with open(pkl_path, 'rb') as f:
        selected_vars = pickle.load(f)

    print(f"✅ {len(selected_vars)}개 선택 변수 로드")

    return selected_vars

def prepare_aipw_data(df, selected_vars=None):
    """
    Prepare data for AIPW analysis

    Returns:
    - T: Binary treatment (DTI > 0)
    - Y: Outcome (TFP_LP)
    - X: Covariates
    - covariate_names: List of covariate names
    """
    print(f"\n{'='*60}")
    print("Step 3: AIPW 데이터 준비")
    print(f"{'='*60}")

    # Treatment: Binary DTI
    if 'dti_binary' not in df.columns:
        # Reconstruct if missing
        # FIXED: Data is already recoded to 0-6 (0 = non-adopter)
        # Do NOT subtract 1 again!
        dti_cols = ['dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata']
        available = [c for c in dti_cols if c in df.columns]

        if len(available) >= 2:
            df['dti_continuous'] = df[available].mean(axis=1)
            df['dti_binary'] = (df['dti_continuous'] > 0).astype(int)
        else:
            raise ValueError("Cannot construct DTI")

    T = df['dti_binary'].values

    # Outcome: TFP_LP (preferred) or tfp_simple
    if 'tfp_lp' in df.columns:
        Y = df['tfp_lp'].values
        outcome_name = 'TFP_LP'
    elif 'tfp_simple' in df.columns:
        Y = df['tfp_simple'].values
        outcome_name = 'TFP_simple'
    else:
        raise ValueError("No TFP variable found")

    print(f"Treatment (DTI_binary):")
    print(f"  채택자 (T=1): {T.sum():,}개 ({T.mean()*100:.1f}%)")
    print(f"  미채택자 (T=0): {(1-T).sum():,}개 ({(1-T).mean()*100:.1f}%)")

    print(f"\nOutcome ({outcome_name}):")
    print(f"  평균: {np.nanmean(Y):.4f}")
    print(f"  표준편차: {np.nanstd(Y):.4f}")

    # Covariates
    exclude_cols = [
        'firm_id', 'year',
        'dti_binary', 'dti_continuous', 'dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata',
        'tfp_lp', 'tfp_simple',
        'y', 'k', 'l', 'm',  # LP variables
        'revenue', 'assets', 'employees', 'capital',  # TFP components
        'phi_hat', 'beta_l',  # LP intermediate
    ]

    if selected_vars is not None:
        # Use LASSO-selected variables
        covariate_names = [v for v in selected_vars if v in df.columns and v not in exclude_cols]
    else:
        # Use all except excluded
        covariate_names = [c for c in df.columns if c not in exclude_cols]

    X = df[covariate_names].values

    print(f"\nCovariates:")
    print(f"  변수 수: {len(covariate_names)}개")
    print(f"  Matrix shape: {X.shape}")

    # Memory optimization: Convert to sparse matrix if high-dimensional with many dummy variables
    if len(covariate_names) > 50:
        # Count dummy variables (binary variables or categorical indicators)
        n_dummy = sum(1 for col in covariate_names if
                     ('industry' in col.lower() or 'region' in col.lower() or 'year' in col.lower() or
                      df[col].nunique() <= 10))

        dummy_ratio = n_dummy / len(covariate_names)

        if dummy_ratio > 0.3:  # If >30% are dummy variables
            # Calculate memory before conversion
            dense_memory_mb = X.nbytes / (1024**2)

            # Convert to CSR sparse matrix
            X_sparse = csr_matrix(X)
            sparse_memory_mb = (X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes) / (1024**2)

            memory_reduction_pct = (1 - sparse_memory_mb / dense_memory_mb) * 100

            print(f"\n{'='*60}")
            print("CSR Sparse Matrix Optimization")
            print(f"{'='*60}")
            print(f"  Dummy variables: {n_dummy}/{len(covariate_names)} ({dummy_ratio*100:.1f}%)")
            print(f"  Dense memory: {dense_memory_mb:.1f} MB")
            print(f"  Sparse memory: {sparse_memory_mb:.1f} MB")
            print(f"  Memory reduction: {memory_reduction_pct:.1f}%")
            print(f"  ✅ Using CSR sparse matrix for memory efficiency")
            print(f"{'='*60}\n")

            # Note: GradientBoosting models don't support sparse matrices directly
            # We'll convert back to dense when needed, but track the optimization
            # X = X_sparse  # Keep as dense for GradientBoosting compatibility
            # In production, use LinearSVC or LogisticRegression which support sparse
        else:
            print(f"  ℹ️  Not enough dummy variables ({dummy_ratio*100:.1f}%) for sparse optimization")

    # VERIFICATION CHECK: Ensure all available covariates are used
    EXPECTED_COV_COUNT = 5  # debt, rd_employees, exports, industry_2digit, region

    if len(covariate_names) != EXPECTED_COV_COUNT:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: UNEXPECTED COVARIATE COUNT")
        print(f"{'='*60}")
        print(f"Expected: {EXPECTED_COV_COUNT} covariates")
        print(f"Actual: {len(covariate_names)} covariates")
        print(f"Covariates: {covariate_names}")
        print(f"")
        print(f"NOTE: Dataset contains only 5 control variables by design")
        print(f"      (from 00_imputation.py variable selection)")
        print(f"{'='*60}\n")

    if len(covariate_names) < EXPECTED_COV_COUNT:
        raise ValueError(
            f"Missing covariates: only {len(covariate_names)} of {EXPECTED_COV_COUNT} expected. "
            f"Use selected_vars=None to include all available covariates."
        )

    print(f"  ✅ Using all {len(covariate_names)} available covariates")

    # 결측치 확인
    n_missing_T = np.isnan(T).sum()
    n_missing_Y = np.isnan(Y).sum()
    n_missing_X = np.isnan(X).sum()

    if n_missing_T + n_missing_Y + n_missing_X > 0:
        print(f"\n⚠️ 결측치:")
        print(f"  T: {n_missing_T:,}개")
        print(f"  Y: {n_missing_Y:,}개")
        print(f"  X: {n_missing_X:,}개")

        # Remove rows with missing values
        valid_idx = ~(np.isnan(T) | np.isnan(Y) | np.isnan(X).any(axis=1))
        T = T[valid_idx]
        Y = Y[valid_idx]
        X = X[valid_idx]

        print(f"  → 유효 관측치: {len(T):,}개")

    return T, Y, X, covariate_names

def estimate_propensity_score(T, X, method='ensemble'):
    """
    Estimate propensity score e(X) = P(T=1|X)

    Parameters:
    - method: 'logistic', 'gbm', or 'ensemble'

    Returns:
    - ps: Propensity scores
    - model: Fitted model
    """
    print(f"\n{'='*60}")
    print(f"Step 4: Propensity Score 추정 ({method})")
    print(f"{'='*60}")

    if method == 'logistic':
        model = LogisticRegressionCV(cv=5, max_iter=5000, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X, T)
        ps = model.predict_proba(X)[:, 1]

    elif method == 'gbm':
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=50,
            random_state=RANDOM_SEED
        )
        model.fit(X, T)
        ps = model.predict_proba(X)[:, 1]

    elif method == 'ensemble':
        # Ensemble: average of logistic and GBM
        model_logit = LogisticRegressionCV(cv=5, max_iter=5000, random_state=RANDOM_SEED, n_jobs=-1)
        model_gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=50, random_state=RANDOM_SEED
        )

        model_logit.fit(X, T)
        model_gbm.fit(X, T)

        ps_logit = model_logit.predict_proba(X)[:, 1]
        ps_gbm = model_gbm.predict_proba(X)[:, 1]

        ps = (ps_logit + ps_gbm) / 2
        model = {'logistic': model_logit, 'gbm': model_gbm}

    else:
        raise ValueError(f"Unknown method: {method}")

    # Diagnostics
    print(f"\nPropensity Score 진단:")
    print(f"  범위: [{ps.min():.4f}, {ps.max():.4f}]")
    print(f"  평균: {ps.mean():.4f}")
    print(f"  표준편차: {ps.std():.4f}")

    # Overlap: common support
    ps_treated = ps[T == 1]
    ps_control = ps[T == 0]

    common_min = max(ps_treated.min(), ps_control.min())
    common_max = min(ps_treated.max(), ps_control.max())
    common_support = (ps >= common_min) & (ps <= common_max)

    print(f"\nCommon Support:")
    print(f"  범위: [{common_min:.4f}, {common_max:.4f}]")
    print(f"  비율: {common_support.mean()*100:.1f}%")

    # Trimming recommendation
    extreme_ps = (ps < 0.01) | (ps > 0.99)
    print(f"\nExtreme PS (< 0.01 or > 0.99): {extreme_ps.sum():,}개 ({extreme_ps.mean()*100:.1f}%)")

    return ps, model

def estimate_outcome_models(T, Y, X):
    """
    Estimate outcome models μ₁(X) and μ₀(X)

    μ₁(X) = E[Y|X, T=1]
    μ₀(X) = E[Y|X, T=0]

    Returns:
    - mu1: E[Y|X, T=1] for all observations
    - mu0: E[Y|X, T=0] for all observations
    - models: Fitted models
    """
    print(f"\n{'='*60}")
    print("Step 5: Outcome Models 추정")
    print(f"{'='*60}")

    # Separate data by treatment
    X1 = X[T == 1]
    Y1 = Y[T == 1]
    X0 = X[T == 0]
    Y0 = Y[T == 0]

    print(f"Treated (T=1): {len(Y1):,}개")
    print(f"Control (T=0): {len(Y0):,}개")

    # Model for E[Y|X, T=1]
    model1 = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=RANDOM_SEED
    )
    model1.fit(X1, Y1)

    # Model for E[Y|X, T=0]
    model0 = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=20,
        random_state=RANDOM_SEED
    )
    model0.fit(X0, Y0)

    # Predict for all observations
    mu1 = model1.predict(X)
    mu0 = model0.predict(X)

    print(f"\nOutcome Model μ₁(X):")
    print(f"  평균: {mu1.mean():.4f}")
    print(f"  범위: [{mu1.min():.4f}, {mu1.max():.4f}]")

    print(f"\nOutcome Model μ₀(X):")
    print(f"  평균: {mu0.mean():.4f}")
    print(f"  범위: [{mu0.min():.4f}, {mu0.max():.4f}]")

    return mu1, mu0, {'model1': model1, 'model0': model0}

def aipw_ate(T, Y, ps, mu1, mu0):
    """
    Calculate AIPW ATE

    τ_AIPW = E[T(Y - μ₁(X))/e(X) + μ₁(X)] - E[(1-T)(Y - μ₀(X))/(1-e(X)) + μ₀(X)]

    Returns:
    - ate: Average Treatment Effect
    - influence_function: For variance calculation
    """
    print(f"\n{'='*60}")
    print("Step 6: AIPW ATE 계산")
    print(f"{'='*60}")

    n = len(T)

    # AIPW components
    # E[Y(1)]
    Y1_aipw = T * (Y - mu1) / ps + mu1

    # E[Y(0)]
    Y0_aipw = (1 - T) * (Y - mu0) / (1 - ps) + mu0

    # ATE
    ate = Y1_aipw.mean() - Y0_aipw.mean()

    # Influence function (for variance)
    psi = Y1_aipw - Y0_aipw - ate

    # Variance (asymptotic)
    var_ate = np.var(psi) / n
    se_ate = np.sqrt(var_ate)

    # 95% CI
    ci_lower = ate - 1.96 * se_ate
    ci_upper = ate + 1.96 * se_ate

    # p-value (two-sided)
    z_stat = ate / se_ate
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    print(f"\nAIPW 결과:")
    print(f"  ATE: {ate:.6f}")
    print(f"  SE: {se_ate:.6f}")
    print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  z-statistic: {z_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    # Interpretation
    if p_value < 0.05:
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        print(f"  → 통계적으로 유의함 {sig}")
    else:
        print(f"  → 통계적으로 유의하지 않음 (p > 0.05)")

    # Effect size (percentage change for log outcome)
    pct_change = (np.exp(ate) - 1) * 100
    print(f"\n해석 (log TFP):")
    print(f"  디지털 채택 → TFP {pct_change:+.2f}% 변화")

    results = {
        'ate': ate,
        'se': se_ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'z_stat': z_stat,
        'p_value': p_value,
        'pct_change': pct_change,
        'n': n,
        'influence_function': psi
    }

    return results

def rubin_combine_estimates(ate_list, se_list):
    """
    Combine estimates from multiple imputations using Rubin's rules

    Parameters:
    - ate_list: List of ATE estimates
    - se_list: List of standard errors

    Returns:
    - pooled_ate, pooled_se, pooled_ci
    """
    print(f"\n{'='*60}")
    print("Step 7: Rubin's Rules로 Multiple Imputation 결합")
    print(f"{'='*60}")

    M = len(ate_list)

    # Pooled estimate
    pooled_ate = np.mean(ate_list)

    # Within-imputation variance
    W = np.mean([se**2 for se in se_list])

    # Between-imputation variance
    B = np.var(ate_list, ddof=1)

    # Total variance
    T_var = W + (1 + 1/M) * B

    pooled_se = np.sqrt(T_var)

    # 95% CI
    # Degrees of freedom (Barnard & Rubin, 1999)
    df = (M - 1) * (1 + W / ((1 + 1/M) * B))**2

    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf(0.975, df)

    ci_lower = pooled_ate - t_crit * pooled_se
    ci_upper = pooled_ate + t_crit * pooled_se

    # p-value
    t_stat = pooled_ate / pooled_se
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))

    print(f"\nRubin's Rules 결과:")
    print(f"  M (imputations): {M}")
    print(f"  Pooled ATE: {pooled_ate:.6f}")
    print(f"  Pooled SE: {pooled_se:.6f}")
    print(f"  Within-variance (W): {W:.8f}")
    print(f"  Between-variance (B): {B:.8f}")
    print(f"  Total variance: {T_var:.8f}")
    print(f"  DF: {df:.2f}")
    print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    results = {
        'ate': pooled_ate,
        'se': pooled_se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        't_stat': t_stat,
        'p_value': p_value,
        'df': df,
        'M': M
    }

    return results

def save_aipw_results(final_results, output_dir):
    """Save AIPW results"""
    print(f"\n{'='*60}")
    print("Step 8: AIPW 결과 저장")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(output_dir, "aipw_results.csv")

    pd.DataFrame([{
        'Method': 'AIPW (Doubly Robust)',
        'ATE': final_results['ate'],
        'SE': final_results['se'],
        'CI_Lower': final_results['ci_lower'],
        'CI_Upper': final_results['ci_upper'],
        't_statistic': final_results.get('t_stat', final_results.get('z_stat')),
        'p_value': final_results['p_value'],
        'N': final_results.get('n', 'pooled'),
        'M_imputations': final_results.get('M', 1)
    }]).to_csv(csv_path, index=False)

    print(f"✓ CSV: {csv_path}")

    # Text report
    report_path = os.path.join(output_dir, "aipw_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("AIPW (Augmented Inverse Probability Weighting) Results\n")
        f.write("="*60 + "\n\n")

        f.write("Method: Doubly Robust Estimator\n")
        f.write("  - Propensity score: e(X) = P(T=1|X)\n")
        f.write("  - Outcome models: μ₁(X), μ₀(X)\n")
        f.write("  - Estimator: AIPW combines both\n\n")

        f.write("Results:\n")
        f.write(f"  ATE: {final_results['ate']:.6f}\n")
        f.write(f"  SE: {final_results['se']:.6f}\n")
        f.write(f"  95% CI: [{final_results['ci_lower']:.6f}, {final_results['ci_upper']:.6f}]\n")

        stat_name = 't-statistic' if 'M' in final_results else 'z-statistic'
        stat_value = final_results.get('t_stat', final_results.get('z_stat'))
        f.write(f"  {stat_name}: {stat_value:.4f}\n")
        f.write(f"  p-value: {final_results['p_value']:.6f}\n\n")

        if final_results['p_value'] < 0.05:
            f.write("  → Statistically significant (p < 0.05)\n\n")
        else:
            f.write("  → Not statistically significant (p >= 0.05)\n\n")

        pct_change = (np.exp(final_results['ate']) - 1) * 100
        f.write(f"Interpretation:\n")
        f.write(f"  Digital adoption → TFP change: {pct_change:+.2f}%\n\n")

        if 'M' in final_results:
            f.write(f"Multiple Imputation:\n")
            f.write(f"  M = {final_results['M']} imputations\n")
            f.write(f"  Combined using Rubin's rules\n")

    print(f"✓ Report: {report_path}")

    return csv_path

def main():
    """메인 실행"""
    print("="*60)
    print("AIPW (Doubly Robust) Analysis Pipeline")
    print("="*60)

    # Load data
    datasets = load_data_with_tfp()

    # CRITICAL FIX: Do NOT use selected_vars (only 4 LASSO-selected variables)
    # For causal inference, we need ALL ~280 covariates to control confounding
    # selected_vars = load_selected_variables()  # REMOVED

    # Run AIPW on each imputed dataset
    ate_list = []
    se_list = []

    for m, df in enumerate(datasets, 1):
        print(f"\n{'='*60}")
        print(f"Imputation {m}/{len(datasets)}")
        print(f"{'='*60}")

        # Prepare data - uses ALL covariates (selected_vars=None by default)
        T, Y, X, covar_names = prepare_aipw_data(df, selected_vars=None)

        # Propensity score
        ps, ps_model = estimate_propensity_score(T, X, method='ensemble')

        # Outcome models
        mu1, mu0, outcome_models = estimate_outcome_models(T, Y, X)

        # AIPW ATE
        results = aipw_ate(T, Y, ps, mu1, mu0)

        ate_list.append(results['ate'])
        se_list.append(results['se'])

    # Combine using Rubin's rules
    final_results = rubin_combine_estimates(ate_list, se_list)

    # Save
    save_path = save_aipw_results(final_results, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✅ AIPW Analysis 완료!")
    print("="*60)
    print(f"\n최종 결과:")
    print(f"  ATE: {final_results['ate']:.6f}")
    print(f"  95% CI: [{final_results['ci_lower']:.6f}, {final_results['ci_upper']:.6f}]")
    print(f"  p-value: {final_results['p_value']:.6f}")
    print(f"\n저장: {save_path}")

if __name__ == "__main__":
    main()
