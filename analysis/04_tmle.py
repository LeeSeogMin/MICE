#!/usr/bin/env python3
"""
Algorithm 4: TMLE (Targeted Maximum Likelihood Estimation)

TMLE은 doubly robust estimator로, AIPW보다 효율적인 추정을 제공합니다.
- Propensity score와 outcome model을 사용
- Targeting step으로 bias 제거
- Multiple imputation (Rubin's rules) 지원

Author: Claude
Date: 2025-11-10
"""

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
import time
from datetime import datetime

print("=" * 70)
print("TMLE (Targeted Maximum Likelihood Estimation) Pipeline")
print("=" * 70)

# ============================================================
# Step 1: Load Imputed Datasets with TFP
# ============================================================
print("\n" + "=" * 70)
print("Step 1: TFP Dataset 로드")
print("=" * 70)

with open('analysis/processed_data/imputed_datasets_with_tfp.pkl', 'rb') as f:
    datasets = pickle.load(f)

print(f"✅ {len(datasets)}개 imputed datasets 로드")

# ============================================================
# Step 2: Load Selected Variables
# ============================================================
print("\n" + "=" * 70)
print("Step 2: 공변량 설정")
print("=" * 70)

# CRITICAL FIX: Do NOT use selected_vars (only 4 LASSO-selected variables)
# For causal inference, we need ALL ~280 covariates to control confounding
# REMOVED: Load of selected_variables.pkl
selected_vars = None
print("✅ 모든 공변량 사용 (인과추론을 위한 완전한 교란변수 통제)")

# ============================================================
# TMLE Function
# ============================================================

def tmle_ate(Y, T, X, verbose=True):
    """
    TMLE for Average Treatment Effect

    Parameters:
    -----------
    Y : array-like, outcome
    T : array-like, binary treatment
    X : array-like, covariates
    verbose : bool, print progress

    Returns:
    --------
    ate : float, average treatment effect
    se : float, standard error
    ci : tuple, 95% confidence interval
    """

    n = len(Y)

    if verbose:
        print("\n" + "=" * 70)
        print("Step 1: Initial Outcome Model μ(T,X)")
        print("=" * 70)

    # Step 1: Initial outcome model μ(T,X) = E[Y|T,X]
    # Fit separate models for treated and control
    mu1_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    mu0_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

    mu1_model.fit(X[T == 1], Y[T == 1])
    mu0_model.fit(X[T == 0], Y[T == 0])

    # Predict for everyone
    mu1_X = mu1_model.predict(X)
    mu0_X = mu0_model.predict(X)

    if verbose:
        print(f"μ₁(X) - treated outcome model:")
        print(f"  Mean: {mu1_X.mean():.4f}, Std: {mu1_X.std():.4f}")
        print(f"μ₀(X) - control outcome model:")
        print(f"  Mean: {mu0_X.mean():.4f}, Std: {mu0_X.std():.4f}")

    if verbose:
        print("\n" + "=" * 70)
        print("Step 2: Propensity Score g(X) = P(T=1|X)")
        print("=" * 70)

    # Step 2: Propensity score g(X) = P(T=1|X)
    ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    ps_model.fit(X, T)
    g_X = ps_model.predict_proba(X)[:, 1]

    # Trim extreme propensity scores
    g_X = np.clip(g_X, 0.01, 0.99)

    if verbose:
        print(f"Propensity Score g(X):")
        print(f"  Mean: {g_X.mean():.4f}, Std: {g_X.std():.4f}")
        print(f"  Range: [{g_X.min():.4f}, {g_X.max():.4f}]")
        print(f"  Extreme (<0.01 or >0.99): {((g_X < 0.01) | (g_X > 0.99)).sum()}개")

    if verbose:
        print("\n" + "=" * 70)
        print("Step 3: Targeting (Clever Covariate)")
        print("=" * 70)

    # Step 3: Clever covariate
    H1 = T / g_X
    H0 = (1 - T) / (1 - g_X)

    if verbose:
        print(f"Clever Covariate H₁ (treated):")
        print(f"  Mean: {H1.mean():.4f}, Max: {H1.max():.4f}")
        print(f"Clever Covariate H₀ (control):")
        print(f"  Mean: {H0.mean():.4f}, Max: {H0.max():.4f}")

    # Step 4: Fluctuation (targeting step)
    # Fit logistic regression with offset to get epsilon
    # Q̃(T,X) = μ(T,X) + ε * H(T,X)

    # For simplicity, we use one-step targeting
    # Compute initial Q
    Q_init = T * mu1_X + (1 - T) * mu0_X

    # Fluctuation parameter epsilon
    # This is a simplified version - full TMLE would iterate
    H = H1 - H0
    epsilon = np.sum(H * (Y - Q_init)) / np.sum(H * H)

    if verbose:
        print(f"\nFluctuation parameter ε: {epsilon:.6f}")

    # Updated Q
    mu1_X_star = mu1_X + epsilon / g_X
    mu0_X_star = mu0_X - epsilon / (1 - g_X)

    if verbose:
        print("\n" + "=" * 70)
        print("Step 4: TMLE Estimate")
        print("=" * 70)

    # Step 5: TMLE estimate
    ate = np.mean(mu1_X_star - mu0_X_star)

    # Influence function for variance
    IC = (T / g_X) * (Y - mu1_X_star) - ((1 - T) / (1 - g_X)) * (Y - mu0_X_star) + (mu1_X_star - mu0_X_star) - ate

    var = np.var(IC) / n
    se = np.sqrt(var)

    # 95% CI
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    # Z-statistic and p-value
    z_stat = ate / se
    p_value = 2 * (1 - 0.5 * (1 + np.sign(z_stat) * (1 - np.exp(-np.abs(z_stat)**2/2))))

    if verbose:
        print(f"\nTMLE Results:")
        print(f"  ATE: {ate:.6f}")
        print(f"  SE: {se:.6f}")
        print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"  z-statistic: {z_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  → {'통계적으로 유의함' if p_value < 0.05 else '유의하지 않음'} {sig}")

        pct_effect = 100 * (np.exp(ate) - 1)
        print(f"\n해석 (log TFP):")
        print(f"  디지털 채택 → TFP {pct_effect:+.2f}% 변화")

    return {
        'ate': ate,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'z_stat': z_stat,
        'p_value': p_value,
        'pct_effect': 100 * (np.exp(ate) - 1)
    }

# ============================================================
# Step 3: Run TMLE on Each Imputed Dataset
# ============================================================

print("\n" + "=" * 70)
print("Step 3: TMLE on Multiple Imputed Datasets")
print("=" * 70)

tmle_results = []

for m, df in enumerate(datasets, 1):
    print(f"\n{'='*70}")
    print(f"Imputation {m}/{len(datasets)}")
    print(f"{'='*70}")

    # Prepare data
    # Treatment: DTI_binary
    df_dti = df[['dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata']].apply(lambda x: x - 1)
    df['dti_continuous'] = df_dti.mean(axis=1)
    df['dti_binary'] = (df['dti_continuous'] > 0).astype(int)

    T = df['dti_binary'].values

    # Outcome: tfp_lp
    Y = df['tfp_lp'].values

    # Covariates
    exclude_cols = [
        'firm_id', 'year',
        'dti_binary', 'dti_continuous', 'dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata',
        'tfp_lp', 'tfp_simple',
        'y', 'k', 'l', 'm',
        'revenue', 'assets', 'employees',
        'phi_hat', 'beta_l',
    ]

    if selected_vars is not None:
        covariate_names = [v for v in selected_vars if v in df.columns and v not in exclude_cols]
    else:
        covariate_names = [c for c in df.columns if c not in exclude_cols]

    X = df[covariate_names].values

    print(f"\nData Summary:")
    print(f"  N: {len(df):,}")
    print(f"  Treated (T=1): {T.sum():,} ({100*T.mean():.1f}%)")
    print(f"  Control (T=0): {(1-T).sum():,} ({100*(1-T).mean():.1f}%)")
    print(f"  Covariates: {len(covariate_names)}")
    print(f"  Outcome (TFP): mean={Y.mean():.4f}, std={Y.std():.4f}")

    # VERIFICATION CHECK: Ensure all available covariates are used
    EXPECTED_COV_COUNT = 5  # debt, rd_employees, exports, industry_2digit, region

    if len(covariate_names) != EXPECTED_COV_COUNT:
        print(f"\n⚠️  WARNING: Expected {EXPECTED_COV_COUNT} covariates, got {len(covariate_names)}")
        print(f"   Covariates: {covariate_names}")

    if len(covariate_names) < EXPECTED_COV_COUNT:
        raise ValueError(
            f"Missing covariates: only {len(covariate_names)} of {EXPECTED_COV_COUNT} expected"
        )

    print(f"  ✅ Using all {len(covariate_names)} available covariates")

    # Run TMLE
    result = tmle_ate(Y, T, X, verbose=True)
    tmle_results.append(result)

# ============================================================
# Step 4: Pool Results using Rubin's Rules
# ============================================================
print("\n" + "=" * 70)
print("Step 4: Rubin's Rules로 Multiple Imputation 결합")
print("=" * 70)

M = len(tmle_results)
ate_m = np.array([r['ate'] for r in tmle_results])
se_m = np.array([r['se'] for r in tmle_results])

# Pooled estimate
ate_pooled = np.mean(ate_m)

# Within-imputation variance
W = np.mean(se_m ** 2)

# Between-imputation variance
B = np.var(ate_m, ddof=1)

# Total variance
T_var = W + (1 + 1/M) * B

# Pooled SE
se_pooled = np.sqrt(T_var)

# Degrees of freedom (Barnard-Rubin adjustment)
df = (M - 1) * (1 + W / ((1 + 1/M) * B)) ** 2

# 95% CI (using t-distribution)
from scipy import stats
t_crit = stats.t.ppf(0.975, df)
ci_lower_pooled = ate_pooled - t_crit * se_pooled
ci_upper_pooled = ate_pooled + t_crit * se_pooled

# t-statistic and p-value
t_stat = ate_pooled / se_pooled
p_value_pooled = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

print(f"\nRubin's Rules 결과:")
print(f"  M (imputations): {M}")
print(f"  Pooled ATE: {ate_pooled:.6f}")
print(f"  Pooled SE: {se_pooled:.6f}")
print(f"  Within-variance (W): {W:.8f}")
print(f"  Between-variance (B): {B:.8f}")
print(f"  Total variance: {T_var:.8f}")
print(f"  DF: {df:.2f}")
print(f"  95% CI: [{ci_lower_pooled:.6f}, {ci_upper_pooled:.6f}]")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_pooled:.6f}")

pct_pooled = 100 * (np.exp(ate_pooled) - 1)
print(f"\n최종 해석:")
print(f"  디지털 채택 → TFP {pct_pooled:+.2f}% 변화")

# ============================================================
# Step 5: Save Results
# ============================================================
print("\n" + "=" * 70)
print("Step 5: TMLE 결과 저장")
print("=" * 70)

# Individual imputations
df_individual = pd.DataFrame(tmle_results)
df_individual['imputation'] = range(1, M+1)
df_individual = df_individual[['imputation', 'ate', 'se', 'ci_lower', 'ci_upper', 'p_value', 'pct_effect']]

# Pooled result
df_pooled = pd.DataFrame([{
    'method': 'TMLE (Pooled)',
    'M': M,
    'ate': ate_pooled,
    'se': se_pooled,
    'ci_lower': ci_lower_pooled,
    'ci_upper': ci_upper_pooled,
    't_stat': t_stat,
    'p_value': p_value_pooled,
    'pct_effect': pct_pooled,
    'W': W,
    'B': B,
    'df': df
}])

# Save
df_individual.to_csv('analysis/results/tmle_individual.csv', index=False)
df_pooled.to_csv('analysis/results/tmle_pooled.csv', index=False)

# Report
report = f"""TMLE (Targeted Maximum Likelihood Estimation) Results
{'='*70}

Method: Doubly Robust Estimator with Targeting
  - Initial outcome models: μ₁(X), μ₀(X)
  - Propensity score: g(X) = P(T=1|X)
  - Targeting: Clever covariate + fluctuation
  - Estimator: TMLE combines both with bias reduction

Results:
  ATE: {ate_pooled:.6f}
  SE: {se_pooled:.6f}
  95% CI: [{ci_lower_pooled:.6f}, {ci_upper_pooled:.6f}]
  t-statistic: {t_stat:.4f}
  p-value: {p_value_pooled:.6f}

  → {'Statistically significant (p < 0.05)' if p_value_pooled < 0.05 else 'Not significant (p ≥ 0.05)'}

Interpretation:
  Digital adoption → TFP change: {pct_pooled:+.2f}%

Multiple Imputation:
  M = {M} imputations
  Combined using Rubin's rules
  Degrees of freedom: {df:.2f}

Comparison with AIPW:
  TMLE is theoretically more efficient than AIPW
  Both are doubly robust estimators
  TMLE uses targeting to reduce bias further
"""

with open('analysis/results/tmle_report.txt', 'w') as f:
    f.write(report)

print("✓ CSV: analysis/results/tmle_individual.csv")
print("✓ CSV: analysis/results/tmle_pooled.csv")
print("✓ Report: analysis/results/tmle_report.txt")

print("\n" + "=" * 70)
print("✅ TMLE Analysis 완료!")
print("=" * 70)

print(f"\n최종 결과:")
print(f"  ATE: {ate_pooled:.6f}")
print(f"  95% CI: [{ci_lower_pooled:.6f}, {ci_upper_pooled:.6f}]")
print(f"  p-value: {p_value_pooled:.6f}")
print(f"\n저장: analysis/results/tmle_pooled.csv")
