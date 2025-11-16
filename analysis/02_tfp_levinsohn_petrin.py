#!/usr/bin/env python3
"""
Phase 1-3: Levinsohn-Petrin TFP Estimation
- 해결: 단순 Cobb-Douglas → Levinsohn-Petrin 방법
- 방법: 중간재를 proxy로 사용하여 생산성 충격 식별
- 산출: LP 방법으로 추정된 TFP
"""

import pandas as pd
import numpy as np
import os
import pickle
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 설정
INPUT_DIR = "analysis/processed_data"
OUTPUT_DIR = "analysis/processed_data"

def load_imputed_datasets():
    """Load all imputed datasets"""
    print(f"\n{'='*60}")
    print("Step 1: Imputed Datasets 로드")
    print(f"{'='*60}")

    pkl_path = os.path.join(INPUT_DIR, "imputed_datasets.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        imputed_datasets = pickle.load(f)

    print(f"✅ {len(imputed_datasets)}개 imputed datasets 로드")
    print(f"  각 dataset: {len(imputed_datasets[0]):,}개 행")

    return imputed_datasets

def prepare_lp_variables(df):
    """
    Prepare variables for Levinsohn-Petrin estimation

    Required:
    - Output (Y): revenue
    - Capital (K): assets or capital
    - Labor (L): employees
    - Intermediates (M): materials + energy + outsourcing

    Returns:
    - df with log-transformed variables
    """
    print(f"\n{'='*60}")
    print("Step 2: LP 변수 준비")
    print(f"{'='*60}")

    # Output
    if 'revenue' in df.columns:
        df['y'] = np.log(df['revenue'].clip(lower=1))
        print(f"✓ Output (y): revenue → log 변환")
    else:
        raise ValueError("'revenue' column not found")

    # Capital (자본)
    if 'capital' in df.columns:
        df['k'] = np.log(df['capital'].clip(lower=1))
        print(f"✓ Capital (k): capital → log 변환")
    elif 'assets' in df.columns:
        df['k'] = np.log(df['assets'].clip(lower=1))
        print(f"✓ Capital (k): assets → log 변환 (대체)")
    else:
        raise ValueError("'capital' or 'assets' column not found")

    # Labor (노동)
    if 'employees' in df.columns:
        df['l'] = np.log(df['employees'].clip(lower=1))
        print(f"✓ Labor (l): employees → log 변환")
    else:
        raise ValueError("'employees' column not found")

    # Intermediates (중간재)
    # M = materials + energy + outsourcing
    intermediate_cols = []

    if 'materials' in df.columns:
        intermediate_cols.append('materials')
    if 'energy' in df.columns:
        intermediate_cols.append('energy')
    if 'outsourcing' in df.columns:
        intermediate_cols.append('outsourcing')

    if len(intermediate_cols) == 0:
        print("⚠️ 중간재 변수 없음 - 단순 TFP로 대체")
        df['m'] = df['l']  # Fallback: use labor
    else:
        # 중간재 합계
        df['m_total'] = df[intermediate_cols].sum(axis=1).clip(lower=1)
        df['m'] = np.log(df['m_total'])
        print(f"✓ Intermediates (m): {intermediate_cols} → log 변환")

    # 결측치 확인
    required_vars = ['y', 'k', 'l', 'm']
    for var in required_vars:
        n_missing = df[var].isna().sum()
        if n_missing > 0:
            print(f"  ⚠️ {var}: {n_missing:,}개 결측")

    return df

def levinsohn_petrin_stage1(df):
    """
    Levinsohn-Petrin Stage 1: Estimate productivity proxy

    Regression:
    y = β_l * l + φ(k, m)

    where φ(k, m) is approximated by polynomial of k and m

    Returns:
    - df with phi_hat (productivity proxy)
    - beta_l (labor coefficient)
    """
    print(f"\n{'='*60}")
    print("Step 3: LP Stage 1 - Productivity Proxy")
    print(f"{'='*60}")

    # 결측치 제거
    df_clean = df[['y', 'k', 'l', 'm']].dropna()
    print(f"유효 관측치: {len(df_clean):,}개")

    # Polynomial features for φ(k, m)
    # Use 3rd order polynomial
    km = df_clean[['k', 'm']].values

    poly = PolynomialFeatures(degree=3, include_bias=False)
    phi_features = poly.fit_transform(km)

    print(f"Polynomial features: {phi_features.shape[1]}개")

    # Regression: y = β_l * l + φ(k, m)
    # φ(k, m) ≈ γ₀ + γ₁*k + γ₂*m + γ₃*k² + γ₄*m² + γ₅*km + ...

    X = np.column_stack([
        df_clean['l'].values,  # Labor
        phi_features            # φ(k, m) polynomial
    ])

    y = df_clean['y'].values

    # OLS
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # Extract β_l (labor coefficient)
    beta_l = model.params[1]  # First variable after constant

    print(f"\nStage 1 결과:")
    print(f"  β_l (labor coef): {beta_l:.4f}")
    print(f"  R²: {model.rsquared:.4f}")

    # Predicted φ(k, m)
    phi_hat = model.fittedvalues - beta_l * df_clean['l'].values

    # Add to DataFrame
    df.loc[df_clean.index, 'phi_hat'] = phi_hat
    df.loc[df_clean.index, 'beta_l'] = beta_l

    return df, beta_l, model

def levinsohn_petrin_stage2(df, beta_l):
    """
    Levinsohn-Petrin Stage 2: Estimate capital coefficient

    Using φ_hat from Stage 1, estimate β_k

    Moment condition:
    E[ξ_t * k_t] = 0

    where ξ_t = y_t - β_l*l_t - β_k*k_t - ω_t

    Returns:
    - beta_k (capital coefficient)
    - TFP estimates
    """
    print(f"\n{'='*60}")
    print("Step 4: LP Stage 2 - Capital Coefficient")
    print(f"{'='*60}")

    # 결측치 제거
    df_clean = df[['y', 'k', 'l', 'phi_hat']].dropna()

    # Objective: Minimize squared residuals
    # ξ = y - β_l*l - β_k*k - ω

    def objective(beta_k):
        """Objective function for β_k estimation"""
        residuals = (df_clean['y']
                    - beta_l * df_clean['l']
                    - beta_k * df_clean['k']
                    - df_clean['phi_hat'])

        return np.sum(residuals ** 2)

    # Initial guess
    beta_k_init = 0.3  # Standard Cobb-Douglas share

    # Minimize
    result = minimize(
        objective,
        x0=[beta_k_init],
        method='L-BFGS-B',
        bounds=[(0.01, 0.99)]  # β_k should be positive and < 1
    )

    beta_k = result.x[0]

    print(f"\nStage 2 결과:")
    print(f"  β_k (capital coef): {beta_k:.4f}")
    print(f"  Optimization success: {result.success}")

    # TFP = y - β_l*l - β_k*k
    df['tfp_lp'] = df['y'] - beta_l * df['l'] - beta_k * df['k']

    # 통계
    tfp_mean = df['tfp_lp'].mean()
    tfp_std = df['tfp_lp'].std()

    print(f"\nTFP (LP) 통계:")
    print(f"  평균: {tfp_mean:.4f}")
    print(f"  표준편차: {tfp_std:.4f}")
    print(f"  유효 관측치: {df['tfp_lp'].notna().sum():,}개")

    return df, beta_k

def compare_tfp_methods(df):
    """
    Compare TFP from different methods

    - TFP_simple: Cobb-Douglas with fixed α, β
    - TFP_LP: Levinsohn-Petrin
    """
    print(f"\n{'='*60}")
    print("Step 5: TFP 방법 비교")
    print(f"{'='*60}")

    # Simple TFP (기존)
    ALPHA = 0.3
    BETA = 0.7

    df['tfp_simple'] = df['y'] - ALPHA * df['k'] - BETA * df['l']

    # 상관관계
    corr = df[['tfp_simple', 'tfp_lp']].corr().iloc[0, 1]

    print(f"\nTFP 비교:")
    print(f"  TFP_simple 평균: {df['tfp_simple'].mean():.4f}")
    print(f"  TFP_LP 평균: {df['tfp_lp'].mean():.4f}")
    print(f"  상관계수: {corr:.4f}")

    # Scatter plot (optional, saved to file)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.scatter(df['tfp_simple'], df['tfp_lp'], alpha=0.3, s=1)
        plt.xlabel('TFP (Simple Cobb-Douglas)')
        plt.ylabel('TFP (Levinsohn-Petrin)')
        plt.title(f'TFP Comparison (corr={corr:.3f})')
        plt.plot([-2, 4], [-2, 4], 'r--', alpha=0.5)  # 45-degree line

        plot_path = os.path.join(OUTPUT_DIR, 'tfp_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Scatter plot: {plot_path}")
        plt.close()
    except:
        print("  ⚠️ Matplotlib 사용 불가")

    return df

def save_tfp_datasets(imputed_datasets_with_tfp, output_dir):
    """Save datasets with TFP estimates"""
    print(f"\n{'='*60}")
    print("Step 6: TFP Datasets 저장")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Pickle
    pkl_path = os.path.join(output_dir, "imputed_datasets_with_tfp.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(imputed_datasets_with_tfp, f)

    pkl_size = os.path.getsize(pkl_path) / (1024**2)
    print(f"✓ Pickle: {pkl_path} ({pkl_size:.1f} MB)")

    # 개별 CSV (첫 번째만)
    csv_path = os.path.join(output_dir, "imputed_m1_with_tfp.csv")
    imputed_datasets_with_tfp[0].to_csv(csv_path, index=False)

    csv_size = os.path.getsize(csv_path) / (1024**2)
    print(f"✓ CSV (m=1): {csv_path} ({csv_size:.1f} MB)")

    # Summary
    summary_path = os.path.join(output_dir, "tfp_estimation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Levinsohn-Petrin TFP Estimation Summary\n")
        f.write("="*60 + "\n\n")

        df_sample = imputed_datasets_with_tfp[0]

        f.write("1. 방법: Levinsohn-Petrin (2003)\n")
        f.write("   - Stage 1: φ(k, m) proxy with 3rd-order polynomial\n")
        f.write("   - Stage 2: β_k optimization with moment conditions\n\n")

        f.write("2. 변수:\n")
        f.write("   - Output (y): log(revenue)\n")
        f.write("   - Capital (k): log(assets or capital)\n")
        f.write("   - Labor (l): log(employees)\n")
        f.write("   - Intermediates (m): log(materials + energy + outsourcing)\n\n")

        f.write("3. 추정 계수:\n")
        if 'beta_l' in df_sample.columns:
            f.write(f"   - β_l (labor): {df_sample['beta_l'].iloc[0]:.4f}\n")
        if 'tfp_lp' in df_sample.columns:
            f.write(f"   - β_k (capital): (from Stage 2)\n")

        f.write("\n4. TFP 통계:\n")
        f.write(f"   - 평균: {df_sample['tfp_lp'].mean():.4f}\n")
        f.write(f"   - 표준편차: {df_sample['tfp_lp'].std():.4f}\n")
        f.write(f"   - 유효 관측치: {df_sample['tfp_lp'].notna().sum():,}\n")

        f.write("\n5. Imputed datasets: {len(imputed_datasets_with_tfp)}\n")

    print(f"✓ Summary: {summary_path}")

    return pkl_path

def main():
    """메인 실행"""
    print("="*60)
    print("Levinsohn-Petrin TFP Estimation Pipeline")
    print("="*60)

    # Step 1: Load imputed datasets
    imputed_datasets = load_imputed_datasets()

    # Process each imputed dataset
    imputed_datasets_with_tfp = []

    for m, df in enumerate(imputed_datasets, 1):
        print(f"\n{'='*60}")
        print(f"Processing Imputation {m}/{len(imputed_datasets)}")
        print(f"{'='*60}")

        # Step 2: Prepare LP variables
        df = prepare_lp_variables(df)

        # Step 3: LP Stage 1
        df, beta_l, model_stage1 = levinsohn_petrin_stage1(df)

        # Step 4: LP Stage 2
        df, beta_k = levinsohn_petrin_stage2(df, beta_l)

        # Step 5: Compare methods
        df = compare_tfp_methods(df)

        imputed_datasets_with_tfp.append(df)

    # Step 6: Save
    save_path = save_tfp_datasets(imputed_datasets_with_tfp, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✅ Levinsohn-Petrin TFP Estimation 완료!")
    print("="*60)
    print(f"\n산출물:")
    print(f"  - {len(imputed_datasets_with_tfp)}개 datasets with TFP_LP")
    print(f"  - 저장: {save_path}")
    print(f"\nPhase 1 완료!")
    print(f"다음 단계: Phase 2 (Doubly Robust Analysis)")
    print(f"  → python analysis/03_propensity_score.py")

if __name__ == "__main__":
    main()
