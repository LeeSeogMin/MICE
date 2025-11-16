#!/usr/bin/env python3
"""
Execute additional analyses using actual data:
1. Dose-response (DTI quartiles) - if variation exists
2. COVID period effects
3. Industry heterogeneity

FIXED VERSION: Uses ALL covariates (281) for proper confounder control
Previous version used only 4 covariates, causing severe omitted variable bias
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

print("="*80)
print("LOADING DATA")
print("="*80)

# Load p101 imputed datasets (113 variables including engineered features)
print("Loading p101 full dataset with 113 variables...")
with open('analysis/processed_data/p101_full/imputed_datasets_with_tfp.pkl', 'rb') as f:
    datasets = pickle.load(f)

df = datasets[0].copy()  # Use first imputation
print(f"N = {len(df):,} observations")
print(f"Total variables: {len(df.columns)}")

# Check columns
print(f"\nColumns: {list(df.columns)[:15]}...")

# Create DTI variables (already recoded: 0 = non-adopter, 2-6 = adopter stages)
df['DTI'] = df[['dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata']].mean(axis=1)
df['DTI_binary'] = (df['DTI'] > 0).astype(int)

# Prepare covariates (use ALL available variables for proper confounder control)
# CRITICAL: Previous version used only 4 LASSO-selected variables
# This caused severe omitted variable bias (sign reversal in pooled vs stratified)
exclude_cols = [
    # Identifiers
    'firm_id', 'year',

    # Treatment variables (DTI)
    'dti_binary', 'DTI_binary', 'dti_continuous', 'DTI',
    'dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata',

    # Outcome variables (TFP)
    'tfp_lp', 'tfp_simple',

    # TFP estimation inputs (should not be used as covariates to avoid multicollinearity)
    'revenue', 'employees', 'capital',  # Used to construct TFP

    # Variables that will be created during analysis
    'period',  # Created for COVID analysis
    'DTI_quartile'  # Created for dose-response
]

# Use ALL columns except excluded ones
covariate_names = [c for c in df.columns if c not in exclude_cols and not c.startswith('_')]

print(f"\n🔧 CRITICAL FIX: Using {len(covariate_names)} covariates")
print(f"   (Previous version incorrectly used only 4 LASSO-selected variables)")
print(f"   Covariate names (first 15): {covariate_names[:15]}...")

# Create full covariate matrix
X_full = df[covariate_names].values
print(f"   Covariate matrix shape: {X_full.shape}")

print(f"\nDTI statistics:")
print(f"  Mean: {df['DTI'].mean():.3f}")
print(f"  Std: {df['DTI'].std():.3f}")
print(f"  Min: {df['DTI'].min():.3f}")
print(f"  Max: {df['DTI'].max():.3f}")
print(f"  Adoption rate: {df['DTI_binary'].mean()*100:.1f}%")

# Check if enough variation for dose-response
print(f"\nDTI percentiles:")
for p in [0, 25, 50, 75, 90, 95, 100]:
    print(f"  {p:3d}%: {df['DTI'].quantile(p/100):.3f}")

# =================================================================
# 1. DOSE-RESPONSE ANALYSIS (if variation exists)
# =================================================================
print("\n" + "="*80)
print("1. DOSE-RESPONSE ANALYSIS")
print("="*80)

# Check if we have enough variation for quartiles
if df['DTI'].std() > 0.5:  # Reasonable variation
    # Create quartiles among adopters
    df_adopters = df[df['DTI_binary'] == 1].copy()

    try:
        df_adopters['DTI_quartile'] = pd.qcut(df_adopters['DTI'], q=4,
                                                labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'],
                                                duplicates='drop')
        print("\n✓ DTI Quartiles created among adopters")
        print(df_adopters['DTI_quartile'].value_counts().sort_index())

        # Run AIPW for each quartile vs non-adopters
        dose_response_results = []

        y_all = df['tfp_lp'].values

        # Baseline: non-adopters
        non_adopter_tfp = df[df['DTI_binary'] == 0]['tfp_lp'].mean()

        for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
            # Define treatment: this quartile vs non-adopters
            T_dose = ((df_adopters['DTI_quartile'] == q).reindex(df.index, fill_value=False)).astype(int).values

            n_treated = T_dose.sum()
            if n_treated < 100:
                continue

            # AIPW estimation with FULL covariate set
            ps_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
            ps_model.fit(X_full, T_dose)
            ps = ps_model.predict_proba(X_full)[:, 1]
            ps = np.clip(ps, 0.01, 0.99)

            mu1_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            mu0_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

            mu1_model.fit(X_full[T_dose == 1], y_all[T_dose == 1])
            mu0_model.fit(X_full[T_dose == 0], y_all[T_dose == 0])

            mu1 = mu1_model.predict(X_full)
            mu0 = mu0_model.predict(X_full)

            aipw_scores = T_dose * (y_all - mu1) / ps - (1 - T_dose) * (y_all - mu0) / (1 - ps) + mu1 - mu0

            ate = np.mean(aipw_scores)
            se = np.std(aipw_scores) / np.sqrt(len(aipw_scores))
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se
            p_value = 2 * (1 - 0.5 * (1 + np.sign(ate/se) * (1 - np.exp(-abs(ate/se)**2/2))))

            dose_response_results.append({
                'Quartile': q,
                'N': n_treated,
                'ATE': ate,
                'SE': se,
                'CI_lower': ci_lower,
                'CI_upper': ci_upper,
                'p_value': p_value,
                'Effect_pct': 100 * (np.exp(ate) - 1)
            })

        df_dose = pd.DataFrame(dose_response_results)
        df_dose.to_csv('analysis/results/dose_response_analysis.csv', index=False)
        print("\n✅ Dose-response results:")
        print(df_dose)

    except Exception as e:
        print(f"\n❌ Dose-response analysis failed: {e}")
        print("   Insufficient variation in DTI for quartile analysis")
else:
    print("\n❌ Dose-response analysis NOT FEASIBLE")
    print(f"   DTI std = {df['DTI'].std():.3f} (too low variation)")

# =================================================================
# 2. COVID PERIOD ANALYSIS
# =================================================================
print("\n" + "="*80)
print("2. COVID PERIOD ANALYSIS")
print("="*80)

if 'year' in df.columns:
    # Create periods
    df['period'] = 'Pre-COVID (2019)'
    df.loc[df['year'].isin([2020, 2021]), 'period'] = 'During-COVID (2020-2021)'
    df.loc[df['year'].isin([2022, 2023]), 'period'] = 'Post-COVID (2022-2023)'

    print("\nPeriod distribution:")
    print(df['period'].value_counts())

    covid_results = []

    for period in ['Pre-COVID (2019)', 'During-COVID (2020-2021)', 'Post-COVID (2022-2023)']:
        mask = df['period'] == period

        if mask.sum() < 500:
            print(f"\n⚠️  {period}: Too few observations ({mask.sum()})")
            continue

        print(f"\n--- {period} ---")

        y_sub = df.loc[mask, 'tfp_lp'].values
        T_sub = df.loc[mask, 'DTI_binary'].values
        X_sub = df.loc[mask, covariate_names].values  # Use FULL covariate set

        n = len(y_sub)
        n_treated = T_sub.sum()

        print(f"N = {n:,}, Treated = {n_treated:,} ({100*n_treated/n:.1f}%)")

        # AIPW with FULL covariates
        ps_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        ps_model.fit(X_sub, T_sub)
        ps = ps_model.predict_proba(X_sub)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)

        mu1_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        mu0_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

        mu1_model.fit(X_sub[T_sub == 1], y_sub[T_sub == 1])
        mu0_model.fit(X_sub[T_sub == 0], y_sub[T_sub == 0])

        mu1 = mu1_model.predict(X_sub)
        mu0 = mu0_model.predict(X_sub)

        aipw_scores = T_sub * (y_sub - mu1) / ps - (1 - T_sub) * (y_sub - mu0) / (1 - ps) + mu1 - mu0

        ate = np.mean(aipw_scores)
        se = np.std(aipw_scores) / np.sqrt(n)
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        p_value = 2 * (1 - 0.5 * (1 + np.sign(ate/se) * (1 - np.exp(-abs(ate/se)**2/2))))

        print(f"ATE = {ate:.6f}, SE = {se:.6f}, p = {p_value:.4f}")
        print(f"Effect = {100 * (np.exp(ate) - 1):+.2f}%")

        covid_results.append({
            'Period': period,
            'N': n,
            'N_treated': n_treated,
            'Treated_pct': 100*n_treated/n,
            'ATE': ate,
            'SE': se,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value,
            'Effect_pct': 100 * (np.exp(ate) - 1)
        })

    df_covid = pd.DataFrame(covid_results)
    df_covid.to_csv('analysis/results/covid_period_analysis.csv', index=False)
    print("\n✅ COVID period results:")
    print(df_covid)

else:
    print("\n❌ COVID period analysis NOT FEASIBLE (no year variable)")

# =================================================================
# 3. INDUSTRY HETEROGENEITY
# =================================================================
print("\n" + "="*80)
print("3. INDUSTRY HETEROGENEITY")
print("="*80)

if 'industry_2digit' in df.columns:
    print(f"\nTotal industries: {df['industry_2digit'].nunique()}")

    # Find industries with sufficient N
    industry_counts = df['industry_2digit'].value_counts()
    large_industries = industry_counts[industry_counts >= 500]

    print(f"Industries with N >= 500: {len(large_industries)}")
    print(large_industries.head(10))

    # Run AIPW for top 5 industries
    industry_results = []

    for industry in large_industries.head(5).index:
        mask = df['industry_2digit'] == industry

        print(f"\n--- Industry {industry:.0f} ---")

        y_sub = df.loc[mask, 'tfp_lp'].values
        T_sub = df.loc[mask, 'DTI_binary'].values
        # Use all covariates except industry_2digit itself (it's constant in subgroup)
        ind_covariates = [c for c in covariate_names if c != 'industry_2digit']
        X_sub = df.loc[mask, ind_covariates].values

        n = len(y_sub)
        n_treated = T_sub.sum()

        print(f"N = {n:,}, Treated = {n_treated:,} ({100*n_treated/n:.1f}%)")

        # AIPW with FULL covariates (except industry itself)
        ps_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        ps_model.fit(X_sub, T_sub)
        ps = ps_model.predict_proba(X_sub)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)

        mu1_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        mu0_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

        mu1_model.fit(X_sub[T_sub == 1], y_sub[T_sub == 1])
        mu0_model.fit(X_sub[T_sub == 0], y_sub[T_sub == 0])

        mu1 = mu1_model.predict(X_sub)
        mu0 = mu0_model.predict(X_sub)

        aipw_scores = T_sub * (y_sub - mu1) / ps - (1 - T_sub) * (y_sub - mu0) / (1 - ps) + mu1 - mu0

        ate = np.mean(aipw_scores)
        se = np.std(aipw_scores) / np.sqrt(n)
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        p_value = 2 * (1 - 0.5 * (1 + np.sign(ate/se) * (1 - np.exp(-abs(ate/se)**2/2))))

        print(f"ATE = {ate:.6f}, SE = {se:.6f}, p = {p_value:.4f}")
        print(f"Effect = {100 * (np.exp(ate) - 1):+.2f}%")

        industry_results.append({
            'Industry': int(industry),
            'N': n,
            'N_treated': n_treated,
            'Treated_pct': 100*n_treated/n,
            'ATE': ate,
            'SE': se,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'p_value': p_value,
            'Effect_pct': 100 * (np.exp(ate) - 1)
        })

    df_industry = pd.DataFrame(industry_results)
    df_industry.to_csv('analysis/results/industry_heterogeneity.csv', index=False)
    print("\n✅ Industry heterogeneity results:")
    print(df_industry)

else:
    print("\n❌ Industry heterogeneity NOT FEASIBLE (no industry variable)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✅ All feasible analyses completed")
print("   Results saved in analysis/results/")
