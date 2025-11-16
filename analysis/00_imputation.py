#!/usr/bin/env python3
"""
Phase 1-1: Multiple Imputation using MICE
- 해결: Complete case (43% 손실) → Multiple Imputation
- 방법: Iterative Imputer (MICE algorithm)
- 산출: 10개 imputed datasets

High-Dimensional Expansion:
- Phase 4 (CURRENT): p=101 full run
- Variable mapping from variable_mapping_p100.py
- Expected runtime: 1-2 hours (based on p=50 test results)
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Import variable mapping for p=101 full run
from variable_mapping_p100 import SELECTED_VARIABLES_P100, ENGINEERED_FEATURES

# 설정
DATA_DIR = "raw data"
OUTPUT_DIR = "analysis/processed_data/p101_full"
ENCODING = 'cp949'
N_IMPUTATIONS = 10
RANDOM_SEED = 42

def load_and_merge_data():
    """5개년 데이터 로드 및 통합"""
    print(f"\n{'='*60}")
    print("Step 1: 5개년 데이터 로드 및 통합")
    print(f"{'='*60}")

    years = ['2019', '2020', '2021', '2022', '2023']
    all_data = []

    for year in years:
        filepath = os.path.join(DATA_DIR, f"{year}_기업활동조사_20251107_29604.csv")

        if not os.path.exists(filepath):
            print(f"⚠️  {year}년 파일 없음: {filepath}")
            continue

        print(f"\n로드: {year}년 데이터")
        df_year = pd.read_csv(filepath, encoding=ENCODING, low_memory=False)
        print(f"  행 수: {len(df_year):,}개, 컬럼: {len(df_year.columns):,}개")

        all_data.append(df_year)

    # 통합
    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ 통합 완료: {len(df):,}개 행, {len(df.columns):,}개 컬럼")

    return df

def filter_manufacturing(df):
    """제조업만 필터링 (KSIC 대분류 C)"""
    print(f"\n{'='*60}")
    print("Step 2: 제조업 필터링")
    print(f"{'='*60}")

    # 제조업 코드 확인
    ksic_col = '주사업_산업대분류코드'

    if ksic_col not in df.columns:
        print(f"❌ 컬럼 '{ksic_col}' 없음")
        print(f"사용 가능한 컬럼: {[c for c in df.columns if '산업' in c][:10]}")
        return df

    df_mfg = df[df[ksic_col] == 'C'].copy()

    print(f"전체: {len(df):,}개")
    print(f"제조업: {len(df_mfg):,}개 ({len(df_mfg)/len(df)*100:.1f}%)")

    return df_mfg

def identify_key_variables(df):
    """
    핵심 변수 식별 및 타입 변환 (p=101 Full Run)

    Uses SELECTED_VARIABLES_P100 from variable_mapping_p100.py
    Total: 101 variables (87 raw covariates + 14 engineered features)
    """
    print(f"\n{'='*60}")
    print("Step 3: 핵심 변수 식별 (p=101 Full Run)")
    print(f"{'='*60}")

    # Core identifiers and treatment variables (not in SELECTED_VARIABLES_P100)
    id_and_treatment_vars = {
        # 식별자
        'firm_id': 'BR_패널키값',
        'year': '조사기준연도',

        # Treatment 관련 (DTI variables)
        'dti_iot': '4차산업혁명_기술활용_IOT_활용단계구분코드',
        'dti_cloud': '4차산업혁명_기술활용_CLOUD_활용단계구분코드',
        'dti_ai': '4차산업혁명_기술활용_AI_활용단계구분코드',
        'dti_bigdata': '4차산업혁명_기술활용_BIGDATA_활용단계구분코드',
    }

    # TFP estimation variables (Levinsohn-Petrin method)
    tfp_vars = {
        'revenue': '경영실적_매출금액영업수익',
        'employees': '상용근로_남여합계종사자수',
        'capital': '재무구조_유형자산',  # Already in SELECTED_VARIABLES_P100 as tangible_assets
        'labor_cost': '비용내역_영업비_인건비',  # Already in SELECTED_VARIABLES_P100 as labor_costs
    }

    # Materials for LP (중간재)
    # Note: These will be computed from SELECTED_VARIABLES_P100 variables
    # materials = cost_of_sales (매출원가) as proxy
    # energy: Can be approximated from cost_of_sales if not directly available
    # outsourcing: Already in SELECTED_VARIABLES_P100 as outsourcing_total

    # Combine all variables
    key_vars = {**id_and_treatment_vars, **tfp_vars, **SELECTED_VARIABLES_P100}

    # Remove duplicates (TFP vars that are also in SELECTED_VARIABLES_P100)
    # capital = tangible_assets (already mapped)
    # labor_cost = labor_costs (already mapped)
    key_vars_dedup = {}
    seen_korean_names = set()

    for var_name, col_name in key_vars.items():
        if col_name not in seen_korean_names:
            key_vars_dedup[var_name] = col_name
            seen_korean_names.add(col_name)

    # 실제 존재하는 변수만 매핑
    available_vars = {}
    missing_vars = []

    for var_name, col_name in key_vars_dedup.items():
        if col_name in df.columns:
            available_vars[var_name] = col_name
        else:
            missing_vars.append((var_name, col_name))

    print(f"\n사용 가능한 변수: {len(available_vars)}개 / {len(key_vars_dedup)}개 요청")
    print(f"  ✓ 식별자 & Treatment: 6개")
    print(f"  ✓ TFP 추정용: {len([v for v in tfp_vars if v in available_vars])}개")
    print(f"  ✓ p=101 Covariates: {len([v for v in SELECTED_VARIABLES_P100 if v in available_vars])}개")

    if missing_vars:
        print(f"\n⚠️  누락된 변수: {len(missing_vars)}개")
        print(f"누락 변수 목록 (상위 10개):")
        for var_name, col_name in missing_vars[:10]:
            print(f"  ✗ {var_name:30s}: {col_name}")

        if len(missing_vars) > 10:
            print(f"  ... (추가 {len(missing_vars)-10}개)")

    # 변수 리네임
    rename_map = {col: var for var, col in available_vars.items()}
    df_renamed = df.rename(columns=rename_map)

    # 핵심 변수만 선택
    selected_cols = list(available_vars.keys())
    df_selected = df_renamed[selected_cols].copy()

    print(f"\n최종 선택된 변수 수: {len(selected_cols)}개")

    return df_selected, available_vars

def convert_to_numeric(df):
    """모든 변수를 숫자형으로 변환"""
    print(f"\n{'='*60}")
    print("Step 4: 숫자형 변환")
    print(f"{'='*60}")

    # 식별자 제외
    id_cols = ['firm_id', 'year']
    numeric_cols = [c for c in df.columns if c not in id_cols]

    # Step 4.1: 범주형 변수 전처리 (Y/N → 1/0)
    # 이진 범주형 변수들: 외부위탁여부, 기업집단대상여부 등
    binary_categorical_vars = [
        'production_outsourcing_domestic',
        'rd_outsourcing_domestic',
        'corporate_group_restricted',
        'stock_market_listing',
        'has_subsidiaries'
    ]

    converted_count = 0
    for col in binary_categorical_vars:
        if col in df.columns:
            # 변환 전 상태 확인
            original_dtype = df[col].dtype
            unique_before = df[col].dropna().unique()[:5]

            # 'Y' → 1, 'N'/'*'/기타 → 0 변환
            df[col] = df[col].apply(lambda x: 1 if x == 'Y' else (0 if pd.notna(x) else x))
            converted_count += 1

            unique_after = df[col].dropna().unique()[:5]
            print(f"  ✓ {col}: {unique_before} → {unique_after}")

    if converted_count > 0:
        print(f"\n범주형 변수 {converted_count}개 변환 완료 (Y/N → 1/0)\n")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 결측치 현황
    missing_summary = df[numeric_cols].isna().sum().sort_values(ascending=False)
    missing_pct = (missing_summary / len(df) * 100).round(1)

    print(f"\n결측치가 많은 상위 10개 변수:")
    top_missing = pd.DataFrame({
        'Variable': missing_summary.head(10).index,
        'Missing': missing_summary.head(10).values,
        'Percent': missing_pct.head(10).values
    })
    print(top_missing.to_string(index=False))

    # Complete case 비율
    complete_cases = df[numeric_cols].notna().all(axis=1).sum()
    complete_pct = complete_cases / len(df) * 100

    print(f"\n전체 관측치: {len(df):,}개")
    print(f"Complete cases: {complete_cases:,}개 ({complete_pct:.1f}%)")
    print(f"→ Complete case 손실: {100 - complete_pct:.1f}%")

    return df

def recode_dti_variables(df):
    """
    DTI 변수를 연도별로 올바르게 재코딩

    **중요**: 2021년부터 설문 방식이 변경되어 연도별 다른 로직 필요

    2019-2020:
      - 값 1-6: 활용 단계 → 원본 유지
      - NA: 미활용 → 0으로 변환

    2021-2023:
      - 값 1: 미활용 → 0으로 변환
      - 값 2-6: 활용 단계 → 원본 유지
      - NA: 미활용/무응답 → 0으로 변환
    """
    print(f"\n{'='*60}")
    print("Step 4.5: DTI 변수 연도별 재코딩")
    print(f"{'='*60}")

    dti_cols = ['dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata']

    print("\n재코딩 전 DTI 분포:")
    for col in dti_cols:
        if col in df.columns:
            non_na = df[col].notna().sum()
            na = df[col].isna().sum()
            total = len(df)
            print(f"  {col}: 응답={non_na} ({100*non_na/total:.1f}%), 결측={na} ({100*na/total:.1f}%)")

    # 연도별 재코딩
    print("\n연도별 재코딩 수행:")

    for year in sorted(df['year'].unique()):
        df_year_mask = (df['year'] == year)
        n_year = df_year_mask.sum()

        print(f"\n  {year}년 (N={n_year}):")

        if year <= 2020:
            # 2019-2020: NA → 0, 1-6 유지
            print(f"    로직: NA → 0, 값 1-6 → 원본 유지")
            for col in dti_cols:
                if col in df.columns:
                    before_na = df.loc[df_year_mask, col].isna().sum()
                    df.loc[df_year_mask & df[col].isna(), col] = 0
                    print(f"      {col}: NA {before_na}개 → 0으로 변환")

        else:  # year >= 2021
            # 2021-2023: 1 → 0, NA → 0, 2-6 유지
            print(f"    로직: 값 1 → 0, NA → 0, 값 2-6 → 원본 유지")
            for col in dti_cols:
                if col in df.columns:
                    before_1 = (df.loc[df_year_mask, col] == 1).sum()
                    before_na = df.loc[df_year_mask, col].isna().sum()

                    # 값 1을 0으로 변환
                    df.loc[df_year_mask & (df[col] == 1), col] = 0
                    # 결측값도 0으로 변환
                    df.loc[df_year_mask & df[col].isna(), col] = 0

                    after_zero = (df.loc[df_year_mask, col] == 0).sum()
                    print(f"      {col}: 값1 {before_1}개 + NA {before_na}개 → {after_zero}개")

    print("\n재코딩 후 전체 DTI 분포:")
    for col in dti_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            nonzero_count = (df[col] > 0).sum()
            total = len(df)
            print(f"  {col}: 0={zero_count} ({100*zero_count/total:.1f}%), >0={nonzero_count} ({100*nonzero_count/total:.1f}%)")

    # DTI 평균 계산 (검증용)
    df['dti_mean'] = df[dti_cols].mean(axis=1)

    print(f"\nDTI 평균 (4개 변수):")
    print(f"  평균: {df['dti_mean'].mean():.4f}")
    print(f"  중앙값: {df['dti_mean'].median():.4f}")
    print(f"  최소: {df['dti_mean'].min():.4f}")
    print(f"  최대: {df['dti_mean'].max():.4f}")

    print(f"\n연도별 처치군 분포 (DTI > 0):")
    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year]
        treated = (df_year['dti_mean'] > 0).sum()
        total = len(df_year)
        print(f"  {year}: {treated}/{total} ({100*treated/total:.1f}%)")

    # dti_mean은 임시 컬럼이므로 제거
    df = df.drop(columns=['dti_mean'])

    print("\n✅ DTI 연도별 재코딩 완료")

    return df

def engineer_features(df):
    """
    Feature Engineering for p=101 Full Run

    Creates 14 engineered features from ENGINEERED_FEATURES dictionary:
    - Financial ratios (3)
    - Innovation intensity (2)
    - Market exposure (2)
    - Human capital quality (2)
    - Year fixed effects (5)

    Note: Features are created BEFORE MICE imputation, so they inherit
    imputed values from their component variables.
    """
    print(f"\n{'='*60}")
    print("Step 4.6: Feature Engineering (14 features for p=101)")
    print(f"{'='*60}")

    # Copy to avoid modifying original
    df = df.copy()

    # Financial ratios
    print("\n1. Financial Ratios (3 features):")
    if 'total_debt' in df.columns and 'total_equity' in df.columns:
        df['debt_to_equity'] = df['total_debt'] / (df['total_equity'] + 1e-6)
        print(f"  ✓ debt_to_equity: mean={df['debt_to_equity'].mean():.3f}")

    if 'quick_assets' in df.columns and 'current_liabilities' in df.columns:
        df['current_ratio'] = df['quick_assets'] / (df['current_liabilities'] + 1e-6)
        print(f"  ✓ current_ratio: mean={df['current_ratio'].mean():.3f}")

    if 'revenue' in df.columns and 'total_assets' in df.columns:
        df['asset_turnover'] = df['revenue'] / (df['total_assets'] + 1e-6)
        print(f"  ✓ asset_turnover: mean={df['asset_turnover'].mean():.3f}")

    # Innovation intensity
    print("\n2. Innovation Intensity (2 features):")
    if 'patents_held' in df.columns and 'employees' in df.columns:
        df['patent_intensity'] = df['patents_held'] / (df['employees'] + 1e-6)
        print(f"  ✓ patent_intensity: mean={df['patent_intensity'].mean():.6f}")

    if 'rd_employees_total' in df.columns and 'employees' in df.columns:
        df['rd_employee_ratio'] = df['rd_employees_total'] / (df['employees'] + 1e-6)
        print(f"  ✓ rd_employee_ratio: mean={df['rd_employee_ratio'].mean():.4f}")

    # Market exposure
    print("\n3. Market Exposure (2 features):")
    if 'direct_export_amount' in df.columns and 'revenue' in df.columns:
        df['export_intensity'] = df['direct_export_amount'] / (df['revenue'] + 1e-6)
        print(f"  ✓ export_intensity: mean={df['export_intensity'].mean():.4f}")

    if 'direct_import_amount' in df.columns and 'revenue' in df.columns:
        df['import_intensity'] = df['direct_import_amount'] / (df['revenue'] + 1e-6)
        print(f"  ✓ import_intensity: mean={df['import_intensity'].mean():.4f}")

    # Human capital quality
    print("\n4. Human Capital Quality (2 features):")
    if 'regular_employees_female' in df.columns and 'employees' in df.columns:
        df['female_employee_ratio'] = df['regular_employees_female'] / (df['employees'] + 1e-6)
        print(f"  ✓ female_employee_ratio: mean={df['female_employee_ratio'].mean():.3f}")

    if 'hq_production_employees_total' in df.columns and 'hq_employees_total' in df.columns:
        df['production_worker_ratio'] = df['hq_production_employees_total'] / (df['hq_employees_total'] + 1e-6)
        print(f"  ✓ production_worker_ratio: mean={df['production_worker_ratio'].mean():.3f}")

    # Year fixed effects (5 features)
    print("\n5. Year Fixed Effects (5 features):")
    if 'year' in df.columns:
        df['year_2020'] = (df['year'] == 2020).astype(int)
        df['year_2021'] = (df['year'] == 2021).astype(int)
        df['year_2022'] = (df['year'] == 2022).astype(int)
        df['year_2023'] = (df['year'] == 2023).astype(int)
        df['covid_period'] = (df['year'] >= 2020).astype(int)

        print(f"  ✓ year_2020: n={df['year_2020'].sum()}")
        print(f"  ✓ year_2021: n={df['year_2021'].sum()}")
        print(f"  ✓ year_2022: n={df['year_2022'].sum()}")
        print(f"  ✓ year_2023: n={df['year_2023'].sum()}")
        print(f"  ✓ covid_period: n={df['covid_period'].sum()}")

    # Count successfully created features
    engineered_cols = [
        'debt_to_equity', 'current_ratio', 'asset_turnover',
        'patent_intensity', 'rd_employee_ratio',
        'export_intensity', 'import_intensity',
        'female_employee_ratio', 'production_worker_ratio',
        'year_2020', 'year_2021', 'year_2022', 'year_2023', 'covid_period'
    ]
    created = [c for c in engineered_cols if c in df.columns]

    print(f"\n✅ Feature Engineering 완료: {len(created)}/14 features created")

    if len(created) < 14:
        missing_features = [c for c in engineered_cols if c not in df.columns]
        print(f"\n⚠️  생성되지 않은 features ({len(missing_features)}):")
        for feat in missing_features:
            print(f"  ✗ {feat}")

    return df

def multiple_imputation_mice(df, n_imputations=10):
    """
    Multiple Imputation using MICE (Multivariate Imputation by Chained Equations)

    중요: DTI 변수는 MICE에서 제외
    - DTI는 이미 재코딩됨 (NA → 0)
    - DTI를 MICE에 포함하면 0값이 다른 값으로 잘못 대체될 수 있음

    Parameters:
    - df: DataFrame with missing values
    - n_imputations: Number of imputed datasets to create

    Returns:
    - List of imputed DataFrames
    """
    print(f"\n{'='*60}")
    print(f"Step 5: Multiple Imputation (MICE)")
    print(f"{'='*60}")

    # 식별자 분리
    id_cols = ['firm_id', 'year']
    ids = df[id_cols].copy()

    # DTI 변수 분리 (MICE에서 제외)
    dti_cols = ['dti_iot', 'dti_cloud', 'dti_ai', 'dti_bigdata']
    dti_data = df[dti_cols].copy()

    # MICE에 포함할 변수: 숫자형 변수에서 식별자와 DTI 제외
    exclude_cols = id_cols + dti_cols
    numeric_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[numeric_cols].copy()

    print(f"\nImputation 설정:")
    print(f"  - 변수 수: {X.shape[1]}개 (DTI 제외)")
    print(f"  - DTI 변수: {len(dti_cols)}개 (MICE 제외, 원본 유지)")
    print(f"  - 관측치 수: {X.shape[0]:,}개")
    print(f"  - Imputation 횟수: {n_imputations}회")
    print(f"  - 알고리즘: Iterative Imputer (MICE)")

    # MICE imputer
    # RandomForestRegressor를 사용하여 더 유연한 관계 포착
    #
    # ⚠️  High-Dimensional Settings (p=50 test):
    # - Current settings: Optimized for p=15, testing at p=50
    # - Expected runtime: 2-3 hours (vs. 45 min for p=15, 6-8 hours for p=101)
    # - If convergence issues detected at p=50, adjust before p=101:
    #   * Increase max_depth to 7-10
    #   * Increase n_estimators to 20
    #   * Increase max_iter to 15-20
    # - Trade-off: Better quality vs. 2-3x longer runtime
    #
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=10,  # 빠른 실행 (p=50 test, p=101에서는 20 권장)
            max_depth=5,      # 얕은 트리 (p=50 test, p=101에서는 7-10 권장)
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        max_iter=10,          # MICE 반복 횟수 (p=50 test, p=101에서는 15-20 권장)
        random_state=RANDOM_SEED,
        verbose=1,
        initial_strategy='median'
    )

    # Multiple imputations
    imputed_datasets = []

    for m in range(n_imputations):
        print(f"\n[{m+1}/{n_imputations}] Imputation 수행 중...")

        # 다른 random seed 사용
        imputer.set_params(random_state=RANDOM_SEED + m)

        # Impute
        X_imputed = imputer.fit_transform(X)

        # DataFrame으로 변환
        # Note: IterativeImputer may drop columns with all missing values
        # Use actual shape of X_imputed to get correct column names
        if X_imputed.shape[1] != len(numeric_cols):
            # Find which columns were kept
            valid_cols = []
            for col in numeric_cols:
                if X[col].notna().any():  # Column has at least one non-missing value
                    valid_cols.append(col)
            df_imputed = pd.DataFrame(X_imputed, columns=valid_cols)
        else:
            df_imputed = pd.DataFrame(X_imputed, columns=numeric_cols)

        # 식별자와 DTI 변수 추가
        df_imputed = pd.concat([
            ids.reset_index(drop=True),
            dti_data.reset_index(drop=True),
            df_imputed
        ], axis=1)

        imputed_datasets.append(df_imputed)

        # 진행상황
        print(f"  ✓ Imputation {m+1} 완료")

    print(f"\n✅ Total {n_imputations}개 imputed datasets 생성 완료")

    return imputed_datasets

def save_imputed_datasets(imputed_datasets, output_dir):
    """Imputed datasets 저장"""
    print(f"\n{'='*60}")
    print("Step 6: Imputed Datasets 저장")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 개별 CSV 저장
    for m, df_imp in enumerate(imputed_datasets):
        filepath = os.path.join(output_dir, f"imputed_m{m+1}.csv")
        df_imp.to_csv(filepath, index=False, encoding='utf-8')

        file_size = os.path.getsize(filepath) / (1024**2)  # MB
        print(f"  ✓ Imputation {m+1}: {filepath} ({file_size:.1f} MB)")

    # Pickle로 전체 저장 (빠른 로드)
    pickle_path = os.path.join(output_dir, "imputed_datasets.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(imputed_datasets, f)

    pickle_size = os.path.getsize(pickle_path) / (1024**2)
    print(f"\n  ✓ Pickle: {pickle_path} ({pickle_size:.1f} MB)")

    # 메타데이터 저장
    meta_path = os.path.join(output_dir, "imputation_metadata.txt")
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write("Multiple Imputation Metadata\n")
        f.write("="*60 + "\n\n")
        f.write(f"Number of imputations: {len(imputed_datasets)}\n")
        f.write(f"Observations per dataset: {len(imputed_datasets[0]):,}\n")
        f.write(f"Variables: {len(imputed_datasets[0].columns)}\n")
        f.write(f"Method: MICE (Iterative Imputer with Random Forest)\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"\nVariables:\n")
        for col in imputed_datasets[0].columns:
            f.write(f"  - {col}\n")

    print(f"  ✓ Metadata: {meta_path}")

    return output_dir

def main():
    """메인 실행"""
    print("="*60)
    print("Multiple Imputation Pipeline")
    print("="*60)

    # Step 1: 데이터 로드
    df = load_and_merge_data()

    # Step 2: 제조업 필터링
    df_mfg = filter_manufacturing(df)

    # Step 3: 핵심 변수 선택
    df_selected, var_map = identify_key_variables(df_mfg)

    # Step 4: 숫자형 변환
    df_numeric = convert_to_numeric(df_selected)

    # Step 4.5: DTI 변수 재코딩 (NA → 0)
    df_recoded = recode_dti_variables(df_numeric)

    # Step 4.6: Feature Engineering (p=101 expansion)
    df_engineered = engineer_features(df_recoded)

    # Step 4.9: MICE 전 데이터 저장 (MAR verification용)
    print(f"\n{'='*60}")
    print("Step 4.9: MICE 전 데이터 저장 (MAR verification용)")
    print(f"{'='*60}")
    pre_mice_path = os.path.join(OUTPUT_DIR, "data_before_mice.pkl")
    df_engineered.to_pickle(pre_mice_path)
    print(f"✅ 저장 완료: {pre_mice_path}")
    print(f"   Shape: {df_engineered.shape}")
    print(f"   Missing values: {df_engineered.isna().sum().sum():,}")

    # Step 5: Multiple Imputation
    imputed_datasets = multiple_imputation_mice(df_engineered, n_imputations=N_IMPUTATIONS)

    # Step 6: 저장
    output_dir = save_imputed_datasets(imputed_datasets, OUTPUT_DIR)

    print("\n" + "="*60)
    print("✅ Multiple Imputation 완료!")
    print("="*60)
    print(f"\n산출물:")
    print(f"  - {N_IMPUTATIONS}개 imputed datasets")
    print(f"  - 저장 위치: {output_dir}/")
    print(f"\n다음 단계: python analysis/01_variable_selection.py")

if __name__ == "__main__":
    main()
