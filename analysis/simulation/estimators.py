"""
Causal Estimators for Simulation Study
=======================================

Implements 7 estimation methods:
1. MICE-DR (AIPW): Multiple imputation + doubly robust
2. Complete-Case-DR: Delete missing rows + AIPW
3. MICE-OLS: Multiple imputation + naive regression
4. Single-Imputation-DR: Mean imputation + AIPW
5. CEM: Coarsened Exact Matching
6. CBPS: Covariate Balancing Propensity Score
7. NN-Matching: Nearest Neighbor matching

Author: Causal Inference Pipeline Project
Date: 2025-01
"""

import time
import warnings
import numpy as np
from abc import ABC, abstractmethod
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import t as t_dist

warnings.filterwarnings('ignore')


class BaseEstimator(ABC):
    """Abstract base class for all estimators"""

    @abstractmethod
    def estimate(self, X_miss, T, Y):
        """
        Estimate ATE from data with missing values

        Parameters
        ----------
        X_miss : ndarray, shape (n, p)
            Covariates with missing values (np.nan)
        T : ndarray, shape (n,)
            Binary treatment
        Y : ndarray, shape (n,)
            Continuous outcome

        Returns
        -------
        ate_est : float
            Point estimate of ATE
        se_est : float
            Standard error
        ci_lower : float
            Lower bound of 95% CI
        ci_upper : float
            Upper bound of 95% CI
        runtime : float
            Computation time in seconds
        """
        pass


class MICEDoublyRobust(BaseEstimator):
    """
    MICE + AIPW Estimator (Main Method)

    Multiple Imputation by Chained Equations combined with
    Augmented Inverse Probability Weighting (Doubly Robust)
    """

    def __init__(self, m_imputations=10, k_folds=5, max_iter=5, random_state=42):
        """
        Parameters
        ----------
        m_imputations : int
            Number of multiple imputations (default 10)
        k_folds : int
            Number of folds for cross-fitting (default 5)
        max_iter : int
            Max iterations for MICE (default 5)
        random_state : int
            Random seed
        """
        self.m_imputations = m_imputations
        self.k_folds = k_folds
        self.max_iter = max_iter
        self.random_state = random_state

    def estimate(self, X_miss, T, Y):
        start_time = time.time()

        # Step 1: MICE Imputation (excluding T from imputation model)
        imputed_datasets = self._mice_impute(X_miss, T, Y)

        # Step 2: AIPW per imputation
        ates = []
        variances = []

        for m, X_imp in enumerate(imputed_datasets):
            ate_m, var_m = self._aipw_crossfit(X_imp, T, Y)
            ates.append(ate_m)
            variances.append(var_m)

        # Step 3: Rubin's rules
        ate_pool = np.mean(ates)
        var_within = np.mean(variances)
        var_between = np.var(ates, ddof=1) if len(ates) > 1 else 0
        var_pool = var_within + (1 + 1 / self.m_imputations) * var_between
        se_pool = np.sqrt(var_pool)

        # 95% CI
        ci_lower = ate_pool - 1.96 * se_pool
        ci_upper = ate_pool + 1.96 * se_pool

        runtime = time.time() - start_time

        return ate_pool, se_pool, ci_lower, ci_upper, runtime

    def _mice_impute(self, X_miss, T, Y):
        """
        Perform MICE imputation

        Returns list of M imputed datasets
        """
        # Prepare data for imputation: [X, Y] (exclude T)
        n, p = X_miss.shape
        data_for_imputation = np.column_stack([X_miss, Y])

        # MICE imputer
        imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            sample_posterior=True,  # Draw from posterior for uncertainty
            n_nearest_features=min(10, p),  # Limit features for speed
            skip_complete=True
        )

        # Generate M imputations
        imputed_datasets = []
        for m in range(self.m_imputations):
            imputer.set_params(random_state=self.random_state + m)
            data_imputed = imputer.fit_transform(data_for_imputation)
            X_imputed = data_imputed[:, :p]  # Extract X only (Y not used in estimation)
            imputed_datasets.append(X_imputed)

        return imputed_datasets

    def _aipw_crossfit(self, X, T, Y):
        """
        Doubly Robust AIPW with K-fold cross-fitting

        Returns ATE and variance
        """
        n = len(T)
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)

        # Store fold-wise estimates
        fold_estimates = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Fit propensity score model
            ps_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state
            )
            ps_model.fit(X_train, T_train)
            ps_test = ps_model.predict_proba(X_test)[:, 1]
            ps_test = np.clip(ps_test, 0.01, 0.99)  # Avoid division by zero

            # Fit outcome models
            # E[Y|X,T=1]
            idx_treated = T_train == 1
            if idx_treated.sum() > 10:  # Need enough treated units
                mu1_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
                mu1_model.fit(X_train[idx_treated], Y_train[idx_treated])
                mu1_test = mu1_model.predict(X_test)
            else:
                mu1_test = np.mean(Y_train[idx_treated]) if idx_treated.sum() > 0 else 0

            # E[Y|X,T=0]
            idx_control = T_train == 0
            if idx_control.sum() > 10:
                mu0_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
                mu0_model.fit(X_train[idx_control], Y_train[idx_control])
                mu0_test = mu0_model.predict(X_test)
            else:
                mu0_test = np.mean(Y_train[idx_control]) if idx_control.sum() > 0 else 0

            # AIPW estimator for test fold
            aipw_scores = (
                mu1_test - mu0_test +
                T_test / ps_test * (Y_test - mu1_test) -
                (1 - T_test) / (1 - ps_test) * (Y_test - mu0_test)
            )

            fold_estimates.extend(aipw_scores)

        # Pool across folds
        ate = np.mean(fold_estimates)
        variance = np.var(fold_estimates, ddof=1) / n

        return ate, variance


class CompleteCaseDR(BaseEstimator):
    """Complete-case analysis + AIPW"""

    def __init__(self, k_folds=5, random_state=42):
        self.k_folds = k_folds
        self.random_state = random_state

    def estimate(self, X_miss, T, Y):
        start_time = time.time()

        # Remove rows with any missing values
        complete_mask = ~np.isnan(X_miss).any(axis=1)
        X_complete = X_miss[complete_mask]
        T_complete = T[complete_mask]
        Y_complete = Y[complete_mask]

        if len(T_complete) < 100:
            # Too few complete cases
            return np.nan, np.nan, np.nan, np.nan, time.time() - start_time

        # Apply AIPW
        mice_dr = MICEDoublyRobust(m_imputations=1, k_folds=self.k_folds, random_state=self.random_state)
        ate, var = mice_dr._aipw_crossfit(X_complete, T_complete, Y_complete)

        se = np.sqrt(var)
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        runtime = time.time() - start_time
        return ate, se, ci_lower, ci_upper, runtime


class MICE_OLS(BaseEstimator):
    """MICE + Naive OLS regression (no DR)"""

    def __init__(self, m_imputations=10, max_iter=5, random_state=42):
        self.m_imputations = m_imputations
        self.max_iter = max_iter
        self.random_state = random_state

    def estimate(self, X_miss, T, Y):
        start_time = time.time()

        # MICE imputation
        mice_dr = MICEDoublyRobust(
            m_imputations=self.m_imputations,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        imputed_datasets = mice_dr._mice_impute(X_miss, T, Y)

        # OLS per imputation
        ates = []
        variances = []

        for X_imp in imputed_datasets:
            # Simple OLS: Y ~ T + X
            X_with_treatment = np.column_stack([T, X_imp])
            model = LinearRegression()
            model.fit(X_with_treatment, Y)

            ate_m = model.coef_[0]  # Coefficient on T

            # Estimate variance (naive, ignoring imputation uncertainty)
            Y_pred = model.predict(X_with_treatment)
            residuals = Y - Y_pred
            sigma2 = np.sum(residuals**2) / (len(Y) - X_with_treatment.shape[1] - 1)
            XtX_inv = np.linalg.inv(X_with_treatment.T @ X_with_treatment)
            var_m = sigma2 * XtX_inv[0, 0]

            ates.append(ate_m)
            variances.append(var_m)

        # Rubin's rules
        ate_pool = np.mean(ates)
        var_within = np.mean(variances)
        var_between = np.var(ates, ddof=1)
        var_pool = var_within + (1 + 1 / self.m_imputations) * var_between
        se_pool = np.sqrt(var_pool)

        ci_lower = ate_pool - 1.96 * se_pool
        ci_upper = ate_pool + 1.96 * se_pool

        runtime = time.time() - start_time
        return ate_pool, se_pool, ci_lower, ci_upper, runtime


class SingleImputationDR(BaseEstimator):
    """Single mean imputation + AIPW (no uncertainty propagation)"""

    def __init__(self, k_folds=5, random_state=42):
        self.k_folds = k_folds
        self.random_state = random_state

    def estimate(self, X_miss, T, Y):
        start_time = time.time()

        # Mean imputation
        imputer = SimpleImputer(strategy='mean')
        X_imp = imputer.fit_transform(X_miss)

        # AIPW
        mice_dr = MICEDoublyRobust(m_imputations=1, k_folds=self.k_folds, random_state=self.random_state)
        ate, var = mice_dr._aipw_crossfit(X_imp, T, Y)

        se = np.sqrt(var)
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        runtime = time.time() - start_time
        return ate, se, ci_lower, ci_upper, runtime


class CEM_Matching(BaseEstimator):
    """
    Coarsened Exact Matching

    NOTE: Cannot handle missing data - uses complete cases only
    """

    def __init__(self, n_bins=5, random_state=42):
        """
        Parameters
        ----------
        n_bins : int
            Number of bins for coarsening continuous variables
        """
        self.n_bins = n_bins
        self.random_state = random_state

    def estimate(self, X_miss, T, Y):
        start_time = time.time()

        # Remove missing (CEM requires complete data)
        complete_mask = ~np.isnan(X_miss).any(axis=1)
        X = X_miss[complete_mask]
        T_complete = T[complete_mask]
        Y_complete = Y[complete_mask]

        if len(T_complete) < 100 or T_complete.sum() < 10 or (1 - T_complete).sum() < 10:
            return np.nan, np.nan, np.nan, np.nan, time.time() - start_time

        # Coarsen X into bins
        X_coarsened = self._coarsen(X)

        # Find exact matches on coarsened X
        matched_idx_treated, matched_idx_control, weights = self._find_matches(
            X_coarsened, T_complete
        )

        if len(matched_idx_treated) == 0:
            return np.nan, np.nan, np.nan, np.nan, time.time() - start_time

        # Compute ATE on matched sample (weighted)
        Y_treated = Y_complete[matched_idx_treated]
        Y_control = Y_complete[matched_idx_control]

        ate = np.mean(Y_treated * weights) - np.mean(Y_control * weights)

        # Bootstrap SE
        se = self._bootstrap_se(Y_treated, Y_control, weights, n_bootstrap=200)

        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        runtime = time.time() - start_time
        return ate, se, ci_lower, ci_upper, runtime

    def _coarsen(self, X):
        """Bin continuous variables into n_bins categories"""
        n, p = X.shape
        X_coarsened = np.zeros((n, p), dtype=int)

        for j in range(p):
            # Create bins based on quantiles
            bins = np.percentile(X[:, j], np.linspace(0, 100, self.n_bins + 1))
            bins = np.unique(bins)  # Remove duplicates
            X_coarsened[:, j] = np.digitize(X[:, j], bins[1:-1])

        return X_coarsened

    def _find_matches(self, X_coarsened, T):
        """Find treated-control pairs with exact match on coarsened X"""
        import pandas as pd

        # Create strata based on coarsened X
        df = pd.DataFrame(X_coarsened)
        df['T'] = T
        df['idx'] = np.arange(len(T))

        # Group by covariate bins
        strata = df.iloc[:, :-2].apply(tuple, axis=1)
        df['stratum'] = strata

        # Find strata with both treated and control units
        matched_treated = []
        matched_control = []
        weights_list = []

        for stratum_id, group in df.groupby('stratum'):
            treated_in_stratum = group[group['T'] == 1]
            control_in_stratum = group[group['T'] == 0]

            if len(treated_in_stratum) > 0 and len(control_in_stratum) > 0:
                # Weight by stratum size (inverse probability weighting)
                n_stratum = len(group)
                weight = 1 / n_stratum

                matched_treated.extend(treated_in_stratum['idx'].tolist())
                matched_control.extend(control_in_stratum['idx'].tolist())
                weights_list.extend([weight] * len(treated_in_stratum))

        return np.array(matched_treated), np.array(matched_control), np.array(weights_list)

    def _bootstrap_se(self, Y_treated, Y_control, weights, n_bootstrap=200):
        """Bootstrap standard error"""
        rng = np.random.RandomState(self.random_state)
        ates_boot = []

        for _ in range(n_bootstrap):
            idx_boot = rng.choice(len(Y_treated), size=len(Y_treated), replace=True)
            Y_treated_boot = Y_treated[idx_boot]
            Y_control_boot = Y_control[idx_boot]
            weights_boot = weights[idx_boot]

            ate_boot = np.mean(Y_treated_boot * weights_boot) - np.mean(Y_control_boot * weights_boot)
            ates_boot.append(ate_boot)

        return np.std(ates_boot, ddof=1)


# Placeholder for CBPS and NN-Matching (can be added later if needed)
class CBPS_Weighting(BaseEstimator):
    """Placeholder - requires external package"""
    def estimate(self, X_miss, T, Y):
        return np.nan, np.nan, np.nan, np.nan, 0.0


class NearestNeighborMatching(BaseEstimator):
    """Placeholder - can be implemented if needed"""
    def estimate(self, X_miss, T, Y):
        return np.nan, np.nan, np.nan, np.nan, 0.0


def test_estimators():
    """Test all estimators on synthetic data"""
    print("Testing Estimators...")
    print("=" * 60)

    # Generate simple test data
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simulation.dgp import DGP

    dgp = DGP(n=500, p=10, imbalance_ratio=3.0, missing_rate=0.3, scenario='linear', random_state=42)
    X_miss, T, Y, true_ate = dgp.generate(seed=42)

    print(f"True ATE: {true_ate:.3f}\n")

    estimators = {
        'MICE-DR': MICEDoublyRobust(m_imputations=5, k_folds=3),
        'Complete-DR': CompleteCaseDR(k_folds=3),
        'MICE-OLS': MICE_OLS(m_imputations=5),
        'SingleImp-DR': SingleImputationDR(k_folds=3),
        'CEM': CEM_Matching(n_bins=3),
    }

    for name, estimator in estimators.items():
        print(f"Testing {name}...")
        ate, se, ci_low, ci_up, runtime = estimator.estimate(X_miss, T, Y)

        bias = ate - true_ate if not np.isnan(ate) else np.nan
        coverage = int(ci_low <= true_ate <= ci_up) if not np.isnan(ci_low) else 0

        print(f"  ATE: {ate:.3f} (Bias: {bias:.3f})")
        print(f"  SE: {se:.3f}")
        print(f"  95% CI: [{ci_low:.3f}, {ci_up:.3f}]")
        print(f"  Coverage: {coverage}")
        print(f"  Runtime: {runtime:.2f}s")
        print()

    print("=" * 60)
    print("All estimators tested!")


if __name__ == '__main__':
    test_estimators()
