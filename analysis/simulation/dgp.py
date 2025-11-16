"""
Data Generating Process for MICE-DR Simulations
================================================

Generates synthetic datasets with:
- High-dimensional covariates (p=10, 50, 100)
- Extreme treatment imbalance (3:1, 10:1, 20:1)
- MAR or MNAR missingness (20%, 40%, 60%)
- Known treatment effect for evaluation

Author: Causal Inference Pipeline Project
Date: 2025-01
"""

import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal


class DGP:
    """Data Generating Process for causal inference simulations"""

    def __init__(
        self,
        n=2000,
        p=50,
        imbalance_ratio=10.0,
        missing_rate=0.4,
        treatment_effect=0.4,
        scenario='linear',
        random_state=None
    ):
        """
        Parameters
        ----------
        n : int
            Sample size (default 2000)
        p : int
            Number of covariates (default 50)
        imbalance_ratio : float
            Control-to-treated ratio (3, 10, or 20)
        missing_rate : float
            Proportion of observations with missing covariates (0.2, 0.4, 0.6)
        treatment_effect : float
            True ATE (default 0.4)
        scenario : str
            'linear', 'nonlinear', or 'mnar'
        random_state : int or None
            Random seed for reproducibility
        """
        self.n = n
        self.p = p
        self.imbalance_ratio = imbalance_ratio
        self.missing_rate = missing_rate
        self.treatment_effect = treatment_effect
        self.scenario = scenario
        self.random_state = random_state

        # Derived parameters
        self.treatment_prevalence = 1 / (1 + imbalance_ratio)
        self.n_confounders_treatment = 5  # First 5 covariates affect treatment
        self.n_confounders_outcome = 10   # First 10 covariates affect outcome

        # Covariate correlation structure
        self.rho = 0.3  # Moderate correlation

        # Calibrate beta_0 to achieve target imbalance
        self._calibrate_treatment_parameters()

    def _calibrate_treatment_parameters(self):
        """
        Find beta_0 such that P(T=1) ≈ treatment_prevalence

        Using logit P(T=1|X) = beta_0 + beta^T X
        When X ~ N(0, I), average PS ≈ expit(beta_0)

        For imbalance_ratio:
        - 3:1   → P(T=1)=0.25 → beta_0 ≈ -1.1
        - 10:1  → P(T=1)=0.09 → beta_0 ≈ -2.4
        - 20:1  → P(T=1)=0.05 → beta_0 ≈ -3.0
        """
        calibration_map = {
            3.0: -1.10,
            10.0: -2.40,
            20.0: -2.94
        }

        if self.imbalance_ratio in calibration_map:
            self.beta_0 = calibration_map[self.imbalance_ratio]
        else:
            # Approximate for other ratios
            target_prob = self.treatment_prevalence
            self.beta_0 = np.log(target_prob / (1 - target_prob))

        # Treatment coefficients (only first 5 covariates matter)
        self.beta_treatment = np.zeros(self.p)
        self.beta_treatment[:self.n_confounders_treatment] = [0.5, 0.5, 0.3, 0.3, 0.2]

    def generate(self, seed=None):
        """
        Generate one complete dataset

        Parameters
        ----------
        seed : int or None
            Random seed (overrides self.random_state if provided)

        Returns
        -------
        X_miss : ndarray, shape (n, p)
            Covariate matrix with missing values (np.nan)
        T : ndarray, shape (n,)
            Binary treatment indicator
        Y : ndarray, shape (n,)
            Continuous outcome
        true_ate : float
            True average treatment effect (for evaluation)
        """
        if seed is None:
            seed = self.random_state

        rng = np.random.RandomState(seed)

        # Step 1: Generate covariates
        X = self._generate_covariates(rng)

        # Step 2: Generate treatment
        T = self._generate_treatment(X, rng)

        # Step 3: Generate outcome
        Y = self._generate_outcome(X, T, rng)

        # Step 4: Induce missingness
        X_miss = self._induce_missingness(X, Y, T, rng)

        return X_miss, T, Y, self.treatment_effect

    def _generate_covariates(self, rng):
        """
        Generate X ~ MVN(0, Σ) with moderate correlation

        Covariance matrix: Σ_ij = ρ^|i-j| (AR(1) structure)
        """
        # Create correlation matrix (AR1 structure)
        indices = np.arange(self.p)
        Sigma = self.rho ** np.abs(indices[:, None] - indices[None, :])

        # Generate multivariate normal
        X = rng.multivariate_normal(mean=np.zeros(self.p), cov=Sigma, size=self.n)

        return X

    def _generate_treatment(self, X, rng):
        """
        Generate binary treatment T ~ Bernoulli(e(X))

        Propensity score:
        logit P(T=1|X) = beta_0 + beta_1*X_1 + ... + beta_5*X_5
        """
        # Propensity score (only first 5 covariates matter)
        logit_ps = self.beta_0 + X @ self.beta_treatment
        ps = expit(logit_ps)

        # Sample treatment
        T = rng.binomial(n=1, p=ps, size=self.n)

        return T

    def _generate_outcome(self, X, T, rng):
        """
        Generate continuous outcome Y

        Linear scenario:
        Y = gamma^T X + tau * T + epsilon

        Nonlinear scenario:
        Y = gamma^T X + tau * T + 0.3 * T * X_1 + 0.2 * X_1^2 + epsilon
        """
        # Outcome coefficients (first 10 covariates are confounders)
        gamma_outcome = np.zeros(self.p)
        gamma_outcome[:self.n_confounders_outcome] = [
            0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1
        ]

        # Base outcome (linear in X)
        Y0 = X @ gamma_outcome + rng.normal(0, 1, size=self.n)

        # Treatment effect
        if self.scenario == 'linear':
            # Constant treatment effect
            treatment_effect_i = self.treatment_effect
        elif self.scenario == 'nonlinear':
            # Heterogeneous treatment effect + interactions
            treatment_effect_i = (
                self.treatment_effect +
                0.3 * X[:, 0] +  # Treatment-covariate interaction
                0.2 * X[:, 0]**2  # Nonlinearity
            )
        else:  # MNAR scenario uses same DGP as linear
            treatment_effect_i = self.treatment_effect

        Y1 = Y0 + treatment_effect_i

        # Observed outcome
        Y = T * Y1 + (1 - T) * Y0

        return Y

    def _induce_missingness(self, X, Y, T, rng):
        """
        Induce missing values in X

        MAR (Missing at Random):
        logit P(R_ij = 1 | X, Y, T) = alpha_0 + alpha_Y * Y + alpha_X * X_1

        MNAR (Missing Not at Random):
        logit P(R_ij = 1 | X_j) = alpha_0 + alpha_X * X_j (missing depends on itself)

        Missingness applies to X[:, 1:] only (keep X_0 always observed)
        Treatment T and outcome Y are always observed
        """
        X_miss = X.copy()

        if self.missing_rate == 0:
            return X_miss

        if self.scenario == 'mnar':
            # MNAR: Missingness depends on the variable itself
            for j in range(1, self.p):  # Keep X_0 always observed
                alpha_0 = -np.log((1 - self.missing_rate) / self.missing_rate)
                alpha_self = 0.5

                logit_miss = alpha_0 + alpha_self * X[:, j]
                prob_miss = expit(logit_miss)
                missing_mask = rng.binomial(n=1, p=prob_miss, size=self.n).astype(bool)

                X_miss[missing_mask, j] = np.nan
        else:
            # MAR: Missingness depends on Y and X_0
            # Target: missing_rate proportion of rows have at least one missing value

            # Calibrate alpha_0 to achieve target missing rate
            alpha_0 = -1.0  # Baseline (will adjust)
            alpha_Y = 0.3
            alpha_X1 = 0.3

            # Standardize Y and X[:, 0] for stability
            Y_std = (Y - Y.mean()) / Y.std()
            X0_std = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()

            logit_miss_row = alpha_0 + alpha_Y * Y_std + alpha_X1 * X0_std
            prob_miss_row = expit(logit_miss_row)

            # Sample which rows will have missingness
            rows_with_missing = rng.binomial(n=1, p=prob_miss_row, size=self.n).astype(bool)

            # Adjust to match exact target missing rate
            current_rate = rows_with_missing.mean()
            if current_rate < self.missing_rate:
                # Add more missing rows randomly
                n_to_add = int((self.missing_rate - current_rate) * self.n)
                available = np.where(~rows_with_missing)[0]
                if len(available) > 0:
                    add_idx = rng.choice(available, size=min(n_to_add, len(available)), replace=False)
                    rows_with_missing[add_idx] = True
            elif current_rate > self.missing_rate:
                # Remove some missing rows
                n_to_remove = int((current_rate - self.missing_rate) * self.n)
                available = np.where(rows_with_missing)[0]
                if len(available) > 0:
                    remove_idx = rng.choice(available, size=min(n_to_remove, len(available)), replace=False)
                    rows_with_missing[remove_idx] = False

            # For rows with missing, randomly set 30-70% of covariates to missing
            # (Keep X_0 always observed)
            for i in np.where(rows_with_missing)[0]:
                n_miss_vars = rng.randint(
                    int(0.3 * (self.p - 1)),
                    int(0.7 * (self.p - 1)) + 1
                )
                miss_vars = rng.choice(range(1, self.p), size=n_miss_vars, replace=False)
                X_miss[i, miss_vars] = np.nan

        return X_miss

    def get_info(self):
        """Return DGP configuration as dict"""
        actual_prevalence = expit(self.beta_0)  # When X=0

        return {
            'n': self.n,
            'p': self.p,
            'imbalance_ratio': self.imbalance_ratio,
            'target_prevalence': self.treatment_prevalence,
            'actual_prevalence_at_X0': actual_prevalence,
            'missing_rate': self.missing_rate,
            'treatment_effect': self.treatment_effect,
            'scenario': self.scenario,
            'beta_0': self.beta_0,
            'n_confounders_treatment': self.n_confounders_treatment,
            'n_confounders_outcome': self.n_confounders_outcome
        }


def test_dgp():
    """Test DGP with various parameter combinations"""
    print("Testing DGP class...")
    print("=" * 60)

    test_configs = [
        {'n': 1000, 'p': 20, 'imbalance_ratio': 3.0, 'missing_rate': 0.2, 'scenario': 'linear'},
        {'n': 2000, 'p': 50, 'imbalance_ratio': 10.0, 'missing_rate': 0.4, 'scenario': 'linear'},
        {'n': 500, 'p': 10, 'imbalance_ratio': 20.0, 'missing_rate': 0.6, 'scenario': 'nonlinear'},
    ]

    for i, config in enumerate(test_configs, 1):
        print(f"\nTest {i}: {config}")
        dgp = DGP(**config, random_state=42)

        # Generate 10 datasets
        treatment_rates = []
        missing_rates = []
        outcome_means = []

        for seed in range(10):
            X_miss, T, Y, true_ate = dgp.generate(seed)

            treatment_rates.append(T.mean())
            missing_rates.append(np.isnan(X_miss).any(axis=1).mean())
            outcome_means.append(Y.mean())

        print(f"  Treatment rate: {np.mean(treatment_rates):.3f} ± {np.std(treatment_rates):.3f}")
        print(f"    (Target: {dgp.treatment_prevalence:.3f})")
        print(f"  Missing rate: {np.mean(missing_rates):.3f} ± {np.std(missing_rates):.3f}")
        print(f"    (Target: {config['missing_rate']:.3f})")
        print(f"  Outcome mean: {np.mean(outcome_means):.3f} ± {np.std(outcome_means):.3f}")
        print(f"  True ATE: {true_ate:.3f}")

        # Check dimensions
        assert X_miss.shape == (config['n'], config['p'])
        assert T.shape == (config['n'],)
        assert Y.shape == (config['n'],)
        assert abs(np.mean(treatment_rates) - dgp.treatment_prevalence) < 0.05, \
            f"Treatment rate {np.mean(treatment_rates):.3f} far from target {dgp.treatment_prevalence:.3f}"
        assert abs(np.mean(missing_rates) - config['missing_rate']) < 0.10, \
            f"Missing rate {np.mean(missing_rates):.3f} far from target {config['missing_rate']:.3f}"

        print("  ✓ Passed")

    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == '__main__':
    test_dgp()
