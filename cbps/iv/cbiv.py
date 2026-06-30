"""
Covariate Balancing Propensity Score for Instrumental Variable Estimates (CBIV)

This module implements the Covariate Balancing Propensity Score (CBPS) methodology
for instrumental variable (IV) settings with treatment noncompliance. CBIV estimates
compliance type probabilities (complier, always-taker, never-taker) using the
generalized method of moments (GMM) framework that simultaneously:

1. **Propensity score estimation**: Fits a multinomial logistic model to predict
   the observed joint distribution of instrument (Z) and treatment (Tr).
2. **Covariate balance optimization**: Ensures that weighted covariate means
   for compliers approximate population means, reducing selection bias.

Principal Stratification Framework
----------------------------------
In IV settings with noncompliance, units are classified into principal strata
based on potential treatment under different instrument values (Angrist et al., 1996):

- Compliers (C): Tr(Z=1)=1, Tr(Z=0)=0 - respond to encouragement
- Always-takers (A): Tr(Z=1)=1, Tr(Z=0)=1 - always treated
- Never-takers (N): Tr(Z=1)=0, Tr(Z=0)=0 - never treated
- Defiers: Tr(Z=1)=0, Tr(Z=0)=1 - excluded by monotonicity assumption

The local average treatment effect (LATE) is identified among compliers. CBIV
provides inverse probability weights (1/π_c) for downstream effect estimation.

Noncompliance Models
--------------------
- **Two-sided** (twosided=True): Models all three compliance types using
  multinomial logistic regression. Appropriate when both always-takers and
  never-takers are present.
- **One-sided** (twosided=False): Models only compliers and never-takers
  using binary logistic regression. Appropriate when always-takers are absent
  (e.g., encouragement designs where treatment access requires encouragement).

References
----------
Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
Journal of the Royal Statistical Society: Series B (Statistical Methodology),
76(1), 243-263. https://doi.org/10.1111/rssb.12027

Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). Identification of
Causal Effects Using Instrumental Variables. Journal of the American
Statistical Association, 91(434), 444-455. https://doi.org/10.1080/01621459.1996.10476902

Hansen, L. P. (1982). Large Sample Properties of Generalized Method of
Moments Estimators. Econometrica, 50(4), 1029-1054. https://doi.org/10.2307/1912775
"""

import warnings
from typing import Optional, Tuple, Dict, Callable

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import statsmodels.api as sm
from statsmodels.genmod.families import Gaussian

from ..utils.numerics import r_ginv_like, pinv_symmetric_psd, symmetrize, numeric_rank

# Constants
PROBS_MIN = 1e-6  # Probability clipping threshold for numerical stability


class CBIVNumericalWarning(UserWarning):
    """
    Warning raised for numerical stability issues during CBIV estimation.

    This warning is issued when compliance probabilities approach extreme values
    and require clipping to the interval [probs_min, 1 - probs_min]. Excessive
    clipping may indicate:

    - Complete or quasi-complete separation in the data
    - Weak instrument (low correlation between Z and Tr)
    - Insufficient sample size relative to the number of covariates
    - Extreme covariate values causing numerical overflow

    When this warning appears, consider:

    1. Checking instrument relevance (first-stage F-statistic > 10)
    2. Reducing the number of covariates or using regularization
    3. Examining the data for outliers or extreme values
    4. Increasing sample size if possible
    """
    pass


class CBIVSummary:
    """Summary object for CBIVResults.

    Encapsulates formatted summary information of CBIV estimation results,
    providing a statsmodels-style summary() return type. Use ``str()`` or
    ``print()`` to obtain formatted text.

    Parameters
    ----------
    coefficients : np.ndarray
        Estimated coefficients.
    fitted_values : np.ndarray
        Fitted compliance type probabilities.
    weights : np.ndarray
        Inverse probability weights (1/π_c).
    deviance : float
        Model deviance.
    converged : bool
        Whether optimization converged.
    J : float
        Hansen J-statistic.
    df : int
        Effective degrees of freedom.
    bal : float
        Covariate balance loss.
    method : str
        Estimation method.
    two_sided : bool
        Whether two-sided noncompliance model was used.
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        fitted_values: np.ndarray,
        weights: np.ndarray,
        deviance: float,
        converged: bool,
        J: float,
        df: int,
        bal: float,
        method: str,
        two_sided: bool,
    ):
        self.coefficients = coefficients
        self.fitted_values = fitted_values
        self.weights = weights
        self.deviance = deviance
        self.converged = converged
        self.J = J
        self.df = df
        self.bal = bal
        self.method = method
        self.two_sided = two_sided

    def __str__(self) -> str:
        """Return formatted CBIV estimation results summary (consistent with CBIVResults.__str__() output)."""
        output = "\nCBIV Estimation Results\n"
        output += "=" * 60 + "\n"

        # Basic information
        n = len(self.weights)
        k = self.coefficients.shape[0]
        output += f"Sample size: {n}\n"
        output += f"Coefficients: {k}\n"
        output += f"Method: {self.method}\n"
        output += f"Two-sided noncompliance: {'Yes' if self.two_sided else 'No'}\n"
        output += f"Converged: {'Yes' if self.converged else 'No'}\n"

        # Statistics
        output += f"\nModel Statistics:\n"
        output += f"  J-statistic: {self.J:.6f}\n"
        output += f"  Deviance: {self.deviance:.6f}\n"
        output += f"  Balance loss: {self.bal:.6f}\n"
        output += f"  Degrees of freedom: {self.df}\n"

        # Fitted values information
        if self.fitted_values is not None:
            if self.two_sided:
                output += f"\nCompliance Probabilities (π_c, π_a, π_n):\n"
                output += f"  Compliers (π_c): min={self.fitted_values[:, 0].min():.4f}, max={self.fitted_values[:, 0].max():.4f}, mean={self.fitted_values[:, 0].mean():.4f}\n"
                output += f"  Always-takers (π_a): min={self.fitted_values[:, 1].min():.4f}, max={self.fitted_values[:, 1].max():.4f}, mean={self.fitted_values[:, 1].mean():.4f}\n"
                output += f"  Never-takers (π_n): min={self.fitted_values[:, 2].min():.4f}, max={self.fitted_values[:, 2].max():.4f}, mean={self.fitted_values[:, 2].mean():.4f}\n"
            else:
                output += f"\nComplier Probabilities (π_c):\n"
                output += f"  Min: {self.fitted_values.min():.4f}\n"
                output += f"  Max: {self.fitted_values.max():.4f}\n"
                output += f"  Mean: {self.fitted_values.mean():.4f}\n"

        # Weight information
        if self.weights is not None:
            output += f"\nComplier Weights (1/π_c):\n"
            output += f"  Min: {self.weights.min():.4f}\n"
            output += f"  Max: {self.weights.max():.4f}\n"
            output += f"  Mean: {self.weights.mean():.4f}\n"

        output += "=" * 60 + "\n"

        return output

    def __repr__(self) -> str:
        n = len(self.weights)
        return f"CBIVSummary(n={n}, converged={self.converged})"


class CBIVResults:
    """
    Container for CBIV estimation results.

    Stores all outputs from the CBIV fitting procedure, including compliance
    type coefficients, estimated probabilities, inverse probability weights,
    and diagnostic statistics for model assessment.

    Attributes
    ----------
    coefficients : np.ndarray
        Estimated coefficients for the compliance type model.

        - Two-sided noncompliance: shape (k, 2) matrix where column 0 contains
          complier coefficients (β_c) and column 1 contains always-taker
          coefficients (β_a). Never-taker probability is the reference category.
        - One-sided noncompliance: shape (k,) vector of complier coefficients.

    fitted_values : np.ndarray
        Estimated compliance type probabilities for each observation.

        - Two-sided: shape (n, 3) matrix with columns [π_c, π_a, π_n] representing
          complier, always-taker, and never-taker probabilities respectively.
          Each row sums to 1.
        - One-sided: shape (n, 1) matrix of complier probabilities π_c.

    weights : np.ndarray
        Inverse probability weights for compliers, computed as 1/π_c.
        Shape (n,). These weights can be used for downstream LATE estimation.

    deviance : float
        Model deviance, computed as -2 times the log-likelihood. Lower values
        indicate better fit to the observed (Z, Tr) distribution.

    converged : bool
        Whether the optimization algorithm converged successfully. If False,
        results should be interpreted with caution.

    J : float
        Hansen's J-statistic for the over-identification test. Under correct
        model specification, J ~ χ²(df) asymptotically. Large J values suggest
        model misspecification or violation of IV assumptions.

    df : int
        Effective degrees of freedom, equal to the rank of the covariate matrix.
        Used as the degrees of freedom for the J-statistic test.

    bal : float
        Covariate balance loss, computed as the GMM objective using only the
        balance moment conditions. Lower values indicate better balance.

    method : str
        Estimation method used: 'over' (over-identified), 'exact' (just-identified),
        or 'mle' (maximum likelihood only).

    two_sided : bool
        Whether two-sided noncompliance model was fitted.

    iterations : int or None
        Maximum number of optimization iterations specified.

    Examples
    --------
    >>> import numpy as np
    >>> from cbps import CBIV
    >>> # Simulate data
    >>> n = 500
    >>> np.random.seed(42)
    >>> X = np.random.randn(n, 2)
    >>> Z = np.random.binomial(1, 0.5, n)
    >>> # Compliance depends on X
    >>> p_comply = 1 / (1 + np.exp(-0.5 - 0.3 * X[:, 0]))
    >>> comply = np.random.binomial(1, p_comply, n)
    >>> Tr = Z * comply  # Simplified one-sided model
    >>> # Fit CBIV
    >>> fit = CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=False)
    >>> print(f"Converged: {fit.converged}")
    >>> print(f"J-statistic: {fit.J:.4f}")
    >>> print(f"Mean weight: {fit.weights.mean():.4f}")
    """

    def __init__(
        self,
        coefficients: np.ndarray,
        fitted_values: np.ndarray,
        weights: np.ndarray,
        deviance: float,
        converged: bool,
        J: float,
        df: int,
        bal: float,
        method: str,
        two_sided: bool,
        iterations: Optional[int] = None,
    ):
        """
        Initialize a CBIVResults instance.

        This constructor is typically not called directly. Use the :func:`CBIV`
        function to fit a model and obtain a CBIVResults object.

        Parameters
        ----------
        coefficients : np.ndarray
            Estimated model coefficients. Shape (k, 2) for two-sided or (k,)
            for one-sided noncompliance.
        fitted_values : np.ndarray
            Estimated compliance probabilities. Shape (n, 3) for two-sided
            with columns [π_c, π_a, π_n], or (n, 1) for one-sided.
        weights : np.ndarray
            Inverse probability weights (1/π_c), shape (n,).
        deviance : float
            Model deviance (-2 × log-likelihood).
        converged : bool
            Optimization convergence status.
        J : float
            Hansen's J-statistic for over-identification test.
        df : int
            Effective degrees of freedom (covariate matrix rank).
        bal : float
            Covariate balance loss from GMM objective.
        method : str
            Estimation method: 'over', 'exact', or 'mle'.
        two_sided : bool
            Whether two-sided noncompliance model was used.
        iterations : int, optional
            Maximum optimization iterations specified.
        """
        self.coefficients = coefficients
        self.fitted_values = fitted_values
        self.weights = weights
        self.deviance = deviance
        self.converged = converged
        self.J = J
        self.df = df
        self.bal = bal
        self.method = method
        self.two_sided = two_sided
        self.iterations = iterations

    def __repr__(self) -> str:
        """Concise representation."""
        n = len(self.weights)
        k = self.coefficients.shape[0]
        return (
            f"CBIVResults(n={n}, k={k}, J={self.J:.6f}, " f"converged={self.converged})"
        )
    
    def __str__(self) -> str:
        """Full string representation (for print)."""
        output = "\nCBIV Estimation Results\n"
        output += "=" * 60 + "\n"
        
        # Basic information
        n = len(self.weights)
        k = self.coefficients.shape[0]
        output += f"Sample size: {n}\n"
        output += f"Coefficients: {k}\n"
        output += f"Method: {self.method}\n"
        output += f"Two-sided noncompliance: {'Yes' if self.two_sided else 'No'}\n"
        output += f"Converged: {'Yes' if self.converged else 'No'}\n"
        
        # Statistics
        output += f"\nModel Statistics:\n"
        output += f"  J-statistic: {self.J:.6f}\n"
        output += f"  Deviance: {self.deviance:.6f}\n"
        output += f"  Balance loss: {self.bal:.6f}\n"
        output += f"  Degrees of freedom: {self.df}\n"
        
        # Fitted values information
        if self.fitted_values is not None:
            if self.two_sided:
                output += f"\nCompliance Probabilities (π_c, π_a, π_n):\n"
                output += f"  Compliers (π_c): min={self.fitted_values[:, 0].min():.4f}, max={self.fitted_values[:, 0].max():.4f}, mean={self.fitted_values[:, 0].mean():.4f}\n"
                output += f"  Always-takers (π_a): min={self.fitted_values[:, 1].min():.4f}, max={self.fitted_values[:, 1].max():.4f}, mean={self.fitted_values[:, 1].mean():.4f}\n"
                output += f"  Never-takers (π_n): min={self.fitted_values[:, 2].min():.4f}, max={self.fitted_values[:, 2].max():.4f}, mean={self.fitted_values[:, 2].mean():.4f}\n"
            else:
                output += f"\nComplier Probabilities (π_c):\n"
                output += f"  Min: {self.fitted_values.min():.4f}\n"
                output += f"  Max: {self.fitted_values.max():.4f}\n"
                output += f"  Mean: {self.fitted_values.mean():.4f}\n"
        
        # Weight information
        if self.weights is not None:
            output += f"\nComplier Weights (1/π_c):\n"
            output += f"  Min: {self.weights.min():.4f}\n"
            output += f"  Max: {self.weights.max():.4f}\n"
            output += f"  Mean: {self.weights.mean():.4f}\n"
        
        output += "=" * 60 + "\n"
        
        return output

    @property
    def p_complier(self) -> np.ndarray:
        """
        Complier probability vector.

        Provides a unified interface to access complier probabilities regardless
        of whether the model used two-sided or one-sided noncompliance.

        Returns
        -------
        np.ndarray
            Estimated complier probabilities π_c, shape (n,).

        Notes
        -----
        For two-sided models, extracts the first column of `fitted_values`.
        For one-sided models, flattens `fitted_values` to a 1D array.

        The inverse of these probabilities equals the `weights` attribute:
        ``weights = 1 / p_complier``.

        Examples
        --------
        >>> fit = CBIV(Tr=Tr, Z=Z, X=X, twosided=True)
        >>> p_c = fit.p_complier
        >>> np.allclose(fit.weights, 1.0 / p_c)  # True
        """
        if self.two_sided:
            return self.fitted_values[:, 0]
        else:
            return self.fitted_values.ravel()
    
    def vcov(self) -> np.ndarray:
        """
        Retrieve the variance-covariance matrix of estimated coefficients.

        Returns
        -------
        np.ndarray
            Variance-covariance matrix of coefficients.

            - Two-sided: shape (2k, 2k), joint covariance of stacked vector [β_c; β_a]
            - One-sided: shape (k, k), covariance of β_c

        Raises
        ------
        AttributeError
            If the variance-covariance matrix was not computed during fitting.

        Notes
        -----
        The variance-covariance matrix is computed using the standard GMM
        sandwich formula:

        .. math::

            \\text{Var}(\\hat{\\beta}) = (G' V^{-1} G)^{-1} / n

        where G is the Jacobian of moment conditions with respect to parameters,
        V is the covariance matrix of moment conditions, and n is the sample size.

        The Jacobian G is computed via numerical differentiation. If optimization
        did not converge, the variance estimates may be unreliable.

        Standard errors can be obtained as the square root of the diagonal:
        ``se = np.sqrt(np.diag(fit.vcov()))``.

        References
        ----------
        Hansen, L. P. (1982). Large Sample Properties of Generalized Method of
        Moments Estimators. Econometrica, 50(4), 1029-1054.

        Examples
        --------
        >>> fit = CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=True)
        >>> V = fit.vcov()
        >>> se = np.sqrt(np.diag(V))
        >>> print(f"Standard errors: {se}")
        """
        if not hasattr(self, '_vcov_matrix'):
            raise AttributeError(
                "Variance-covariance matrix not available. "
                "This may occur if computation failed during fitting. "
                "Re-fit the model to attempt computation."
            )
        return self._vcov_matrix
    
    @property
    def var(self) -> np.ndarray:
        """
        Variance-covariance matrix of coefficients (property alias).

        This property provides convenient access to the variance-covariance
        matrix, equivalent to calling :meth:`vcov`.

        Returns
        -------
        np.ndarray
            Variance-covariance matrix, same as ``vcov()``.

        See Also
        --------
        vcov : Method returning the same variance-covariance matrix.
        """
        return self.vcov()

    def summary(self) -> 'CBIVSummary':
        """
        Generate a summary object for CBIV estimation results.

        Returns a ``CBIVSummary`` instance. Use ``str()`` or ``print()``
        to obtain formatted text summary.

        Returns
        -------
        CBIVSummary
            Object containing estimation result summary information.

        Examples
        --------
        >>> fit = CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=False)
        >>> summary = fit.summary()
        >>> print(summary)  # Formatted output
        >>> type(summary)   # CBIVSummary
        """
        return CBIVSummary(
            coefficients=self.coefficients,
            fitted_values=self.fitted_values,
            weights=self.weights,
            deviance=self.deviance,
            converged=self.converged,
            J=self.J,
            df=self.df,
            bal=self.bal,
            method=self.method,
            two_sided=self.two_sided,
        )


def CBIV(
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    Tr: np.ndarray | pd.Series | None = None,
    Z: np.ndarray | pd.DataFrame | None = None,
    X: np.ndarray | pd.DataFrame | None = None,
    iterations: int = 1000,
    method: str = "over",
    twostep: bool = True,
    twosided: bool = True,
    probs_min: float = 1e-6,
    warn_clipping: bool = True,
    clipping_warn_threshold: float = 0.05,
    verbose: int = 0,
) -> CBIVResults:
    """
    Covariate Balancing Propensity Score for Instrumental Variable Estimates.

    Estimates compliance type propensity scores in an instrumental variable framework,
    simultaneously optimizing covariate balance and propensity score prediction.

    Parameters
    ----------
    formula : str, optional
        IV formula string in format "treatment ~ covariates | instruments".
        Example: "treat ~ x1 + x2 | z1 + z2"
        - treatment: binary treatment variable (0/1)
        - covariates: pre-treatment covariates (intercept added automatically)
        - instruments: binary instrument variable (0/1)
    data : pd.DataFrame, optional
        DataFrame containing treatment, covariates, and instruments (required with formula).
    Tr : np.ndarray or pd.Series, shape (n,), optional
        Binary treatment variable (0/1) (matrix interface, mutually exclusive with formula).
    Z : np.ndarray or pd.DataFrame, shape (n,) or (n, 1), optional
        Binary instrument variable (0/1, encouragement) (matrix interface).
    X : np.ndarray or pd.DataFrame, shape (n, p), optional
        Pre-treatment covariate matrix (without intercept) (matrix interface).
    iterations : int, default=1000
        Maximum number of optimization iterations.
    method : str, default="over"
        Estimation method:
        - "over": Over-identified model (propensity score + covariate balance conditions)
        - "exact": Exactly-identified model (covariate balance conditions only)
        - "mle": Maximum likelihood estimation (propensity score conditions only)
    twostep : bool, default=True
        GMM estimation mode:
        
        - **True** (default): Two-step GMM estimator.
          Pre-computes weight matrix invV and uses fixed invV during optimization.
          Faster and numerically stable.
        
        - **False**: Continuously updating GMM estimator (Hansen et al. 1996).
          Re-computes invV at each iteration.
          Better finite-sample properties in theory, but 5-10x slower.
    
    twosided : bool, default=True
        Whether to allow two-sided noncompliance:
        - True: Two-sided noncompliance (with always-takers and never-takers)
        - False: One-sided noncompliance (never-takers only, π_a=0)
    probs_min : float, default=1e-6
        Probability clipping boundary. Compliance probabilities are constrained
        to [probs_min, 1-probs_min] interval.
        
        - Default 1e-6 maintains numerical stability
        - Lowering this value may cause numerical instability
        - Raising this value increases bias but improves stability
    warn_clipping : bool, default=True
        Whether to warn when proportion of clipped compliance probabilities
        exceeds threshold.
        
        - True: Issue warning (recommended for transparency)
        - False: Silent operation
        
        This feature helps identify numerical issues such as complete
        or quasi-complete separation.
    clipping_warn_threshold : float, default=0.05
        Minimum clipping proportion to trigger warning (between 0 and 1).
        
        - Default 0.05 means warning when >5% of probabilities are clipped
        - Set to 0.0 to warn on any clipping
        - Set to 1.0 to never warn (equivalent to warn_clipping=False)
    verbose : int, default=0
        Controls output verbosity during optimization.
        
        - 0: Silent mode, only warnings and errors
        - 1: Basic optimization info (iterations, convergence status)
        - 2: Detailed diagnostics (loss, gradients per iteration)

    Returns
    -------
    CBIVResults
        Result object containing coefficients, fitted values, weights, etc.

    Notes
    -----
    **Principal Stratification (Three-State Model)**:
    
    - Compliers: Accept treatment when Z=1, refuse when Z=0
    - Always-takers: Accept treatment regardless of Z
    - Never-takers: Refuse treatment regardless of Z

    **Instrumental Variable (IV) Assumptions**:
    
    CBIV assumes the following conditions are satisfied:
    
    1. **Relevance**: Instrument Z is correlated with treatment Tr.
       Recommend verifying first-stage correlation before using CBIV:
       
       >>> import numpy as np
       >>> corr = np.corrcoef(Z, Tr)[0, 1]
       >>> if abs(corr) < 0.1:
       ...     print("Warning: Weak instrument detected.")
       
       Or run first-stage regression and check F-statistic (should be > 10,
       see Stock & Yogo 2005).
    
    2. **Exclusion Restriction**: Z affects outcome Y only through Tr.
    
    3. **Monotonicity**: No defiers (individuals who refuse treatment when
       Z=1 but accept when Z=0).
    
    Weak instruments lead to finite-sample bias and inference failure.
    See Staiger & Stock (1997) and Stock & Yogo (2005) for weak IV diagnostics.
    
    **Optimization Behavior**:
    
    - CBIV executes silently by default
    - Uses multiple starting points internally to find optimal solution
    - Falls back to random initialization with warning if GLM initialization fails

    Examples
    --------
    **Basic usage with matrix interface:**

    >>> import numpy as np
    >>> from cbps import CBIV
    >>> # Simulate IV data with noncompliance
    >>> np.random.seed(42)
    >>> n = 500
    >>> X = np.random.randn(n, 2)  # Covariates
    >>> Z = np.random.binomial(1, 0.5, n)  # Instrument (randomized)
    >>> # Generate compliance: Pr(comply | X) depends on X
    >>> p_comply = 1 / (1 + np.exp(-0.5 - 0.3 * X[:, 0]))
    >>> comply = np.random.binomial(1, p_comply, n)
    >>> Tr = Z * comply  # Treatment = instrument * compliance
    >>> # Fit CBIV model (one-sided noncompliance)
    >>> fit = CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=False)
    >>> print(f"Converged: {fit.converged}")
    >>> print(f"J-statistic: {fit.J:.4f}")

    **Formula interface (recommended for DataFrames):**

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'treat': Tr, 'z': Z, 'x1': X[:, 0], 'x2': X[:, 1]
    ... })
    >>> fit = CBIV(formula="treat ~ x1 + x2 | z", data=df,
    ...            method='over', twosided=False)

    **Two-sided noncompliance model:**

    >>> # When both always-takers and never-takers are present
    >>> fit = CBIV(Tr=Tr, Z=Z, X=X, method='over', twosided=True)
    >>> print(fit.fitted_values.shape)  # (n, 3): [π_c, π_a, π_n]
    >>> print(fit.coefficients.shape)   # (k, 2): [β_c, β_a]

    **Estimation methods:**

    >>> # Over-identified (default): combines balance + score conditions
    >>> fit_over = CBIV(Tr=Tr, Z=Z, X=X, method='over')
    >>> # Exactly-identified: balance conditions only
    >>> fit_exact = CBIV(Tr=Tr, Z=Z, X=X, method='exact')
    >>> # Maximum likelihood: score conditions only
    >>> fit_mle = CBIV(Tr=Tr, Z=Z, X=X, method='mle')

    **Accessing results:**

    >>> fit = CBIV(Tr=Tr, Z=Z, X=X, twosided=False)
    >>> weights = fit.weights           # IPW weights for LATE estimation
    >>> p_c = fit.p_complier            # Complier probabilities
    >>> se = np.sqrt(np.diag(fit.vcov()))  # Standard errors

    **Input Interfaces**:

    CBIV supports two input methods:

    1. **Formula interface**: ``CBIV(formula="treat ~ x1 + x2 | z", data=df)``
       Uses patsy for formula parsing. Intercept is added automatically.

    2. **Matrix interface**: ``CBIV(Tr=Tr, Z=Z, X=X)``
       Pass arrays directly. Note that an intercept column is added internally.

    The interfaces are mutually exclusive; specify either formula+data or Tr+Z+X.

    See Also
    --------
    CBIVResults : Result container returned by this function.

    References
    ----------
    Imai, K. and Ratkovic, M. (2014). Covariate Balancing Propensity Score.
    Journal of the Royal Statistical Society: Series B (Statistical Methodology),
    76(1), 243-263. https://doi.org/10.1111/rssb.12027

    Angrist, J. D., Imbens, G. W., and Rubin, D. B. (1996). Identification of
    Causal Effects Using Instrumental Variables. Journal of the American
    Statistical Association, 91(434), 444-455.

    Staiger, D. and Stock, J. H. (1997). Instrumental Variables Regression with
    Weak Instruments. Econometrica, 65(3), 557-586.

    Stock, J. H. and Yogo, M. (2005). Testing for Weak Instruments in Linear IV
    Regression. In: Andrews, D. W. K. and Stock, J. H. (eds.) Identification and
    Inference for Econometric Models: Essays in Honor of Thomas Rothenberg.
    Cambridge University Press, pp. 80-108.
    """
    # ========== Step 1: Parameter Validation and Interface Selection ==========
    
    # Import required modules
    import pandas as pd
    
    # Mutual exclusivity check: formula and matrix interface cannot be used together
    if formula is not None and (Tr is not None or Z is not None or X is not None):
        raise ValueError(
            "Cannot specify both 'formula' and matrix parameters (Tr/Z/X). "
            "Please use either:\n"
            "  1. Formula interface: CBIV(formula='treat ~ x1 + x2 | z1 + z2', data=df)\n"
            "  2. Matrix interface: CBIV(Tr=treat_array, Z=z_array, X=X_matrix)\n"
            f"\nReceived:\n"
            f"  formula = {repr(formula)}\n"
            f"  Tr = {'<provided>' if Tr is not None else 'None'}\n"
            f"  Z = {'<provided>' if Z is not None else 'None'}\n"
            f"  X = {'<provided>' if X is not None else 'None'}"
        )
    
    # Parameter completeness check
    if formula is None and (Tr is None or Z is None or X is None):
        raise ValueError(
            "Must provide either:\n"
            "  1. Formula interface: formula + data\n"
            "  2. Matrix interface: Tr + Z + X\n"
            f"\nReceived:\n"
            f"  formula = {repr(formula)}\n"
            f"  Tr = {'<provided>' if Tr is not None else 'None'}\n"
            f"  Z = {'<provided>' if Z is not None else 'None'}\n"
            f"  X = {'<provided>' if X is not None else 'None'}"
        )
    
    # Formula interface: requires data parameter
    if formula is not None and data is None:
        raise ValueError(
            "data parameter is required when using formula interface. "
            "Please provide a pandas DataFrame containing the variables in your formula.\n"
            f"Example: CBIV(formula='{formula}', data=your_dataframe)"
        )
    
    # Formula interface: validate data type
    if formula is not None and not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame when using formula interface. "
            f"Got: {type(data).__name__}. "
            f"If you have a dict, convert it: pd.DataFrame(your_dict)"
        )
    
    # Formula interface: validate formula type and format
    if formula is not None:
        if not isinstance(formula, str):
            raise TypeError(
                f"formula must be a string, got {type(formula).__name__}. "
                f"Received: formula={formula}\n\n"
                f"Example: 'treat ~ x1 + x2 | z1 + z2'"
            )
        
        # Validate formula contains '~' and '|'
        if '~' not in formula:
            raise ValueError(
                f"Formula must contain '~' to separate treatment from covariates. "
                f"Got: '{formula}'. "
                f"Example: 'treat ~ x1 + x2 | z1 + z2'"
            )
        
        if '|' not in formula:
            raise ValueError(
                f"IV formula must contain '|' to separate covariates from instruments. "
                f"Got: '{formula}'. "
                f"Format: 'treatment ~ covariates | instruments'\n"
                f"Example: 'treat ~ x1 + x2 | z1 + z2'"
            )
    
    # ========== Step 2: Formula Parsing (if using formula interface) ==========
    
    if formula is not None:
        # Parse IV formula: "treatment ~ covariates | instruments"
        # Split into three parts
        if '|' not in formula:
            raise ValueError(
                f"IV formula must contain '|' to separate covariates from instruments. "
                f"Got: '{formula}'"
            )
        
        # Split formula
        parts = formula.split('|')
        if len(parts) != 2:
            raise ValueError(
                f"IV formula must have exactly one '|' separator. "
                f"Got: '{formula}'\n"
                f"Format: 'treatment ~ covariates | instruments'"
            )
        
        main_formula = parts[0].strip()  # "treatment ~ covariates"
        instrument_formula = parts[1].strip()  # "instruments"
        
        # Validate main_formula contains '~'
        if '~' not in main_formula:
            raise ValueError(
                f"Main formula part must contain '~'. "
                f"Got: '{main_formula}'\n"
                f"Full formula: '{formula}'"
            )
        
        # Use patsy to parse main formula (treatment ~ covariates)
        from patsy import dmatrices, PatsyError
        
        try:
            # Parse treatment and covariates
            # dmatrices returns (y, X), where y is treatment, X is covariates (with intercept)
            y_matrix, X_matrix = dmatrices(main_formula, data, return_type='dataframe')
            
            # Extract treatment (y_matrix may have multiple columns, take first)
            if y_matrix.shape[1] == 1:
                Tr = y_matrix.iloc[:, 0].to_numpy()
            else:
                # If patsy one-hot encoded categorical treatment, needs reverse transform
                # Simplified handling: take first column
                warnings.warn(
                    f"Treatment variable was one-hot encoded by patsy. Using first column. "
                    f"Consider using matrix interface for more control.",
                    UserWarning
                )
                Tr = y_matrix.iloc[:, 0].to_numpy()
            
            # Extract covariates (X_matrix already includes intercept)
            X = X_matrix.to_numpy()
            
        except PatsyError as e:
            raise ValueError(
                f"Failed to parse formula '{main_formula}': {e}\n"
                f"Please check that all variables exist in the data."
            )
        
        # Parse instruments
        # Construct a simple formula to extract instrument variables
        # Use "0 + instrument_vars" to avoid adding intercept
        instrument_formula_patsy = "0 + " + instrument_formula
        
        try:
            from patsy import dmatrix
            Z_matrix = dmatrix(instrument_formula_patsy, data, return_type='dataframe')
            Z = Z_matrix.to_numpy()
            
            # CBIV only supports single instrument; if multiple columns, use first and warn
            if Z.shape[1] > 1:
                warnings.warn(
                    f"Formula specified {Z.shape[1]} instruments, but CBIV only uses the first one. "
                    f"Multiple instruments are not supported in the current CBIV implementation.",
                    UserWarning
                )
                Z = Z[:, 0:1]  # Keep 2D shape
            
        except PatsyError as e:
            raise ValueError(
                f"Failed to parse instruments '{instrument_formula}': {e}\n"
                f"Please check that all instrument variables exist in the data."
            )
    
    # ========== Step 3: Matrix Interface Parameter Validation ==========
    
    # Validate iterations parameter
    if not isinstance(iterations, (int, np.integer)):
        raise TypeError(
            f"CBIV: iterations must be an integer, got {type(iterations).__name__}. "
            f"Received: iterations={iterations}"
        )
    if iterations < 1:
        raise ValueError(
            f"CBIV: iterations must be ≥1 (at least one optimization step required). "
            f"Received: iterations={iterations}"
        )
    if iterations > 100000:
        warnings.warn(
            f"CBIV: iterations={iterations} is very large and may take a long time. "
            f"Consider using a smaller value (default is 1000).",
            UserWarning
        )
    
    Tr = np.asarray(Tr).ravel()
    Z = np.asarray(Z)
    X = np.asarray(X)

    # Ensure Z and X are 2D arrays
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(Tr)
    # Use shape[0] for dimension check
    if Z.shape[0] != n or X.shape[0] != n:
        raise ValueError(
            f"Tr, Z, X must have same number of rows, got {n}, {Z.shape[0]}, {X.shape[0]}"
        )
    
    # CBIV only uses single instrument variable
    # If Z has multiple columns, warn and use only the first
    if Z.shape[1] > 1:
        warnings.warn(
            f"CBIV: Z has {Z.shape[1]} columns but CBIV only uses the first column as the instrument. "
            f"Multiple instruments are not supported in the current CBIV implementation.",
            UserWarning
        )
    
    # Extract first column as instrument variable (Z becomes (n,) vector)
    Z = Z[:, 0]

    if not np.all(np.isin(Tr, [0, 1])):
        raise ValueError("Tr must be binary (0/1)")

    if not np.all(np.isin(Z, [0, 1])):
        raise ValueError("Z must be binary (0/1)")

    if method not in ["over", "exact", "mle"]:
        raise ValueError(f"method must be 'over', 'exact', or 'mle', got {method}")
    
    # Validate additional parameters
    if not isinstance(probs_min, (int, float)) or probs_min <= 0 or probs_min >= 0.5:
        raise ValueError(
            f"CBIV: probs_min must be in (0, 0.5), got {probs_min}. "
            f"Recommended range: [1e-8, 1e-4]."
        )
    
    if not isinstance(warn_clipping, bool):
        raise TypeError(f"CBIV: warn_clipping must be bool, got {type(warn_clipping).__name__}")
    
    if not isinstance(clipping_warn_threshold, (int, float)) or \
       clipping_warn_threshold < 0 or clipping_warn_threshold > 1:
        raise ValueError(
            f"CBIV: clipping_warn_threshold must be in [0, 1], got {clipping_warn_threshold}"
        )

    # Compute pZ (proportion with Z=1)
    pZ = Z.mean()
    
    # Validate pZ range to avoid division by zero
    if pZ <= 0.0 or pZ >= 1.0:
        if pZ == 0.0:
            raise ValueError(
                f"CBIV: Instrument Z has no variation. All observations have Z=0.\n"
                f"CBIV requires variation in the instrument (0 < P(Z=1) < 1).\n"
                f"This results in pZ=0.0, which causes division by zero in the weighting formula."
            )
        elif pZ == 1.0:
            raise ValueError(
                f"CBIV: Instrument Z has no variation. All observations have Z=1.\n"
                f"CBIV requires variation in the instrument (0 < P(Z=1) < 1).\n"
                f"This results in pZ=1.0, which causes division by zero in the weighting formula."
            )
        else:
            # Should not reach here in theory, but for completeness
            raise ValueError(
                f"CBIV: Invalid pZ value: {pZ}. "
                f"Expected 0 < pZ < 1, but got pZ={pZ}."
            )

    # Set method flags
    score_only = method == "mle"
    bal_only = method == "exact"

    # X matrix preprocessing
    # 1. Add intercept column, remove constant columns
    X = np.column_stack([np.ones(n), X[:, X.std(axis=0) > 0]])

    # 2. Save original X and compute statistics
    X_orig = X.copy()
    x_sd = X[:, 1:].std(axis=0)
    x_mean = X[:, 1:].mean(axis=0)

    # 3. Standardize non-intercept columns
    X[:, 1:] = (X[:, 1:] - x_mean) / x_sd

    # 4. Compute effective degrees of freedom k (matrix rank)
    XtX = X.T @ X
    k = int(np.floor(np.trace(XtX @ r_ginv_like(XtX)) + 0.1))
    
    # Detect covariate collinearity
    p = X.shape[1]
    if k < p:
        raise ValueError(
            f"CBIV: Covariate matrix X is rank-deficient after preprocessing.\n"
            f"Effective rank k={k}, Columns={p}. Perfect collinearity detected.\n"
            f"Please remove linearly dependent covariates before calling CBIV.\n"
            f"Hint: Check for duplicate columns, constant multiples, or linear combinations."
        )

    # ========== Internal Helper Functions ==========
    
    def _check_and_warn_clipping(
        probs_before_clip: np.ndarray,
        probs_after_clip: np.ndarray,
        probs_min: float,
        label: str,
        warn: bool,
        threshold: float,
    ) -> None:
        """
        Detect and warn about probability clipping.
        
        Parameters
        ----------
        probs_before_clip : np.ndarray
            Probabilities before clipping.
        probs_after_clip : np.ndarray
            Probabilities after clipping.
        probs_min : float
            Probability clipping boundary.
        label : str
            Probability type label (e.g., "complier probability").
        warn : bool
            Whether to issue warning.
        threshold : float
            Minimum clipping proportion to trigger warning (0-1).
        """
        if not warn:
            return
        
        n = len(probs_before_clip)
        # Count samples clipped to lower and upper bounds
        clipped_low = np.sum(probs_after_clip <= probs_min)
        clipped_high = np.sum(probs_after_clip >= 1 - probs_min)
        n_clipped = clipped_low + clipped_high
        clipping_rate = n_clipped / n
        
        if clipping_rate > threshold:
            warnings.warn(
                f"\nCBIV Numerical Warning: {n_clipped}/{n} ({clipping_rate:.1%}) {label} "
                f"clipped to bounds [{probs_min:.1e}, {1-probs_min:.1e}].\n"
                f"  - Clipped to lower bound: {clipped_low} ({clipped_low/n:.1%})\n"
                f"  - Clipped to upper bound: {clipped_high} ({clipped_high/n:.1%})\n\n"
                f"This may indicate complete or quasi-complete separation.\n\n"
                f"Recommendations:\n"
                f"  (1) Check data quality and sample size\n"
                f"  (2) Check instrument Z relevance (weak instrument?)\n"
                f"  (3) Consider increasing sample size or reducing covariates\n"
                f"  (4) To adjust tolerance: use probs_min parameter (current={probs_min:.1e})\n"
                f"  (5) To disable this warning: use warn_clipping=False\n",
                CBIVNumericalWarning,
                stacklevel=4
            )

    def _compute_compliance_probs_twosided(
        beta_curr: np.ndarray, 
        probs_min: float = PROBS_MIN,
        warn_clipping: bool = False,
        clipping_warn_threshold: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute three compliance probabilities for two-sided noncompliance.

        Parameters
        ----------
        beta_curr : np.ndarray, shape (2*k,)
            Parameter vector, first k are β_c, last k are β_a.
        probs_min : float
            Probability clipping lower bound.

        Returns
        -------
        probs_c : np.ndarray, shape (n,)
            Complier probability.
        probs_a : np.ndarray, shape (n,)
            Always-taker probability.
        probs_n : np.ndarray, shape (n,)
            Never-taker probability.

        Notes
        -----
        Key steps:
        1. Compute baseline_prob = 1 / (1 + exp(X@β_c) + exp(X@β_a))
        2. Compute three probabilities: π_c, π_a, π_n
        3. Clip to [probs_min, 1-probs_min]
        4. Renormalize to ensure sum equals 1

        Numerical stability: uses log-sum-exp trick to avoid overflow.
        """
        beta_c = beta_curr[:k]
        beta_a = beta_curr[k:]

        # Compute linear predictors
        eta_c = X @ beta_c
        eta_a = X @ beta_a

        # Use log-sum-exp trick to compute softmax (numerically stable)
        # log(1 + exp(eta_c) + exp(eta_a)) = log(exp(0) + exp(eta_c) + exp(eta_a))
        # = max_eta + log(exp(0-max_eta) + exp(eta_c-max_eta) + exp(eta_a-max_eta))
        max_eta = np.maximum(np.maximum(0.0, eta_c), eta_a)
        log_sum = max_eta + np.log(
            np.exp(0.0 - max_eta) + np.exp(eta_c - max_eta) + np.exp(eta_a - max_eta)
        )

        # Compute three probabilities (in log space)
        # probs_c = exp(eta_c) / (1 + exp(eta_c) + exp(eta_a))
        #         = exp(eta_c - log_sum)
        probs_c_raw = np.exp(eta_c - log_sum)
        probs_a_raw = np.exp(eta_a - log_sum)
        probs_n_raw = np.exp(0.0 - log_sum)

        # Clip to [probs_min, 1-probs_min]
        probs_c = np.clip(probs_c_raw, probs_min, 1.0 - probs_min)
        probs_a = np.clip(probs_a_raw, probs_min, 1.0 - probs_min)
        probs_n = np.clip(probs_n_raw, probs_min, 1.0 - probs_min)

        # Renormalize to ensure sum equals 1
        sums = probs_c + probs_a + probs_n
        probs_c = probs_c / sums
        probs_a = probs_a / sums
        probs_n = probs_n / sums
        
        # Check and warn about clipping (only when explicitly passed)
        if warn_clipping:
            _check_and_warn_clipping(probs_c_raw, probs_c, probs_min, 
                                     "complier probability (π_c)", warn_clipping, clipping_warn_threshold)
            _check_and_warn_clipping(probs_a_raw, probs_a, probs_min, 
                                     "always-taker probability (π_a)", warn_clipping, clipping_warn_threshold)
            _check_and_warn_clipping(probs_n_raw, probs_n, probs_min, 
                                     "never-taker probability (π_n)", warn_clipping, clipping_warn_threshold)

        return probs_c, probs_a, probs_n

    def _compute_compliance_probs_onesided(
        beta_curr: np.ndarray, 
        probs_min: float = PROBS_MIN,
        warn_clipping: bool = False,
        clipping_warn_threshold: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute compliance probabilities for one-sided noncompliance.

        Parameters
        ----------
        beta_curr : np.ndarray, shape (k,)
            Parameter vector β_c.
        probs_min : float
            Lower bound for probability clipping.
        warn_clipping : bool
            Whether to issue clipping warnings.
        clipping_warn_threshold : float
            Minimum clipping proportion to trigger warning.

        Returns
        -------
        probs_c : np.ndarray, shape (n,)
            Complier probabilities.
        probs_n : np.ndarray, shape (n,)
            Never-taker probabilities (= 1 - π_c).

        Notes
        -----
        One-sided noncompliance assumes π_a = 0 (no always-takers).
        Formula: π_c = clip(sigmoid(X @ β_c), probs_min, 1-probs_min)

        Uses scipy.special.expit for numerical stability.
        """
        # Use stable sigmoid function (avoids overflow)
        # expit(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
        eta = X @ beta_curr
        probs_c_raw = scipy.special.expit(eta)

        # Clip probabilities
        probs_c = np.clip(probs_c_raw, probs_min, 1.0 - probs_min)

        # π_n = 1 - π_c (one-sided noncompliance, no renormalization needed)
        probs_n = 1.0 - probs_c
        
        # Check and warn about clipping if requested
        if warn_clipping:
            _check_and_warn_clipping(probs_c_raw, probs_c, probs_min, 
                                     "complier probability (π_c)", warn_clipping, clipping_warn_threshold)

        return probs_c, probs_n

    def _gmm_func_twosided(
        beta_curr: np.ndarray, invV: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        GMM objective function for two-sided noncompliance.

        Parameters
        ----------
        beta_curr : np.ndarray, shape (2*k,)
            Parameter vector: first k elements are β_c, last k are β_a.
        invV : np.ndarray, optional
            Inverse of V matrix. If None, it will be computed.

        Returns
        -------
        dict
            Dictionary containing 'loss' and 'invV'.

        Notes
        -----
        GMM loss: loss = gbar' @ invV @ gbar
        - gbar: 6K-dimensional moment condition vector
        - V: 6K × 6K covariance matrix
        - invV: Moore-Penrose pseudoinverse of V
        """
        # Compute three compliance probabilities
        probs_c, probs_a, probs_n = _compute_compliance_probs_twosided(beta_curr)

        # Compute combined probabilities (avoid redundant computation)
        s_ca = probs_c + probs_a  # compliers + always-takers
        s_cn = probs_c + probs_n  # compliers + never-takers

        # ========== Construct gbar vector (6K-dimensional) ==========
        # First 2K: propensity score conditions
        # Note: no eps added for consistency (probability clipping provides sufficient protection)
        t1 = (
            Z * Tr / (1.0 - probs_n)
            + (1.0 - Z) * (1.0 - Tr) / (1.0 - probs_a)
            - 1.0
        ) * probs_c
        t2 = (
            Z * Tr / (1.0 - probs_n) + (1.0 - Z) * Tr / probs_a - 1.0
        ) * probs_a

        g1 = (X.T @ t1) / n  # K-dimensional
        g2 = (X.T @ t2) / n  # K-dimensional

        # Last 4K: covariate balance conditions
        w1 = Z * Tr / (pZ * s_ca) - 1.0
        w2 = (1.0 - Z) * Tr / ((1.0 - pZ) * probs_a) - 1.0
        w3 = Z * (1.0 - Tr) / (pZ * probs_n) - 1.0
        w4 = (1.0 - Z) * (1.0 - Tr) / ((1.0 - pZ) * s_cn) - 1.0

        W = np.column_stack([w1, w2, w3, w4])  # (n, 4)
        w_del = (X.T @ W) / n  # (k, 4)

        # Concatenate gbar (6K-dimensional): [g1, g2, flattened 4 columns of w_del]
        gbar = np.concatenate([g1, g2, w_del.ravel(order="F")])

        # ========== Construct V matrix (6K × 6K) ==========
        if invV is None:
            # Compute all weight vectors
            # Consistent with original implementation, no eps added
            w11 = (
                pZ / (1.0 - probs_n) + (1.0 - pZ) / (1.0 - probs_a) - 1.0
            ) * probs_c**2
            w12 = (pZ / s_ca - 1.0) * probs_a * probs_c
            w13 = probs_c * (1.0 / s_ca - 1.0)
            w14 = -probs_c
            w15 = -probs_c
            w16 = probs_c * (1.0 / s_cn - 1.0)

            w22 = (
                pZ / (1.0 - probs_n) + (1.0 - pZ) / probs_a - 1.0
            ) * probs_a**2
            w23 = probs_a * (1.0 / s_ca - 1.0)
            w24 = probs_a * (1.0 / probs_a - 1.0)
            w25 = -probs_a
            w26 = -probs_a

            w33 = 1.0 / (pZ * s_ca) - 1.0
            w34 = -np.ones(n)
            w35 = -np.ones(n)
            w36 = -np.ones(n)

            w44 = 1.0 / ((1.0 - pZ) * probs_a) - 1.0
            w45 = -np.ones(n)
            w46 = -np.ones(n)

            w55 = 1.0 / (pZ * probs_n) - 1.0
            w56 = -np.ones(n)

            w66 = 1.0 / ((1.0 - pZ) * s_cn) - 1.0

            # Helper function: compute X'diag(w)X
            def XtXw(w: np.ndarray) -> np.ndarray:
                return X.T @ (X * w[:, None])

            # Compute all K × K blocks
            B11 = XtXw(w11) / n
            B12 = XtXw(w12) / n
            B13 = XtXw(w13) / n
            B14 = XtXw(w14) / n
            B15 = XtXw(w15) / n
            B16 = XtXw(w16) / n

            B22 = XtXw(w22) / n
            B23 = XtXw(w23) / n
            B24 = XtXw(w24) / n
            B25 = XtXw(w25) / n
            B26 = XtXw(w26) / n

            B33 = XtXw(w33) / n
            B34 = XtXw(w34) / n
            B35 = XtXw(w35) / n
            B36 = XtXw(w36) / n

            B44 = XtXw(w44) / n
            B45 = XtXw(w45) / n
            B46 = XtXw(w46) / n

            B55 = XtXw(w55) / n
            B56 = XtXw(w56) / n

            B66 = XtXw(w66) / n

            # Construct 6 × 6 block matrix (exploiting symmetry)
            V = np.block(
                [
                    [B11, B12, B13, B14, B15, B16],
                    [B12.T, B22, B23, B24, B25, B26],
                    [B13.T, B23.T, B33, B34, B35, B36],
                    [B14.T, B24.T, B34.T, B44, B45, B46],
                    [B15.T, B25.T, B35.T, B45.T, B55, B56],
                    [B16.T, B26.T, B36.T, B46.T, B56.T, B66],
                ]
            )
            # Numerical symmetrization to prevent floating-point errors
            V = symmetrize(V)
            # Warn if V is rank-deficient (only when invV is first constructed)
            r = numeric_rank(V)
            if r < V.shape[0]:
                warnings.warn(
                    f"GMM weighting matrix V is rank-deficient (rank={r} < {V.shape[0]}). Using pseudoinverse.",
                    RuntimeWarning,
                )
            # Compute pseudoinverse using specialized version for symmetric (semi-)positive definite matrices
            invV = pinv_symmetric_psd(V)

        # Compute GMM loss
        loss = float(gbar @ invV @ gbar)

        return {"loss": loss, "invV": invV}

    def _gmm_func_onesided(
        beta_curr: np.ndarray, invV: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        GMM objective function for one-sided noncompliance.

        Parameters
        ----------
        beta_curr : np.ndarray, shape (k,)
            Parameter vector β_c
        invV : np.ndarray, optional
            Inverse of V matrix. If None, will be computed.

        Returns
        -------
        dict
            Dictionary containing 'loss' and 'invV'

        Notes
        -----
        GMM loss: loss = gbar' @ invV @ gbar
        - gbar: 3K-dimensional moment conditions vector
        - V: 3K×3K covariance matrix
        - invV: Moore-Penrose pseudoinverse of V
        """
        # Compute two compliance probabilities
        probs_c, probs_n = _compute_compliance_probs_onesided(beta_curr)

        # ========== Construct gbar vector (3K-dimensional) ==========
        # First K: propensity score conditions
        g1 = (X.T @ (Tr * Z * (1.0 - probs_c) - Z * (1.0 - Tr) * probs_c)) / n

        # Next 2K: covariate balance conditions
        w1 = Z * Tr / (pZ * probs_c) - 1.0
        w2 = Z * (1.0 - Tr) / (pZ * probs_n) - 1.0

        W = np.column_stack([w1, w2])  # (n, 2)
        w_del = (X.T @ W) / n  # (k, 2)

        # Concatenate gbar (3K-dimensional): [g1, flattened w_del columns]
        gbar = np.concatenate([g1, w_del.ravel(order="F")])

        # ========== Construct V matrix (3K×3K) ==========
        if invV is None:
            # Compute all weight vectors
            w11 = pZ * probs_c * (1.0 - probs_c)
            w12 = 1.0 - probs_c
            w13 = -probs_c

            w22 = 1.0 / (pZ * probs_c) - 1.0
            w23 = -np.ones(n)

            w33 = 1.0 / (pZ * probs_n) - 1.0

            # Helper function: compute X'diag(w)X
            def XtXw(w: np.ndarray) -> np.ndarray:
                return X.T @ (X * w[:, None])

            # Compute all K×K blocks
            B11 = XtXw(w11) / n
            B12 = XtXw(w12) / n
            B13 = XtXw(w13) / n

            B22 = XtXw(w22) / n
            B23 = XtXw(w23) / n

            B33 = XtXw(w33) / n

            # Construct 3×3 block matrix (exploiting symmetry)
            V = np.block([[B11, B12, B13], [B12.T, B22, B23], [B13.T, B23.T, B33]])
            # Numerical symmetrization to prevent FP errors from breaking symmetry
            V = symmetrize(V)
            # Warn if V is rank-deficient (only on first invV construction)
            r = numeric_rank(V)
            if r < V.shape[0]:
                warnings.warn(
                    f"GMM weighting matrix V is rank-deficient (rank={r} < {V.shape[0]}). Using pseudoinverse.",
                    RuntimeWarning,
                )
            # Compute pseudoinverse using specialized version for symmetric (semi-)positive definite matrices
            invV = pinv_symmetric_psd(V)

        # Compute GMM loss
        loss = float(gbar @ invV @ gbar)

        return {"loss": loss, "invV": invV}

    def _mle_gradient_twosided(beta_curr: np.ndarray) -> np.ndarray:
        """
        MLE gradient for two-sided noncompliance.

        Returns a 2K-length gradient vector.

        Uses numerically stable log-sum-exp trick to avoid exp overflow.
        """
        beta_c = beta_curr[:k]
        beta_a = beta_curr[k:]

        # Compute log probabilities (numerically stable)
        logit_c = X @ beta_c
        logit_a = X @ beta_a

        # log(1 + exp(logit_c) + exp(logit_a)) using log-sum-exp
        max_logit = np.maximum(np.maximum(logit_c, logit_a), 0.0)
        log_sum = max_logit + np.log(
            np.exp(-max_logit) + np.exp(logit_c - max_logit) + np.exp(logit_a - max_logit)
        )

        # Probability = exp(logit) / (1 + exp(logit_c) + exp(logit_a))
        log_probs_c = logit_c - log_sum
        log_probs_a = logit_a - log_sum
        log_probs_n = -log_sum

        probs_c = np.exp(log_probs_c)
        probs_a = np.exp(log_probs_a)
        probs_n = np.exp(log_probs_n)

        # Clip and normalize
        probs_c = np.clip(probs_c, PROBS_MIN, 1 - PROBS_MIN)
        probs_a = np.clip(probs_a, PROBS_MIN, 1 - PROBS_MIN)
        probs_n = np.clip(probs_n, PROBS_MIN, 1 - PROBS_MIN)

        sums = probs_c + probs_a + probs_n
        probs_c = probs_c / sums
        probs_a = probs_a / sums
        probs_n = probs_n / sums

        # Gradient computation
        # Probabilities are already clipped to [PROBS_MIN, 1-PROBS_MIN],
        # so denominators are safe. No eps needed (consistent with R).
        grad_c = -X.T @ ((Z * Tr / (probs_c + probs_a) + (1 - Z) * (1 - Tr) / (1 - probs_a) - 1) * probs_c)
        grad_a = -X.T @ ((Z * Tr / (probs_c + probs_a) + (1 - Z) * Tr / probs_a - 1) * probs_a)

        return np.concatenate([grad_c, grad_a])

    def _mle_gradient_onesided(beta_curr: np.ndarray) -> np.ndarray:
        """
        MLE gradient for one-sided noncompliance.

        Returns a K-length gradient vector.
        """
        # Compute probabilities
        probs = scipy.special.expit(X @ beta_curr)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)

        # Gradient computation
        # Probabilities are already clipped to [PROBS_MIN, 1-PROBS_MIN],
        # so denominators are safe. No eps needed (consistent with R).
        grad = -X.T @ ((Z * Tr / probs - Z * (1 - Tr) / (1 - probs)) * probs * (1 - probs))

        return grad

    def _bal_gradient_twosided(beta_curr: np.ndarray, invV: np.ndarray) -> np.ndarray:
        """
        Balance gradient for two-sided noncompliance.

        Returns a 2K-length gradient vector.
        """
        beta_c = beta_curr[:k]
        beta_a = beta_curr[k:]

        # Compute probabilities using numerically stable log-sum-exp trick
        logit_c = X @ beta_c
        logit_a = X @ beta_a

        # log(1 + exp(logit_c) + exp(logit_a)) using log-sum-exp
        max_logit = np.maximum(np.maximum(logit_c, logit_a), 0.0)
        log_sum = max_logit + np.log(
            np.exp(-max_logit) + np.exp(logit_c - max_logit) + np.exp(logit_a - max_logit)
        )

        # Probability = exp(logit) / (1 + exp(logit_c) + exp(logit_a))
        log_probs_c = logit_c - log_sum
        log_probs_a = logit_a - log_sum
        log_probs_n = -log_sum

        probs_c = np.exp(log_probs_c)
        probs_a = np.exp(log_probs_a)
        probs_n = np.exp(log_probs_n)

        # Clip and normalize
        probs_c = np.clip(probs_c, PROBS_MIN, 1 - PROBS_MIN)
        probs_a = np.clip(probs_a, PROBS_MIN, 1 - PROBS_MIN)
        probs_n = np.clip(probs_n, PROBS_MIN, 1 - PROBS_MIN)

        sums = probs_c + probs_a + probs_n
        probs_c = probs_c / sums
        probs_a = probs_a / sums
        probs_n = probs_n / sums

        # Compute intermediate variables
        Ac = -probs_c * probs_n / ((probs_c + probs_a) ** 2)
        Bc = probs_c / probs_a
        Cc = -probs_c * probs_a / ((1 - probs_a) ** 2)
        Dc = probs_c / probs_n
        Aa = -probs_a * probs_n / ((probs_c + probs_a) ** 2)
        Ba = -(1 - probs_a) / probs_a
        Ca = probs_a / (1 - probs_a)
        Da = probs_a / probs_n

        # Compute weight matrix
        w_curr = np.column_stack([
            Z * Tr / (pZ * (probs_c + probs_a)) - 1,
            (1 - Z) * Tr / ((1 - pZ) * probs_a) - 1,
            Z * (1 - Tr) / (pZ * probs_n) - 1,
            (1 - Z) * (1 - Tr) / ((1 - pZ) * (probs_c + probs_n)) - 1
        ])

        w_curr_del = (1.0 / n) * X.T @ w_curr
        wbar = w_curr_del.ravel(order='F')

        # Compute derivative of weights w.r.t. beta_c
        dw_beta_c = (1.0 / n) * np.column_stack([
            (X * (Z * Tr / pZ * Ac)[:, None]).T @ X,
            (X * ((1 - Z) * Tr / (1 - pZ) * Bc)[:, None]).T @ X,
            (X * (Z * (1 - Tr) / pZ * Dc)[:, None]).T @ X,
            (X * ((1 - Z) * (1 - Tr) / (1 - pZ) * Cc)[:, None]).T @ X
        ])

        # Compute derivative of weights w.r.t. beta_a
        dw_beta_a = (1.0 / n) * np.column_stack([
            (X * (Z * Tr / pZ * Aa)[:, None]).T @ X,
            (X * ((1 - Z) * Tr / (1 - pZ) * Ba)[:, None]).T @ X,
            (X * (Z * (1 - Tr) / pZ * Da)[:, None]).T @ X,
            (X * ((1 - Z) * (1 - Tr) / (1 - pZ) * Ca)[:, None]).T @ X
        ])

        # Extract the last 4K×4K sub-block of invV
        invV_sub = invV[2 * k:, 2 * k:]

        # Compute gradient
        out_1 = 2 * dw_beta_c @ invV_sub @ wbar
        out_2 = 2 * dw_beta_a @ invV_sub @ wbar

        return np.concatenate([out_1, out_2])

    def _bal_gradient_onesided(beta_curr: np.ndarray, invV: np.ndarray) -> np.ndarray:
        """
        Balance gradient for one-sided noncompliance.

        Returns a K-length gradient vector.
        """
        # Compute probabilities
        probs = scipy.special.expit(X @ beta_curr)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)

        # Compute weight matrix
        w_curr = np.column_stack([
            Z * Tr / (pZ * probs) - 1,
            Z * (1 - Tr) / (pZ * (1 - probs)) - 1
        ])

        w_curr_del = (1.0 / n) * X.T @ w_curr
        wbar = w_curr_del.ravel(order='F')

        # Compute derivative of weights w.r.t. beta
        dw_beta = (1.0 / n) * np.column_stack([
            (X * (-Z * Tr * (1 - probs) / (pZ * probs))[:, None]).T @ X,
            (X * (Z * (1 - Tr) * probs / (pZ * (1 - probs)))[:, None]).T @ X
        ])

        # Extract the last 2K×2K sub-block of invV
        invV_sub = invV[k:, k:]

        # Compute gradient
        grad = 2 * dw_beta @ invV_sub @ wbar

        return grad

    def _gmm_gradient_twosided(beta_curr: np.ndarray, invV: np.ndarray) -> np.ndarray:
        """
        GMM gradient for two-sided noncompliance.

        Returns a 2K-length gradient vector.
        """
        beta_c = beta_curr[:k]
        beta_a = beta_curr[k:]

        # Compute probabilities using numerically stable log-sum-exp trick
        logit_c = X @ beta_c
        logit_a = X @ beta_a

        # log(1 + exp(logit_c) + exp(logit_a)) using log-sum-exp
        max_logit = np.maximum(np.maximum(logit_c, logit_a), 0.0)
        log_sum = max_logit + np.log(
            np.exp(-max_logit) + np.exp(logit_c - max_logit) + np.exp(logit_a - max_logit)
        )

        # Probability = exp(logit) / (1 + exp(logit_c) + exp(logit_a))
        log_probs_c = logit_c - log_sum
        log_probs_a = logit_a - log_sum
        log_probs_n = -log_sum

        probs_c = np.exp(log_probs_c)
        probs_a = np.exp(log_probs_a)
        probs_n = np.exp(log_probs_n)

        # Clip and normalize
        probs_c = np.clip(probs_c, PROBS_MIN, 1 - PROBS_MIN)
        probs_a = np.clip(probs_a, PROBS_MIN, 1 - PROBS_MIN)
        probs_n = np.clip(probs_n, PROBS_MIN, 1 - PROBS_MIN)

        sums = probs_c + probs_a + probs_n
        probs_c = probs_c / sums
        probs_a = probs_a / sums
        probs_n = probs_n / sums

        # Probability clipping provides sufficient protection
        s_ca = probs_c + probs_a
        s_cn = probs_c + probs_n

        # Compute weight matrix
        w_curr = np.column_stack([
            Z * Tr / (pZ * s_ca) - 1,
            (1 - Z) * Tr / ((1 - pZ) * probs_a) - 1,
            Z * (1 - Tr) / (pZ * probs_n) - 1,
            (1 - Z) * (1 - Tr) / ((1 - pZ) * s_cn) - 1
        ])

        w_curr_del = (1.0 / n) * X.T @ w_curr

        # Compute gbar
        gbar = np.concatenate([
            (1.0 / n) * X.T @ ((Z * Tr / (1 - probs_n) + (1 - Z) * (1 - Tr) / (1 - probs_a) - 1) * probs_c),
            (1.0 / n) * X.T @ ((Z * Tr / (1 - probs_n) + (1 - Z) * Tr / probs_a - 1) * probs_a),
            w_curr_del.ravel('F')  # Use Fortran order for consistency with objective function
        ])

        # Compute intermediate variables
        Ac = -probs_c * probs_n / (s_ca ** 2)
        Bc = probs_c / probs_a
        Cc = -probs_c * probs_a / ((1 - probs_a) ** 2)
        Dc = probs_c / probs_n
        Aa = -probs_a * probs_n / (s_ca ** 2)
        Ba = -(1 - probs_a) / probs_a
        Ca = probs_a / (1 - probs_a)
        Da = probs_a / probs_n

        # Compute dgbar
        # First row: derivative w.r.t. beta_c
        dgbar_c1 = (X * (probs_c * (Z * Tr * Ac + (1 - Z) * (1 - Tr) * Cc +
                                     (Z * Tr / s_ca + (1 - Z) * (1 - Tr) / (1 - probs_a) - 1) *
                                     (1 - probs_c)))[:, None]).T @ X
        dgbar_c2 = (X * (probs_a * (Z * Tr * Ac + (1 - Z) * Tr * Bc -
                                     (Z * Tr / s_ca + (1 - Z) * Tr / probs_a - 1) *
                                     probs_c))[:, None]).T @ X
        dgbar_c3 = (X * (Z * Tr / pZ * Ac)[:, None]).T @ X
        dgbar_c4 = (X * ((1 - Z) * Tr / (1 - pZ) * Bc)[:, None]).T @ X
        dgbar_c5 = (X * (Z * (1 - Tr) / pZ * Dc)[:, None]).T @ X
        dgbar_c6 = (X * ((1 - Z) * (1 - Tr) / (1 - pZ) * Cc)[:, None]).T @ X

        dgbar_c = np.column_stack([dgbar_c1, dgbar_c2, dgbar_c3, dgbar_c4, dgbar_c5, dgbar_c6])

        # Second row: derivative w.r.t. beta_a
        dgbar_a1 = (X * (probs_c * (Z * Tr * Aa + (1 - Z) * (1 - Tr) * Ca -
                                     (Z * Tr / s_ca + (1 - Z) * (1 - Tr) / (1 - probs_a) - 1) *
                                     probs_a))[:, None]).T @ X
        dgbar_a2 = (X * (probs_a * (Z * Tr * Aa + (1 - Z) * Tr * Ba +
                                     (Z * Tr / s_ca + (1 - Z) * Tr / probs_a - 1) *
                                     (1 - probs_a)))[:, None]).T @ X
        dgbar_a3 = (X * (Z * Tr / pZ * Aa)[:, None]).T @ X
        dgbar_a4 = (X * ((1 - Z) * Tr / (1 - pZ) * Ba)[:, None]).T @ X
        dgbar_a5 = (X * (Z * (1 - Tr) / pZ * Da)[:, None]).T @ X
        dgbar_a6 = (X * ((1 - Z) * (1 - Tr) / (1 - pZ) * Ca)[:, None]).T @ X

        dgbar_a = np.column_stack([dgbar_a1, dgbar_a2, dgbar_a3, dgbar_a4, dgbar_a5, dgbar_a6])

        dgbar = (1.0 / n) * np.vstack([dgbar_c, dgbar_a])

        # Compute gradient: 2 * dgbar %*% invV %*% gbar
        grad = 2 * dgbar @ invV @ gbar

        return grad

    def _gmm_gradient_onesided(beta_curr: np.ndarray, invV: np.ndarray) -> np.ndarray:
        """
        GMM gradient for one-sided noncompliance.

        Returns a K-length gradient vector.
        """
        # Compute probabilities
        probs = scipy.special.expit(X @ beta_curr)
        probs = np.clip(probs, PROBS_MIN, 1 - PROBS_MIN)

        # Probability clipping provides sufficient protection
        # Compute weight matrix
        w_curr = np.column_stack([
            Z * Tr / (pZ * probs) - 1,
            Z * (1 - Tr) / (pZ * (1 - probs)) - 1
        ])

        w_curr_del = (1.0 / n) * X.T @ w_curr

        # Compute gbar
        # gbar is a 3k-dimensional vector: [gbar_part1, w_curr_del[:,0], w_curr_del[:,1]]
        gbar = np.concatenate([
            (1.0 / n) * X.T @ (Z * (Tr - probs)),
            w_curr_del.ravel('F')  # Fortran order (column-major)
        ])

        # Compute dgbar: (k, 3k) matrix containing three (k, k) blocks
        # Simplification: -Z*Tr - Z*(1-Tr) = -Z
        dgbar_1 = (X * (-Z * probs * (1 - probs))[:, None]).T @ X
        dgbar_2 = (X * (-Z * Tr * (1 - probs) / (pZ * probs))[:, None]).T @ X
        dgbar_3 = (X * (Z * (1 - Tr) * probs / (pZ * (1 - probs)))[:, None]).T @ X

        dgbar = (1.0 / n) * np.column_stack([dgbar_1, dgbar_2, dgbar_3])

        # Compute gradient
        grad = 2 * dgbar @ invV @ gbar

        return grad

    def _mle_loss_twosided(beta_curr: np.ndarray) -> float:
        """
        MLE loss function (negative log-likelihood) for two-sided noncompliance.

        Parameters
        ----------
        beta_curr : np.ndarray, shape (2*k,)
            Parameter vector.

        Returns
        -------
        float
            Negative log-likelihood.
        """
        # Compute three compliance probabilities using numerically stable log-sum-exp
        beta_c = beta_curr[:k]
        beta_a = beta_curr[k:]

        # Compute log probabilities (numerically stable)
        logit_c = X @ beta_c
        logit_a = X @ beta_a

        # log(1 + exp(logit_c) + exp(logit_a)) using log-sum-exp
        max_logit = np.maximum(np.maximum(logit_c, logit_a), 0.0)
        log_sum = max_logit + np.log(
            np.exp(-max_logit) + np.exp(logit_c - max_logit) + np.exp(logit_a - max_logit)
        )

        # Probability = exp(logit) / (1 + exp(logit_c) + exp(logit_a))
        log_probs_c = logit_c - log_sum
        log_probs_a = logit_a - log_sum
        log_probs_n = -log_sum

        probs_c = np.exp(log_probs_c)
        probs_a = np.exp(log_probs_a)
        # Note: use 1-probs_c-probs_a for probs_n
        probs_n = 1.0 - probs_c - probs_a

        # Clip probabilities
        eps = PROBS_MIN
        probs_c = np.clip(probs_c, eps, 1.0 - eps)
        probs_a = np.clip(probs_a, eps, 1.0 - eps)
        probs_n = np.clip(probs_n, eps, 1.0 - eps)

        # Renormalize
        sums = probs_c + probs_a + probs_n
        probs_c = probs_c / sums
        probs_a = probs_a / sums
        probs_n = probs_n / sums

        # Negative log-likelihood
        # Probabilities are already clipped to [PROBS_MIN, 1-PROBS_MIN] and renormalized,
        # so log arguments are safe. No eps needed (consistent with R).
        loss = -np.sum(
            Z * Tr * np.log(probs_c + probs_a)
            + Z * (1.0 - Tr) * np.log(probs_n)
            + (1.0 - Z) * Tr * np.log(probs_a)
            + (1.0 - Z) * (1.0 - Tr) * np.log(1.0 - probs_a)
        )

        return float(loss)

    def _mle_loss_onesided(beta_curr: np.ndarray) -> float:
        """
        MLE loss function for one-sided noncompliance (negative log-likelihood).

        Parameters
        ----------
        beta_curr : np.ndarray, shape (k,)
            Parameter vector.

        Returns
        -------
        float
            Negative log-likelihood.
        """
        # Compute compliance probabilities using logistic function
        probs_c = scipy.special.expit(X @ beta_curr)
        probs_c = np.clip(probs_c, PROBS_MIN, 1.0 - PROBS_MIN)

        # Negative log-likelihood
        # Probabilities are already clipped to [PROBS_MIN, 1-PROBS_MIN],
        # so log arguments are safe. No eps needed (consistent with R).
        loss = -np.sum(
            Z * Tr * np.log(probs_c)
            + Z * (1.0 - Tr) * np.log(1.0 - probs_c)
        )

        return float(loss)

    def _bal_loss_twosided(beta_curr: np.ndarray, invV: np.ndarray) -> float:
        """
        Balance loss function for two-sided noncompliance.

        Parameters
        ----------
        beta_curr : np.ndarray, shape (2*k,)
            Parameter vector.
        invV : np.ndarray
            Inverse of V matrix (6K×6K).

        Returns
        -------
        float
            Balance loss.

        Notes
        -----
        invV needs to be sliced to the last 4K×4K submatrix.
        """
        # Compute three compliance probabilities
        probs_c, probs_a, probs_n = _compute_compliance_probs_twosided(beta_curr)

        # Compute combined probabilities
        s_ca = probs_c + probs_a
        s_cn = probs_c + probs_n

        # Compute balance weights for two-sided noncompliance
        # No epsilon added to denominators (fitted values already clipped)
        w1 = Z * Tr / (pZ * s_ca) - 1.0
        w2 = (1.0 - Z) * Tr / ((1.0 - pZ) * probs_a) - 1.0
        w3 = Z * (1.0 - Tr) / (pZ * probs_n) - 1.0
        w4 = (1.0 - Z) * (1.0 - Tr) / ((1.0 - pZ) * s_cn) - 1.0

        W = np.column_stack([w1, w2, w3, w4])
        w_del = (X.T @ W) / n  # (k, 4)
        wbar = w_del.ravel(order="F")  # 4K-dimensional

        # Slice invV to last 4K×4K submatrix
        invV_sub = invV[2 * k :, 2 * k :]

        # Compute GMM loss
        loss = float(wbar @ invV_sub @ wbar)

        return loss

    def _bal_loss_onesided(beta_curr: np.ndarray, invV: np.ndarray) -> float:
        """
        Balance loss function for one-sided noncompliance.

        Parameters
        ----------
        beta_curr : np.ndarray, shape (k,)
            Parameter vector.
        invV : np.ndarray
            Inverse of V matrix (3K×3K).

        Returns
        -------
        float
            Balance loss.

        Notes
        -----
        invV needs to be sliced to the last 2K×2K submatrix.
        """
        # Compute probabilities
        probs_c, probs_n = _compute_compliance_probs_onesided(beta_curr)

        # Compute balance weights for one-sided noncompliance
        # No epsilon added to denominators (fitted values already clipped)
        w1 = Z * Tr / (pZ * probs_c) - 1.0
        w2 = Z * (1.0 - Tr) / (pZ * probs_n) - 1.0

        W = np.column_stack([w1, w2])
        w_del = (X.T @ W) / n  # (k, 2)
        wbar = w_del.ravel(order="F")  # 2K-dimensional

        # Slice invV to last 2K×2K submatrix
        invV_sub = invV[k:, k:]

        # Compute GMM loss
        loss = float(wbar @ invV_sub @ wbar)

        return loss

    # ========== Initialization ==========
    
    # Detect perfect or near-perfect compliance
    # When Tr has no variation in a Z subset, GLM fitting will fail
    Z1_mask = Z == 1
    Z0_mask = Z == 0
    
    # Check perfect compliance: variance of Tr in each Z subset
    Tr_Z1_var = np.var(Tr[Z1_mask]) if Z1_mask.sum() > 1 else 0.0
    Tr_Z0_var = np.var(Tr[Z0_mask]) if Z0_mask.sum() > 1 else 0.0
    
    perfect_compliance_Z1 = Tr_Z1_var < 1e-10  # Tr has no variation when Z=1
    perfect_compliance_Z0 = Tr_Z0_var < 1e-10  # Tr has no variation when Z=0
    
    if perfect_compliance_Z1 and perfect_compliance_Z0:
        # Perfect compliance: Tr is completely determined by Z
        warnings.warn(
            "Perfect compliance detected: Tr is completely determined by Z. "
            "All units are compliers (π_c≈1, π_a≈0, π_n≈0). "
            "Using fallback initialization with small random perturbation.",
            UserWarning
        )
        # Use initial values close to perfect compliance
        if twosided:
            # Two-sided: β_c and β_a both approach -∞ (making π_a and π_n approach 0)
            p = X.shape[1]
            beta_c_init = np.random.randn(p) * 0.01 - 5.0  # logit(π_c) ≈ 5, π_c≈0.993
            beta_a_init = np.random.randn(p) * 0.01 - 5.0  # logit(π_a) ≈ -5, π_a≈0.007
            beta_init = np.concatenate([beta_c_init, beta_a_init])
        else:
            # One-sided: β_c approaches +∞ (making π_c approach 1)
            p = X.shape[1]
            beta_init = np.random.randn(p) * 0.01 + 5.0  # logit(π_c) ≈ 5, π_c≈0.993
    elif twosided:
        # Two-sided noncompliance initialization
        # Use try-except to handle possible GLM failures
        try:
            # Step 1: Fit never-takers model on Z=1 subset
            if not perfect_compliance_Z1:
                glm_n = sm.GLM(1 - Tr[Z1_mask], X[Z1_mask], family=Gaussian()).fit()
                beta_n0 = glm_n.params
            else:
                # When Z=1, Tr is all 1, 1-Tr is all 0, use zero vector
                beta_n0 = np.zeros(X.shape[1])

            # Step 2: Fit always-takers model on Z=0 subset
            if not perfect_compliance_Z0:
                glm_a = sm.GLM(Tr[Z0_mask], X[Z0_mask], family=Gaussian()).fit()
                beta_a0 = glm_a.params
            else:
                # When Z=0, Tr is all 0, use zero vector
                beta_a0 = np.zeros(X.shape[1])

            # Step 3: Compute initial compliance probabilities
            # Use numerically stable log-sum-exp trick
            logit_a0 = X @ beta_a0
            logit_n0 = X @ beta_n0

            # log(1 + exp(logit_a0) + exp(logit_n0)) using log-sum-exp
            max_logit = np.maximum(np.maximum(logit_a0, logit_n0), 0.0)
            log_sum = max_logit + np.log(
                np.exp(-max_logit) + np.exp(logit_a0 - max_logit) + np.exp(logit_n0 - max_logit)
            )

            # Probability = exp(logit) / (1 + exp(logit_a0) + exp(logit_n0))
            # Note: here c corresponds to never-taker (logit=0), a to always-taker, n to complier
            # Variable names follow the standard compliance type convention
            p_hat_a0 = np.exp(logit_a0 - log_sum)
            p_hat_n0 = np.exp(logit_n0 - log_sum)
            p_hat_c0 = np.exp(-log_sum)

            # Clip probabilities
            eps = PROBS_MIN
            p_hat_a0 = np.clip(p_hat_a0, eps, 1.0 - eps)
            p_hat_n0 = np.clip(p_hat_n0, eps, 1.0 - eps)
            p_hat_c0 = np.clip(p_hat_c0, eps, 1.0 - eps)

            # Renormalize
            sums = p_hat_c0 + p_hat_a0 + p_hat_n0
            p_hat_c0 = p_hat_c0 / sums
            p_hat_a0 = p_hat_a0 / sums
            p_hat_n0 = p_hat_n0 / sums

            # Step 4: Get initial beta via linear regression
            # Linear regression on log(p/(1-p))
            # Note: p_hat is already clipped to [PROBS_MIN, 1-PROBS_MIN], so no extra eps needed
            logit_c = np.log(p_hat_c0 / (1.0 - p_hat_c0))
            logit_a = np.log(p_hat_a0 / (1.0 - p_hat_a0))

            beta_c_init = np.linalg.lstsq(X, logit_c, rcond=None)[0]
            beta_a_init = np.linalg.lstsq(X, logit_a, rcond=None)[0]

            beta_init = np.concatenate([beta_c_init, beta_a_init])
        except (ValueError, np.linalg.LinAlgError) as e:
            # GLM failed, use fallback initialization
            warnings.warn(
                f"GLM initialization failed ({e}). Using fallback initialization.",
                UserWarning
            )
            p = X.shape[1]
            beta_c_init = np.random.randn(p) * 0.1
            beta_a_init = np.random.randn(p) * 0.1
            beta_init = np.concatenate([beta_c_init, beta_a_init])
    else:
        # One-sided noncompliance initialization
        # Fit logistic regression on Z=1 subset
        if not perfect_compliance_Z1:
            try:
                glm_c = sm.GLM(Tr[Z1_mask], X[Z1_mask], family=Gaussian()).fit()
                beta_init = glm_c.params
            except (ValueError, np.linalg.LinAlgError) as e:
                # GLM failed, use fallback initialization
                warnings.warn(
                    f"GLM initialization failed ({e}). Using fallback initialization.",
                    UserWarning
                )
                beta_init = np.random.randn(X.shape[1]) * 0.1
        else:
            # Tr is all 1 or all 0 when Z=1, cannot fit
            # Use small random values for initialization
            warnings.warn(
                "No variation in Tr when Z=1. Using fallback initialization.",
                UserWarning
            )
            beta_init = np.random.randn(X.shape[1]) * 0.1

    # ========== Optimization Loop ==========

    # Step 1: MLE optimization
    if twosided:
        mle_loss_func = _mle_loss_twosided
        mle_grad_func = _mle_gradient_twosided
    else:
        mle_loss_func = _mle_loss_onesided
        mle_grad_func = _mle_gradient_onesided

    # Verbose output for optimization progress
    if verbose >= 1:
        print(f"\n[CBIV] Starting MLE optimization (iterations={iterations})...")
    
    mle_result = scipy.optimize.minimize(
        mle_loss_func, beta_init, method="BFGS", jac=mle_grad_func,
        options={"maxiter": iterations, "gtol": 1e-8, "disp": verbose >= 2}
    )
    beta_mle = mle_result.x
    mle_converged = mle_result.success
    
    if verbose >= 1:
        print(f"[CBIV] MLE optimization: converged={mle_converged}, nit={mle_result.nit}, loss={mle_result.fun:.6f}")

    # Step 2: Compute inverse weight matrix
    if twosided:
        gmm_result = _gmm_func_twosided(beta_mle, invV=None)
    else:
        gmm_result = _gmm_func_onesided(beta_mle, invV=None)

    this_invV = gmm_result["invV"]

    # ========== Continuously Updating GMM vs Two-Step GMM Branch ==========
    # 
    # Two-step GMM (twostep=True, default):
    #   - Use pre-computed fixed this_invV
    #   - Fast, stable
    # 
    # Continuously updating GMM (twostep=False):
    #   - Re-compute invV at each optimization call (pass invV=None)
    #   - Theoretically better finite-sample properties (Hansen et al. 1996)
    #   - 5-10x slower, may be numerically unstable
    
    if twostep:
        # Two-step GMM: Use pre-computed fixed invV (current behavior)
        if twosided:
            bal_loss_func = lambda b: _bal_loss_twosided(b, this_invV)
            bal_grad_func = lambda b: _bal_gradient_twosided(b, this_invV)
            gmm_loss_func = lambda b: _gmm_func_twosided(b, this_invV)["loss"]
            gmm_grad_func = lambda b: _gmm_gradient_twosided(b, this_invV)
        else:
            bal_loss_func = lambda b: _bal_loss_onesided(b, this_invV)
            bal_grad_func = lambda b: _bal_gradient_onesided(b, this_invV)
            gmm_loss_func = lambda b: _gmm_func_onesided(b, this_invV)["loss"]
            gmm_grad_func = lambda b: _gmm_gradient_onesided(b, this_invV)
    else:
        # Continuously updating GMM: GMM loss function re-computes invV each time (new feature)
        # Note: balance loss still uses fixed invV (because bal_loss function doesn't support invV=None)
        warnings.warn(
            "Using Continuous Updating GMM (twostep=False). "
            "This may be 5-10x slower for CBIV but has better finite sample properties. "
            "See Hansen et al. (1996) for theory.",
            UserWarning
        )
        
        if twosided:
            # Balance loss uses fixed invV (bal_loss function doesn't support dynamic invV)
            bal_loss_func = lambda b: _bal_loss_twosided(b, this_invV)
            bal_grad_func = None  # Continuously updating mode doesn't use analytical gradient
            # GMM loss uses continuous updating (dynamic invV)
            gmm_loss_func = lambda b: _gmm_func_twosided(b, invV=None)["loss"]
            gmm_grad_func = None
        else:
            # Balance loss uses fixed invV
            bal_loss_func = lambda b: _bal_loss_onesided(b, this_invV)
            bal_grad_func = None
            # GMM loss uses continuous updating (dynamic invV)
            gmm_loss_func = lambda b: _gmm_func_onesided(b, invV=None)["loss"]
            gmm_grad_func = None

    # Step 3: Choose final optimization based on method
    if score_only:
        # method="mle": Use MLE result only
        gmm_opt = mle_result
        beta_opt = beta_mle
        if verbose >= 1:
            print(f"[CBIV] Using method='mle', no additional optimization needed")
    elif bal_only:
        # method="exact": Optimize balance loss
        # Try two starting points and select the better result
        
        if verbose >= 1:
            print(f"[CBIV] method='exact': Optimizing balance conditions...")
        
        # Start from beta_init
        bal_init_result = scipy.optimize.minimize(
            bal_loss_func, beta_init, method="BFGS", jac=bal_grad_func,
            options={"maxiter": iterations, "gtol": 1e-8, "disp": verbose >= 2}
        )
        # Start from beta_mle
        bal_mle_result = scipy.optimize.minimize(
            bal_loss_func, beta_mle, method="BFGS", jac=bal_grad_func,
            options={"maxiter": iterations, "gtol": 1e-8, "disp": verbose >= 2}
        )

        # Select better result from two starting points
        if bal_init_result.fun > bal_mle_result.fun:
            gmm_opt = bal_mle_result
            beta_opt = bal_mle_result.x
            if verbose >= 1:
                print(f"[CBIV] Selected MLE starting point: loss={bal_mle_result.fun:.6f}")
        else:
            gmm_opt = bal_init_result
            beta_opt = bal_init_result.x
            if verbose >= 1:
                print(f"[CBIV] Selected init starting point: loss={bal_init_result.fun:.6f}")
    else:
        # method="over": Optimize GMM loss
        # Try two starting points and select the better result

        if verbose >= 1:
            print(f"[CBIV] method='over': Step 1 - Balance optimization...")
        
        # Step 1: Compute beta_bal from two starting points
        # Start from beta_init
        bal_init_result = scipy.optimize.minimize(
            bal_loss_func, beta_init, method="BFGS", jac=bal_grad_func,
            options={"maxiter": iterations, "gtol": 1e-8, "disp": verbose >= 2}
        )
        # Start from beta_mle
        bal_mle_result = scipy.optimize.minimize(
            bal_loss_func, beta_mle, method="BFGS", jac=bal_grad_func,
            options={"maxiter": iterations, "gtol": 1e-8, "disp": verbose >= 2}
        )

        # Select better result from two starting points
        if bal_init_result.fun > bal_mle_result.fun:
            beta_bal = bal_mle_result.x
            if verbose >= 1:
                print(f"[CBIV]   Selected MLE start: loss={bal_mle_result.fun:.6f}")
        else:
            beta_bal = bal_init_result.x
            if verbose >= 1:
                print(f"[CBIV]   Selected init start: loss={bal_init_result.fun:.6f}")

        if verbose >= 1:
            print(f"[CBIV] method='over': Step 2 - GMM optimization...")
        
        # Step 2: GMM optimization from two starting points
        # Start from beta_mle
        gmm_mle_result = scipy.optimize.minimize(
            gmm_loss_func, beta_mle, method="BFGS", jac=gmm_grad_func,
            options={"maxiter": iterations, "gtol": 1e-8, "disp": verbose >= 2}
        )
        # Start from beta_bal
        gmm_bal_result = scipy.optimize.minimize(
            gmm_loss_func, beta_bal, method="BFGS", jac=gmm_grad_func,
            options={"maxiter": iterations, "gtol": 1e-8, "disp": verbose >= 2}
        )

        # Select better result from two starting points
        if gmm_mle_result.fun > gmm_bal_result.fun:
            gmm_opt = gmm_bal_result
            beta_opt = gmm_bal_result.x
            if verbose >= 1:
                print(f"[CBIV]   Selected balance start: converged={gmm_bal_result.success}, loss={gmm_bal_result.fun:.6f}")
        else:
            gmm_opt = gmm_mle_result
            beta_opt = gmm_mle_result.x
            if verbose >= 1:
                print(f"[CBIV]   Selected MLE start: converged={gmm_mle_result.success}, loss={gmm_mle_result.fun:.6f}")

    # ========== Compute Final Statistics ==========

    # Reshape beta to matrix
    if twosided:
        beta_opt_matrix = beta_opt.reshape(k, 2, order="F")  # (k, 2)
    else:
        beta_opt_matrix = beta_opt.reshape(k, 1, order="F")  # (k, 1)

    # Compute J-statistic
    if twosided:
        J_opt = _gmm_func_twosided(beta_opt, this_invV)["loss"]
        bal_loss_opt = _bal_loss_twosided(beta_opt, this_invV)
    else:
        J_opt = _gmm_func_onesided(beta_opt, this_invV)["loss"]
        bal_loss_opt = _bal_loss_onesided(beta_opt, this_invV)

    # ========== Coefficient Inverse Transform and Final Probability Computation ==========

    if twosided:
        # Two-sided noncompliance
        # Step 1: Compute raw probabilities
        # Use numerically stable log-sum-exp trick
        logit_c_opt = X @ beta_opt_matrix[:, 0]
        logit_a_opt = X @ beta_opt_matrix[:, 1]

        # log(1 + exp(logit_c) + exp(logit_a)) using log-sum-exp
        max_logit = np.maximum(np.maximum(logit_c_opt, logit_a_opt), 0.0)
        log_sum = max_logit + np.log(
            np.exp(-max_logit) + np.exp(logit_c_opt - max_logit) + np.exp(logit_a_opt - max_logit)
        )

        # Probability = exp(logit) / (1 + exp(logit_c) + exp(logit_a))
        pi_c_opt_raw = np.exp(logit_c_opt - log_sum)
        pi_a_opt_raw = np.exp(logit_a_opt - log_sum)
        pi_n_opt_raw = np.exp(-log_sum)

        # Step 2: Clip probabilities to avoid numerical issues
        pi_c_opt = np.clip(pi_c_opt_raw, probs_min, 1.0 - probs_min)
        pi_a_opt = np.clip(pi_a_opt_raw, probs_min, 1.0 - probs_min)
        pi_n_opt = np.clip(pi_n_opt_raw, probs_min, 1.0 - probs_min)

        # Step 3: Renormalize to ensure probabilities sum to 1
        sums = pi_c_opt + pi_a_opt + pi_n_opt
        fitted_values = np.column_stack(
            [pi_c_opt / sums, pi_a_opt / sums, pi_n_opt / sums]
        )
        
        # Enhanced: detect and warn about clipping of final fitted_values
        _check_and_warn_clipping(pi_c_opt_raw, fitted_values[:, 0], probs_min,
                                 "final complier probability (π_c)", warn_clipping, clipping_warn_threshold)
        _check_and_warn_clipping(pi_a_opt_raw, fitted_values[:, 1], probs_min,
                                 "final always-taker probability (π_a)", warn_clipping, clipping_warn_threshold)
        _check_and_warn_clipping(pi_n_opt_raw, fitted_values[:, 2], probs_min,
                                 "final never-taker probability (π_n)", warn_clipping, clipping_warn_threshold)

        # Step 4: Coefficient inverse transform
        # Divide coefficients except intercept by standard deviation
        beta_opt_matrix[1:, :] = beta_opt_matrix[1:, :] / x_sd[:, None]

        # Intercept adjustment
        if k > 2:
            # Matrix multiplication for intercept adjustment
            beta_opt_matrix[0, :] = beta_opt_matrix[0, :] - (
                x_mean @ beta_opt_matrix[1:, :]
            )
        else:
            # Scalar multiplication for intercept adjustment
            beta_opt_matrix[0, :] = (
                beta_opt_matrix[0, :] - x_mean * beta_opt_matrix[1, :]
            )

        # Compute deviance (-2 * log-likelihood)
        # Note: No epsilon added inside log argument (fitted_values already clipped)
        deviance = -2.0 * np.sum(
            Z * Tr * np.log(fitted_values[:, 0] + fitted_values[:, 1])
            + Z * (1.0 - Tr) * np.log(fitted_values[:, 2])
            + (1.0 - Z) * Tr * np.log(fitted_values[:, 1])
            + (1.0 - Z) * (1.0 - Tr) * np.log(1.0 - fitted_values[:, 1])
        )

        # Weights: 1/π_c
        weights = 1.0 / fitted_values[:, 0]
    else:
        # One-sided noncompliance
        # Step 1: Compute final probabilities
        # Use stable sigmoid function
        fitted_values_c_raw = scipy.special.expit(X @ beta_opt_matrix[:, 0])
        fitted_values_c = np.clip(fitted_values_c_raw, probs_min, 1.0 - probs_min)
        
        # Return (n, 1) for one-sided
        # Note: Two-sided returns (n, 3) matrix, one-sided returns (n, 1) matrix
        # p_complier property handles both cases uniformly by extracting/flattening
        fitted_values = fitted_values_c[:, None]  # shape (n, 1)
        
        # Enhanced: detect and warn about clipping of final fitted_values
        _check_and_warn_clipping(fitted_values_c_raw, fitted_values, probs_min,
                                 "final complier probability (π_c)", warn_clipping, clipping_warn_threshold)

        # Step 2: Compute deviance
        # Formula: -2*sum(Z*Tr*log(fitted.values) + Z*(1-Tr)*log(1-fitted.values))
        # fitted_values is (n,1), need to flatten to (n,)
        fitted_values_flat = fitted_values[:, 0]
        deviance = -2.0 * np.sum(
            Z * Tr * np.log(fitted_values_flat)
            + Z * (1.0 - Tr) * np.log(1.0 - fitted_values_flat)
        )

        # Step 3: Coefficient inverse transform
        beta_opt_matrix[1:, 0] = beta_opt_matrix[1:, 0] / x_sd
        beta_opt_matrix[0, 0] = beta_opt_matrix[0, 0] - np.sum(
            x_mean * beta_opt_matrix[1:, 0]
        )

        # One-sided returns (k,) vector, not (k, 1) matrix
        # Creates (k,1) matrix
        # No extraction needed, use beta.opt directly
        # coef() method converts to vector
        # Return (k,) vector for one-sided
        beta_opt_vector = beta_opt_matrix.ravel()
        
        # Weights: 1/π_c
        # weights should be (n,) vector for consistency
        # weights are vectors in both implementations
        # Even if fitted_values is (n,1), weights should be (n,)
        weights = (1.0 / fitted_values).ravel()

    # ========== Compute variance-covariance matrix ==========
    # GMM variance formula: Var(β) = (G' invV G)^{-1} / n
    # where G is the Jacobian of moment conditions
    
    def _compute_vcov_matrix(beta_opt: np.ndarray, invV: np.ndarray) -> np.ndarray:
        """
        Compute variance-covariance matrix for GMM estimator
        
        Parameters
        ----------
        beta_opt : np.ndarray
            Optimal parameter vector
        invV : np.ndarray
            Weight matrix (pseudoinverse of V)
            
        Returns
        -------
        np.ndarray
            Variance-covariance matrix
            
        Notes
        -----
        Use numerical differentiation to compute Jacobian G = ∂gbar/∂β
        Then apply GMM variance formula: Var(β) = (G' invV G)^{-1} / n
        """
        # Parameter dimension
        p = len(beta_opt)
        
        # Compute Jacobian using numerical differentiation
        eps = 1e-7
        
        # Compute moment conditions at current parameters
        if twosided:
            gbar_current = _gmm_func_twosided(beta_opt, invV)
            # Extract gbar (recompute to get gbar vector)
            probs_c, probs_a, probs_n = _compute_compliance_probs_twosided(beta_opt)
            s_ca = probs_c + probs_a
            s_cn = probs_c + probs_n
            
            t1 = (Z * Tr / (1.0 - probs_n) + (1.0 - Z) * (1.0 - Tr) / (1.0 - probs_a) - 1.0) * probs_c
            t2 = (Z * Tr / (1.0 - probs_n) + (1.0 - Z) * Tr / probs_a - 1.0) * probs_a
            g1 = (X.T @ t1) / n
            g2 = (X.T @ t2) / n
            
            w1 = Z * Tr / (pZ * s_ca) - 1.0
            w2 = (1.0 - Z) * Tr / ((1.0 - pZ) * probs_a) - 1.0
            w3 = Z * (1.0 - Tr) / (pZ * probs_n) - 1.0
            w4 = (1.0 - Z) * (1.0 - Tr) / ((1.0 - pZ) * s_cn) - 1.0
            W = np.column_stack([w1, w2, w3, w4])
            w_del = (X.T @ W) / n
            
            gbar = np.concatenate([g1, g2, w_del.ravel(order="F")])
        else:
            gbar_current = _gmm_func_onesided(beta_opt, invV)
            # Extract gbar
            probs_c, probs_n = _compute_compliance_probs_onesided(beta_opt)
            g1 = (X.T @ (Tr * Z * (1.0 - probs_c) - Z * (1.0 - Tr) * probs_c)) / n
            
            w1 = Z * Tr / (pZ * probs_c) - 1.0
            w2 = Z * (1.0 - Tr) / (pZ * probs_n) - 1.0
            W = np.column_stack([w1, w2])
            w_del = (X.T @ W) / n
            
            gbar = np.concatenate([g1, w_del.ravel(order="F")])
        
        # Moment condition dimension
        m = len(gbar)
        
        # Initialize Jacobian matrix G (m × p)
        G = np.zeros((m, p))
        
        # Compute numerical derivative for each parameter
        for j in range(p):
            beta_plus = beta_opt.copy()
            beta_plus[j] += eps
            beta_minus = beta_opt.copy()
            beta_minus[j] -= eps
            
            # Compute moment conditions after perturbation
            if twosided:
                probs_c_p, probs_a_p, probs_n_p = _compute_compliance_probs_twosided(beta_plus)
                s_ca_p = probs_c_p + probs_a_p
                s_cn_p = probs_c_p + probs_n_p
                
                t1_p = (Z * Tr / (1.0 - probs_n_p) + (1.0 - Z) * (1.0 - Tr) / (1.0 - probs_a_p) - 1.0) * probs_c_p
                t2_p = (Z * Tr / (1.0 - probs_n_p) + (1.0 - Z) * Tr / probs_a_p - 1.0) * probs_a_p
                g1_p = (X.T @ t1_p) / n
                g2_p = (X.T @ t2_p) / n
                
                w1_p = Z * Tr / (pZ * s_ca_p) - 1.0
                w2_p = (1.0 - Z) * Tr / ((1.0 - pZ) * probs_a_p) - 1.0
                w3_p = Z * (1.0 - Tr) / (pZ * probs_n_p) - 1.0
                w4_p = (1.0 - Z) * (1.0 - Tr) / ((1.0 - pZ) * s_cn_p) - 1.0
                W_p = np.column_stack([w1_p, w2_p, w3_p, w4_p])
                w_del_p = (X.T @ W_p) / n
                
                gbar_plus = np.concatenate([g1_p, g2_p, w_del_p.ravel(order="F")])
                
                probs_c_m, probs_a_m, probs_n_m = _compute_compliance_probs_twosided(beta_minus)
                s_ca_m = probs_c_m + probs_a_m
                s_cn_m = probs_c_m + probs_n_m
                
                t1_m = (Z * Tr / (1.0 - probs_n_m) + (1.0 - Z) * (1.0 - Tr) / (1.0 - probs_a_m) - 1.0) * probs_c_m
                t2_m = (Z * Tr / (1.0 - probs_n_m) + (1.0 - Z) * Tr / probs_a_m - 1.0) * probs_a_m
                g1_m = (X.T @ t1_m) / n
                g2_m = (X.T @ t2_m) / n
                
                w1_m = Z * Tr / (pZ * s_ca_m) - 1.0
                w2_m = (1.0 - Z) * Tr / ((1.0 - pZ) * probs_a_m) - 1.0
                w3_m = Z * (1.0 - Tr) / (pZ * probs_n_m) - 1.0
                w4_m = (1.0 - Z) * (1.0 - Tr) / ((1.0 - pZ) * s_cn_m) - 1.0
                W_m = np.column_stack([w1_m, w2_m, w3_m, w4_m])
                w_del_m = (X.T @ W_m) / n
                
                gbar_minus = np.concatenate([g1_m, g2_m, w_del_m.ravel(order="F")])
            else:
                probs_c_p, probs_n_p = _compute_compliance_probs_onesided(beta_plus)
                g1_p = (X.T @ (Tr * Z * (1.0 - probs_c_p) - Z * (1.0 - Tr) * probs_c_p)) / n
                
                w1_p = Z * Tr / (pZ * probs_c_p) - 1.0
                w2_p = Z * (1.0 - Tr) / (pZ * probs_n_p) - 1.0
                W_p = np.column_stack([w1_p, w2_p])
                w_del_p = (X.T @ W_p) / n
                
                gbar_plus = np.concatenate([g1_p, w_del_p.ravel(order="F")])
                
                probs_c_m, probs_n_m = _compute_compliance_probs_onesided(beta_minus)
                g1_m = (X.T @ (Tr * Z * (1.0 - probs_c_m) - Z * (1.0 - Tr) * probs_c_m)) / n
                
                w1_m = Z * Tr / (pZ * probs_c_m) - 1.0
                w2_m = Z * (1.0 - Tr) / (pZ * probs_n_m) - 1.0
                W_m = np.column_stack([w1_m, w2_m])
                w_del_m = (X.T @ W_m) / n
                
                gbar_minus = np.concatenate([g1_m, w_del_m.ravel(order="F")])
            
            # Numerical derivative: ∂gbar/∂β_j
            G[:, j] = (gbar_plus - gbar_minus) / (2 * eps)
        
        # Compute Variance-covariance matrix: Var(β) = (G' invV G)^{-1} / n
        # Note: invV is already the pseudoinverse of V
        GtinvVG = G.T @ invV @ G
        
        try:
            # Try using specialized inverse for symmetric positive definite matrix
            vcov_matrix = pinv_symmetric_psd(GtinvVG) / n
        except (np.linalg.LinAlgError, ValueError):
            # If failed, use general pseudoinverse
            vcov_matrix = np.linalg.pinv(GtinvVG) / n
        
        # Ensure symmetry
        vcov_matrix = symmetrize(vcov_matrix)
        
        return vcov_matrix
    
    # Compute vcov matrix
    try:
        vcov_matrix = _compute_vcov_matrix(beta_opt, this_invV)
    except Exception as e:
        # If vcov computation fails, warn but do not interrupt
        warnings.warn(
            f"Failed to compute variance-covariance matrix: {e}. "
            f"vcov() method will not be available for this fit.",
            RuntimeWarning
        )
        vcov_matrix = None
    
    # ========== Return value encapsulation ==========
    # Verbose final summary
    if verbose >= 1:
        print(f"\n[CBIV] Optimization complete:")
        print(f"  - Method: {method}")
        print(f"  - Two-sided: {twosided}")
        print(f"  - Converged: {gmm_opt.success}")
        print(f"  - J statistic: {J_opt:.6f}")
        print(f"  - Balance loss: {bal_loss_opt:.6f}")
        if twosided:
            print(f"  - Mean p_complier: {fitted_values[:, 0].mean():.4f}")
            print(f"  - Mean p_always: {fitted_values[:, 1].mean():.4f}")
            print(f"  - Mean p_never: {fitted_values[:, 2].mean():.4f}")
        else:
            print(f"  - Mean p_complier: {fitted_values.mean():.4f}")
    
    # Add method and two_sided fields

    if twosided:
        result = CBIVResults(
            coefficients=beta_opt_matrix,
            fitted_values=fitted_values,
            weights=weights,
            deviance=float(deviance),
            converged=bool(gmm_opt.success),
            J=float(J_opt),
            df=k,
            bal=float(bal_loss_opt),
            method=method,
            two_sided=True,
            iterations=iterations,
        )
        # Add vcov matrix (if computed successfully)
        if vcov_matrix is not None:
            result._vcov_matrix = vcov_matrix
        return result
    else:
        result = CBIVResults(
            coefficients=beta_opt_vector,  # Returns (k,) vector for one-sided
            fitted_values=fitted_values,  # Returns (n, 1) matrix for one-sided
            weights=weights,
            deviance=float(deviance),
            converged=bool(gmm_opt.success),
            J=float(J_opt),
            df=k,
            bal=float(bal_loss_opt),
            method=method,
            two_sided=False,
            iterations=iterations,
        )
        # Add vcov matrix (if computed successfully)
        if vcov_matrix is not None:
            result._vcov_matrix = vcov_matrix
        return result
