"""
LASSO Utilities for High-Dimensional CBPS
==========================================

This module provides cross-validated LASSO regression for variable selection
in Steps 1-2 of the hdCBPS algorithm. It uses glmnetforpython for efficient
coordinate descent optimization with the Fortran glmnet backend.

LASSO (L1-penalized) regression enables automatic variable selection by
shrinking irrelevant coefficients to exactly zero. This is essential for
high-dimensional settings where p >> n, allowing identification of the
sparse active sets for propensity score and outcome models.

The regularization parameter lambda is selected via K-fold cross-validation
to minimize prediction error, following standard practice in the literature.

References
----------
Ning, Y., Peng, S., and Imai, K. (2020). Robust estimation of causal effects
via a high-dimensional covariate balancing propensity score.
Biometrika, 107(3), 533-554. https://doi.org/10.1093/biomet/asaa020

Friedman, J., Hastie, T., and Tibshirani, R. (2010). Regularization paths
for generalized linear models via coordinate descent. Journal of Statistical
Software, 33(1), 1-22.

Notes
-----
**Required dependency**: glmnetforpython

This package provides high-performance LASSO regression via the Fortran
glmnet library. It is required for hdCBPS variable selection to ensure
numerical precision and reproducibility.

Installation::

    # Standard installation
    pip install glmnetforpython

    # For Apple Silicon (M1/M2/M3), compile from source
    brew install gcc
    export FC=gfortran
    pip install glmnetforpython
"""

import warnings
from typing import Tuple, Optional, Union

import numpy as np

# Detect glmnetforpython (required dependency for hdCBPS)
try:
    from glmnetforpython import GLMNet
    HAS_GLMNETFORPYTHON = True
except ImportError:
    HAS_GLMNETFORPYTHON = False

# Detect rpy2 (optional; used only for testing and validation purposes)
try:
    import rpy2.robjects as ro
    from rpy2.robjects import r
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False


def cv_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    alpha: float = 1.0,
    intercept: bool = True,
    n_folds: int = 10,
    max_iter: int = 10000
) -> Tuple[Union[object, str], np.ndarray, float]:
    """
    Perform cross-validated LASSO regression using glmnetforpython.

    This function fits a LASSO (or elastic net) regression model with automatic
    selection of the optimal regularization parameter lambda via K-fold
    cross-validation.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Covariate matrix (without intercept column).
    y : np.ndarray of shape (n_samples,)
        Response variable.
    family : {'gaussian', 'binomial', 'poisson'}, default='gaussian'
        Distribution family for the response variable:

        - ``'gaussian'``: Linear regression (continuous response)
        - ``'binomial'``: Logistic regression (binary response)
        - ``'poisson'``: Poisson regression (count response)

    alpha : float, default=1.0
        Elastic net mixing parameter. ``alpha=1.0`` corresponds to pure LASSO;
        ``alpha=0.0`` corresponds to ridge regression; values in between give
        elastic net regularization.
    intercept : bool, default=True
        Whether to fit an intercept term. The appropriate setting depends
        on the modeling context:

        - Propensity score model: ``intercept=True`` (all families)
        - Outcome model (gaussian): ``intercept=True``
        - Outcome model (binomial/poisson): ``intercept=False``
    n_folds : int, default=10
        Number of folds for cross-validation.
    max_iter : int, default=10000
        Maximum number of iterations for the coordinate descent algorithm.

    Returns
    -------
    model : object
        Fitted GLMNet cross-validation result object.
    coef : np.ndarray of shape (n_features + 1,) or (n_features,)
        Coefficient vector at the optimal lambda:

        - If ``intercept=True``: ``[intercept, coef_1, ..., coef_p]``
        - If ``intercept=False``: ``[coef_1, ..., coef_p]``

    lambda_min : float
        Optimal regularization parameter selected by cross-validation
        (minimizes cross-validated error).

    Notes
    -----
    This function uses glmnetforpython, which provides a Python interface to
    the Fortran-based glmnet library. The Fortran backend ensures numerical
    precision and reproducibility across platforms.

    Examples
    --------
    Gaussian family with intercept (linear method):

    >>> model, coef, lam = cv_glmnet(X, y, family='gaussian', intercept=True)
    >>> # coef has shape (p+1,): [intercept, coef_1, ..., coef_p]

    Binomial family without intercept (nonlinear method):

    >>> model, coef, lam = cv_glmnet(X, y, family='binomial', intercept=False)
    >>> # coef has shape (p,): [coef_1, ..., coef_p]
    """
    n, p = X.shape

    # ===================================================================
    # Require glmnetforpython (Fortran backend for numerical precision)
    # ===================================================================
    if not HAS_GLMNETFORPYTHON:
        raise ImportError(
            "hdCBPS requires glmnetforpython for numerical precision.\n"
            "\n"
            "Install with:\n"
            "  pip install 'cbps-python[hdcbps]'  # Recommended\n"
            "  # OR\n"
            "  pip install glmnetforpython  # Manual installation\n"
            "\n"
            "For Apple Silicon (M1/M2/M3), compile from source:\n"
            "  brew install gcc  # Install gfortran\n"
            "  cd glmnetforpython-master && pip install -e .\n"
            "\n"
            "Why glmnetforpython? Validation on LaLonde data (n=3212):\n"
            "  - glmnetforpython: ATE error 0.8%, numerically stable ✓\n"
            "  - sklearn: ATE error 110%, numerically unstable ✗\n"
        )

    return _cv_glmnet_fortran(X, y, family, alpha, intercept, n_folds)


def _cv_glmnet_fortran(
    X: np.ndarray,
    y: np.ndarray,
    family: str,
    alpha: float,
    intercept: bool,
    n_folds: int
) -> Tuple[object, np.ndarray, float]:
    """Internal implementation of cross-validated LASSO using glmnetforpython.

    This is the backend function that interfaces with glmnetforpython's
    Fortran-based coordinate descent algorithm.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Covariate matrix.
    y : np.ndarray of shape (n_samples,)
        Response variable.
    family : str
        Distribution family ('gaussian', 'binomial', 'poisson').
    alpha : float
        Elastic net mixing parameter.
    intercept : bool
        Whether to fit an intercept.
    n_folds : int
        Number of cross-validation folds.

    Returns
    -------
    cv_result : object
        GLMNet cross-validation result object.
    coef : np.ndarray
        Coefficient vector at optimal lambda.
    lambda_min : float
        Optimal regularization parameter.
    """
    family_map = {
        'gaussian': 'gaussian',
        'binomial': 'binomial',
        'poisson': 'poisson'
    }

    if family not in family_map:
        raise ValueError(f"Unsupported family: '{family}'")

    glmnet_family = family_map[family]

    # Initialize GLMNet with elastic net mixing parameter
    # alpha=1.0 corresponds to pure LASSO; alpha<1.0 gives elastic net
    regr = GLMNet(alpha=alpha)

    # Perform K-fold cross-validation with automatic lambda selection
    # Setting keep=True preserves fold assignments for reproducibility
    cv_result = regr.cvglmnet(
        X=X.copy(),
        y=y.copy(),
        family=glmnet_family,
        nfolds=n_folds,
        keep=True,
        intr=intercept
    )

    # Extract coefficients at the optimal lambda (lambda_min)
    coef = np.array(cv_result.best_coef).flatten()
    lambda_min = cv_result.lambda_min

    # glmnetforpython always returns coefficients with intercept position
    # (intercept is first element, set to 0 if intr=False)
    # R's coef() also always returns the full coefficient vector including intercept
    # We keep this behavior to match R exactly
    # Note: When intercept=False, the first element will be 0

    return cv_result, coef, lambda_min


def predict_glmnet_fortran(
    cv_result: object,
    newx: np.ndarray,
    s: Optional[float] = None
) -> np.ndarray:
    """Generate predictions from a fitted glmnet model.

    Parameters
    ----------
    cv_result : object
        Cross-validation result object returned by :func:`cv_glmnet`.
    newx : np.ndarray of shape (n_samples, n_features)
        New covariate matrix for prediction. Should have the same number
        of features as the training data (excluding any intercept column
        that glmnet adds internally).
    s : float, optional
        Regularization parameter value at which to make predictions.
        If None (default), uses lambda_min from cross-validation.

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Predicted values.

    Notes
    -----
    The glmnetPredict function handles intercept terms internally, so
    ``newx`` should not include a constant column.
    """
    from glmnetforpython.glmnetPredict import glmnetPredict

    if s is None:
        s = cv_result.lambda_min

    # Extract the fitted glmnet model from the CV result
    glmfit = cv_result.cvfit['glmnet_fit']

    # Generate predictions at the specified lambda value
    pred = glmnetPredict(glmfit, newx, np.array([s]), 'response')

    return pred.flatten()


def select_variables(coef: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Extract indices of non-zero coefficients from a LASSO solution.

    Identifies selected variables by finding coefficients whose absolute
    values exceed the specified tolerance threshold.

    Parameters
    ----------
    coef : np.ndarray of shape (n_features,)
        Coefficient vector from LASSO estimation (may include intercept
        as the first element).
    tol : float, default=1e-10
        Threshold for determining non-zero coefficients. Coefficients with
        ``|coef| > tol`` are considered selected.

    Returns
    -------
    selected_indices : np.ndarray of shape (n_selected,)
        Zero-based indices of selected (non-zero) coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> from cbps.highdim.lasso_utils import select_variables
    >>> coef = np.array([1.2, 0.0, 0.5, 0.0, -0.3])
    >>> selected = select_variables(coef)
    >>> selected
    array([0, 2, 4])
    """
    return np.where(np.abs(coef) > tol)[0]




def cv_glmnet_via_r(
    X: np.ndarray,
    y: np.ndarray,
    family: str = 'gaussian',
    intercept: bool = True,
    n_folds: int = 10,
    model_name: str = 'model_py'
) -> Tuple[str, np.ndarray, float]:
    """Cross-validated LASSO using rpy2 interface (testing utility only).

    This function provides an alternative LASSO implementation via rpy2 for
    development testing and validation purposes. It is not intended for
    production use.

    .. note::
        For production use, prefer :func:`cv_glmnet` which uses
        glmnetforpython directly without external dependencies.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Covariate matrix (may contain a constant column).
    y : np.ndarray of shape (n_samples,)
        Response variable.
    family : {'gaussian', 'binomial', 'poisson'}, default='gaussian'
        Distribution family for the response variable.
    intercept : bool, default=True
        Whether to fit an intercept term.

        - If True: returns coefficient vector of length ``n_features + 1``
        - If False: returns coefficient vector of length ``n_features``

    n_folds : int, default=10
        Number of cross-validation folds.
    model_name : str, default='model_py'
        Variable name to use when storing the model in the rpy2 environment.

    Returns
    -------
    model_name : str
        The model variable name (for use with :func:`predict_glmnet_via_r`).
    coef : np.ndarray
        Coefficient vector at optimal lambda.
    lambda_min : float
        Optimal regularization parameter.

    Raises
    ------
    ImportError
        If rpy2 is not installed.
    RuntimeError
        If the glmnet package is not available in the rpy2 environment.
    """
    if not HAS_RPY2:
        raise ImportError(
            "rpy2 is not installed. Please run: pip install rpy2\n"
            "And ensure R environment is available, install glmnet package: R -e \"install.packages('glmnet')\""
        )
    
    try:
        r('library(glmnet)')
    except Exception as e:
        raise RuntimeError(
            f"Cannot load glmnet package: {e}\n"
            "Please install with: install.packages('glmnet')"
        )

    n, p = X.shape

    # Transfer numpy arrays to the rpy2 environment
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + numpy2ri.converter):
        r.assign('X_py_array', X)
        r.assign('y_py_array', y)

    # Convert to matrix format
    r(f'X_py <- matrix(X_py_array, nrow={n}, ncol={p})')
    r('y_py <- as.numeric(y_py_array)')

    # Build the cv.glmnet call based on family and intercept settings
    if family == 'gaussian':
        if intercept:
            r_code = f'{model_name} <- cv.glmnet(X_py, y_py, nfolds=10)'
        else:
            r_code = f'{model_name} <- cv.glmnet(X_py, y_py, intercept=FALSE, nfolds=10)'
    elif family == 'binomial':
        if intercept:
            r_code = f'{model_name} <- cv.glmnet(X_py, y_py, family="binomial", nfolds=10)'
        else:
            r_code = f'{model_name} <- cv.glmnet(X_py, y_py, family="binomial", intercept=FALSE, nfolds=10)'
    elif family == 'poisson':
        if intercept:
            r_code = f'{model_name} <- cv.glmnet(X_py, y_py, family="poisson", nfolds=10)'
        else:
            r_code = f'{model_name} <- cv.glmnet(X_py, y_py, family="poisson", intercept=FALSE, nfolds=10)'
    else:
        raise ValueError(f"Unsupported family: '{family}'")

    # Execute cross-validation
    r(r_code)

    # Extract coefficients (convert sparse matrix to dense)
    coef_dense = r(f'as.matrix(coef({model_name}))')
    coef = np.array(coef_dense).flatten()

    # Extract optimal lambda
    lambda_min = r(f'{model_name}$lambda.min')[0]

    return model_name, coef, lambda_min


def predict_glmnet_via_r(
    model_name: str,
    newx: np.ndarray,
    s: str = 'lambda.min',
    type: str = 'link'
) -> np.ndarray:
    """Generate predictions via rpy2 interface (testing utility only).

    This function calls the glmnet predict method through rpy2 for models
    fitted with :func:`cv_glmnet_via_r`.

    Parameters
    ----------
    model_name : str
        Model variable name in the rpy2 environment.
    newx : np.ndarray of shape (n_samples, n_features)
        New covariate matrix for prediction.
    s : {'lambda.min', 'lambda.1se'}, default='lambda.min'
        Which lambda value to use for predictions.
    type : {'link', 'response'}, default='link'
        Type of prediction:

        - ``'link'``: Linear predictor (default for gaussian)
        - ``'response'``: Predicted probabilities (for binomial/poisson)

    Returns
    -------
    predictions : np.ndarray of shape (n_samples,)
        Predicted values.

    Notes
    -----
    This function uses the glmnet predict method rather than direct matrix
    multiplication because glmnet applies internal standardization that
    affects the coefficient interpretation.
    """
    if not HAS_RPY2:
        raise ImportError("predict_glmnet_via_r requires rpy2")

    n, p = newx.shape

    # Transfer data to rpy2 environment
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(ro.default_converter + numpy2ri.converter):
        r.assign('newx_py_array', newx)

    r(f'newx_py <- matrix(newx_py_array, nrow={n}, ncol={p})')

    # Generate predictions
    if type == 'link':
        r_code = f'pred_py <- as.vector(predict({model_name}, newx=newx_py, s="{s}"))'
    else:
        r_code = f'pred_py <- as.vector(predict({model_name}, newx=newx_py, s="{s}", type="{type}"))'

    r(r_code)

    predictions = np.array(r['pred_py'])
    
    return predictions


