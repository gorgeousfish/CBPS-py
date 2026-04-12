"""
Taylor Approximation Functions for Empirical Likelihood.

This module provides modified logarithm functions with second-order Taylor
series approximation for numerical stability in empirical likelihood
optimization. When the argument falls below a threshold (typically 1/N),
the Taylor approximation prevents log(0) singularities.

The key functions are:

- ``llog``: Modified log with Taylor branch for small arguments
- ``llogp``: Derivative of llog for gradient-based optimization

Mathematical Background
-----------------------
During empirical likelihood optimization, the objective involves
:math:`\\sum_i \\log w_i` where weights :math:`w_i = 1/(1 - \\gamma^T g_i)`.
When the denominator approaches zero, the logarithm diverges. The Taylor
approximation around :math:`\\epsilon = 1/N` ensures smooth optimization:

.. math::

    \\log(z) \\approx \\log(\\epsilon) - 1.5 + 2(z/\\epsilon) - 0.5(z/\\epsilon)^2
    \\quad \\text{for } z < \\epsilon

This approximation:

1. Matches the true log at :math:`z = \\epsilon`
2. Has continuous first derivative at the boundary
3. Prevents numerical overflow during BFGS iterations

References
----------
Fong, C., Hazlett, C., and Imai, K. (2018). Covariate balancing propensity
score for a continuous treatment: Application to the efficacy of political
advertisements. The Annals of Applied Statistics, 12(1), 156-177.
https://doi.org/10.1214/17-AOAS1101

See Section 3.3.2: "when the argument to the logarithmic function falls
below 1/N, we instead use the second order Taylor series approximation
to the log around the point 1/N."
"""

import numpy as np


def llog(z: np.ndarray, eps: float) -> np.ndarray:
    """
    Modified logarithm with second-order Taylor approximation for small values.

    This function returns :math:`\\log(z)` when :math:`z \\geq \\epsilon`, and a
    second-order Taylor series approximation when :math:`z < \\epsilon`. The
    approximation prevents numerical issues when optimizing the empirical
    likelihood objective.

    Parameters
    ----------
    z : np.ndarray
        Input array. NaN values are preserved in the output.
    eps : float
        Threshold below which Taylor approximation is used. In npCBPS,
        this is typically set to :math:`1/N` where N is the sample size.

    Returns
    -------
    np.ndarray
        Element-wise modified log values with the same shape as input.

    Notes
    -----
    **Taylor expansion formula:**

    For :math:`z < \\epsilon`:

    .. math::

        \\text{llog}(z) = \\log(\\epsilon) - 1.5 + 2\\frac{z}{\\epsilon}
                         - 0.5\\left(\\frac{z}{\\epsilon}\\right)^2

    For :math:`z \\geq \\epsilon`:

    .. math::

        \\text{llog}(z) = \\log(z)

    **Derivation:**

    The standard second-order Taylor expansion of :math:`\\log(z)` around
    :math:`a = \\epsilon` is:

    .. math::

        \\log(z) \\approx \\log(a) + \\frac{z-a}{a} - \\frac{(z-a)^2}{2a^2}

    Expanding and simplifying yields the coefficients -1.5, 2, and 0.5.

    **Boundary continuity:**

    At :math:`z = \\epsilon`, the Taylor branch evaluates to
    :math:`\\log(\\epsilon) - 1.5 + 2 - 0.5 = \\log(\\epsilon)`, matching
    the standard log branch exactly.

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Section 3.3.2.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([0.005, 0.01, 0.1, 1.0])
    >>> eps = 0.01
    >>> result = llog(z, eps)
    >>> # z < eps uses Taylor approximation
    >>> # z >= eps uses standard log
    >>> np.isclose(result[1], np.log(eps))
    True
    """
    ans = z.copy()
    avoid_na = ~np.isnan(z)
    lo = (z < eps) & avoid_na

    # Taylor approximation branch (z < eps)
    ans[lo] = np.log(eps) - 1.5 + 2 * z[lo]/eps - 0.5 * (z[lo]/eps)**2

    # Standard log branch (z >= eps)
    ans[~lo] = np.log(z[~lo])

    return ans


def llogp(z: np.ndarray, eps: float) -> np.ndarray:
    """
    Derivative of the modified logarithm function.

    Computes the exact derivative of :func:`llog` for use in gradient-based
    optimization algorithms such as BFGS.

    Parameters
    ----------
    z : np.ndarray
        Input array. NaN values are preserved in the output.
    eps : float
        Threshold matching the one used in :func:`llog`.

    Returns
    -------
    np.ndarray
        Element-wise derivative values with the same shape as input.

    Notes
    -----
    **Derivative formula:**

    For :math:`z < \\epsilon`:

    .. math::

        \\frac{d}{dz}\\text{llog}(z) = \\frac{2}{\\epsilon}
                                      - \\frac{z}{\\epsilon^2}

    For :math:`z \\geq \\epsilon`:

    .. math::

        \\frac{d}{dz}\\text{llog}(z) = \\frac{1}{z}

    **Derivation:**

    Taking the derivative of the Taylor branch:

    .. math::

        \\frac{d}{dz}\\left[\\log(\\epsilon) - 1.5 + \\frac{2z}{\\epsilon}
        - \\frac{z^2}{2\\epsilon^2}\\right]
        = \\frac{2}{\\epsilon} - \\frac{z}{\\epsilon^2}

    **Boundary continuity:**

    At :math:`z = \\epsilon`, both branches yield :math:`1/\\epsilon`.

    References
    ----------
    Fong, C., Hazlett, C., and Imai, K. (2018). Section 3.3.2.

    Examples
    --------
    >>> import numpy as np
    >>> z = np.array([0.005, 0.01, 0.1])
    >>> eps = 0.01
    >>> deriv = llogp(z, eps)
    >>> # Verify numerically
    >>> h = 1e-8
    >>> numerical = (llog(z + h, eps) - llog(z - h, eps)) / (2 * h)
    >>> np.allclose(deriv, numerical, rtol=1e-5)
    True
    """
    ans = z.copy()
    avoid_na = ~np.isnan(z)
    lo = (z < eps) & avoid_na

    # Taylor derivative branch (z < eps)
    ans[lo] = 2/eps - z[lo]/eps**2

    # Standard derivative branch (z >= eps)
    ans[~lo] = 1/z[~lo]

    return ans
