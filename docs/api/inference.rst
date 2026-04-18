Inference
=========

The ``cbps.inference`` subpackage provides standard-error tools that
propagate propensity-score estimation uncertainty through downstream
treatment-effect calculations.

Asymptotic Variance of Treatment Effects
----------------------------------------

:func:`cbps.AsyVar` returns the asymptotic variance and confidence
interval for the ATE under binary treatment. Two variance formulae are
available via the ``method`` argument:

- ``method='CBPS'`` — full sandwich variance (Fan et al., 2022, Eq. 2.4).
- ``method='oCBPS'`` — semiparametric efficiency bound (Hahn, 1998;
  Fan et al., 2022, Eq. 2.6), which is the correct variance for the
  AIPW-based ATE estimator computed by default inside :func:`cbps.AsyVar`.

.. autofunction:: cbps.AsyVar
   :no-index:

Variance of Weighted Outcome Regressions
----------------------------------------

:func:`cbps.vcov_outcome` corrects the variance-covariance matrix of a
weighted outcome regression for continuous-treatment CBPS (Fong, Hazlett,
and Imai, 2018, Section 3.2) so that it accounts for the uncertainty in
the estimated CBGPS weights rather than treating them as fixed.

.. autofunction:: cbps.vcov_outcome

Low-level Asymptotic Variance Backend
-------------------------------------

.. autofunction:: cbps.inference.asyvar.asy_var
