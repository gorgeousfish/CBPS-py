Configuration
=============

.. currentmodule:: cbps

.. versionadded:: 0.1.0

This module provides centralized numerical configuration and logging control
for the CBPS package. The :class:`~cbps.constants.NumericalConfig` dataclass
defines all stability constants used throughout the estimation algorithms,
and :func:`~cbps.logging_config.set_verbosity` controls package-level log output.

Numerical Configuration
-----------------------

.. automodule:: cbps.constants
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`NumericalConfig` dataclass groups all numerical stability
constants used across CBPS estimators. A global default instance
:data:`DEFAULT_CONFIG` is provided and used internally by all algorithms.

**Configuration Groups**

- **Propensity Score Clipping**: Bounds to prevent extreme probabilities
- **Column Detection**: Threshold for identifying constant columns
- **Optimization**: Convergence tolerances and finite difference steps
- **SVD / Matrix**: Thresholds for singular value decomposition operations

**Example**

.. code-block:: python

   from cbps.constants import NumericalConfig, DEFAULT_CONFIG

   # Inspect default configuration
   print(f"Propensity score bounds: [{DEFAULT_CONFIG.probs_min}, {DEFAULT_CONFIG.probs_max}]")
   print(f"Optimization tolerance: {DEFAULT_CONFIG.optim_xtol}")
   print(f"SVD threshold (MSM): {DEFAULT_CONFIG.svd_threshold_msm}")

   # Create a custom configuration (e.g., for sensitivity analysis)
   custom_config = NumericalConfig(
       probs_min=1e-4,         # Less aggressive clipping
       ndeps=1e-4,             # Finer finite differences
       optim_xtol=1e-10        # Tighter convergence
   )
   print(f"Custom probs_max: {custom_config.probs_max}")

Logging Control
---------------

.. automodule:: cbps.logging_config
   :members:
   :undoc-members:

The :func:`set_verbosity` function controls the amount of diagnostic output
produced by the CBPS package during estimation.

**Verbosity Levels**

- **Level 0** (default): Only warnings are shown. Suitable for production use.
- **Level 1**: Progress messages (INFO level). Shows optimization progress and key decisions.
- **Level 2**: Full diagnostics (DEBUG level). Shows detailed numerical information for debugging.

**Example**

.. code-block:: python

   from cbps import set_verbosity, CBPS
   from cbps.datasets import load_lalonde

   # Enable progress messages
   set_verbosity(1)

   df = load_lalonde(dehejia_wahba_only=True)
   fit = CBPS('treat ~ age + educ + re74 + re75', data=df, att=1)

   # Enable full debug output
   set_verbosity(2)
   fit2 = CBPS('treat ~ age + educ', data=df, att=0)

   # Restore silent mode
   set_verbosity(0)
