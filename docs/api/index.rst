API Reference
=============

This section documents the public API of the ``cbps`` package, organised by
estimator. Each page is generated from the source docstrings via
``sphinx.ext.autodoc`` and follows NumPy conventions.

All user-facing functions and classes are available directly under the
``cbps`` namespace, for example:

.. code-block:: python

   import cbps

   fit = cbps.CBPS(formula='treat ~ age + educ', data=df, att=1)
   bal = cbps.balance(fit)

Estimators
----------

.. toctree::
   :maxdepth: 1

   cbps
   cbmsm
   npcbps
   hdcbps
   cbiv

Inference
---------

.. toctree::
   :maxdepth: 1

   inference

Diagnostics and Plotting
------------------------

.. toctree::
   :maxdepth: 1

   diagnostics

Datasets
--------

.. toctree::
   :maxdepth: 1

   datasets

scikit-learn Integration
------------------------

.. toctree::
   :maxdepth: 1

   sklearn

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
