Installation
============

Basic Installation
------------------

The CBPS Python package can be installed using pip:

.. code-block:: bash

   pip install cbps-python

The package requires Python 3.10 or newer. The core install pulls in the
following runtime dependencies (lower bounds reflect the minimum versions
tested, no upper bounds are pinned):

- numpy >= 1.22.0
- scipy >= 1.8.0
- pandas >= 1.4.0
- statsmodels >= 0.13.0
- patsy >= 0.5.0

Optional Feature Extras
-----------------------

The package ships four user-facing extras that can be combined with the
``pip install`` command:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Extra
     - Install command
     - What it adds
   * - ``plots``
     - ``pip install 'cbps-python[plots]'``
     - ``matplotlib>=3.5.0`` for balance and weight-distribution plots
   * - ``sklearn``
     - ``pip install 'cbps-python[sklearn]'``
     - ``scikit-learn>=1.0`` for the
       :class:`cbps.sklearn.CBPSEstimator` Pipeline/GridSearch-compatible
       wrapper
   * - ``hdcbps``
     - ``pip install 'cbps-python[hdcbps]'``
     - ``glmnetforpython>=1.0`` for the :func:`cbps.hdCBPS` LASSO backend
   * - ``all``
     - ``pip install 'cbps-python[all]'``
     - ``plots`` + ``sklearn`` + ``hdcbps`` (no development tooling)

Extras for contributors (``dev``, ``test``, ``docs``) are documented in
`pyproject.toml`_.

.. _pyproject.toml: https://github.com/gorgeousfish/CBPS-py/blob/main/pyproject.toml

Requirements for hdCBPS
-----------------------

If you plan to use high-dimensional CBPS (``hdCBPS``), you need to install the ``glmnetforpython`` package:

.. code-block:: bash

   pip install glmnetforpython

.. note::
   ``glmnetforpython`` is **required** for ``hdCBPS`` but **optional** for all other CBPS functions.
   If you don't need high-dimensional variable selection, you can skip this dependency.

Apple Silicon (M1/M2/M3) Installation
--------------------------------------

``glmnetforpython`` ships a compiled Fortran extension. On Apple Silicon,
PyPI often provides a matching pre-built wheel, in which case
``pip install glmnetforpython`` just works. The steps below apply when the
wheel is unavailable for your Python / macOS combination and pip falls back
to a source build.

1. Install Homebrew (if not already installed):

   .. code-block:: bash

      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. Install GCC (which ships with ``gfortran``):

   .. code-block:: bash

      brew install gcc

3. Set the Fortran compiler environment variable in the current shell:

   .. code-block:: bash

      export FC=gfortran

4. Install the package. This is the same command as the non-ARM case; the
   environment variable tells pip which Fortran compiler to use:

   .. code-block:: bash

      pip install glmnetforpython

5. If PyPI still cannot build a wheel for your environment, install the
   package from source:

   .. code-block:: bash

      git clone https://github.com/thierrymoudiki/glmnetforpython.git
      pip install -e ./glmnetforpython

6. Verify the installation:

   .. code-block:: python

      import glmnetforpython
      print("glmnetforpython installed successfully!")

.. tip::
   Add ``export FC=gfortran`` to your ``~/.zshrc`` or ``~/.bash_profile`` so
   the variable is available in new shells.

Development Installation
------------------------

To install the package in development mode (for contributing or testing):

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/gorgeousfish/CBPS-py.git
      cd CBPS-py

2. Install in editable mode with development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

This will install additional development tools:

- pytest (testing)
- black (code formatting)
- mypy (type checking)
- sphinx (documentation)
- jupyter (notebooks)

Verifying Installation
----------------------

To verify that the package is installed correctly:

.. code-block:: python

   import cbps
   print(f"CBPS version: {cbps.__version__}")
   
   # Test basic functionality
   from cbps.datasets import load_lalonde
   data = load_lalonde(dehejia_wahba_only=True)
   print(f"Loaded LaLonde data: {data.shape}")
   
   # Test CBPS estimation
   fit = cbps.CBPS(formula='treat ~ age + educ', data=data, att=1)
   print(f"CBPS converged: {fit.converged}")

Expected output:

.. code-block:: text

   CBPS version: 0.1.0
   Loaded LaLonde data: (445, 11)
   CBPS converged: True

Troubleshooting
---------------

**Problem:** ``ModuleNotFoundError: No module named 'glmnetforpython'`` when using ``hdCBPS``

**Solution:** Install glmnetforpython:

.. code-block:: bash

   pip install glmnetforpython

**Problem:** ``glmnetforpython`` installation fails on Apple Silicon

**Solution:** Install GCC and set FC environment variable (see above).

**Problem:** ``ImportError: cannot import name 'CBPS'``

**Solution:** Make sure you installed ``cbps-python``, not ``cbps``:

.. code-block:: bash

   pip uninstall cbps  # Remove wrong package
   pip install cbps-python  # Install correct package

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade cbps-python

Uninstallation
--------------

To uninstall the package:

.. code-block:: bash

   pip uninstall cbps-python

Next Steps
----------

- See :doc:`quickstart` for a quick introduction
- See :doc:`tutorials/index` for the three replication tutorials
- See :doc:`api/index` for the full API reference generated from the source docstrings

