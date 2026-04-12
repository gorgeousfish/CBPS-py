Installation
============

Basic Installation
------------------

The CBPS Python package can be installed using pip:

.. code-block:: bash

   pip install cbps-python

This will install the core package with all required dependencies:

- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- statsmodels >= 0.13.0
- numdifftools >= 0.9.0
- patsy >= 0.5.0

For visualization features (diagnostic plots), install with:

.. code-block:: bash

   pip install 'cbps-python[plots]'

This adds matplotlib >= 3.4.0 for balance plots and weight distribution visualization.

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

On Apple Silicon Macs, ``glmnetforpython`` requires a Fortran compiler. Follow these steps:

1. Install Homebrew (if not already installed):

   .. code-block:: bash

      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. Install GCC (includes gfortran):

   .. code-block:: bash

      brew install gcc

3. Set the Fortran compiler environment variable:

   .. code-block:: bash

      export FC=gfortran

4. Install glmnetforpython:

   .. code-block:: bash

      pip install glmnetforpython

5. Verify installation:

   .. code-block:: python

      import glmnet
      print("glmnetforpython installed successfully!")

.. tip::
   Add ``export FC=gfortran`` to your ``~/.zshrc`` or ``~/.bash_profile`` to make it permanent.

Development Installation
------------------------

To install the package in development mode (for contributing or testing):

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/gorgeousfish/cbps-python.git
      cd cbps-python

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

**Problem:** ``ModuleNotFoundError: No module named 'glmnet'`` when using ``hdCBPS``

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
- See :doc:`tutorials/index` for comprehensive tutorials
- See :doc:`api/index` for API reference

