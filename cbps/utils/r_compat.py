"""
rpy2 and pandas 2.x Compatibility Utilities

This module provides compatibility patches for using rpy2 with pandas 2.x,
where the deprecated ``DataFrame.iteritems()`` and ``Series.iteritems()``
methods were removed. The rpy2 pandas2ri converter relies on these methods,
causing AttributeError in pandas 2.x environments.

This module is primarily used for cross-validation testing and is not
required for normal CBPS functionality.

Usage
-----
Call ``ensure_rpy2_compatibility()`` before importing rpy2::

    from cbps.utils.r_compat import ensure_rpy2_compatibility
    ensure_rpy2_compatibility()
    
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

Notes
-----
The compatibility patch maps ``iteritems()`` to ``items()``, which is the
pandas 2.x replacement. This patch is idempotent and safe to call multiple
times.
"""

import pandas as pd


def ensure_rpy2_compatibility():
    """
    Apply compatibility patches for rpy2 with pandas 2.x.
    
    Maps the removed ``iteritems()`` methods to ``items()`` on both
    DataFrame and Series classes. This is required because rpy2's
    pandas2ri converter uses these deprecated methods.
    
    Notes
    -----
    - Idempotent: safe to call multiple times
    - No effect on pandas 1.x (where iteritems() exists)
    - Applied at the class level to DataFrame and Series
    
    Examples
    --------
    >>> from cbps.utils.r_compat import ensure_rpy2_compatibility
    >>> ensure_rpy2_compatibility()
    >>> # rpy2 can now be safely imported
    """
    # Check if patch is needed (pandas 2.x lacks iteritems)
    if not hasattr(pd.DataFrame, 'iteritems'):
        # Add DataFrame.iteritems as an alias for items
        pd.DataFrame.iteritems = pd.DataFrame.items
        
    if not hasattr(pd.Series, 'iteritems'):
        # Add Series.iteritems as an alias for items
        pd.Series.iteritems = pd.Series.items


def check_rpy2_available():
    """
    Check rpy2 availability and apply compatibility patches.
    
    Attempts to import rpy2 and the CBPS package from the R environment.
    Automatically applies pandas 2.x compatibility patches before import.
    
    This function is primarily used for internal testing and validation.
    
    Returns
    -------
    available : bool
        True if rpy2 and required packages are available.
    components : tuple of (robjects, pandas2ri, cbps_package) or None
        If available, returns the imported rpy2 components.
        If not available, returns None.
    
    Examples
    --------
    >>> from cbps.utils.r_compat import check_rpy2_available
    >>> available, components = check_rpy2_available()
    >>> if available:
    ...     ro, pandas2ri, cbps_pkg = components
    """
    try:
        # Apply compatibility patches first
        ensure_rpy2_compatibility()
        
        # Attempt to import rpy2
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        
        # Activate pandas conversion
        pandas2ri.activate()
        
        # Attempt to import CBPS package
        cbps_r = importr('CBPS')
        
        return True, (ro, pandas2ri, cbps_r)
        
    except ImportError as e:
        # rpy2 or required packages not installed
        return False, None
    except Exception as e:
        # Other initialization errors
        return False, None