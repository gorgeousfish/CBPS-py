# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CBPS'
copyright = '2025-2026, Xuanyu Cai, Wenli Xu'
author = 'Xuanyu Cai, Wenli Xu'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Automatic API documentation
    'sphinx.ext.napoleon',     # NumPy docstring support
    'sphinx.ext.mathjax',      # LaTeX formula rendering
    'sphinx.ext.viewcode',     # Source code links
    'sphinx.ext.intersphinx',  # Cross-references (numpy, scipy, etc.)
    'sphinx.ext.todo',         # TODO directives support
    'numpydoc',                # NumPy-style docstring rendering
    'nbsphinx',                # Jupyter Notebook integration
    'myst_parser',             # Markdown support
    'sphinx_copybutton',       # Copy button for code blocks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# numpydoc settings
numpydoc_show_class_members = False  # Suppress autosummary stub warnings

# Suppress specific warnings
suppress_warnings = ['ref.citation']  # numpydoc auto-numbered citations in References sections

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
}

# MathJax configuration
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# nbsphinx configuration
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = False  # Fail build on notebook errors for production quality
nbsphinx_timeout = 600  # Timeout for notebook execution (if enabled)

# MyST parser configuration
myst_enable_extensions = [
    'dollarmath',  # Enable $...$ and $$...$$ for math
    'amsmath',     # Enable AMS math environments
    'deflist',     # Enable definition lists
    'colon_fence', # Enable ::: fences
]

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Master document
master_doc = 'index'

# Language
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.extend([
    '**.ipynb_checkpoints',
    'build',
    '_build',
])

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'CBPS.tex', 'CBPS Documentation',
     'Xuanyu Cai, Wenli Xu', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'cbps', 'CBPS Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'CBPS', 'CBPS Documentation',
     author, 'CBPS', 'Covariate Balancing Propensity Score',
     'Miscellaneous'),
]

