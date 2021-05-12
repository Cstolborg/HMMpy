import hmmpy
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'HMMpy'
copyright = '2021, Christian Stolborg, Mathias Joergensen'
author = 'Christian Stolborg & Mathias Joergensen'


# -- General configuration ---------------------------------------------------


extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_math_dollar'
]

autodoc_default_options = {'members': None, 'inherited-members': None}


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'description':
        'Hidden Markov Models for unsupervised learning',
    'github_user': 'Cstolborg',
    'github_repo': 'HMMpy',
    'github_banner': True,
    'github_button': False,
    'code_font_size': '80%',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']