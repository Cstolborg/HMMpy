import os
import sys

import hmmpy

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

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

if html_theme == 'alabaster':
    html_theme_options = {
        'description':
            'Hidden Markov Models for unsupervised learning',
        'github_user': 'Cstolborg',
        'github_repo': 'HMMpy',
        'github_banner': True,
        'github_button': False,
        'code_font_size': '80%',
    }
elif html_theme == 'sphinx_rtd_theme':
    html_theme_options = {
        'display_version': True,
        'collapse_navigation': False,
        'sticky_navigation': False,
    }

#html_context = {
#    'display_github': True,
#    'github_user': 'Cstolborg',
#    'github_repo': 'hmmpy'
#}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']