# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import pathlib
import sys

import determined_ai_sphinx_theme

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# -- Project information -----------------------------------------------------

project = "YogaDL"
html_title = "YogaDL Documentation"
copyright = "2020, Determined AI"
author = "hello@determined.ai"

# The version info for the project you"re documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.1"

# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    # "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "m2r",
]

autosummary_generate = True
autoclass_content = "class"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples", "requirements.txt"]

# The suffix of source filenames.
source_suffix = {".rst": "restructuredtext", ".txt": "restructuredtext"}

highlight_language = "none"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
# html_static_path = ["_static"]

# -- HTML theme settings ------------------------------------------------

html_show_sourcelink = False
html_show_sphinx = False
html_last_updated_fmt = None
# html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]}

html_theme_path = [determined_ai_sphinx_theme.get_html_theme_path()]
html_theme = "determined_ai_sphinx_theme"
# html_logo = "assets/images/logo.png"
# html_favicon = "assets/images/favicon.ico"

html_theme_options = {
    "analytics_id": "UA-110089850-1",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

language = "en"

todo_include_todos = True

html_use_index = True
html_domain_indices = True

# -- Sphinx Gallery settings -------------------------------------------

sphinx_gallery_conf = {
    # Subsections are sorted by number of code lines per example. Override this
    # to sort via the explicit ordering.
    # "within_subsection_order": CustomOrdering,
    # "download_all_examples": True,
    "plot_gallery": False,
    "min_reported_time": float("inf"),
}
