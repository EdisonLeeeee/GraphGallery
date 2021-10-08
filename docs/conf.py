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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))


# # -- Project information -----------------------------------------------------

# project = 'GraphGallery'
# copyright = '2021, Jintang Li'
# author = 'Jintang Li'

# # The full version, including alpha/beta/rc tags
# release = 'latest'


# # -- General configuration ---------------------------------------------------

# # Add any Sphinx extension module names here, as strings. They can be
# # extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# # ones.
# extensions = ['sphinx.ext.todo',
#               'sphinx.ext.viewcode',
#               'sphinx.ext.autodoc',
#               'sphinx.ext.napoleon',
#               'sphinx.ext.autosummary',
#               'sphinx.ext.mathjax',
#               'sphinx.ext.viewcode',
#               'sphinx.ext.githubpages']

# source_suffix = ['.rst', '.md']


# # Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# # List of patterns, relative to source directory, that match files and
# # directories to ignore when looking for source files.
# # This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []


# # -- Options for HTML output -------------------------------------------------

# # The theme to use for HTML and HTML Help pages.  See the documentation for
# # a list of builtin themes.
# #
# # html_theme = 'alabaster'
# import sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


import datetime
import sphinx_rtd_theme
import doctest
import graphgallery

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

autosummary_generate = True
templates_path = ['_templates']

source_suffix = '.rst'
master_doc = 'index'

author = 'Jintang Li'
project = 'graphgallery'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = graphgallery.__version__
release = graphgallery.__version__

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'navigation_depth': 2,
}

html_logo = '../imgs/graphgallery.svg'
html_static_path = ['_static']
# html_context = {'css_files': ['_static/css/custom.css']}
rst_context = {'graphgallery': graphgallery}

add_module_names = False


def setup(app):
    def skip(app, what, name, obj, skip, options):
        members = [
            '__init__',
            '__repr__',
            '__weakref__',
            '__dict__',
            '__module__',
        ]
        return True if name in members else skip

    def rst_jinja_render(app, docname, source):
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect('autodoc-skip-member', skip)
    app.connect("source-read", rst_jinja_render)
