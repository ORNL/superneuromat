# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'SuperNeuroMAT'
copyright = '2025, Prasanna Date, Chathika Gunaratne, Shruti Kulkarni, Robert Patton, Mark Coletti'
author = 'Prasanna Date, Chathika Gunaratne, Shruti Kulkarni, Robert Patton, Mark Coletti'
release = '1.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", ]
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # support for NumPy and Google style docstrings
    'sphinx.ext.linkcode',  # generate links to externally-hosted source code
    # "sphinx.ext.viewcode",  # generate html source code previews
    'sphinx.ext.intersphinx',  #  reference external sphinx docs
    'sphinx.ext.ifconfig',
    'sphinx_copybutton',
    'sphinx.ext.todo',
    'sphinx_design',
    'sphinx_togglebutton',
    # 'sphinx_tags',
    'numpydoc',  # Needs to be loaded *after* autodoc.
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.duration',
    "myst_parser",  # for myst support in sphinx design buttons
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = "SuperNeuroMAT"

github_version = "docs-live"

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "kenblu24",
    "github_repo": "superneuromat",
    "github_version": "docs-live",
    "doc_path": "doc/build/html/",
}

github_project_url = "https://github.com/kenblu24/superneuromat"


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return f"{github_project_url}/blob/{github_version}/src/{filename}.py"


numpydoc_class_members_toctree = False

html_last_updated_fmt = "%b %d, %Y"

html_theme_options = {
    "use_edit_page_button": True,
    "footer_end": ["theme-version", "last-updated"],

    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_project_url,
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        # {
        #     "name": "PyPI",
        #     "url": pypi_project_url,
        #     "icon": "fa-brands fa-python",
        #     "type": "fontawesome",
        # },
   ],
    # "search_as_you_type": True,
}
# html_show_sourcelink = False

html_css_files = [
    "css/custom.css",
]

