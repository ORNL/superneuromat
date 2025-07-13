# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'SuperNeuroMAT'
copyright = '2025, Prasanna Date, Chathika Gunaratne, Shruti Kulkarni, Robert Patton, Mark Coletti'
author = 'Prasanna Date, Chathika Gunaratne, Shruti Kulkarni, Robert Patton, Mark Coletti'

# to change version, set [project] -> version in pyproject.toml
# and re-install superneuromat via pip or uv
try:
    import superneuromat
    release = superneuromat.__version__  # may be 'unknown'
except ImportError:
    release = 'latest'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

graphviz_output_format = "svg"

todo_include_todos = True
copybutton_exclude = '.linenos, .gp, .go'
copybutton_remove_prompts = True
copybutton_only_copy_prompt_lines = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'numba': ('https://numba.readthedocs.io/en/stable', None),
    'pypa': ('https://packaging.python.org/en/latest/', None),
    'cuquantum': ('https://docs.nvidia.com/cuda/cuquantum/latest', None),
    # blocked by data-apis/array-api#428
    # 'array-api': ('https://data-apis.org/array-api/2021.12/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static', 'schema']
html_title = "SuperNeuroMAT"

github_version = "main"

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "ORNL",
    "github_repo": "superneuromat",
    "github_version": "main",
    "doc_path": "doc/build/html/",
}

github_project_url = "https://github.com/ORNL/superneuromat"


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
    "logo": {
        # Because the logo is also a homepage link, including "home" in the
        # alt text is good practice
        "alt_text": "SuperNeuroMAT Documentation",
        "image_light": "superneuro-pcg-arrow.svg",
        "image_dark": "superneuro-pcg-arrow.svg",
    },
    # "search_as_you_type": True,
}
# html_show_sourcelink = False

html_css_files = [
    "css/custom.css",
]
