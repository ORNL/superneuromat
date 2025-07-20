# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib as il
import inspect
import pathlib as pl
from functools import lru_cache


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
    'sphinx.ext.intersphinx',  # reference external sphinx docs
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
html_static_path = ['_static']
html_title = "SuperNeuroMAT"
html_extra_path = ['includeroot/']

github_version = "main"

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "ORNL",
    "github_repo": "superneuromat",
    "github_version": "main",
    "doc_path": "doc/build/html/",
}

github_project_url = "https://github.com/ORNL/superneuromat"


# LINKCODE RESOLVE
# https://stackoverflow.com/a/79705449/2712730

this_dir = pl.Path(__file__).parent  # location of conf.py
# project_root should be set to the root of the git repo
project_root = this_dir.parent.parent
# module_src_abs_paths should be a list of absolute paths to folders which contain modules
module_src_abs_paths = [project_root / 'src']


@lru_cache(maxsize=10)
def get_toplevel_module(modulename):
    toplevel_name = modulename.split('.')[0]
    toplevel_module = il.import_module(toplevel_name)
    return [pl.Path(path) for path in toplevel_module.__spec__.submodule_search_locations or []]


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    modulename, fullname = info.get('module', None), info.get('fullname', None)
    if not modulename and not fullname:
        return None
    return linkcode_resolve_py(modulename, fullname)


def linkcode_resolve_py(modulename, fullname):
    filepath = None

    # first, let's get the file where the object is defined

    # import the module containing a reference to the object
    module = il.import_module(modulename)

    # We don't know if the object is a class, module, function, method, etc.
    # The module name given might also not be where the object code is.
    # For instance, if `module` imports `obj` from `module.submodule.obj`.
    objname = fullname.split('.')[0]  # first level object is guaranteed to be in module
    obj = getattr(module, objname)  # get the object, i.e. `module.obj`
    # inspect will find the canonical module for the object
    realmodule = obj if inspect.ismodule(obj) else inspect.getmodule(obj)
    if realmodule is None or realmodule.__file__ is None:
        return
    abspath = pl.Path(realmodule.__file__)  # absolute path to the file containing the object
    # If the package was installed via pip, then the abspath here is
    # probably in a site-packages folder.

    # Let's find the abspath relative to the location of the top-level module.
    toplevel_paths = get_toplevel_module(modulename)

    # There may be multiple top-level paths, so pick the first one that matches
    # the absolute path of the file we want to link to.
    for toplevel_path in toplevel_paths:
        toplevel_path = toplevel_path.parent
        if abspath.is_relative_to(toplevel_path):
            filepath = abspath.relative_to(toplevel_path)
            break

    if filepath is None:
        print(f"Could not find {abspath} relative to {toplevel_paths}.")
        return  # Comment this line out to find out what sphinx file causes a "Could not find" error

    # Now let's make it relative to the same directory in the correct src folder.
    for src_path in module_src_abs_paths:
        if not (src_path / filepath).exists():
            msg = f"Could not find {filepath} in {src_path}"
            raise FileNotFoundError(msg)
    src_rel = src_path.relative_to(project_root)  # get rid of the path anchor
    # src_rel is now the relative path from the project_root folder to the correct module folder
    filepath = (src_rel / filepath).as_posix()

    # now, let's try to get the line number where the object is defined

    # If fullname is something like `MyClass.property`, getsourcelines() will fail.
    # In this case, let's return the next best thing, which in this case is the line number of the class.
    name_parts = fullname.split('.')  # get the different components to check

    obj = module  # start with the module
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except TypeError:
        lineno = None  # default to no line number
    # try getting line number for each component and stop on failure
    for child_name in name_parts:
        try:  # get the name of the object for error messages
            name = obj.__name__
        except AttributeError:
            name = objname
        try:
            child = getattr(obj, child_name)  # get the next level object
        except AttributeError:
            print(f"Couldn't load {name}.{child_name} to get its line number; fallback to {name} at line {lineno}")
        try:
            lineno = inspect.getsourcelines(child)[1]  # getsourcelines returns [str, int]
        except Exception:
            # getsourcelines throws TypeError if the object is not a class, module, function, method
            # i.e. if it's a @property, float, etc.
            print(f"{name}.{child_name} of type {type(child).__name__} can't be inspected for line number; fallback to {name} at line {lineno}")
            break  # if we can't get the line number, let it be that of the previous
        obj = child  # update the object to the next level

    suffix = f"#L{lineno}" if lineno else ""
    return f"{github_project_url}/blob/{github_version}/{filepath}{suffix}"


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
