************
Installation
************

This page describes how to install SuperNeuroMAT.

Setting up your environment
===========================

.. note::
   If you're on Ubuntu or a Debian-based Linux distribution, you may need to use ``pyenv``
   to install :fab:`python` Python 3.11 or later.
   See the `pyenv installation instructions <https://github.com/pyenv/pyenv#installation>`_.
   You need to do this before creating the virtual environment.

.. important::
   :fab:`windows` Windows users: please **DO NOT** use Python from the :fab:`microsoft` Microsoft Store.
   Instead, download and install the latest version of Python from the `Python website <https://www.python.org/downloads/>`_.
   Make sure to check the box to add Python to your PATH:

   .. card::
      :far:`square-check` Add Python to PATH

   If you didn't do this when installing Python, you'll need to add it manually.
   See :ref:`python:setting-envvars` or `How to Add Python to PATH <https://realpython.com/add-python-to-path/>`_.
   Or, uninstall and reinstall Python (You may need to re-\ ``pip install`` system Python packages).




To install SuperNeuroMAT, we recommend using ``uv``.

.. dropdown:: Install UV for faster installs
   :color: secondary
   :open:

   .. code-block:: bash
      :caption: Install ``uv`` <https://github.com/pyuv/uv> for faster installs

      pip install uv -U

   The ``-U`` flag is shorthand for ``--upgrade``.
   
   You can preface most ``pip install`` commands with ``uv`` for *much* faster installation.
   ``uv pip install`` may not work for some packages. If you get an error, try using regular ``pip install`` first.

The recommended way to install SuperNeuroMAT is with a **virtual environment**.

Virtual environments are isolated Python environments that allow you to install
packages without affecting your system Python installation.

First, you need to choose a location to store your virtual environment.

.. code-block:: bash

    mkdir foovenv
    cd foovenv

.. tab-set::
   :class: sd-width-content-min
   :sync-group: uv

   .. tab-item:: uv
      :sync: uv     

      .. code-block:: bash
         :caption: Create a virtual environment

         uv venv
         

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash
         :caption: Create a virtual environment

         pip install virtualenv
         virtualenv .venv --prompt .

This will create a virtual environment ``.venv`` folder in your current directory.

Now, we need to activate the virtual environment.

.. _activate-venv:

Activating the virtual environment
----------------------------------

Once you have created your virtual environment, you need to activate it.

Make sure you're in the directory where you created the virtual environment.
In this example we're in the ``foovenv/`` folder.

.. tab-set::
   :class: sd-width-content-min
   :sync-group: os

   .. tab-item:: :fab:`windows` Windows
      :sync: windows

      .. code-block:: bat

         .venv\Scripts\activate

   .. tab-item:: :fab:`linux` Linux / :fab:`apple` macOS / :fab:`windows`\ :fab:`linux` WSL
      :sync: posix

      .. code-block:: bash

         source .venv/bin/activate

.. note::

   The above activation command is for the default shell environments, such as ``bash``, ``zsh``, or ``sh`` on Unix, or ``cmd`` and ``powershell`` on Windows.
   If you're using a different shell, such as ``fish`` or ``Nushell``, you may need to use a different activation file.

   .. tab-set::
      :class: sd-width-content-min
      :sync-group: shell

      .. tab-item:: fish
         :sync: fish

         .. code-block:: fish

            source .venv/bin/activate.fish
            

      .. tab-item:: Nushell
         :sync: nushell

         .. tab-set::
            :class: sd-width-content-min
            :sync-group: os

            .. tab-item:: :fab:`windows` Windows
               :sync: windows

               .. code-block:: powershell

                  overlay use .venv\Scripts\activate.nu

            .. tab-item:: :fab:`linux` Linux / :fab:`apple` macOS / :fab:`windows`\ :fab:`linux` WSL
               :sync: posix

               .. code-block:: bash

                  overlay use .venv/bin/activate.nu

You should see the name of your virtual environment in parentheses at the beginning of your terminal prompt:

.. tab-set::
   :class: sd-width-content-min
   :sync-group: os

   .. tab-item:: :fab:`windows` Windows
      :sync: windows

      .. code-block:: doscon

         (foovenv) C:\foovenv> 

   .. tab-item:: :fab:`linux` Linux / :fab:`apple` macOS / :fab:`windows`\ :fab:`linux` WSL
      :sync: posix

      .. code-block:: bash

         (foovenv) user@host:~/foovenv$

To deactivate the virtual environment, use the ``deactivate`` command:

.. code-block:: bash

   deactivate

.. _regular-install:

Installing SuperNeuroMAT
==============================

.. tab-set::
   :class: sd-width-content-min
   :sync-group: uv

   .. tab-item:: uv
      :sync: uv

      .. code-block:: bash

         uv pip install superneuromat

      .. note::
         
         It's possible to install SuperNeuroMAT to the global system Python installation
         with the ``--system`` flag. However, this is not recommended, as it may cause
         conflicts with other packages.

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         pip install superneuromat



While you're here, let's also install ``pyreadline3`` which makes the ``python`` shell much more user-friendly.

.. tab-set::
   :class: sd-width-content-min
   :sync-group: uv

   .. tab-item:: uv
      :sync: uv

      .. code-block:: bash

         uv pip install pyreadline3

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         pip install pyreadline3

If the installation was successful, you should be able to open a ``python`` shell and import the package:

.. code-block:: python-console
   :caption: ``python``

   Python 3.10.0 (or newer)
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import superneuromat
   >>> 

If you installed ``pyreadline3``, you can exit the ``python`` shell with :kbd:`Ctrl+C` to stop
currently running commands and then :kbd:`Ctrl+D` or ``quit()`` to quit the python REPL.

.. _snm-install-numba:

Installing with Numba support
=============================

If you want to use SuperNeuroMAT with the ``'jit'`` backend or with the CUDA ``'gpu'`` backend, you'll need to install
`Numba <https://numba.readthedocs.io/en/stable/>`_.

.. admonition:: Don't forget to activate the virtual environment!

   :ref:`activate-venv`

.. tab-set::
   :class: sd-width-content-min
   :sync-group: uv

   .. tab-item:: uv
      :sync: uv

      For just the ``'jit'`` backend:

      .. code-block:: bash

         uv pip install superneuromat[jit]

      For both the ``'jit'`` and ``'gpu'`` backends:

      .. code-block:: bash

         uv pip install superneuromat[cuda]

   .. tab-item:: pip
      :sync: pip

      For just the ``'jit'`` backend:

      .. code-block:: bash

         pip install superneuromat[jit]

      For both the ``'jit'`` and ``'gpu'`` backends:

      .. code-block:: bash

         pip install superneuromat[cuda]

To use CUDA with Numba installed this way, you'll also need to install the `CUDA SDK <https://developer.nvidia.com/cuda-downloads>`_ from NVIDIA.
See `Installing using pip on x86/x86_64 Platforms <https://numba.pydata.org/numba-doc/0.42.0/user/installing.html#installing-using-pip-on-x86-x86-64-platforms>`_.

Once you have ``numba.cuda.is_available()``, SuperNeuroMAT will be able to use the CUDA ``'gpu'`` backend.

.. code-block:: bash

   python -c "import numba.cuda; assert numba.cuda.is_available()"

.. seealso::

   Don't know if you need these?
   See :doc:`/guide/speed` to learn more about SuperNeuroMAT's different backends.

Development Installations
=========================

If you intend to contribute to SuperNeuroMAT, you should follow the
:doc:`installation guide for development </devel/install>` instead.

.. button-ref:: /devel/install
   :color: primary
   :expand:

WSL Installation
================

Although SuperNeuroMAT works natively on :fab:`windows` Windows, you can also install SuperNeuroMAT
in a :fab:`windows`\ :fab:`linux` Windows Subsystem for Linux (WSL) environment.

First, you need to install WSL.

.. toctree::
   :maxdepth: 2

   install-wsl

Then, follow the :ref:`regular-install` or :doc:`/devel/install` instructions as if you were on Linux.

-----

.. card::
   :link: /guide/firstrun
   :link-type: doc
   :link-alt: First Run Tutorial
   :margin: 3

   Finished installing? Check out the :doc:`/guide/firstrun` tutorial.  :fas:`circle-chevron-right;float-right font-size-1_7em`
