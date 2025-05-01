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

The recommended way to install SuperNeuroMAT is with a **virtual environment**.

Virtual environments are isolated Python environments that allow you to install
packages without affecting your system Python installation.

First, you need to choose a location to store your virtual environment.

.. code-block:: bash

    mkdir foovenv
    cd foovenv

To create a virtual environment, we recommend using ``virtualenv``, but you can
still use ``venv`` if you prefer.

.. tab-set::
   :class: sd-width-content-min
   :sync-group: venv

   .. tab-item:: virtualenv (recommended)

      .. code-block:: bash

        pip install virtualenv
        virtualenv .

   .. tab-item:: venv

      .. code-block:: bash

        python -m venv .

This will create a virtual environment in your current directory.

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

      Make sure you're in the directory where you created the virtual environment, which should
      contain ``Scripts\\`` and ``Lib\\`` and ``pyvenv.cfg`` among other things. You can see what's
      in your current folder by typing ``dir`` in the command prompt.

      .. code-block:: bat

         .\Scripts\activate

   .. tab-item:: :fab:`linux` Linux / :fab:`apple` macOS / :fab:`windows`\ :fab:`linux` WSL
      :sync: posix

      Make sure you're in the directory where you created the virtual environment, which should
      have a ``bin/`` and ``lib/`` and ``pyvenv.cfg`` among other things. You can see what's
      in your current folder by typing ``ls`` in the command prompt.

      .. code-block:: bash

         source bin/activate

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

To install SuperNeuroMAT, we recommend using ``uv``.
You can preface most ``pip install`` commands with ``uv`` for *much* faster installation.

.. tab-set::
   :class: sd-width-content-min
   :sync-group: uv

   .. tab-item:: uv
      :sync: uv

      .. code-block:: bash

         pip install uv
         uv pip install https://github.com/kenblu24/superneuromat.git@main

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         pip install https://github.com/kenblu24/superneuromat.git@main


If the installation was successful, you should be able to open a ``python`` shell and import the package:

.. code-block:: python-console
   :caption: ``python``

   Python 3.11.0 (or newer)
   Type "help", "copyright", "credits" or "license" for more information.
   >>> from superneuromat import SNN
   >>> 

.. _snm-install-numba:

Installing with Numba support
=============================

If you want to use SuperNeuroMAT with the ``'jit'`` backend or with the CUDA ``'gpu'`` backend, you'll need to install
`Numba <https://numba.readthedocs.io/en/stable/>`_.

.. tab-set::
   :class: sd-width-content-min
   :sync-group: uv

   .. tab-item:: uv
      :sync: uv

      For just the ``'jit'`` backend:

      .. code-block:: bash

         uv pip install git+https://github.com/kenblu24/superneuromat.git@main[jit]

      For both the ``'jit'`` and ``'gpu'`` backends:

      .. code-block:: bash

         uv pip install git+https://github.com/kenblu24/superneuromat.git@main[cuda]

   .. tab-item:: pip
      :sync: pip

      For just the ``'jit'`` backend:

      .. code-block:: bash

         pip install git+https://github.com/kenblu24/superneuromat.git@main[jit]

      For both the ``'jit'`` and ``'gpu'`` backends:

      .. code-block:: bash

         pip install git+https://github.com/kenblu24/superneuromat.git@main[cuda]

To use CUDA with Numba installed this way, you'll also need to install the `CUDA SDK <https://developer.nvidia.com/cuda-downloads>`_ from NVIDIA.
See `Installing using pip on x86/x86_64 Platforms <https://numba.pydata.org/numba-doc/0.42.0/user/installing.html#installing-using-pip-on-x86-x86-64-platforms>`_.

Once you have ``numba.cuda.is_available()``, SuperNeuroMAT will be able to use the CUDA ``'gpu'`` backend.

.. code-block:: bash

   python -c "import numba.cuda; assert numba.cuda.is_available()"

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
