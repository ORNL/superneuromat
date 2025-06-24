**************************
Installing for Development
**************************

This page describes how to install SuperNeuroMAT for development.

.. admonition:: This is a guide for **internal contributors** to SuperNeuroMAT.
   
   It assumes you're added to the github repository as a collaborator
   and uses **SSH URLs** for git. External contributors should fork the repository <https://github.com/ornl/superneuromat/fork>

.. seealso::
   This guide assumes you're using a Unix-like operating system, such as :fab:`linux` Linux or :fab:`apple` macOS.
   If you're on :fab:`windows` Windows, please use WSL :doc:`/guide/install-wsl` and then follow this guide
   as if you were on Linux (because you are). Then, see :ref:`wsl-post-install` for further setup.


Installing git
==============


If you're contributing to SuperNeuroMAT, you should probably be using :fab:`git-alt` :fab:`git` git version control.

.. button-link:: https://git-scm.com/downloads
   :ref-type: myst
   :color: primary

   :fab:`git-alt` Download git :fas:`arrow-up-right-from-square`

Once you have git, make sure you can run ``git`` from the command line. If not, you may need to restart your terminal.

.. _snm-install-ssh-keys:

SSH keys
========

This guide uses SSH URLs for git. You'll need to have an SSH key set up and added to your GitHub account
to clone, pull, or push to/from the remote repository.
If you don't have an SSH key on your system, you'll need to generate one.

.. code-block:: console
   :caption: "Check if you have an SSH key already"

   $ ls -A ~/.ssh
   id_rsa  id_rsa.pub  known_hosts

You should see ``id_rsa.pub`` or ``id_ed25519.pub`` in the output.
If you don't, you'll need to generate a new SSH key.

.. caution::
   Be aware that existing SSH keys may be used by other applications. If you delete or overwrite an existing key,
   you may need to re-add it wherever it was used.

.. code-block:: console
   :caption: Generate a new SSH key

   $ ssh-keygen

Copy & paste the contents of your ``id_rsa.pub`` or ``id_ed25519.pub`` file into your GitHub account's SSH keys page.
Make sure to give the key a descriptive title; you won't be able to change it later.

.. button-link:: https://github.com/settings/ssh/new
   :ref-type: myst
   :color: secondary

   :fas:`key` Add :fab:`github` GitHub SSH key :fas:`arrow-up-right-from-square`


See the `GitHub documentation <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`_ for more information.


Python Installation
===================

If you're on :fab:`ubuntu` Ubuntu or a :fab:`debian` Debian-based Linux distribution, you should use ``pyenv``
to install :fab:`python` Python 3.11 or later.

This allows you to install any Python version you want, without affecting your system Python installation.
See the `pyenv installation instructions <https://github.com/pyenv/pyenv#installation>`_.

.. code-block:: bash
   :caption: Install & switch to Python>=3.11

   pyenv install 3.13
   pyenv global 3.13

Then, make sure we're actually using the right version of Python.
You should see something similar to this:

.. code-block:: console
   :caption: Check the python version and make sure ``_ctypes`` is available

   $ which python
   /home/username/.pyenv/shims/python
   $ python --version
   Python 3.13.0
   $ python -c "import _ctypes"
   $ pip --version
   pip 24.2 from /home/username/.pyenv/versions/3.13.0/lib/python3.13/site-packages/pip (python 3.13)


.. hint::
   This needs to be done before creating the virtual environment, as ``venv`` or ``virtualenv``
   will use whatever version of Python it finds when you run it. Running ``which python`` may help you know more.

   If you already made the virtual environment, the easiest way to fix this is to delete the virtual environment and start over.

.. _snm-install-editable:

Downloading & Installing as editable
====================================

We recommend using UV which provides environment tools and faster installs.

.. dropdown:: Install UV for faster installs
   :color: secondary
   :open:

   .. code-block:: bash
      :caption: Install ``uv`` <https://github.com/pyuv/uv> for faster installs

      pip install uv -U

   The ``-U`` flag is shorthand for ``--upgrade``.
   
   You can preface most ``pip install`` commands with ``uv`` for *much* faster installation.
   ``uv pip install`` may not work for some packages. If you get an error, try using regular ``pip install`` first.


First, let's make a project folder and **virtual environment**. Pick a place
to store your virtual environment, and then ``cd`` into it.

.. code-block:: bash
   :caption: Make a project folder and virtual environment

   mkdir snm
   cd snm

Next, we can create the virtual environment.

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

Now, we need to activate the virtual environment.

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

.. admonition:: Activating fish, Nushell, or PowerShell

   The above activation command is for the default shell environments, such as ``bash``, ``zsh``, or ``sh`` on Unix, or ``cmd`` and ``powershell`` on Windows.
   If you're using a different shell, such as ``fish`` or ``Nushell``, or if you're using PowerShell and have activation issues, you may need to use a different activation file.

   .. tab-set::
      :class: sd-width-content-min
      :sync-group: shell

      .. tab-item:: fish
         :sync: fish

         .. code-block:: fish

            source .venv/bin/activate.fish

      .. tab-item:: PowerShell
         :sync: powershell

         .. code-block:: powershell

            .venv\bin\activate.ps1
            

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

You can deactivate the virtual environment with the ``deactivate`` command.

Now, let's `git clone` the SuperNeuroMAT repository.

.. code-block:: bash
   :caption: git clone the SuperNeuroMAT repository (using HTTPS URL) and ``cd`` into it

   git clone https://github.com/ORNL/superneuromat.git
   cd superneuromat


.. admonition:: SSH URLs

   GitHub won't let you push to HTTPS remote URLs using password authentication. If you choose to use the HTTPS URL as shown above,
   you'll need to create a `personal access token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>`_
   and use that as the password every time you push.

   However, if you successfully set up your SSH key in the `section above <snm-install-ssh-keys>`_, and have contributor-level permissions on GitHub,
   you can use the SSH URL instead.

   .. code-block:: bash
      :caption: git clone the SuperNeuroMAT repository (using SSH URL) and ``cd`` into it

      git clone git@github.com:ORNL/superneuromat.git
      cd superneuromat

   Again, if you're not an internal contributor, you'll need to fork the <https://github.com/ornl/superneuromat/fork> repository and use the URL for your fork.

It's finally time to install SuperNeuroMAT into our virtual environment:

We'll use a ``pip --editable`` install allows you to make changes to the code and see the effects immediately.

.. tab-set::
   :class: sd-width-content-min
   :sync-group: uv

   .. tab-item:: uv
      :sync: uv

      .. code-block:: bash

         uv pip install -e .[docs,jit]

   .. tab-item:: pip
      :sync: pip

      .. code-block:: bash

         pip install -e .[docs,jit]

The ``.`` refers to the current directory, and the ``[docs,jit]`` refers to the optional dependencies.
``[docs]`` refers to the dependencies for building the documentation, and ``[jit]`` refers to the ``'jit'`` backend dependencies.

The other optional dependency is ``[cuda]``, which refers to the ``'gpu'`` backend dependencies. The ``[cuda]`` dependency set
includes everything in the ``[jit]`` dependency set, plus the ``numba-cuda`` package. It also has the most stringent requirements
on package versioning, i.e. it requires a specific version of ``numpy`` to be installed, and this requirement is not managed by
the ``numba`` package.

All these dependencies are specified in the ``superneuromat/pyproject.toml`` file, in the ``[project]`` ``dependencies`` section,
and the ``[project.optional-dependencies]`` section.

Note that the ``[cuda]`` dependency testing must be done manually, as we have not setup ``tox`` to work with system CUDA installs.

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

If you installed ``pyreadline3`` or are using Python 3.13 or newer, you can exit the ``python`` shell with :kbd:`Ctrl+C` to stop
currently running commands and then :kbd:`Ctrl+D`. Or you can type ``quit()`` to quit the python REPL.

-----

.. card::
   :link: /guide/firstrun
   :link-type: doc
   :link-alt: First Run Tutorial
   :margin: 3

   Finished installing? Check out the :doc:`/guide/firstrun` tutorial.  :fas:`circle-chevron-right;float-right font-size-1_7em`