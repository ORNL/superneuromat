**************************************
Installing Windows Subsystem for Linux
**************************************

While most of SuperNeuroMAT works the same on :fab:`windows` Windows as it does on :fab:`linux` Linux,
you may need to use SuperNeuroMAT alongside Linux-only software.

If you're on :fab:`windows` Windows but need a :fab:`linux` Linux environment, you can use
:fab:`windows`\ :fab:`linux` Windows Subsystem for Linux (WSL).
WSL 2 is basically a virtualized Linux environment, but it's nicely integrated into Windows.
You can access files and folders on your Windows machine from WSL, and vice versa. You can even
run GUI applications from WSL and have them appear alongside other programs on your Windows desktop.

.. seealso::

   Microsoft's `What is the Windows Subsystem for Linux? <https://learn.microsoft.com/en-us/windows/wsl/about>`_
   has a good overview of WSL and its features.

Installing WSL
==============

These days, installing WSL is pretty easy. You should be able to install WSL with just
one command from an Administrator cmd or PowerShell prompt.

.. code-block:: console
   :caption: Install WSL

   wsl --install

This will install WSL and use :fab:`ubuntu` Ubuntu as the default Linux distribution.

This guide assumes you'll be using :fab:`ubuntu` Ubuntu, but if you want to use another Linux distribution,
or change any other settings, you can follow the `WSL installation instructions <https://learn.microsoft.com/en-us/windows/wsl/install>`_.

.. button-link:: https://learn.microsoft.com/en-us/windows/wsl/install
   :ref-type: myst
   :color: primary

   :fab:`microsoft` Install WSL :fas:`arrow-up-right-from-square`

.. _wsl-install-old:

.. dropdown:: Old Installation Method

   Prior to November 2022, the default installation method was to use the optional WSL/lxss component.
   The Windows Store version is now the default, and this change was documented here:

   `The Windows Subsystem for Linux in the Microsoft Store is now generally available on Windows 10 and 11 <https://devblogs.microsoft.com/commandline/the-windows-subsystem-for-linux-in-the-microsoft-store-is-now-generally-available-on-windows-10-and-11/>`_

   If you want to use the older installation method, see `Manual installation steps for older versions of WSL <https://learn.microsoft.com/en-us/windows/wsl/install-manual>`_
   or you can follow the instructions below.

   .. div:: h5

      Enabling the Windows Feature

   .. tab-set::
      :class: sd-width-content-min
      :sync-group: gui_cli

      .. tab-item:: :fas:`gears` Windows GUI
         :sync: windows

         Open Windows :fas:`magnifying-glass` Search, type :samp:`features` and select :guilabel:`Turn Windows features on or off`.
         Scroll down and check the :far:`square-check` :guilabel:`Windows Subsystem for Linux box`\ .

         .. image:: /i/windowsfeatures_wsl_vmp.png
            :scale: 80 %
            :loading: lazy
            :alt: Windows Features Dialog with Windows Subsystem for Linux checked

         Then, click :guilabel:`OK` and :fas:`power-off` restart your computer.

      .. tab-item:: :fas:`terminal` PowerShell
         :sync: cli

         Open a PowerShell prompt as an :fas:`shield` Administrator.

         .. code-block:: pwsh-session
            :caption: :fas:`shield` Administrator: Windows PowerShell

            PS C:\> Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
            PS C:\> Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
            Do you want to restart the computer to complete this operation now?
            [Y] Yes [N] No [?] Help (default is "Y"):

         Type :kbd:`Y` and press :kbd:`Enter` to :fas:`power-off` restart the computer.

      .. tab-item:: :fas:`terminal` cmd or other terminal
         :sync: cli-cmd

         Open a PowerShell prompt as an :fas:`shield` Administrator.

         .. code-block:: pwsh-session
            :caption: :fas:`shield` Administrator: Windows PowerShell

            C:\> dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
            C:\> dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all

         Then, :fas:`power-off` restart your computer.

   .. div:: h5

      Installing WSL

   You shouldn't need to manually update the WSL kernel or set WSL to version 2, as these should
   already be at the latest version, but if you want to do so manually, you can follow the
   instructions to `Download the Linux kernel update package <https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package>`_
   and also Set WSL 2 as your default version.

   Now to install WSL:

   .. tab-set::
      :class: sd-width-content-min
      :sync-group: gui_cli

      .. tab-item:: :fas:`gears` Windows GUI
         :sync: windows

         Open the Microsoft Store and search for :samp:`WSL`.
         Then click :guilabel:`Get` or :guilabel:`Install`.

         .. button-link:: https://aka.ms/wslstorepage
            :ref-type: myst
            :color: primary

            :fab:`microsoft` Get WSL :fas:`arrow-up-right-from-square`

         Then, search for your desired distribution on the store and install it.

         .. button-link:: https://apps.microsoft.com/search?query=linux
            :ref-type: myst
            :color: primary

            :fas:`magnifying-glass` :fab:`microsoft` Search for Linux Distributions :fas:`arrow-up-right-from-square`

         

      .. tab-item:: :fas:`terminal` Windows Terminal, cmd, pwsh, etc.
         :sync: cli

         Open a terminal (wt, cmd, pwsh, etc.) as an :fas:`shield` Administrator.
         Then, get the list of available distributions with ``wsl -l -o`` and 
         install the desired distribution with ``wsl --install --distribution {distribution_name}``.

         .. code-block:: console
            :caption: :fas:`shield` Terminal
         
            > wsl --list --online
            > wsl --install --inbox --distribution <DISTRIBUTION_NAME> 
            
         
Once it's installed, you'll need to boot the WSL environment for first-time setup.

You can type :samp:`wsl` in the :fas:`terminal` terminal or :fas:`magnifying-glass` search for it in the Start Menu.

Choose your desired :bdg-success-line:`username` and :bdg-info-line:`password`.
Make sure to remember your password! It's not connected to your Windows password.


Common Installation Errors
--------------------------

Microsoft has some troubleshooting steps for common WSL errors:

.. button-link:: https://learn.microsoft.com/en-us/windows/wsl/troubleshooting
   :shadow:

   Troubleshooting Windows Subsystem for Linux

You might also try the :ref:`old installation method <wsl-install-old>`\ .

.. button-link:: https://learn.microsoft.com/en-us/windows/wsl/install-manual
   :shadow:

   Manual installation steps for older versions of WSL

If you're having trouble with GUI applications, see this guide: https://github.com/microsoft/wslg/wiki/Diagnosing-%22cannot-open-display%22-type-issues-with-WSLg

If you're still having problems, try forcing X11 forwarding instead of wayland:

#. Disable WSLg: https://github.com/microsoft/wslg/discussions/523 
#. Install a windows X server. You have several choices:

   * https://github.com/Opticos/GWSL-Source (Works best, but prebuilt binaries are paid)
   * https://github.com/marchaesen/vcxsrv (Free)
   * `Xming <https://sourceforge.net/projects/xming/>`_ (Free, old)

#. Read `these answers on StackOverflow <https://stackoverflow.com/questions/61110603/how-to-set-up-working-x11-forwarding-on-wsl2>`_
   to set up X11 forwarding.

While troubleshooting, you can use ``xeyes`` and/or ``glxgears`` to test if gui apps are working.
You may need to install them if they didn't come with your distribution.

.. code-block:: bash

   sudo apt install xorg-x11-apps mesa-utils libgl1-mesa-dri


.. _wsl-post-install:

Post-Installation
=================

If you're new to WSL, Microsoft has a short lesson on getting aquainted with Linux
and how to use it for development.

.. button-link:: https://learn.microsoft.com/en-us/training/modules/developing-in-wsl/
   :ref-type: myst
   :color: secondary

   :fab:`microsoft` WSL Tutorial :fas:`arrow-up-right-from-square`

Here are some more resources for getting started with WSL:

* `Best practices for set up <https://learn.microsoft.com/en-us/windows/wsl/setup/environment>`_  :bdg-primary:`highly recommended`
* `Get started using Git on Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-git>`_
* `Run Linux GUI apps on the Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps>`_
* `Getting started with Linux and Bash <Getting started with Linux and Bash>`_  :bdg-primary:`highly recommended`
* `VSCode: Developing in WSL <https://code.visualstudio.com/docs/remote/wsl>`_


-----

.. |_| unicode:: 0xA0 
   :trim:

.. grid:: 2
   :gutter: 3

   .. grid-item-card::
      :link: /guide/install
      :link-type: doc
      :link-alt: Installation Tutorial

      :fas:`chevron-left;font-size-1_2em`\ |_|\ |_|\ Install SuperNeuroMAT

   .. grid-item-card::
      :link: /devel/install
      :link-type: doc
      :link-alt: Development Installation Tutorial

      :fas:`chevron-left;font-size-1_2em`\ |_|\ |_|\ Development Install Guide

   .. grid-item-card::
      :link: /guide/firstrun
      :link-type: doc
      :link-alt: First Run Tutorial
      :columns: 12

      Already installed SuperNeuroMAT? Check out the :doc:`/guide/firstrun` tutorial.  :fas:`circle-chevron-right;float-right font-size-1_7em`

