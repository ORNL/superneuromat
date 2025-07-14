.. SuperNeuroMAT documentation master file, created by
   sphinx-quickstart on Mon Mar 17 21:53:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SuperNeuroMAT |release| documentation
=====================================

SuperNeuroMAT is a Python package for simulating and analyzing spiking neural networks.

Unlike its sister package, `SuperNeuroABM <https://github.com/ORNL/superneuroabm>`_, SuperNeuroMAT uses a matrix-based representation
of the network, which allows for more efficient simulation and GPU acceleration.

SuperNeuroMAT focuses on super-fast computation of Leaky Integrate and Fire **(LIF)** spiking neuron models with STDP.

.. warning::
   Both the documentation and the simulator software are under development.
   Please report any issues with the software or documentation to the `GitHub issue tracker <https://github.com/ORNL/superneuromat/issues>`_.

Get Started
===========

.. tab-set::
   :class: sd-width-content-min

   .. tab-item:: pip

      .. code-block:: bash

         pip install superneuromat

   .. tab-item:: other

      .. rst-class:: section-toc
      .. toctree::
         :maxdepth: 2

         guide/install

For more detailed instructions, see the :doc:`installation guide <guide/install>`, which covers
virtual environments, faster installation with uv, installing support for CUDA GPU acceleration, and more.

Then, you can import the :py:mod:`superneuromat` package:

.. code-block:: python
   :caption: Python

   from superneuromat import SNN

For those coming from older versions of SuperNeuroMAT, see the :doc:`migration guide <guide/migration2>`.

Cite SuperNeuroMAT
------------------

Thank you for your interest in SuperNeuroMAT!
If you use it in your research, please cite the following paper:

.. include:: guide/cite_snm.rst

.. include:: guide/cites_snm.rst

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guide/index
   api/index
   devel/index

