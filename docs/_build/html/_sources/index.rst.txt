.. SuperNeuroMAT documentation master file, created by
   sphinx-quickstart on Fri Apr 18 21:50:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SuperNeuroMAT's documentation!
=========================================

SuperNeuroMAT is a matrix-based simulator for simulating spiking neural networks, which are used in neuromorphic computing. It is one of the fastest, if not the fastest, simlators for simulating spiking neural networks.

Some salient features of SuperNeuroMAT are:

#. Support for leaky integrate and fire neuron model with the following parameters: neuron threshold, neuron leak, and neuron refractory period
#. Support for Spiking-Time-Dependent Plasticity (STDP) synapses with weights and delays
#. No restrictions on connectivity of the neurons, all-to-all connections as well as self connections possible
#. Constant leak supported
#. STDP learning can be configured to turn on/off positive updates and negative updates
#. Excessive synaptic delay can slow down the execution of the simulation, so try to avoid as much as possible
#. Leak refers to the constant amount by which the internal state (membrane potential) of a neuron changes in each time step of the simulation; therefore, zero leak means the neuron fully retains the value in its internal state, and infinite leak means the neuron never retains the value in its internal state
#. STDP implementation is extremely fast
#. The model of neuromorphic computing supported in SuperNeuroMAT is Turing-complete
#. All underlying computational operations are matrix-based and currently supported on CPUs


.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. include:: installation.rst
.. include:: usage.rst
.. include:: development.rst
.. include:: citation.rst
.. include:: superneuromat.rst



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`