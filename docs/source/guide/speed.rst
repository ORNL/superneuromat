************************
Considerations for Speed
************************

SuperNeuroMAT uses a bimodal representation of the SNN. The user-facing API, the frontend,
is designed for fast and easy building of the SNN, but the backend uses its own internal
representation. Most of the translation between frontend and backend representations is
handled by the :py:meth:`~superneuromat.SNN.simulate()` function.

Choosing a Backend
==================

There are several different backends available, each with certain advantages and disadvantages.
SuperNeuroMAT automatically chooses a backend based on several factors:

* Availability of ``'jit'`` and ``'gpu'`` backends
* Number of neurons in the SNN
* Number of synapses in the SNN
* Number of time steps in the simulation

However, the 
If you want to force SuperNeuroMAT to use a specific backend and sparsity mode, you can set the
:py:attr:`~superneuromat.SNN.backend` and :py:attr:`~superneuromat.SNN.sparse` attributes.

Here are the available backends:

* ``'auto'``: Choose the fastest backend available based on the above factors.
* ``'cpu'``: Use the CPU backend.

   The CPU backend is the most widely supported backend, but its dense representation
   is often the slowest implementation.
   The CPU backend is also the only backend that supports sparse representations.

* ``'jit'``: Use the JIT backend.

   The ``'jit'`` backend uses just-in-time compilation to avoid some of the overhead
   of python. However, there is an initial compilation cost at the first simulation call.
   It is chosen when the number of neurons is large or the number of time steps is large.
   Currently, it does not support sparse mode.

* ``'gpu'``: Use the GPU backend.

   The ``'gpu'`` backend uses the CUDA GPU to accelerate the simulation.
   It tends to be faster than the ``'jit'`` backend, and excels at processing dense
   networks with many neurons, many synapses, and many ``simulate()`` time steps.
   However, having many STDP time steps may cause slow operation, and GPU memory tends
   to be more limited than CPU memory, limiting the size of the SNN that can be simulated.

   SuperNeuroMAT does not check for GPU memory size, so you should make sure you have enough
   GPU memory available before running a simulation with this backend.

   Currently, it does not support sparse mode.

If :py:attr:`~superneuromat.SNN.sparse` is ``True``, SuperNeuroMAT will use a sparse representation
of the SNN. This is only available using the ``'cpu'`` backend. If the backend is ``'auto'``,
the ``'cpu'`` backend will be used so that the sparse mode is available.

Sparse mode excels at simulating networks with very few synapses and is less likely to be
affected by having many STDP time steps.


Multiprocessing and Multi-Threading
===================================

Many of SuperNeuroMAT's backends use NumPy and SciPy to perform matrix operations.
These libraries in turn use BLAS and LAPACK libraries, which tend to be C-based linear algebra libraries.

Many implementations of these libraries will multi-thread matrix functions behind the scenes.
This means that you might not see a linear performance increase by implementing concurrent processing
of multiple SuperNeuroMAT models, as a single model may already be using multiple threads.

You may want to see Jason Brownlee's
`Whicn NumPy Functions are Multithreaded <https://superfastpython.com/multithreaded-numpy-functions/>`_
post for more information.

.. note::

   This applies to the ``'cpu'`` and ``'jit'`` backends, but not the ``'gpu'`` backend, which
   may still benefit from ```python.multiprocessing``, as long as your GPU has enough memory and cores.

Memory Usage
============

The size of the SNN is determined by the number of neurons and synapses in the SNN.
SuperNeuroMAT needs to store properties of each neuron and synapse, so for large networks,
it may help to understand the different possible representations of the SNN.

Sparse vs. Dense
----------------

An SNN is made of neurons and synapses. A synapse is a connection between two neurons,
so to represent a synapse, we need to store the ID of its pre- and post-synaptic neurons, as well as
the weight of the synapse.

One way to represent a synapse is to store the weight of the synapse in a dense matrix, the column index
and row index represent the pre- and post-synaptic neurons, and the value of the matrix represents the weight.
This is the dense representation, and it tends to be quite large, as it stores the weight of every possible
connection between every neuron in the network, even if we don't want those connections. This means that
the dense weight matrix always represents ``num_neurons ** 2`` synapses even if we have fewer synapses than that.

Another way to represent a synapse is to store the weight of the synapse in a sparse matrix, which is a
matrix that only stores the non-zero values of the matrix. This is the sparse representation, and it tends
to be much smaller than the dense representation, as it only stores the weight of the connections that
we actually make.

However, a sparse representation has some overhead, as we need to store which connections we care about.
This is fine if the number of synapses is much smaller than the number of neurons, but if we have lots of
synapses, we might end up using more memory to store the addresses of the source and destination neurons
than a dense matrix, which doesn't use any memory for the addresses.

Frontend
--------

The frontend representation of the SNN uses a sparse representation as Python list objects.

There are several lists which are used to represent the SNN.

.. rubric:: Neuron Properties

* :py:attr:`~superneuromat.SNN.neuron_thresholds`: The threshold value for each neuron.
* :py:attr:`~superneuromat.SNN.neuron_leaks`: The leak value for each neuron.
* :py:attr:`~superneuromat.SNN.neuron_reset_states`: The reset state for each neuron.
* :py:attr:`~superneuromat.SNN.neuron_refractory_periods`: The refractory period for each neuron.
* :py:attr:`~superneuromat.SNN.neuron_refractory_periods_state`: The refractory period state for each neuron.

.. rubric:: Synapse Properties

* :py:attr:`~superneuromat.SNN.pre_synaptic_neuron_ids`: The ID of the pre-synaptic neuron for each synapse.
* :py:attr:`~superneuromat.SNN.post_synaptic_neuron_ids`: The ID of the post-synaptic neuron for each synapse.
* :py:attr:`~superneuromat.SNN.synaptic_weights`: The weight of each synapse.
* :py:attr:`~superneuromat.SNN.synaptic_delays`: The delay of the synapse.
* :py:attr:`~superneuromat.SNN.enable_stdp`: The STDP enabled state of each synapse.

Backend
-------

The backend representation consists of NumPy or SciPy arrays.

For most properties, the backend representation is a NumPy vector, that is, a 1-dimensional array.
For example, the :py:attr:`~superneuromat.SNN.neuron_thresholds` property is a 1-dimensional array where each
element is the threshold value for a neuron.

However, for the :py:attr:`~superneuromat.SNN.synaptic_weights` and :py:attr:`~superneuromat.SNN.enable_stdp`
properties, the backend representation may be sparse or dense, as we need to represent a property for each connection between neurons.

In dense mode, a the synaptic weights are stored as a 2-dimensional array where each element is the weight of a synapse.
In sparse mode, the synaptic weights are stored as a sparse matrix, where the pre- and post-synaptic neuron IDs
are stored as vectors, and the weight of the synapse is stored as a vector. This only stores the weights of the connections
that are active in the network.

.. card:: :fas:`memory` Releasing backend memory

   When you call :py:meth:`~superneuromat.SNN.simulate()` or :py:meth:`~superneuromat.SNN.setup()`, SuperNeuroMAT
   will create the internal representation. During multiprocessing, a Python process may not exit cleanly, or Python's reference
   counting may not be accurate. This can result in Python failing to garbage collect the internal representation of completed processes.
   To avoid this, you can call :py:meth:`~superneuromat.SNN.release_mem()` after :py:meth:`~superneuromat.SNN.simulate()`

   .. warning::

      :py:meth:`~superneuromat.SNN.release_mem()` is a low-level function and has few safeguards.
      Use with caution. It will call :ref:`del` on all numpy internal state variables.

      This means that errors will be raised if called before :py:meth:`~superneuromat.SNN.setup()` or :py:meth:`~superneuromat.SNN.simulate()`,
      or if called more than once in a row.

      .. code-block:: python

         snn.simulate()
         # only call release_mem() after setting up the SNN
         snn.release_mem()

Synaptic Delays
===============

Synaptic delays are implemented by adding a chain of proxy neurons.

A delay of 10 between neuron A and neuron B would add 9 proxy neurons between A and B.
This may result in severe performance degradation. Consider using sparse representation
or agent-based SNN simulators in networks with high delay times.

