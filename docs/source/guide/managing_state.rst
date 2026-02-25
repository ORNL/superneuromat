==================
Managing SNN State
==================

.. _reset-snn:

Resetting the SNN
-----------------

Resetting the SNN is as simple as calling :py:meth:`~superneuromat.SNN.reset()`.
By default, this will reset the neuron states, zero refractory periods, spike train, and input spikes.

.. code-block:: python

    snn.reset()

This is roughly equivalent to:

.. code-block:: python

    snn.reset_neuron_states()
    snn.zero_refractory_periods()
    snn.clear_spike_train()
    snn.clear_input_spikes()
    snn.restore()

.. warning::

    This method does not reset the synaptic weights or STDP parameters.
    SuperNeuroMAT also does not automatically store the initial neuron state values, such as
    the ``initial_state`` and ``refractory_state`` parameters of :py:meth:`create_neuron`.
    When ``reset()`` is called, if those states are not memoized, each neuron's charge
    state will be set to its reset state in :py:attr:`neuron_reset_states`,
    and the refractory countdown in :py:attr:`neuron_refractory_periods_state` will be set to zero.

    If this is not desirable, consider manually copying the parameters you care
    about so you can assign them later, or using :py:meth:`memoize` to store a snapshot
    to return to when :py:meth:`restore()` or :py:meth:`reset()` is called, or manually
    calling only the individual functions that you need (shown above).

If you want more granular control, you can call the individual methods:

.. rubric:: Reset Functions

.. autosummary::
    :nosignatures:

    ~superneuromat.SNN.reset_neuron_states
    ~superneuromat.SNN.zero_refractory_periods
    ~superneuromat.SNN.clear_spike_train
    ~superneuromat.SNN.clear_input_spikes
    ~superneuromat.SNN.restore

.. rubric:: Related Functions

.. autosummary::
    :nosignatures:

    ~superneuromat.SNN.reset
    ~superneuromat.SNN.zero_neuron_states
    ~superneuromat.SNN.activate_all_refractory_periods

.. note::

    By default, the :py:attr:`~superneuromat.SNN.synaptic_weights` are not reset
    by the :py:meth:`~superneuromat.SNN.reset()` function. If you want to reset the weights, you
    must first save them by memoizing them.


Memoization
-----------

If you would like to save the state of the SNN, you can use the memoization features of SuperNeuroMAT.

The :py:meth:`~superneuromat.SNN.memoize()` function will save variables of your choosing to
a shadow copy. This shadow copy will be restored when you call :py:meth:`~superneuromat.SNN.reset()`
or :py:meth:`~superneuromat.SNN.restore()`.

.. code-block:: python

    snn.memoize(snn.synaptic_weights)
    snn.memoize(snn.enable_stdp)

    snn.simulate()

    snn.reset()

.. rubric:: Memo Functions

.. autosummary::
    :nosignatures:

    ~superneuromat.SNN.memoize
    ~superneuromat.SNN.unmemoize
    ~superneuromat.SNN.clear_memos
    ~superneuromat.SNN.restore

If you find yourself needing to store a full copy of the SNN, you can use the
:py:meth:`~superneuromat.SNN.copy()` function.

.. code-block:: python

    snn.copy()

This will return a :py:func:`~copy.deepcopy` of the SNN, which you can then use to restore the state of the original SNN:

.. code-block:: python

    original = snn.copy()

    snn.simulate()

    snn = original.copy()

The full list of SNN variables which can be memoized is available in the :py:attr:`~superneuromat.SNN.eqvars` attribute.


.. _low-level-control:

Manual Setup
------------

SuperNeuroMAT uses a bimodal representation of the SNN. The user-facing API is designed for
fast and easy building of the SNN, but the internal representation is designed for fast
and efficient simulation. When you call :py:meth:`~superneuromat.SNN.simulate()`, SuperNeuroMAT
needs to convert the user-facing API to the internal representation.
And then, once the simulation is complete, SuperNeuroMAT needs to convert the internal
representation back to the user-facing API. This setup and write-back process is implemented
in several setup and finalization functions.

Normally, there is no need to call any setup functions; SuperNeuroMAT will automatically
setup the SNN for you when you call :py:meth:`~superneuromat.SNN.simulate()`. However, in some cases,
you may want to call the setup functions yourself.

If you want granular control over the SNN setup, you can turn off the automatic setup
and finalization functions by setting :py:attr:`~superneuromat.SNN.manual_setup` to ``True``.
Otherwise, you'll get the following warning:

.. code-block:: bash

    warning: setup() called without snn.manual_setup = True. setup() will be called again in simulate().

Then, you can call the setup functions yourself:

.. code-block:: python

   snn.manual_setup = True

   snn.setup()  # call this before setting input spikes
   snn.setup_input_spikes()  # convert input spikes to internal representation

   snn.simulate(time_steps)

   # convert internal representations of neuron states, synaptic weights, and
   # refractory period states to the user-facing API
   self.devec()
   # remove the input spikes from the spike train
   self.consume_input_spikes(time_steps)

Keep in mind that once you've called :py:meth:`~superneuromat.SNN.setup()`, the SNN
will create internal numpy-based matrix representations of many model parameters to be
used in simulation. This means that if you want to modify the SNN before or after the
simulation, you'll need to call the setup or finalization functions respectively, or
the internal representations will be out of sync.

.. admonition:: Spike Train

   For the ``'cpu'`` and ``'jit'`` backends, the output spike train is always in a
   user-facing representation, so no finalization of the spike train is necessary if
   you want to read or modify the spike train after simulation.
