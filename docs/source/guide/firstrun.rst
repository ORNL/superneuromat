***********
Basic Usage
***********

.. currentmodule:: superneuromat

.. |_| unicode:: 0xA0 
   :trim:

Now that you've gone through :doc:`installing SuperNeuroMAT </guide/install>`\ , let's get started with some basic usage.


Creating a neuromorphic model
=============================

To create a neuromorphic model, you can use the :py:class:`~superneuromat.SNN` class.

Let's create a simple neuromorphic model with two neurons and one synapse.

We'll use the :py:meth:`~superneuromat.SNN.create_neuron()` and :py:meth:`~superneuromat.SNN.create_synapse()` methods
to add neurons and synapses to the SNN.

Then, we'll queue a spike from neuron A to neuron B. This can be done with the :py:meth:`Neuron.add_spike()` method,
or the :py:meth:`SNN.add_spike()` method.

.. code-block:: python
   :caption: Python

   from superneuromat import SNN

   snn = SNN()

   a = snn.create_neuron(threshold=1.0, leak=0)
   b = snn.create_neuron(threshold=1.0, leak=0)

   snn.create_synapse(a, b, weight=1.0, delay=1)

   a.add_spike(time=1, value=1.0)

   snn.simulate(time_steps=2)

   print(snn.neurons)


The :py:meth:`simulate()` function simulates the neuromorphic circuit for a given number of time steps.
During simulation, the input spikes we queued up are sent to the neurons.
Here, we sent a spike of value `1.0` to neuron ``a``. The ``time`` parameter of :py:meth:`Neuron.add_spike()` is the
number of time steps to wait before sending the spike, so it is sent *after* the *first* time step.

We can see that ``print(snn.neurons)`` shows us the state of the neurons:

.. code-block:: bash
   :caption: Result

   Neuron Info (2):
   idx         state          thresh         leak  ref per       spikes
     0             1               1            0    0   0       [--]
     1             0               1            0    0   0       [--]

By default, each neuron created has a charge state of `0.0`. After sending a spike of value `1.0` to neuron ``a``,
you should see that neuron ``a`` now has a weight of `1.0`, which is barely not enough for the neuron to spike.

Integrate and Fire
==================

Let's add another spike to neuron ``a``:

.. code-block:: python
   :caption: Python

   a.add_spike(0, 1.0)

   snn.simulate(2)

   print(snn)


You should see something like this:

.. code-block:: bash
   :caption: Result

   SNN with 2 neurons and 1 synapses @ 0x1e000f00ba4
   STDP is globally  enabled with 0 time steps
   apos: []
   aneg: []

   Neuron Info(2):
   idx          state          thresh         leak  ref per       spikes
      0             0               1            0    0   0       [--┴-]
      1             1               1            0    0   0       [----]

   Synapse Info (1):
   idx      pre   post         weight     delay   stdp_enabled
      0       0      1              1         1   -

   Input Spikes:
   Time:  Spike-value    Destination


   Spike Train:
   t:  0 1
   0: [│ │ ]
   1: [│ │ ]
   2: [├─│ ]
   3: [│ │ ]
   1 spikes since last reset

.. grid:: 1

   .. grid-item-card:: :fas:`info` |_| |_| **What just happened?**

      We sent another spike of value `1.0` to neuron ``a``. During ``snn.simulate(2)``,
      neuron ``a``'s state increased from `1.0` to `2.0`. Since ``a.state > a.threshold``,
      the neuron spiked, and its charge state was reset to `0.0`.

      Because neuron ``a`` is connected to neuron ``b``, the spike travels through
      the synapse and is sent to neuron ``b``, and ``b.state`` increased from `0.0` to `1.0`.

This is the Integrate-and-Fire part of the Leaky Integrate and Fire (**LIF**) model, which is a heavily simplified model of how biological neurons
operate. We'll talk about Leak, Refractory period, and Reset states later. For now, let's move on to the STDP part of the model.

Learning with STDP
==================

Neural networks often need to be trained to be useful, but Spiking Neural Networks (SNNs)
cannot learn like other neural networks which use gradient descent, because spikes are
non-differentiable.

SNNs can learn using other mechanisms, such as gradient surrogates or evolutionary algorithms,
but the primary method of learning supported by SuperNeuroMAT is Spike-Timing-Dependent
Plasticity (STDP).

STDP is a mechanism that allows neurons to adapt their firing rates based on the timing of
spikes. It is a form of synaptic plasticity, which means that the strength of the synapses
between neurons can change over time.

The theory is simple: Connections between neurons that fire at the same time should be stronger,
and connections between neurons that fire at different times should be weaker.
This is a simplified version of `Hebb's rule <https://en.wikipedia.org/wiki/Hebbian_theory>`_,
often referred to by the adage, "Neurons that fire together, wire together."

To use STDP, you need to set the amount that the synaptic weight should change if
neurons fire at the same time. This is done with the :py:attr:`~SNN.apos` attribute.

.. code-block:: python
   :caption: Python

   from superneuromat import SNN

   snn = SNN()

   snn.apos = [0.1]

   a = snn.create_neuron(threshold=1.0, leak=0)
   b = snn.create_neuron(threshold=1.0, leak=0)

   snn.create_synapse(a, b, weight=1.0, delay=1, stdp_enabled=True)

   a.add_spike(time=0, value=2.0)
   b.add_spike(time=0, value=2.0)
   a.add_spike(time=1, value=2.0)
   b.add_spike(time=1, value=2.0)

   snn.simulate(time_steps=2)

   print(snn.neurons)
   print(snn.synapses)

.. code-block:: bash
   :caption: Result

   Neuron Info (2):
   idx         state          thresh         leak  ref per       spikes
     0             0               1            0    0   0       [┴┴]
     1             0               1            0    0   0       [┴┴]

   Synapse Info (1):
   idx     pre   post         weight     delay   stdp_enabled
     0       0      1            1.1         1   Y

As you can see, the synapse between neurons ``a`` and ``b`` has a weight of `1.1`.

.. note::

   The STDP update takes place across two timesteps. This is because `Hebb's rule <https://en.wikipedia.org/wiki/Hebbian_theory>`_
   applies to causal relationships. A neuron can't directly cause another neuron to fire in the same timestep as SuperNeuroMAT
   doesn't allow for zero-delay synapses. So, STDP applies to the current spike output, and the spike output
   from previous timesteps.

   To see the second set of spikes update the weight, run ``snn.simulate()`` again, and the synapse will have a weight of `1.2`.

You can choose a different amount of positive or negative update to occur between the current timestep and
any previous timestep:

.. code-block:: python
   :caption: Python

   snn.add_spike(time=0, neuron_id=0, value=2.0)

   snn.apos = [1.0, 0.0]
   snn.aneg = [0.0, 0.1]

   snn.simulate(time_steps=3)

   print(snn.synapses)

You can also temporarily disable STDP without setting ``apos`` or ``aneg`` to be empty:

.. code-block:: python

   snn.stdp_positive_update = False  # turns off positive update
   snn.stdp_negative_update = False  # turns off negative update
   snn.stdp = False  # turns off STDP entirely

   # prevent a particular synapse's weights from changing
   snn.synapses[0].stdp_enabled = False  # turns off STDP for synapse 0
   # you can also do this when creating the synapse
   snn.create_synapse(0, 1, stdp_enabled=False)  # False is the default

