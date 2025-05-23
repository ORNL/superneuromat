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

.. note::

   It is possible to send a spike to a neuron with any floating-point value, but
   when a neuron spikes, the value of the spike is always `1.0` before it gets multiplied
   by the synaptic weight.

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


Encoding Recipes
================

Rate Encoding
-------------

One method of encoding a non-binary value is to use a rate-coded spike train.

For example, if you want to encode a value of ``5``, you can send a spike
train of ``[1.0, 1.0, 1.0, 1.0, 1.0]``.

.. code-block:: python

   neuron.add_spikes(np.ones(5) * 2)

This is equivalent to:

.. code-block:: python

   for i in range(5):
       neuron.add_spike(i, 2.0)

.. note::

   The ``add_spikes()`` (plural) method is only available on neurons.

   Additionally, both the :py:meth:`Neuron.add_spike()` and :py:meth:`SNN.add_spike()` methods
   will happily send a ``0.0`` valued spike, :py:meth:`Neuron.add_spikes()` will ignore them.

One modification is to use randomly-distributed spikes instead of a sending all the spikes at the start
of the encoding period. Here's how this might be implemented to send a value between `0.0` and `1.0`
over a period of `10` time steps using :py:meth:`numpy.random.Generator.geometric()`:

.. code-block:: python

   rng = np.random.default_rng(seed=None)  # use system time as seed
   vec = rng.geometric(p=0.5, size=10)
   neuron.add_spikes(vec)

The ``Neuron.add_spikes()`` method also allows you to send spikes after a specific number of time_steps:

.. code-block:: python

   neuron.add_spikes(np.ones(5), time_offset=10)

First-to-spike
--------------

Let's say the first three neurons in our network are the input neurons.

In the first-to-spike encoding scheme, we want to spike each input neuron at least once,
but we want them to fire in a specific order.

.. todo::
   
   Add an example of first-to-spike encoding.


.. _snm-decoding-recipes:

Decoding Recipes
================

Rate Decoding
-------------

One method of decoding a non-binary value is to simply
count the number of spikes that arrive at each neuron.

You can use the :py:meth:`numpy.ndarray.sum()` to get the number of spikes that have been
emitted by each neuron as a vector:

.. code-block:: python

   snn.ispikes.sum(0)

So if your last 5 neurons are your output neurons, you can sum the spikes
emitted by each of them to get the decoded value:

.. code-block:: python

   snn.ispikes.sum(0)[-5:]

Or perhaps we only care about the last 3 timesteps:

.. code-block:: python

   snn.ispikes[-3:, -5:].sum(0)

While we're here, :py:attr:`SNN.ispikes` is a :py:class:`numpy.ndarray` of the number of spikes,
so it's useful for other things:

Get the index of the neuron that emitted the most spikes in the SNN:

.. code-block:: python

   snn.ispikes.sum(0).argmax()

Get the number of spikes emitted per timestep:

.. code-block:: python

   snn.ispikes.sum(1)

Get the total number of spikes emitted:

.. code-block:: python

   snn.ispikes.sum()

First-to-spike Decoding
-----------------------

Another method of decoding a non-binary value is to use the first-to-spike decoding scheme.

This is a population coding scheme, where each neuron is associated with a value. The
first neuron to spike in the population is the output of the network.

Let's say the last three neurons in our network are the output neurons.

.. code-block:: python

   order = []
   for time in range(decoding_start, decoding_end):
       for neuron in snn.neurons[-3:]:
           if neuron.spikes[time]:
               order.append(neuron.idx)
   
The first element, ``order[0]``, is the index of the neuron that emitted the first spike.

.. caution::

   In this encoding scheme, multiple neurons might have spiked at the same time.
   The above code will return the lowest-indexed neuron of them.

   It is also possible that no neuron has spiked and ``order`` will be empty.

   You need to decide for yourself how you want to handle these possibilities.

Reading out other information
-----------------------------

You can also read out other information on a neuron-by-neuron basis
to see what they are after the simulation:

.. code-block:: python

   for neuron in snn.neurons:
       neuron.idx  # the index of the neuron
       neuron.state  # the charge level of the neuron
       neuron.weight  # the weight of the synapse connecting the neuron to the output neuron

You can see what is available by looking at the :py:class:`Neuron` class.

If you'd like to print out information about the network, there are many methods
to help you visualize things.

If you're interested in just a single neuron or synapse, the classes :py:class:`Neuron` and :py:class:`Synapse`
have methods to print out information about them.

If you're looking for information about the entire network, 
see the many methods of the :py:class:`SNN` class in :ref:`Inspecting the SNN <inspecting-the-snn>`.