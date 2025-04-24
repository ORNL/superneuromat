***********
Basic Usage
***********

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

   print(snn)

You should see that Neuron A now has a weight of 1.0, which is barely not enough for the neuron to spike.
Let's add another spike to neuron A:

.. code-block:: python
   :caption: Python

   a.add_spike(0, 1.0)

   snn.simulate(2)

   print(snn)

You should see something like this:

.. code-block:: bash

   <SNN with 2 neurons and 1 synapses @ 0x1e000f00ba4>
   STDP is globally  enabled with 0 time steps
   apos: []
   aneg: []

   idx         state          thresh         leak  ref per       spikes
      0             0               1            0    0   0       [--┴-]
      1             1               1            0    0   0       [----]

   Synapse Info:
   idx     pre   post         weight     delay   stdp_enabled
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

