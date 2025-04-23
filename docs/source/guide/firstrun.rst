***********
Basic Usage
***********

Now that you've gone through :doc:`installing SuperNeuroMAT </guide/install>`\ , let's get started with some basic usage.


Creating a neuromorphic model
=============================

To create a neuromorphic model, you can use the :py:class:`~superneuromat.SNN` class.

Let's create a simple neuromorphic model with two neurons and one synapse.

We'll use the :py:meth:`~superneuromat.SNN.add_neuron()` and :py:meth:`~superneuromat.SNN.add_synapse()` methods
to add neurons and synapses to the SNN.

Then, we'll queue a spike from neuron A to neuron B. This can be done with the :py:meth:`Neuron.add_spike()` method,
or the :py:meth:`SNN.add_spike()` method.

.. code-block:: python
   :caption: Python

   from superneuromat import SNN

   snn = SNN()

   a = snn.add_neuron(threshold=1.0, leak=0)
   b = snn.add_neuron(threshold=1.0, leak=0)

   snn.add_synapse(a, b, weight=1.0, delay=1)

   a.add_spike(time=0, value=1.0)

   snn.simulate(3)

   print(snn)



