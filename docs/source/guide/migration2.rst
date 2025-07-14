**********************************************
Migrating from older versions of SuperNeuroMAT
**********************************************

.. currentmodule:: superneuromat


This page describes the necessary changes to migrate your code from SuperNeuroMAT v1.x.x to the latest version, v\ |release|.

Only breaking changes are covered on this page.

For more information on the changes, see the GitHub release notes: https://github.com/ORNL/superneuromat/releases.


``2.0.x`` to ``3.x.x``
======================

.. dropdown:: Remove calls to :py:meth:`SNN.setup`.

   Outside of manual setup mode, :py:meth:`SNN.setup` will now raise a warning, as 
   :py:meth:`SNN.simulate()` now sets up the :py:class:`SNN` automatically.

   If you need to manually setup the SNN, see :ref:`low-level-control`.


.. dropdown:: Remove ``time_steps`` parameter from :py:meth:`SNN.stdp_setup`.

   :py:meth:`SNN.stdp_setup` no longer accepts a ``time_steps`` parameter.

   Simply remove the ``time_steps`` parameter from the call. The number of STDP time steps will be
   inferred from the length of the ``apos`` and ``aneg`` lists. However, if both positive and negative
   updates are enabled, you must still keep both lists the same length.

   .. code-block:: python
      :emphasize-lines: 2

      snn.stdp_setup(
         # time_steps=3,  # remove this
         Apos=[1.0, 0.5, 0.25],
         Aneg=[-0.1, -0.05, -0.025],
      )

.. dropdown:: Add ``.idx`` when assigning :py:attr:`SNN.create_neuron` and :py:attr:`SNN.create_synapse`.

   :py:meth:`SNN.create_neuron` and :py:meth:`SNN.create_synapse` now return a :py:class:`Neuron` or :py:class:`Synapse` object.

   Previously, :py:meth:`SNN.create_neuron` and :py:meth:`SNN.create_synapse` returned the **id** of the created
   neuron or synapse as an ``int``.

   In v3, :py:class:`Neuron` and :py:class:`Synapse` classes were added. These allow for easier manipulation of the SNN,
   and now :py:meth:`SNN.create_neuron` and :py:meth:`SNN.create_synapse` return these object instances upon creation.

   However, the **id** of the created neuron or synapse can still be easily retrieved with the :py:attr:`Neuron.idx`
   and :py:attr:`Synapse.idx` attributes.

   .. code-block:: python

      neuron_id = snn.create_neuron().idx
      synapse_id = snn.create_synapse(neuron_id, neuron_id).idx

.. dropdown:: Make all elements of ``aneg`` negative in :py:meth:`SNN.stdp_setup`.

   The STDP :py:attr:`SNN.aneg` parameter now expects a list of negative values to result in the same behavior
   as ``stdp_setup(aneg)`` in v1.x.x.

   i.e. Change ``aneg=[1.0]`` to ``aneg=[-1.0]``.

   ``aneg`` represents the post-excitatory response. In SuperNeuroMAT, synapses which do not see a causal link 
   (pre-synaptic neuron fires, then post-synaptic neuron fires) typically results in a negative (inhibitory)
   update to the weight of that synapse. However, this change was made to allow for positive weight updates in
   this situation. You will now receive a warning if you use ``aneg`` with a negative value, but this can be turned
   off with the :py:attr:`SNN.allow_incorrect_stdp_sign` parameter or ``SNMAT_ALLOW_INCORRECT_STDP_SIGN`` environment
   variable.

.. dropdown:: Use ``snn.ispikes.sum()`` instead of ``snn.num_spikes``.

   The :py:attr:`SNN.num_spikes` attribute has been removed.

   See :py:attr:`SNN.ispikes` and :ref:`snm-decoding-recipes`.

.. dropdown:: Restoring weights requires memoization before :py:meth:`SNN.reset`.

   Previously, on :py:meth:`SNN.reset`, the weights of the network would be restored to the values they were
   set to at synapse creation time. This behavior has changed.

   If you want to restore the weights of a network, you must first memoize them.
   See :ref:`reset-snn` and :doc:`managing_state`. This system allows for more flexibility
   in when the weights are stored and recalled.


``1.x.x`` to ``2.0.x``
======================

.. dropdown:: ``NeuromorphicModel()`` is now :py:class:`SNN()`.

   You should now use ``snn = SNN()`` to create a new SNN.

   .. code-block:: python
      :caption: Old

      from superneuromat import NeuromorphicModel
      snn = NeuromorphicModel()

      # even older installations may have used
      from superneuromat.neuromorphicmodel import NeuromorphicModel

   .. code-block:: python
      :caption: New

      from superneuromat import SNN
      snn = SNN()

      # or
      import superneuromat as snm
      snn = snm.SNN()

.. highlights::
   
   Change the ``enable_stdp`` parameter to ``stdp_enabled`` in :py:meth:`SNN.create_synapse`.


.. dropdown:: ``pip install superneuromat`` or update your ``PYTHONPATH``.
   :open:

   Using path hacks to import SuperNeuroMAT is deprecated.
   Instead, you should use a pip editable install, and we recommend installing this
   in a virtual environment: :ref:`snm-install-editable`.

   However, it may still be possible to add your local copy of SuperNeuroMAT to your
   ``PYTHONPATH`` environment variable. This is not recommended, as it may cause
   conflicts with other packages.

   Please be aware that the path to the package has changed. The SuperNeuroMAT project
   now uses a ``src`` layout rather than a flat layout.

   This means that the path to the package changed from:

   ``superneuromat/``

   to:

   ``superneuromat/src/superneuromat``

   See :doc:`pypa:discussions/src-layout-vs-flat-layout` for more information.
