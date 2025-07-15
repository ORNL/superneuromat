superneuromat.SNN
=================

.. currentmodule:: superneuromat



.. py:class:: SNN()

   :param num_neurons: The number of neurons in the SNN.
   :type num_neurons: int
   :param neurons: List of neurons in the SNN.
   :type neurons: NeuronList
   :param num_synapses: The number of synapses in the SNN.
   :type num_synapses: int
   :param synapses: List of synapses in the SNN.
   :type synapses: SynapseList
   :param spike_train: List of output spike trains for each time step. Index by ``spike_train[t][neuron_id]``.
   :type spike_train: list
   :param ispikes: Binary numpy array of input spikes for each time step. Index by ``ispikes[t, neuron_id]``.
   :type ispikes: numpy.ndarray[(int, int), bool]

   .. rubric:: STDP Parameters

   :param stdp: if ``stdp`` is ``False``, STDP will be disabled globally.
   :type stdp: bool, default=True
   :param apos: List of STDP parameters per time step for excitatory update of weights.
   :type apos: list, default=[]
   :param aneg: List of STDP parameters per time step for inhibitory update of weights.
   :type aneg: list, default=[]
   :param stdp_positive_update: If ``False``, disable excitatory STDP update globally.
   :type stdp_positive_update: bool | ~typing.Any, default=True
   :param stdp_negative_update: If ``False``, disable inhibitory STDP update globally.
   :type stdp_negative_update: bool | ~typing.Any, default=True
   :param stdp_time_steps: Number of time steps over which STDP updates will be made. This is determined
      from the length of apos or aneg.
   :type stdp_time_steps: int

   .. rubric:: Config Parameters

   :param default_dtype: Default data type for internal numpy representation of all floating-point arrays.
   :type default_dtype: type | numpy.dtype, default=numpy.float64
   :param default_bool_dtype: Default data type for internal numpy representation of stdp_enabled values and spike vectors.
   :type default_bool_dtype: type | numpy.dtype, default=bool
   :param gpu: If ``False``, disable GPU acceleration. By default, this is ``True`` if ``numba.cuda.is_available()``.
   :type gpu: bool
   :param manual_setup: If ``True``, disable automatic setup of SNN when :py:meth:`simulate()` is called.
   :type manual_setup: bool, default=False
   :param allow_incorrect_stdp_sign: If ``True``, allow negative values in :py:attr:`apos` and positive values in :py:attr:`aneg`.
   :type allow_incorrect_stdp_sign: bool, default=False
   :param allow_signed_leak: If ``True``, allow negative values in :py:attr:`neuron_leaks`. Normally, positive leak will result in charge decay.
   :type allow_signed_leak: bool, default=False
   :param backend: The backend to use for simulation. Can be ``'auto'``, ``'cpu'``, ``'jit'``, or ``'gpu'``.
      Can be overridden on a per-\ :py:meth:`simulate()` basis.
   :type backend: str, default='auto'
   :param sparse: If ``True``, use sparse representation of SNN. If ``'auto'``, use sparse representation if ``num_neurons > 100``.
      Sparse representation is only available using the ``'cpu'`` backend.
   :type sparse: bool | str | ~typing.Any, default=False
   :param last_used_backend: The backend most recently used for simulation.
   :type last_used_backend: str, default=None
   :param is_sparse: If ``True``, the SNN is being internally represented using a sparse representation.
   :type is_sparse: bool, default=False

   .. rubric:: Data Attributes

   These attributes are not intended to be modified directly by end users.

   :type neuron_thresholds: list
   :param neuron_thresholds:
      List of neuron thresholds
   :type neuron_leaks: list
   :param neuron_leaks:
      List of neuron leaks
      defined as the amount by which the internal states of the neurons are pushed towards the neurons' reset states
   :type neuron_reset_states: list
   :param neuron_reset_states:
      List of neuron reset states
   :type neuron_refractory_periods: list
   :param neuron_refractory_periods:
      List of neuron refractory periods
   :type pre_synaptic_neuron_ids: list
   :param pre_synaptic_neuron_ids:
      List of pre-synaptic neuron IDs
   :type post_synaptic_neuron_ids: list
   :param post_synaptic_neuron_ids:
      List of post-synaptic neuron IDs
   :type synaptic_weights: list
   :param synaptic_weights:
      List of synaptic weights
   :type synaptic_delays: list
   :param synaptic_delays:
      List of synaptic delays
   :type enable_stdp: list
   :param enable_stdp:
      List of Boolean values denoting whether STDP learning is enabled on each synapse
   :type input_spikes: dict
   :param input_spikes:
      Dictionary of input spikes for each time step.





   .. warning::

      Delay is implemented by adding a chain of proxy neurons.

      A delay of 10 between neuron A and neuron B would add 9 proxy neurons between A and B.
      This may result in severe performance degradation. Consider using sparse representation
      or agent-based SNN simulators in networks with high delay times.

   .. hint::

      *  Leak brings the internal state of the neuron closer to the reset state.

         The leak value is the amount by which the internal state of the neuron is pushed towards its reset state.

      *  Deletion of neurons may result in unexpected behavior.

      *  Input spikes can have a value

      *  All neurons are monitored by default

      
   .. automethod:: SNN.__init__

Methods
~~~~~~~

.. autosummary::

   SNN.__init__

.. rubric:: Building the SNN

.. autosummary::
   :toctree: _gen/
   :nosignatures:

   ~SNN.add_spike
   ~SNN.create_neuron
   ~SNN.create_synapse
   ~SNN.stdp_setup
   ~SNN.set_stdp_enabled_from_mat
   ~SNN.set_weights_from_mat


.. _inspecting-the-snn:

.. rubric:: Inspecting the SNN

.. autosummary::
   :toctree: _gen/
   :nosignatures:

   ~SNN.get_neuron_df
   ~SNN.neuron_spike_totals
   ~SNN.get_synapse
   ~SNN.get_synapse_df
   ~SNN.get_synapses_by_post
   ~SNN.get_synapses_by_pre
   ~SNN.get_synaptic_id
   ~SNN.get_synaptic_ids_by_post
   ~SNN.get_synaptic_ids_by_pre
   ~SNN.get_weights_df
   ~SNN.input_spikes_info
   ~SNN.get_input_spikes_df
   ~SNN.get_stdp_enabled_df
   ~SNN.last_used_backend
   ~SNN.pretty
   ~SNN.pretty_print
   ~SNN.pretty_spike_train
   ~SNN.print_spike_train
   ~SNN.short
   ~SNN.neuron_info
   ~SNN.stdp_info
   ~SNN.synapse_info
   ~SNN.stdp_enabled_mat
   ~SNN.stdp_enabled_sparse
   ~SNN.weight_mat
   ~SNN.weight_sparsity
   ~SNN.weights_sparse


.. rubric:: Managing Model State

.. autosummary::
   :toctree: _gen/
   :nosignatures:

   ~SNN.clear_input_spikes
   ~SNN.clear_memos
   ~SNN.clear_spike_train
   ~SNN.consume_input_spikes
   ~SNN.copy
   ~SNN.devec
   ~SNN.memoize
   ~SNN.recommend
   ~SNN.recommend_sparsity
   ~SNN.release_mem
   ~SNN.reset
   ~SNN.reset_neuron_states
   ~SNN.reset_refractory_periods
   ~SNN.restore
   ~SNN.setup
   ~SNN.setup_input_spikes
   ~SNN.simulate
   ~SNN.simulate_cpu
   ~SNN.simulate_cpu_jit
   ~SNN.simulate_gpu
   ~SNN.unmemoize
   ~SNN.zero_neuron_states
   ~SNN.zero_refractory_periods
   ~SNN.rebuild_connection_ids
   ~SNN.to_json
   ~SNN.saveas_json
   ~SNN.from_jsons


Attributes
~~~~~~~~~~

.. rubric:: Attributes

.. autosummary::
   :toctree: _gen/

   ~SNN.num_neurons
   ~SNN.num_synapses
   ~SNN.ispikes
   ~SNN.backend
   ~SNN.disable_performance_warnings
   ~SNN.eqvars
   ~SNN._internal_vars
   ~SNN.is_sparse
   ~SNN.sparse
   ~SNN.stdp_time_steps

   .. ~SNN.dbin
   .. ~SNN.dd
   ~SNN.jit_threshold
   ~SNN.gpu_threshold
   ~SNN.sparsity_threshold
