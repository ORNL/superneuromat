def count_synaptic_events(self) -> int:
    """
    Count synaptic events of a system initiated by its neuron spikes, counting all synapses whose presynaptic neurons have fired.

    Returns
    -------
    int
        Total number of synaptic events initiated during the
        recorded spike history.


    """
    spike_train = self.ispikes
    pre_ids = np.asarray(self.pre_synaptic_neuron_ids)

    total_events = 0

    for timestep_spikes in spike_train:
        firing_neurons = np.flatnonzero(timestep_spikes)
        total_events += np.isin(pre_ids, firing_neurons).sum()

    return int(total_events)