
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neuromorphicmodel import NeuromorphicModel
else:
    NeuromorphicModel = object()


class Neuron:
    def __init__(self, model: NeuromorphicModel, idx: int):
        self.m = model
        self.idx = idx

    @property
    def threshold(self):
        return self.m.neuron_thresholds[self.idx]

    @threshold.setter
    def threshold(self, value):
        self.m.neuron_thresholds[self.idx] = value

    @property
    def leak(self):
        return self.m.neuron_leaks[self.idx]

    @leak.setter
    def leak(self, value):
        self.m.neuron_leaks[self.idx] = value

    @property
    def reset_state(self):
        return self.m.neuron_reset_states[self.idx]

    @reset_state.setter
    def reset_state(self, value):
        self.m.neuron_reset_states[self.idx] = value

    @property
    def refractory_period(self):
        return self.m.neuron_refractory_periods[self.idx]

    @refractory_period.setter
    def refractory_period(self, value):
        self.m.neuron_refractory_periods[self.idx] = value

    def spikes(self):
        return self.m.spike_train[:, self.idx]

    def add_spike(self, time: int, value: float = 1.0):
        self.m.add_spike(time, self.idx, value)

    def __eq__(self, x):
        if isinstance(x, Neuron):
            return self.idx == x.idx and self.m is x.m
        else:
            return False

    def __repr__(self):
        return f"<Neuron {self.idx}>"


class NeuronList:
    def __init__(self, model: NeuromorphicModel):
        self.m = model

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Neuron(self.m, idx)
        elif isinstance(idx, slice):
            return [Neuron(self.m, i) for i in range(idx.start, idx.stop, idx.step)]
        else:
            raise TypeError("Invalid index type")

    def __len__(self):
        return len(self.m.neuron_thresholds)

    def __iter__(self):
        return NeuronIterator(self.m)


class NeuronIterator:
    def __init__(self, model: NeuromorphicModel):
        self.m = model
        self.iter = iter(range(len(self.m.neuron_thresholds)))

    def __next__(self):
        next_idx = next(self.iter)
        return Neuron(self.m, next_idx)


class Synapse:
    def __init__(self, model: NeuromorphicModel, idx: int):
        self.m = model
        self.idx = idx

    @property
    def pre(self):
        return self.m.pre_synaptic_neuron_ids[self.idx]

    @property
    def post(self):
        return self.m.post_synaptic_neuron_ids[self.idx]

    @property
    def delay(self):
        return self.m.synaptic_delays[self.idx]

    @delay.setter
    def delay(self, value):
        self.m.synaptic_delays[self.idx] = value

    @property
    def stdp_enabled(self):
        return self.m.enable_stdp[self.idx]

    @stdp_enabled.setter
    def stdp_enabled(self, value):
        self.m.enable_stdp[self.idx] = value

    @property
    def weight(self):
        return self.m.synaptic_weights[self.idx]

    @weight.setter
    def weight(self, value):
        self.m.synaptic_weights[self.idx] = value

    def __eq__(self, x):
        if isinstance(x, Synapse):
            return self.idx == x.idx and self.m is x.m
        else:
            return False

    def __repr__(self):
        return f"<Synapse {self.idx}>"


class SynapseList:
    def __init__(self, model: NeuromorphicModel):
        self.m = model

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Synapse(self.m, idx)
        elif isinstance(idx, slice):
            return [Synapse(self.m, i) for i in range(idx.start, idx.stop, idx.step)]
        else:
            raise TypeError("Invalid index type")

    def __len__(self):
        return len(self.m.synaptic_weights)

    def __iter__(self):
        return SynapseIterator(self.m)


class SynapseIterator:
    def __init__(self, model: NeuromorphicModel):
        self.m = model
        self.iter = iter(range(len(self.m.synaptic_weights)))

    def __next__(self):
        next_idx = next(self.iter)
        return Synapse(self.m, next_idx)
