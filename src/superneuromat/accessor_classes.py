from .util import is_intlike

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
        self.m.neuron_thresholds[self.idx] = float(value)

    @property
    def leak(self):
        return self.m.neuron_leaks[self.idx]

    @leak.setter
    def leak(self, value):
        self.m.neuron_leaks[self.idx] = value

    @property
    def reset_state(self):
        return int(self.m.neuron_reset_states[self.idx])

    @reset_state.setter
    def reset_state(self, value):
        if not is_intlike(value):
            raise TypeError("reset_state must be int")
        self.m.neuron_reset_states[self.idx] = int(value)

    @property
    def state(self):
        return self.m.neuron_states[self.idx]

    @state.setter
    def state(self, value):
        self.m.neuron_states[self.idx] = float(value)

    @property
    def refractory_state(self):
        return int(self.m.neuron_refractory_periods_state[self.idx])

    @refractory_state.setter
    def refractory_state(self, value):
        if not is_intlike(value):
            raise TypeError("refractory_state must be int")
        self.m.neuron_refractory_periods_state[self.idx] = int(value)

    @property
    def refractory_period(self):
        return int(self.m.neuron_refractory_periods[self.idx])

    @refractory_period.setter
    def refractory_period(self, value):
        if not is_intlike(value):
            raise TypeError("refractory_period must be int")
        self.m.neuron_refractory_periods[self.idx] = int(value)

    def spikes(self):
        return self.m.ispikes[:, self.idx]

    def add_spike(self, time: int, value: float = 1.0):
        self.m.add_spike(time, self.idx, value)

    def spikes_str(self, max_steps=10, use_unicode=True):
        return self._spikes_str(self.spikes(), max_steps, use_unicode)

    @classmethod
    def _spikes_str(cls, spikes, max_steps=10, use_unicode=True):
        c0 = '-' if use_unicode else '_'
        c1 = '┴' if use_unicode else 'l'
        sep = '' if use_unicode else ' '
        ellip = '⋯' if use_unicode else '...'
        if len(spikes) > max_steps:
            fi = max_steps // 2 - 1
            li = max_steps // 2 + 1
            first = spikes[:4] if use_unicode else spikes[:fi - 1]
            last = spikes[-li:] if use_unicode else spikes[-li - 1:]
            s = sep.join([c1 if x else c0 for x in first] + [ellip] + [c1 if x else c0 for x in last])
        else:
            s = sep.join([c1 if x else c0 for x in spikes])
        return f"[{s}]"

    def __eq__(self, x):
        if isinstance(x, Neuron):
            return self.idx == x.idx and self.m is x.m
        else:
            return False

    def __repr__(self):
        return f"<Virtual Neuron {self.idx} on model at {hex(id(self.m))}>"

    def info(self):
        return ' | '.join([
            f"id: {self.idx:>5d}",
            f"state: {self.state:f}",
            f"thresh: {self.threshold:>f}",
            f"leak: {self.leak:7f}",
            f"ref_state: {self.refractory_state:>3d}",
            f"ref_period: {self.refractory_period:>3d}",
        ])

    def __str__(self):
        return f"<Neuron {self.info()}>"

    def info_row(self):
        return ''.join([
            f"{self.idx:>5d}\t",
            f"{self.state:>11.9g}\t",
            f"{self.threshold:>11.9g}\t",
            f"{self.leak:>8.6g}  ",
            f"{self.refractory_state:>3d} ",
            f"{self.refractory_period:>3d}\t",
            self.spikes_str(max_steps=10),
        ])

    @classmethod
    def row_header(cls):
        return "  idx         state          thresh         leak  ref per       spikes"

    @classmethod
    def row_cont(cls):
        return "  ...           ...             ...          ...  ... ...       [...]"


class NeuronList:
    def __init__(self, model: NeuromorphicModel):
        self.m = model

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Neuron(self.m, idx)
        elif isinstance(idx, slice):
            indices = list(range(self.m.num_neurons))[idx]
            return [Neuron(self.m, i) for i in indices]
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
        return f"<Virtual Synapse {self.idx} on model at {hex(id(self.m))}>"

    def info(self):
        return ' | '.join([
            f"id: {self.idx:>5d}",
            f"pre: {self.pre:>5d}",
            f"post: {self.post:>5d}",
            f"\tweight: {self.weight:f}\t",
            f"delay: {self.delay:>3d}",
            f"stdp {' en' if self.stdp_enabled else 'dis'}abled",
        ])

    def __str__(self):
        return self.info()

    def info_row(self):
        return ''.join([
            f"{self.idx:>5d}\t",
            f"{self.pre:>5d}  ",
            f"{self.post:>5d}\t",
            f"{self.weight:>12f}\t",
            f"{self.delay:>5d}\t",
            f"{'X' if self.stdp_enabled else '-'}",
        ])

    @staticmethod
    def row_header():
        return "  idx\t  pre   post\t      weight\tdelay\tstdp_enabled"

    @staticmethod
    def row_cont():
        return "  ...\t  ...    ...\t         ...\t  ...\t..."


class SynapseList:
    def __init__(self, model: NeuromorphicModel):
        self.m = model

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return Synapse(self.m, idx)
        elif isinstance(idx, slice):
            indices = list(range(self.m.num_synapses))[idx]
            return [Synapse(self.m, i) for i in indices]
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
