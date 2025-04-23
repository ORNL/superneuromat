from .util import is_intlike

from typing import TYPE_CHECKING
import numpy

if TYPE_CHECKING:
    from .neuromorphicmodel import SNN
    from typing import overload
else:
    class SNN:
        def __repr__(self):
            return "SNN"


class Neuron:
    """Accessor Class for Neurons in SNNs


    .. warning::

        Instances of Neurons are created at access time and are not unique.
        Multiple instances of this class may be created for the same neuron on the SNN.
        To test for equality, use ``==`` instead of ``is``.

    """
    def __init__(self, model: SNN, idx: int):
        self.m = model
        #: The index of this neuron in the SNN.
        self.idx = idx

    @property
    def threshold(self) -> float:
        """The > threshold value for this neuron to spike."""
        return self.m.neuron_thresholds[self.idx]

    @threshold.setter
    def threshold(self, value: float):
        self.m.neuron_thresholds[self.idx] = float(value)

    @property
    def leak(self) -> float:
        """The amount by which the internal state of this neuron is pushed towards its reset state."""
        return self.m.neuron_leaks[self.idx]

    @leak.setter
    def leak(self, value: float):
        self.m.neuron_leaks[self.idx] = value

    @property
    def reset_state(self) -> float:
        """The charge state of this neuron immediately after spiking."""
        return self.m.neuron_reset_states[self.idx]

    @reset_state.setter
    def reset_state(self, value: float):
        self.m.neuron_reset_states[self.idx] = float(value)

    @property
    def state(self) -> float:
        """The charge state of this neuron."""
        return self.m.neuron_states[self.idx]

    @state.setter
    def state(self, value) -> float:
        self.m.neuron_states[self.idx] = float(value)

    @property
    def refractory_state(self) -> float:
        """The remaining number of time steps for which this neuron is in its refractory period."""
        return self.m.neuron_refractory_periods_state[self.idx]

    @refractory_state.setter
    def refractory_state(self, value: float):
        self.m.neuron_refractory_periods_state[self.idx] = value

    @property
    def refractory_period(self) -> float:
        """The number of time steps for which this neuron should be in its refractory period."""
        return self.m.neuron_refractory_periods[self.idx]

    @refractory_period.setter
    def refractory_period(self, value: float):
        self.m.neuron_refractory_periods[self.idx] = float(value)

    @property
    def spikes(self) -> numpy.ndarray[(int), bool] | list:
        """A vector of the spikes that have been emitted by this neuron."""
        if self.m.spike_train:
            return self.m.ispikes[:, self.idx]
        else:
            return []

    def add_spike(self, time: int, value: float = 1.0):
        """Queue a spike to be sent to this Neuron.

        Parameters
        ----------
        time : int
            The number of time_steps until the spike is sent.
        value : float, default=1.0
            The value of the spike.
        """
        self.m.add_spike(time, self.idx, value)

    def connect_child(self, child, weight: float = 1.0, delay: int = 1, stdp_enabled: bool = False):
        """Connect this neuron to a child neuron.

        Parameters
        ----------
        child : Neuron | int
            The child neuron that will receive the spikes from this neuron.
        weight : float, default=1.0
            The weight of the synapse connecting this neuron to the child.
        delay : int, default=1
            The delay of the synapse connecting this neuron to the child.
        stdp_enabled : bool, default=False
            If ``True``, enable STDP learning on the synapse connecting this neuron to the child.
        """
        if isinstance(child, Neuron):
            child = child.idx
        self.m.create_synapse(self.idx, child, weight=weight, delay=delay, stdp_enabled=stdp_enabled)

    def connect_parent(self, parent, weight: float = 1.0, delay: int = 1, stdp_enabled: bool = False):
        """Connect this neuron to a parent neuron.

        Parameters
        ----------
        parent : Neuron | int
            The parent neuron that will send spikes to this neuron.
        weight : float, default=1.0
            The weight of the synapse connecting the parent to this neuron.
        delay : int, default=1
            The delay of the synapse connecting the parent to this neuron.
        stdp_enabled : bool, default=False
            If ``True``, enable STDP learning on the synapse connecting the parent to this neuron.
        """
        if isinstance(parent, Neuron):
            parent = parent.idx
        self.m.create_synapse(parent, self.idx, weight=weight, delay=delay, stdp_enabled=stdp_enabled)

    def spikes_str(self, max_steps=10, use_unicode=True):
        """Returns a pretty string of the spikes that have been emitted by this neuron.

        Parameters
        ----------
        max_steps : int | None, default=10
            Limits the number of steps which will be included.
            If limited, only a total of ``max_steps`` first and last steps will be included.
        use_unicode : bool, default=True
            If ``True``, use unicode characters to represent spikes.
            Otherwise fallback to ascii characters.
        """
        return self._spikes_str(self.spikes, max_steps, use_unicode)

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
        """Check if two Neuron instances represent the same neuron in the SNN."""
        if isinstance(x, Neuron):
            return self.idx == x.idx and self.m is x.m
        else:
            return False

    def __repr__(self):
        return f"<Virtual Neuron {self.idx} on model at {hex(id(self.m))}>"

    def info(self):
        """Returns a string containing information about this neuron."""
        return ' | '.join([
            f"id: {self.idx:d}",
            f"state: {self.state:f}",
            f"thresh: {self.threshold:f}",
            f"leak: {self.leak:f}",
            f"ref_state: {self.refractory_state:d}",
            f"ref_period: {self.refractory_period:d}",
        ])

    def __str__(self):
        return f"<Neuron {self.info()}>"

    def info_row(self):
        """Returns a string containing information about this neuron for use in a table."""
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
    """Redirects indexing to the SNN's neurons.

    Returns a :py:class:`Neuron` or a list of Neurons.

    This is used to allow for the following syntax:

    .. code-block:: python

        snn.neurons[0]
        snn.neurons[1:10]
    """
    def __init__(self, model: SNN):
        self.m = model

    if TYPE_CHECKING:
        @overload
        def __getitem__(self, idx: int) -> Neuron: ...
        @overload
        def __getitem__(self, idx: slice) -> list[Neuron]: ...

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
    def __init__(self, model: SNN):
        self.m = model
        self.iter = iter(range(len(self.m.neuron_thresholds)))

    def __next__(self):
        next_idx = next(self.iter)
        return Neuron(self.m, next_idx)


class Synapse:
    """Synapse accessor class for synapses in an SNN


    .. warning::

        Instances of Synapse are created at access time and are not unique.
        Multiple instances of this class may be created for the same synapse on the SNN.
        To test for equality, use ``==`` instead of ``is``.

    """
    def __init__(self, model: SNN, idx: int):
        self.m = model
        #: The index of this synapse in the SNN.
        self.idx = idx

    @property
    def pre(self) -> Neuron:
        """The pre-synaptic neuron of this synapse."""
        pre = self.m.pre_synaptic_neuron_ids[self.idx]
        return self.m.neurons[pre]

    @property
    def pre_idx(self) -> int:
        """The index of the pre-synaptic neuron of this synapse."""
        return self.m.pre_synaptic_neuron_ids[self.idx]

    @property
    def post(self) -> Neuron:
        """The post-synaptic neuron of this synapse."""
        post = self.m.post_synaptic_neuron_ids[self.idx]
        return self.m.neurons[post]

    @property
    def post_idx(self) -> int:
        """The index of the post-synaptic neuron of this synapse."""
        return self.m.post_synaptic_neuron_ids[self.idx]

    @property
    def delay(self) -> int:
        """The delay of before a spike is sent to the post-synaptic neuron."""
        return self.m.synaptic_delays[self.idx]

    @delay.setter
    def delay(self, value):
        if not is_intlike(value):
            raise TypeError("delay must be an integer")
        self.m.synaptic_delays[self.idx] = int(value)

    @property
    def stdp_enabled(self) -> bool:
        """If ``True``, STDP learning is enabled on this synapse."""
        return self.m.enable_stdp[self.idx]

    @stdp_enabled.setter
    def stdp_enabled(self, value):
        self.m.enable_stdp[self.idx] = bool(value)

    @property
    def weight(self) -> float:
        return self.m.synaptic_weights[self.idx]

    @weight.setter
    def weight(self, value: float):
        self.m.synaptic_weights[self.idx] = float(value)

    def __eq__(self, x):
        """Check if two Synapse instances represent the same synapse in the SNN."""
        if isinstance(x, Synapse):
            return self.idx == x.idx and self.m is x.m
        else:
            return False

    def __repr__(self):
        return f"<Virtual Synapse {self.idx} on model at {hex(id(self.m))}>"

    def info(self):
        """Returns a string containing information about this synapse."""
        return ' | '.join([
            f"id: {self.idx:d}",
            f"pre: {self.pre:d}",
            f"post: {self.post:d}",
            f"weight: {self.weight:g}",
            f"delay: {self.delay:d}",
            f"stdp {'en' if self.stdp_enabled else 'dis'}abled",
        ])

    def __str__(self):
        return f"<Synapse {self.info()}>"

    def info_row(self):
        """Returns a string containing information about this synapse for use in a table."""
        return ''.join([
            f"{self.idx:>5d}\t",
            f"{self.pre:>5d}  ",
            f"{self.post:>5d}\t",
            f"{self.weight:>11.9g}\t",
            f"{self.delay:>5d}\t",
            f"{'X' if self.stdp_enabled else '-'}",
        ])

    @staticmethod
    def row_header():
        return "  idx\t  pre   post\t     weight\tdelay\tstdp_enabled"

    @staticmethod
    def row_cont():
        return "  ...\t  ...    ...\t        ...\t  ...\t..."


class SynapseList:
    """Redirects indexing to the SNN's synapses.

    Returns a :py:class:`Synapse` or a list of Synapses.

    This is used to allow for the following syntax:

    .. code-block:: python

        snn.synapses[0]
        snn.synapses[1:10]
    """
    def __init__(self, model: SNN):
        self.m = model

    if TYPE_CHECKING:
        @overload
        def __getitem__(self, idx: int) -> Synapse: ...
        @overload
        def __getitem__(self, idx: slice) -> list[Synapse]: ...

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
    def __init__(self, model: SNN):
        self.m = model
        self.iter = iter(range(len(self.m.synaptic_weights)))

    def __next__(self):
        next_idx = next(self.iter)
        return Synapse(self.m, next_idx)
