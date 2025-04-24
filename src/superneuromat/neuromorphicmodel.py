import math
import copy
import warnings
import numpy as np
from scipy.sparse import csc_array  # scipy is also used for BLAS + numpy (dense matrix)
from .util import getenvbool, is_intlike, pretty_spike_train
from .accessor_classes import Neuron, Synapse, NeuronList, SynapseList

from typing import Any

try:
    import numba
    from numba import cuda
    from .numba_jit import lif_jit, stdp_update_jit
    from .numba_jit import stdp_update_jit_apos, stdp_update_jit_aneg
except ImportError:
    numba = None

"""Spiking Neural Network model implementing LIF and STDP using matrix representations.

"""

GPU_AVAILABLE = numba and cuda.is_available()


def check_numba():
    msg = """Numba is not installed. Please install Numba to use this feature.
    You can install JIT support for SuperNeuroMAT with `pip install superneuromat[jit]`,
    or install Numba manually with `pip install numba`.
    """
    global numba
    if numba is None:
        try:
            import numba
        except ImportError as err:
            raise ImportError(msg) from err


def check_gpu():
    msg = """GPU support is not installed. Please install Numba to use this feature.
    You can install JIT support for SuperNeuroMAT with `pip install superneuromat[gpu]`,
    or see the install instructions for GPU support at https://ornl.github.io/superneuromat/guide/install.html#gpu-support.
    """
    global numba
    if numba is None:
        try:
            from numba import cuda
        except ImportError as err:
            raise ImportError(msg) from err


class SNN:
    """Defines a spiking neural network with neurons and synapses"""

    gpu_threshold = 100.0
    jit_threshold = 50.0
    sparsity_threshold = 0.1
    disable_performance_warnings = True

    def __init__(self):
        """Initialize the SNN"""

        self.default_dtype: type | np.dtype = np.float64
        self.default_bool_dtype: type | np.dtype = bool

        # Neuron parameters
        self.neuron_leaks = []
        self.neuron_states = []
        self.neuron_thresholds = []
        self.neuron_reset_states = []
        self.neuron_refractory_periods = []
        self.neuron_refractory_periods_state = []

        # Synapse parameters
        self.pre_synaptic_neuron_ids = []
        self.post_synaptic_neuron_ids = []
        self.synaptic_weights = []
        self.synaptic_delays = []
        self.enable_stdp = []

        # Input spikes (can have a value)
        self.input_spikes = {}

        # Spike trains (monitoring all neurons)
        self.spike_train = []

        # STDP Parameters
        #: If False, stdp will be disabled globally.
        self.stdp = True
        self.apos = []
        self.aneg = []
        self._stdp_Apos = np.array([])
        self._stdp_Aneg = np.array([])
        self._do_stdp = False
        self.stdp_positive_update = True
        self.stdp_negative_update = True
        self._stdp_type = 'global'

        self.neurons = NeuronList(self)
        self.synapses = SynapseList(self)
        self.connection_ids = {}

        self.gpu = GPU_AVAILABLE
        # default backend setting. can be overridden at simulate time.
        self._backend = getenvbool('SNMAT_BACKEND', default='auto')
        self._last_used_backend = None
        self._sparse = 'auto'  # default sparsity setting.
        # self._return_sparse = False  # whether to return spikes sparsely
        self._is_sparse = False  # whether internal SNN representation is currently sparse
        self.manual_setup = False

        self.allow_incorrect_stdp_sign = getenvbool('SNMAT_ALLOW_INCORRECT_STDP_SIGN', default=False)
        self.allow_signed_leak = getenvbool('SNMAT_ALLOW_SIGNED_LEAK', default=False)

        self.memoized = {}

    def last_used_backend(self):
        """Get the last-used backend for simulation.

        Returns
        -------
        str
            Returns 'cpu' | 'jit' | 'gpu'

        .. seealso::

           :py:meth:`backend`
           :py:meth:`is_sparse`

        """
        return self._last_used_backend

    @property
    def backend(self):
        """Set the backend to be used for simulation.

        Parameters
        ----------
        use : str

        Raises
        ------
        ValueError
            if ``use`` is not one of ``'auto'``, ``'cpu'``, ``'jit'``, or ``'gpu'``.
        """
        return self._backend

    @backend.setter
    def backend(self, use):
        if not isinstance(use, str) or use.lower() not in ['auto', 'cpu', 'jit', 'gpu']:
            msg = f"Invalid backend: {use}"
            raise ValueError(msg)
        use = use.lower()
        if use == 'jit':
            check_numba()
        elif use == 'gpu':
            check_gpu()
        self._backend = use

    @property
    def dd(self):
        """Alias for self.default_dtype"""
        return self.default_dtype

    @property
    def dbin(self):
        """Alias for self.default_bool_dtype"""
        return self.default_bool_dtype

    @property
    def sparse(self):
        """Returns True if either user has requested sparse, or if SNN is sparse internally.

        To check the user-specified sparsity, see :py:attr:`_sparse`.

        Parameters
        ----------
        sparse : bool | str | Any
            If one of ``1``, ``'1'``, ``True``, ``'true'``, or ``'sparse'``,
            the SNN will be internally represented using a sparse representation.

            If one of ``0``, ``'0'``, ``False``, ``'false'``, or ``'dense'``,
            the SNN will be internally represented using a dense representation.

            If ``'auto'``, the sparsity will be determined at setup-time via :py:meth:`recommend_sparsity()`.
        """
        return self._sparse or self._is_sparse

    @property
    def is_sparse(self):
        """Returns True if SNN is sparse internally."""
        return self._is_sparse

    @staticmethod
    def _parse_sparsity(sparsity: bool | str | Any) -> bool | str:
        if isinstance(sparsity, str):
            sparsity = sparsity.lower()
            if sparsity in ('true', '1', 'sparse'):
                return True
            elif sparsity == ('false', '0', 'dense'):
                return False
            elif sparsity != 'auto':
                msg = f"Invalid sparse value: {sparsity!r}"
                raise ValueError(msg)
            return 'auto'
        else:
            return bool(sparsity)
        # returns True, False, or 'auto'

    @sparse.setter
    def sparse(self, sparse: bool | str | Any):
        """Sets the requested sparsity setting."""
        self._sparse = self._parse_sparsity(sparse)

    @property
    def num_neurons(self):
        """Returns the number of neurons in the SNN.

        Equivalent to ``len(self.neuron_thresholds)``.
        """
        return len(self.neuron_thresholds)

    @property
    def num_synapses(self):
        """Returns the number of synapses in the SNN.

        Equivalent to ``len(self.pre_synaptic_neuron_ids)``.
        """
        return len(self.pre_synaptic_neuron_ids)

    def get_synapses_by_pre(self, pre_id: int | Neuron):
        """Returns a list of synapses with the given pre-synaptic neuron."""
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        return [idx for idx, pre in enumerate(self.pre_synaptic_neuron_ids) if pre == pre_id]

    def get_synapses_by_post(self, post_id: int | Neuron):
        """Returns a list of synapses with the given post-synaptic neuron."""
        if isinstance(post_id, Neuron):
            post_id = post_id.idx
        return [idx for idx, post in enumerate(self.post_synaptic_neuron_ids) if post == post_id]

    def get_synapse(self, pre_id: int | Neuron, post_id: int | Neuron) -> Synapse | None:
        """Returns the synapse that connects the given pre- and post-synaptic neurons.

        Parameters
        ----------
        pre_id : int | Neuron
        post_id : int | Neuron

        Returns
        -------
        Synapse | None
            If no matching synapse exists, returns ``None``.
        """
        if (idx := self.get_synapse_id(pre_id, post_id)):
            return self.synapses[idx]

    def get_synapse_id(self, pre_id: int | Neuron, post_id: int | Neuron) -> int | None:
        """Returns the id of the synapse connecting the given pre- and post-synaptic neurons.

        Parameters
        ----------
        pre_id : int | Neuron
        post_id : int | Neuron

        Returns
        -------
        int | None
            If no matching synapse exists, returns ``None``.
        """
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if isinstance(post_id, Neuron):
            post_id = post_id.idx
        return self.connection_ids.get((pre_id, post_id), None)

    @property
    def stdp_time_steps(self) -> int:
        """Returns the number of time steps over which STDP updates will be made.

        If STDP is not enabled, returns ``0``.
        """
        if self.stdp_positive_update and self.stdp_negative_update:
            assert len(self._stdp_Apos) == len(self._stdp_Aneg)
        if self.stdp_positive_update:
            return len(self._stdp_Apos)
        elif self.stdp_negative_update:
            return len(self._stdp_Aneg)
        else:
            return 0

    def get_weights_df(self):
        """Returns a DataFrame containing information about synapse weights."""
        # TODO: test this
        import pandas as pd
        return pd.DataFrame({
            "Pre Neuron ID": self.pre_synaptic_neuron_ids,
            "Post Neuron ID": self.post_synaptic_neuron_ids,
            "Weight": self.synaptic_weights,
        })

    def get_stdp_enabled_df(self):
        """Returns a DataFrame containing information about STDP enabled synapses."""
        import pandas as pd
        return pd.DataFrame({
            "Pre Neuron ID": self.pre_synaptic_neuron_ids,
            "Post Neuron ID": self.post_synaptic_neuron_ids,
            "STDP Enabled": self.enable_stdp,
        })

    def get_neuron_df(self):
        """Returns a DataFrame containing information about neurons."""
        import pandas as pd
        return pd.DataFrame({
            "Neuron ID": list(range(self.num_neurons)),
            "Threshold": self.neuron_thresholds,
            "Leak": self.neuron_leaks,
            "Reset State": self.neuron_reset_states,
            "Refractory Period": self.neuron_refractory_periods,
        })

    def get_synapse_df(self):
        """Returns a DataFrame containing information about synapses."""
        import pandas as pd
        return pd.DataFrame({
            "Pre Neuron ID": self.pre_synaptic_neuron_ids,
            "Post Neuron ID": self.post_synaptic_neuron_ids,
            "Weight": self.synaptic_weights,
            "Delay": self.synaptic_delays,
            "STDP Enabled": self.enable_stdp,
        })

    def get_input_spikes_df(self):
        """Returns a DataFrame containing information about input spikes."""
        import pandas as pd
        df = pd.DataFrame()
        for time, spikes in self.input_spikes.items():
            for neuron, value in zip(spikes["nids"], spikes["values"]):
                df.loc[time, neuron] = value
        return df.fillna(0.0)

    @property
    def ispikes(self) -> np.ndarray[(int, int), bool]:
        """Convert the output spike train to a dense binary :py:class:`numpy.ndarray`.

        Returns
        -------
        numpy.ndarray[(int, int), bool]

        The ``dtype`` of the array will be :py:attr:`default_bool_dtype`.
        """
        return np.asarray(self.spike_train, dtype=self.dbin)

    def neuron_spike_totals(self, time_index=None):
        """Returns a vector of the number of spikes emitted from each neuron since last reset.

        Equivalent to ``snn.ispikes.sum(0)``.

        Parameters
        ----------
        time_index : _type_, default=None

        Returns
        -------
        numpy.ndarray[(int), bool]

        .. seealso::

            :py:meth:`ispikes`
            :py:meth:`numpy.sum`

        """
        # TODO: make time_index reference global/model time, not just ispikes index, which may have been cleared
        if time_index is None:
            return np.sum(self.ispikes, axis=0)
        else:
            return np.sum(self.ispikes[time_index], axis=0)

    def __str__(self):
        return self.pretty()

    def pretty_print(self, **kwargs):
        print(self.pretty(), **kwargs)

    def short(self):
        return f"<SNN with {self.num_neurons} neurons and {self.num_synapses} synapses @ {hex(id(self))}>"

    def stdp_info(self):
        return '\n'.join([
            f"STDP is globally {' en' if self.stdp else 'dis'}abled" + f" with {self.stdp_time_steps} time steps",
            f"apos: {self.apos}",
            f"aneg: {self.aneg}",
        ])

    def neuron_info(self, max_neurons=None):
        if max_neurons is None or self.num_neurons <= max_neurons:
            rows = (neuron.info_row() for neuron in self.neurons)
        else:
            fi = max_neurons // 2
            first = [neuron.info_row() for neuron in self.neurons[:fi]]
            last = [neuron.info_row() for neuron in self.neurons[-fi:]]
            rows = first + [Neuron.row_cont()] + last
        return '\n'.join([
            "Neuron Info:",
            Neuron.row_header(),
            '\n'.join(rows),
        ])

    def synapse_info(self, max_synapses=None):
        if max_synapses is None or self.num_synapses <= max_synapses:
            rows = (synapse.info_row() for synapse in self.synapses)
        else:
            fi = max_synapses // 2
            first = [synapse.info_row() for synapse in self.synapses[:fi]]
            last = [synapse.info_row() for synapse in self.synapses[-fi:]]
            rows = first + [Synapse.row_cont()] + last
        return '\n'.join([
            "Synapse Info:",
            Synapse.row_header(),
            '\n'.join(rows),
        ])

    def input_spikes_info(self, max_entries=None):

        def row(time, nid, value):
            return f"{time:>5d}: \t{value:>11.9g} -> {self.neurons[nid]!r}"

        all_spikes = []
        for time, spikes in self.input_spikes.items():
            for neuron, value in zip(spikes["nids"], spikes["values"]):
                all_spikes.append((time, neuron, value))
        if max_entries is None or len(self.input_spikes) <= max_entries:
            rows = (row(time, nid, value) for time, nid, value in all_spikes)
        else:
            fi = max_entries // 2
            first = [row(time, nid, value) for time, nid, value in all_spikes[:fi]]
            last = [row(time, nid, value) for time, nid, value in all_spikes[-fi:]]
            rows = first + [f"  ...   {len(all_spikes) - (fi * 2)} rows hidden    ..."] + last
        return '\n'.join([
            "Input Spikes:",
            f" Time:  {'Spike-value':>11s}    Destination",
            '\n'.join(rows),
        ])

    def pretty(self):
        lines = [
            self.short(),
            self.stdp_info(),
            '',
            self.neuron_info(30),
            '',
            self.synapse_info(40),
            '',
            self.input_spikes_info(30),
            '',
            "Spike Train:",
            self.pretty_spike_train(),
            f"{self.ispikes.sum()} spikes since last reset",
        ]
        return '\n'.join(lines)

    def create_neuron(
        self,
        threshold: float = 0.0,
        leak: float = np.inf,
        reset_state: float = 0.0,
        refractory_period: int = 0,
        refractory_state: int = 0,
        initial_state: float | None = 0.0,
    ) -> Neuron:
        """
        Create a neuron in the SNN.

        Parameters
        ----------
        threshold : float, default=0.0
            Neuron threshold; the neuron spikes if its internal state is strictly
            greater than the neuron threshold
        leak : float, default=numpy.inf
            Neuron leak; the amount by which the internal state of the neuron is
            pushed towards its reset state
        reset_state : float, default=0.0
            Reset state of the neuron; the value assigned to the internal state
            of the neuron after spiking
        refractory_period : int, default=0
            Refractory period of the neuron; the number of time steps for which
            the neuron remains in a dormant state after spiking

        Returns
        -------
        Neuron
            The :py:class:`Neuron` object.

        Raises
        ------
        TypeError
            If `threshold`, `leak`, or `reset_state` is not a float or int, or if
            `refractory_period` is not an int.
        ValueError
            If `leak` is less than 0.0 or `refractory_period` is less than 0.


        .. hint::

           :py:attr:`Neuron.idx` is the ID of the created neuron.

        """
        # Type errors
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be int or float")

        if not isinstance(leak, (int, float)):
            raise TypeError("leak must be int or float")
        if not self.allow_signed_leak and leak < 0.0:
            raise ValueError("leak must be grater than or equal to zero")

        if not is_intlike(refractory_period):
            raise TypeError("refractory_period must be int")
        refractory_period = int(refractory_period)
        if refractory_period < 0:
            raise ValueError("refractory_period must be greater than or equal to zero")

        if not is_intlike(refractory_state):
            raise TypeError("refractory_state must be int")
        refractory_state = int(refractory_state)
        if refractory_state < 0:
            raise ValueError("refractory_state must be greater than or equal to zero")

        if not isinstance(initial_state, (int, float)):
            raise TypeError("initial_state must be int or float")

        # Add neurons to SNN
        self.neuron_thresholds.append(float(threshold))
        self.neuron_leaks.append(float(leak))
        self.neuron_reset_states.append(float(reset_state))
        self.neuron_refractory_periods.append(refractory_period)
        self.neuron_refractory_periods_state.append(refractory_state)
        self.neuron_states.append(reset_state if initial_state is None else float(initial_state))

        # Return neuron ID
        return Neuron(self, self.num_neurons - 1)

    def create_synapse(
        self,
        pre_id: int | Neuron,
        post_id: int | Neuron,
        weight: float = 1.0,
        delay: int = 1,
        stdp_enabled: bool | Any = False
    ) -> Synapse:
        """Creates a synapse in the SNN

        Creates synapse connecting a pre-synaptic neuron to a post-synaptic neuron
        with a given set of synaptic parameters (weight, delay and stdp_enabled)

        Parameters
        ----------
        pre_id : int | Neuron
            ID of the pre-synaptic neuron (spike sender).
        post_id : int | Neuron
            ID of the post-synaptic neuron (spike destination).
        weight : float, default=1.0
            Synaptic weight; weight is multiplied to the incoming spike.
        delay : int, default=1
            Synaptic delay; number of time steps by which the outgoing signal of the syanpse is delayed by.
        enable_stdp : bool | Any, default=False
            If True, stdp will be enabled on the synapse, allowing the weight of this synapse to be updated.

        Raises
        ------
        TypeError
            if:

            * pre_id or post_id is not neuron or neuron ID (int).
            * weight is not a float.
            * delay cannot be cast to int.

        ValueError
            if:

            * pre_id or post_id is not a valid neuron or neuron ID.
            * delay is less than or equal to 0
            * Synapse with the given pre- and post-synaptic neurons already exists


        .. seealso::

           :py:meth:`Neuron.connect_child`, :py:meth:`Neuron.connect_parent`
        """

        # Type errors
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if isinstance(post_id, Neuron):
            post_id = post_id.idx

        if not isinstance(pre_id, (int, float)) or not is_intlike(pre_id):
            raise TypeError("pre_id must be int")
        pre_id = int(pre_id)

        if not isinstance(post_id, (int, float)) or not is_intlike(post_id):
            raise TypeError("post_id must be int")
        post_id = int(post_id)

        if not isinstance(weight, (int, float)):
            raise TypeError("weight must be a float")

        if not isinstance(delay, (int, float)) or not is_intlike(delay):
            raise TypeError("delay must be an integer")
        delay = int(delay)

        # Value errors
        if pre_id < 0:
            raise ValueError("pre_id must be greater than or equal to zero")
        elif not pre_id < self.num_neurons:
            msg = f"Added synapse to non-existent pre-synaptic Neuron {pre_id}."
            raise warnings.warn(msg, stacklevel=2)

        if post_id < 0:
            raise ValueError("post_id must be greater than or equal to zero")
        if not post_id < self.num_neurons:
            msg = f"Added synapse to non-existent post-synaptic Neuron {post_id}."
            raise warnings.warn(msg, stacklevel=2)

        if delay <= 0:
            raise ValueError("delay must be greater than or equal to 1")

        if (idx := self.get_synapse_id(pre_id, post_id)) is not None:
            msg = f"Synapse already exists: {self.synapses[idx]!s}"
            raise RuntimeError(msg)

        # Collect synapse parameters
        if delay == 1:
            self.pre_synaptic_neuron_ids.append(pre_id)
            self.post_synaptic_neuron_ids.append(post_id)
            self.synaptic_weights.append(weight)
            self.synaptic_delays.append(delay)
            self.enable_stdp.append(stdp_enabled)
            self.connection_ids[(pre_id, post_id)] = self.num_synapses - 1
        else:
            for _d in range(int(delay) - 1):
                temp_id = self.create_neuron()
                self.create_synapse(pre_id, temp_id)
                pre_id = temp_id

            self.create_synapse(pre_id, post_id, weight=weight, stdp_enabled=stdp_enabled)

        # Return synapse ID
        return Synapse(self, self.num_synapses - 1)

    def add_spike(
        self,
        time: int,
        neuron_id: int | Neuron,
        value: float = 1.0
    ) -> None:
        """Adds an external spike in the SNN

        Parameters
        ----------
        time : int
            The time step at which the external spike is added
        neuron_id : Neuron | int
            The neuron for which the external spike is added
        value : float
            The value of the external spike (default: 1.0)

        Raises
        ------
        TypeError
            if:

            * time cannot be precisely cast to int
            * neuron_id is not a Neuron or neuron ID (int)
            * value is not an int or float


        .. seealso::

           :py:meth:`Neuron.add_spike`
        """

        # Type errors
        if not is_intlike(time):
            raise TypeError("time must be int")
        time = int(time)

        if isinstance(neuron_id, Neuron):
            neuron_id = neuron_id.idx
        if not is_intlike(neuron_id):
            raise TypeError("neuron_id must be int")
        neuron_id = int(neuron_id)

        if not isinstance(value, (int, float)):
            raise TypeError("value must be int or float")

        # Value errors
        if time < 0:
            raise ValueError("time must be greater than or equal to zero")

        if neuron_id < 0:
            raise ValueError("neuron_id must be greater than or equal to zero")
        elif neuron_id >= self.num_neurons:
            msg = f"Added spike to non-existent Neuron {neuron_id} at time {time}."
            raise warnings.warn(msg, stacklevel=2)

        # Add spikes
        if time in self.input_spikes:
            self.input_spikes[time]["nids"].append(neuron_id)
            self.input_spikes[time]["values"].append(value)

        else:
            self.input_spikes[time] = {}
            self.input_spikes[time]["nids"] = [neuron_id]
            self.input_spikes[time]["values"] = [value]

    def stdp_setup(
        self,
        Apos: list | None = None,
        Aneg: list | None = None,
        positive_update: bool | Any = None,
        negative_update: bool | Any = None,
        time_steps=None,
    ) -> None:
        """Setup the Spike-Time-Dependent Plasticity (STDP) parameters

        Parameters
        ----------
        Apos : list, default=[1.0, 0.5, 0.25]
            List of parameters for excitatory STDP updates
        Aneg : list, default=[-1.0, -0.5, -0.25]
            List of parameters for inhibitory STDP updates
        positive_update : bool
            Boolean parameter indicating whether excitatory STDP update should be enabled
        negative_update : bool
            Boolean parameter indicating whether inhibitory STDP update should be enabled

        Raises
        ------
        TypeError

            * Apos is not a list
            * Aneg is not a list
            * positive_update is not a bool
            * negative_update is not a bool

        ValueError

            * Number of elements in Apos is not equal to that of Aneg
            * The elements of Apos, Aneg are not int or float
            * The elements of Apos, Aneg are not greater than or equal to 0.0

        RuntimeError

                * enable_stdp is not set to True on any of the synapses

        """

        if Apos is None and Aneg is None:
            Apos = [1.0, 0.5, 0.25]
            Aneg = [-1.0, -0.5, -0.25]

        # Collect STDP parameters
        self.stdp = True

        if positive_update or Apos is not None:
            if not isinstance(Apos, (list, np.ndarray)):
                raise TypeError("Apos should be a list")
            Apos: list
            if isinstance(Apos, list) and not all([isinstance(x, (int, float)) for x in Apos]):
                raise ValueError("All elements in Apos should be int or float")
            if positive_update is None:
                positive_update = True
            self.apos = np.asarray(Apos, self.dd)

        if negative_update or Aneg is not None:
            if not isinstance(Aneg, (list, np.ndarray)):
                raise TypeError("Aneg should be a list")
            Aneg: list
            if isinstance(Aneg, list) and not all([isinstance(x, (int, float)) for x in Aneg]):
                raise ValueError("All elements in Aneg should be int or float.")
            if negative_update is None:
                negative_update = True
            self.aneg = np.asarray(Aneg, self.dd)

        self.stdp_positive_update = bool(positive_update)
        self.stdp_negative_update = bool(negative_update)

        if positive_update and negative_update:
            if len(Apos) != len(Aneg):  # pyright: ignore[reportArgumentType]
                raise ValueError("Apos and Aneg should have the same size.")

        if not self.allow_incorrect_stdp_sign:
            if positive_update and not all([x >= 0.0 for x in Apos]):
                raise ValueError("All elements in Apos should be positive")
            if negative_update and not all([x <= 0.0 for x in Aneg]):
                raise ValueError("All elements in Aneg should be negative")

        if not any(self.enable_stdp):
            raise warnings.warn("STDP is not enabled on any synapse.", RuntimeWarning, stacklevel=2)
        if time_steps is not None:
            warnings.warn("time_steps on stdp_setup() is deprecated and has no effect. It will be removed in a future version.",
                          FutureWarning, stacklevel=2)

    def setup(self, **kwargs):
        """Setup the SNN for simulation.

        Parameters
        ----------
        dtype : bool | numpy.dtype, default=None
            The dtype to be used for floating-point arrays on this setup().
        sparse : bool | str | Any, default=None
            Whether to use a sparse representation for the SNN.
            See :py:attr:`sparse` for more information.
        """
        if not self.manual_setup:
            warnings.warn("setup() called without snn.manual_setup = True. setup() will be called again in simulate().",
                          RuntimeWarning, stacklevel=2)
        self._setup(**kwargs)

    def set_weights_from_mat(self, mat: np.ndarray[(int, int), float] | np.ndarray | csc_array):
        """Set the synaptic weights from a matrix.

        Parameters
        ----------
        mat : np.ndarray[(int, int), float] | np.ndarray | csc_array
            The matrix should have shape ``(num_neurons, num_neurons)``
            and be indexed by ``mat[pre_id, post_id]``.
        """
        self.synaptic_weights = list(mat[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])

    def weight_mat(self, dtype=None):
        """Create a dense weight matrix from the synaptic weights.

        This is used during :py:meth:`setup()` to create the internal weight matrix.

        Parameters
        ----------
        dtype : type | numpy.dtype, default=None
            The data type of the weight matrix. If None, :py:attr:`default_dtype` is used.

        Returns
        -------
        np.ndarray[(num_neurons, num_neurons), dtype]
        """
        dtype = self.dd if dtype is None else dtype
        mat = np.zeros((self.num_neurons, self.num_neurons), dtype)
        mat[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.synaptic_weights
        return mat

    def weights_sparse(self, dtype=None):
        """Create a sparse weight matrix from the synaptic weights.

        This is used during :py:meth:`setup()` to create the internal weight matrix.

        Parameters
        ----------
        dtype : type | numpy.dtype, default=None
            The data type of the weight matrix. If None, :py:attr:`default_dtype` is used.

        Returns
        -------
        scipy.sparse.csc_matrix[(num_neurons, num_neurons), dtype]
        """
        dtype = self.dd if dtype is None else dtype
        return csc_array(
            (self.synaptic_weights, (self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids)),
            shape=[self.num_neurons, self.num_neurons], dtype=dtype
        )

    def weight_sparsity(self):
        return self.num_synapses / (self.num_neurons ** 2)

    def stdp_enabled_mat(self, dtype=None):
        """Create a boolean dense matrix which indicates whether STDP is enabled on each synapse.

        This is used during :py:meth:`setup()`.

        Parameters
        ----------
        dtype : type | numpy.dtype, default=None
            The data type of the matrix. If None, :py:attr:`default_bool_dtype` is used.

        Returns
        -------
        np.ndarray[(num_neurons, num_neurons), dtype]
        """
        dtype = self.dbin if dtype is None else dtype
        mat = np.zeros((self.num_neurons, self.num_neurons), dtype)
        mat[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.enable_stdp
        return mat

    def set_stdp_enabled_from_mat(self, mat: np.ndarray[(int, int), bool] | np.ndarray):
        self.enable_stdp = list(mat[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])

    def stdp_enabled_sparse(self, dtype=None):
        dtype = self.dbin if dtype is None else dtype
        return csc_array(
            (self.enable_stdp, (self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids)),
            shape=[self.num_neurons, self.num_neurons], dtype=dtype
        )

    def _set_sparse(self, sparse):
        if sparse is not None:
            self.sparse = sparse
        else:
            self._is_sparse = self._sparse

    def _setup(self, dtype=None, sparse=None):
        """Setup the SNN for simulation. Not intended to be called by end user."""

        if dtype is not None:
            self.default_dtype = dtype

        if sparse is None:
            sparse = self.sparse  # use instance sparsity setting
        else:
            sparse = self._parse_sparsity(sparse)  # canonicalize sparsity setting
        # sparse is now either True, False, or 'auto'

        if sparse == 'auto':
            self._is_sparse = self.recommend_sparsity() if self.backend in ('cpu', 'auto') else False
        else:
            self._is_sparse = sparse

        # Create numpy arrays for neuron state variables
        self._neuron_thresholds = np.asarray(self.neuron_thresholds, self.dd)
        self._neuron_leaks = np.asarray(self.neuron_leaks, self.dd)
        self._neuron_reset_states = np.asarray(self.neuron_reset_states, self.dd)
        self._neuron_refractory_periods_original = np.asarray(self.neuron_refractory_periods, self.dd)
        self._internal_states = np.asarray(self.neuron_states, self.dd)

        self._neuron_refractory_periods = np.asarray(self.neuron_refractory_periods_state, self.dd)

        # Create numpy arrays for synapse state variables
        self._weights = self.weights_sparse() if self._is_sparse else self.weight_mat()
        anystdp = self.stdp and any(self.enable_stdp)
        self._do_positive_update = anystdp and self.stdp_positive_update and any(self.apos)
        self._do_negative_update = anystdp and self.stdp_negative_update and any(self.aneg)

        self._do_stdp = self._do_positive_update or self._do_negative_update

        # Create numpy arrays for STDP state variables
        if self._do_stdp:
            self._stdp_enabled_synapses = self.stdp_enabled_sparse() if self._is_sparse else self.stdp_enabled_mat()

            if self._do_positive_update and self._do_negative_update:
                if len(self.apos) != len(self.aneg):
                    raise ValueError("apos and aneg must be the same length")
            if self._do_positive_update:
                self._stdp_Apos = np.asarray(self.apos, self.dd)
            if self._do_negative_update:
                self._stdp_Aneg = np.asarray(self.aneg, self.dd)

        # Create numpy array for input spikes for each timestep
        self._input_spikes = np.zeros((1, self.num_neurons), self.dd)

        # Create numpy vector for spikes for each timestep
        if len(self.spike_train) > 0:
            self._spikes = np.asarray(self.spike_train[-1], self.dbin)
        else:
            self._spikes = np.zeros(self.num_neurons, self.dbin)

    def devec(self):
        """Copy the internal state variables back to the public-facing canonical representations."""
        # De-vectorize from numpy arrays to lists
        self.neuron_states: list[float] = self._internal_states.tolist()
        self.neuron_refractory_periods_state: list[float] = self._neuron_refractory_periods.tolist()

        # Update weights if STDP was enabled
        if self._do_stdp:
            if self._is_sparse:  # for now doing the same thing seems to work
                self.set_weights_from_mat(self._weights)
            else:
                self.set_weights_from_mat(self._weights)

    def zero_neuron_states(self):
        self.neuron_states = np.zeros(self.num_neurons, self.dd).tolist()

    def zero_refractory_periods(self):
        self.neuron_refractory_periods_state = np.zeros(self.num_neurons, self.dd).tolist()

    def reset_neuron_states(self):
        self.neuron_states = copy.copy(self.neuron_reset_states)

    def reset_refractory_periods(self):
        self.neuron_refractory_periods_state = copy.copy(self.neuron_refractory_periods)

    def clear_spike_train(self):
        self.spike_train = []

    def clear_input_spikes(self):
        self.input_spikes = {}

    def reset(self):
        """Reset the SNN's neuron states, refractory periods, spike train, and input spikes.

        Equivalent to:

        .. code-block:: python

            snn.reset_neuron_states()
            snn.reset_refractory_periods()
            snn.clear_spike_train()
            snn.clear_input_spikes()

        .. warning::

            This method does not reset the synaptic weights or STDP parameters.
            Instead, consider copying the parameters you care about so you can assign them later.

            See :ref:`reset-snn` for more information.

        """
        if 'neuron_states' not in self.memoized:
            self.reset_neuron_states()
        if 'neuron_refractory_periods_state' not in self.memoized:
            self.reset_refractory_periods()
        if 'spike_train' not in self.memoized:
            self.clear_spike_train()
        if 'input_spikes' not in self.memoized:
            self.clear_input_spikes()
        self.restore()

    def restore(self, *args):
        """Restore model variables to their memoized states.

        Parameters
        ----------
        *args : str | ~typing.Any
            If provided, only restore the given variables or variable names.

        Raises
        ------
        ValueError
            If the given variables or variable names are disallowed.


        .. seealso::

           :py:meth:`memoize`

        Examples
        --------

            .. code-block:: python

                restore("synaptic_weights")
                restore(snn.synaptic_weights)

                restore()  # restore all memoized variables
        """
        if args:
            keys = [self._get_self_eqvar(arg) for arg in args]
        for key, value in self.memoized.items():
            if key not in self.eqvars:
                msg = f"Invalid key: {key}"
                raise ValueError(msg)
            if args:
                if key in keys:
                    keys.remove(key)
                else:
                    continue
            setattr(self, key, value)
        if args and keys:
            msg = f"Invalid keys: {keys}"
            raise ValueError(msg)

    def setup_input_spikes(self, time_steps: int):
        self._input_spikes = np.zeros((time_steps + 1, self.num_neurons), self.dd)
        for t, spikes_dict in self.input_spikes.items():
            if t > time_steps:
                continue
            for neuron_id, amplitude in zip(spikes_dict["nids"], spikes_dict["values"]):
                self._input_spikes[t][neuron_id] = amplitude

    def consume_input_spikes(self, time_steps: int):
        """Consumes/deletes input spikes for the given number of time steps."""
        self.input_spikes = {t - time_steps: v for t, v in self.input_spikes.items()
                             if t >= time_steps}

    def release_mem(self):
        """Delete internal variables created during computation. Doesn't delete model.

        This is a low-level function and has few safeguards. Use with caution.
        It will call :ref:`del` on all numpy internal state variables.

        .. warning::

            ``release_mem()`` should be called at most once after :py:meth:`setup()` or :py:meth:`simulate()`.
            Do not call it before :py:meth:`setup()` or twice in a row.

        This function may alleviate memory leaks in some cases.
        """
        del self._neuron_thresholds, self._neuron_leaks, self._neuron_reset_states, self._internal_states
        del self._neuron_refractory_periods, self._neuron_refractory_periods_original, self._weights
        del self._input_spikes, self._spikes
        del self._stdp_Apos, self._stdp_Aneg
        if hasattr(self, "_stdp_enabled_synapses"):
            del self._stdp_enabled_synapses
        if hasattr(self, "_output_spikes"):
            del self._output_spikes
        self._is_sparse = False

    def recommend(self, time_steps: int):
        """Recommend a backend to use based on network size and continuous sim time steps."""
        score = self.num_neurons ** 2 * time_steps / 1e6

        if self.gpu and score > self.gpu_threshold and not self.recommend_sparsity():
            return 'gpu'
        elif numba and score > self.jit_threshold and not self.recommend_sparsity():
            return 'jit'
        return 'cpu'

    def recommend_sparsity(self):
        return self.weight_sparsity() < self.sparsity_threshold and self.num_neurons > 100

    def simulate(self, time_steps: int = 1, callback=None, use=None, sparse=None, **kwargs) -> None:
        """Simulate the neuromorphic spiking neural network

        Parameters
        ----------
        time_steps : int
            Number of time steps for which the neuromorphic circuit is to be simulated
        callback : function, optional
            Function to be called after each time step, by default None
        use : str, default=None
            Which backend to use. Can be 'auto', 'cpu', 'jit', or 'gpu'.
            If None, SNN.backend will be used, which is 'auto' by default.
            'auto' will choose a backend based on the network size and time steps.

        Raises
        ------
        TypeError
            If ``time_steps`` is not an int.
        ValueError
            If ``time_steps`` is less than or equal to zero.
        """

        # Type errors
        if not is_intlike(time_steps):
            raise TypeError("time_steps must be int")
        time_steps = int(time_steps)

        # Value errors
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than zero")

        if use is None:
            use = self.backend

        if isinstance(use, str):
            use = use.lower()
        if use == 'auto':
            use = self.recommend(time_steps)
        elif not use:
            use = 'cpu'

        if not self.manual_setup:
            self._setup(sparse=sparse)
            self.setup_input_spikes(time_steps)
        elif sparse is not None:
            msg = "simulate() received sparsity argument in manual_setup mode."
            msg += " Pass sparse to setup() instead."
            raise ValueError(msg)

        if use == 'jit':
            self.simulate_cpu_jit(time_steps, callback, **kwargs)
        elif use == 'gpu':
            self.simulate_gpu(time_steps, callback, **kwargs)
        elif use == 'cpu' or not use:  # if use is falsy, use cpu
            self.simulate_cpu(time_steps, callback, **kwargs)
        else:
            msg = f"Invalid backend: {use}"
            raise ValueError(msg)

    def simulate_cpu_jit(self, time_steps: int = 1, callback=None) -> None:
        # print("Using CPU with Numba JIT optimizations")
        self._last_used_backend = 'jit'
        check_numba()
        if self._is_sparse:
            raise ValueError("Sparse simulations are only supported on the CPU.")

        self._spikes = self._spikes.astype(self.dd)

        for tick in range(time_steps):
            if callback is not None:
                if callable(callback):
                    callback(self, tick, time_steps)

            lif_jit(
                tick,
                self._input_spikes,
                self._spikes,
                self._internal_states,  # modified in-place
                self._neuron_thresholds,
                self._neuron_leaks,
                self._neuron_reset_states,
                self._neuron_refractory_periods,
                self._neuron_refractory_periods_original,
                self._weights,
            )

            self.spike_train.append(self._spikes.astype(self.dbin))  # COPY
            t = min(self.stdp_time_steps, len(self.spike_train) - 1)
            spikes = np.array(self.spike_train[~t - 1:], dtype=self.dd)

            if self._do_stdp:
                if self._do_positive_update and self._do_negative_update:
                    stdp_update_jit(
                        t,
                        spikes,
                        self._weights,
                        self._stdp_Apos,
                        self._stdp_Aneg,
                        self._stdp_enabled_synapses,
                    )
                elif self._do_positive_update:
                    stdp_update_jit_apos(t, spikes, self._weights, self._stdp_Apos, self._stdp_enabled_synapses)
                elif self._do_negative_update:
                    stdp_update_jit_aneg(t, spikes, self._weights, self._stdp_Aneg, self._stdp_enabled_synapses)

        if not self.manual_setup:
            self.devec()
            self.consume_input_spikes(time_steps)

    def simulate_cpu(self, time_steps: int = 1000, callback=None) -> None:
        self._last_used_backend = 'cpu'

        if self._do_stdp:
            if not self._do_positive_update:
                self._stdp_Apos = np.zeros(len(self._stdp_Aneg), self.dd)
            if not self._do_negative_update:
                self._stdp_Aneg = np.zeros(len(self._stdp_Apos), self.dd)
            self._Aneg = np.array(self._stdp_Aneg[::-1])
            self._Asum = (np.asarray(self._stdp_Apos[::-1]) - np.asarray(self._Aneg)
                          ).reshape((-1, 1))

        # Simulate
        for tick in range(time_steps):
            if callback is not None:
                if callable(callback):
                    callback(self, tick, time_steps)

            # Leak: internal state > reset state
            indices = self._internal_states > self._neuron_reset_states
            self._internal_states[indices] = np.maximum(
                self._internal_states[indices] - self._neuron_leaks[indices], self._neuron_reset_states[indices]
            )

            # Leak: internal state < reset state
            indices = self._internal_states < self._neuron_reset_states
            self._internal_states[indices] = np.minimum(
                self._internal_states[indices] + self._neuron_leaks[indices], self._neuron_reset_states[indices]
            )

            # # Zero out _input_spikes
            # self._input_spikes -= self._input_spikes

            # # Include input spikes for current tick
            # if tick in self.input_spikes:
            #     self._input_spikes[self.input_spikes[tick]["nids"]] = self.input_spikes[tick]["values"]

            # Internal state
            self._internal_states += self._input_spikes[tick] + (self._weights.T @ self._spikes)

            # Compute spikes
            self._spikes = np.greater(self._internal_states, self._neuron_thresholds).astype(self.dbin)

            # Refractory period: Compute indices of neuron which are in their refractory period
            indices = np.greater(self._neuron_refractory_periods, 0)

            # For neurons in their refractory period, zero out their spikes and decrement refractory period by one
            self._spikes[indices] = 0
            self._neuron_refractory_periods[indices] -= 1

            # For spiking neurons, turn on refractory period
            mask = self._spikes.astype(bool)
            self._neuron_refractory_periods[mask] = self._neuron_refractory_periods_original[mask]

            # Reset internal states
            self._internal_states[mask] = self._neuron_reset_states[mask]

            # Append spike train
            self.spike_train.append(self._spikes)

            # STDP Operations
            t = min(self.stdp_time_steps, len(self.spike_train) - 1)

            if self._do_stdp and t > 0:
                if self._is_sparse:
                    Sprev = csc_array(self.spike_train[-t - 1:-1],
                                      shape=[t, self.num_neurons], dtype=self.dd)
                    Scurr = csc_array([self.spike_train[-1]] * t,
                                      shape=[t, self.num_neurons], dtype=self.dd)
                else:
                    Sprev = np.asarray(self.spike_train[-t - 1:-1], dtype=self.dd)
                    Scurr = np.asarray([self.spike_train[-1]] * t, dtype=self.dd)

                self._weights += ((((self._Asum[-t:] * Sprev).T @ Scurr) * self._stdp_enabled_synapses)
                                + (self._Aneg[-t:].sum() * self._stdp_enabled_synapses))
                if self._is_sparse:
                    self._weights = self._weights.astype(self.dd)

        if not self.manual_setup:
            self.devec()
            self.consume_input_spikes(time_steps)

    def simulate_gpu(self, time_steps: int = 1, callback=None) -> None:
        """Simulate the neuromorphic circuit using the GPU backend.

        Parameters
        ----------
        time_steps : int, optional
        callback : _type_, optional
            WARNING: If using the GPU backend, this callback will
            not be able to modify the SNN via ``self`` .
        """
        # print("Using CUDA GPU via Numba")
        self._last_used_backend = 'gpu'
        check_gpu()
        if self._is_sparse:
            raise ValueError("Sparse simulations are only supported on the CPU.")
        from .gpu import cuda as gpu
        if self.disable_performance_warnings:
            gpu.disable_numba_performance_warnings()

        if self._do_stdp:
            apos = self._stdp_Apos
            aneg = self._stdp_Aneg
            if self._do_positive_update and not self._do_negative_update:
                aneg = np.zeros(len(self._stdp_Apos), self.dd)
            elif self._do_negative_update and not self._do_positive_update:
                apos = np.zeros(len(self._stdp_Aneg), self.dd)
            assert len(apos) == len(aneg), "apos and aneg must be the same length"
            stdp_enabled = cuda.to_device(self._stdp_enabled_synapses)

        post_synapse = cuda.to_device(np.zeros(self.num_neurons, self.dd))
        output_spikes = cuda.to_device(self._spikes.astype(self.dbin))
        states = cuda.to_device(self._internal_states)
        thresholds = cuda.to_device(self._neuron_thresholds)
        leaks = cuda.to_device(self._neuron_leaks)
        reset_states = cuda.to_device(self._neuron_reset_states)
        refractory_periods = cuda.to_device(self._neuron_refractory_periods)
        refractory_periods_original = cuda.to_device(self._neuron_refractory_periods_original)
        weights = cuda.to_device(self._weights)

        v_tpb = min(self.num_neurons, 32)
        v_blocks = math.ceil(self.num_neurons / v_tpb)

        m_tpb = (v_tpb, v_tpb)
        m_blocks = (v_blocks, v_blocks)

        for tick in range(time_steps):
            if callback is not None:
                if callable(callback):
                    callback(self, tick, time_steps)

            input_spikes = cuda.to_device(self._input_spikes[tick])

            gpu.post_synaptic[v_blocks, v_tpb](
                weights,
                output_spikes,
                post_synapse,
            )

            gpu.lif[v_blocks, v_tpb](
                input_spikes,
                output_spikes,
                post_synapse,
                states,
                thresholds,
                leaks,
                reset_states,
                refractory_periods,
                refractory_periods_original,
            )

            self.spike_train.append(output_spikes.copy_to_host())

            if self._do_stdp:
                for i in range(min(self.stdp_time_steps, len(self.spike_train) - 1)):
                    prev_spikes = cuda.to_device(self.spike_train[~i - 1].astype(self.dbin))
                    gpu.stdp_update[m_blocks, m_tpb](weights, prev_spikes, output_spikes,
                                    stdp_enabled, apos[i], aneg[i])

        self._weights = weights.copy_to_host()
        self._neuron_refractory_periods = refractory_periods.copy_to_host()
        self._internal_states = states.copy_to_host()

        if not self.manual_setup:
            self.devec()
            self.consume_input_spikes(time_steps)

    def print_spike_train(
        self,
        max_steps: int | None = None,
        max_neurons: int | None = None,
        use_unicode=True,
    ):
        """Prints the spike train.

        Parameters
        ----------
        max_steps : int | None, optional
            Limits the number of steps which will be printed.
            If limited, only a total of ``max_steps`` first and last steps will be printed.
        max_neurons : int | None, optional
            Limits the number of neurons which will be printed.
            If limited, only a total of ``max_neurons`` first and last neurons will be printed.
        use_unicode : bool, default=True
            If ``True``, use unicode characters to represent spikes.
            Otherwise fallback to ascii characters.
        """
        print(self.pretty_spike_train(max_steps, max_neurons, use_unicode))

    def pretty_spike_train(
        self,
        max_steps: int | None = 11,
        max_neurons: int | None = 28,
        use_unicode=True,
    ):
        """Returns a pretty string of the spike train.

        Parameters
        ----------
        max_steps : int | None, optional
            Limits the number of steps which will be included.
            If limited, only a total of ``max_steps`` first and last steps will be included.
        max_neurons : int | None, optional
            Limits the number of neurons which will be included.
            If limited, only a total of ``max_neurons`` first and last neurons will be included.
        use_unicode : bool, default=True
            If ``True``, use unicode characters to represent spikes.
            Otherwise fallback to ascii characters.
        """
        return '\n'.join(pretty_spike_train(self.spike_train, max_steps, max_neurons, use_unicode))

    def copy(self):
        """Returns a copy of the SNN

        Returns
        -------
        SNN
            A copy of the SNN created by :py:meth:`copy.deepcopy()`.
        """

        return copy.deepcopy(self)

    #: The list of public variables which define the SNN model.
    eqvars = [
        'default_dtype',
        'num_neurons',
        'num_synapses',
        'neuron_leaks',
        'neuron_states',
        'neuron_thresholds',
        'neuron_reset_states',
        'neuron_refractory_periods',
        'neuron_refractory_periods_state',
        'pre_synaptic_neuron_ids',
        'post_synaptic_neuron_ids',
        'synaptic_weights',
        'synaptic_delays',
        'enable_stdp',
        'input_spikes',
        'spike_train',
        'stdp', 'apos', 'aneg',
        'stdp_positive_update', 'stdp_negative_update',
        'sparse',
        'backend',
        'manual_setup',
    ]

    def __eq__(self, other):
        """Checks if two SNNs are equal."""
        if not isinstance(other, SNN):
            return False
        for var in self.eqvars:
            a = getattr(self, var)
            b = getattr(other, var)
            if isinstance(a, (np.ndarray, csc_array)):
                if not np.array_equal(a, b):
                    return False
            else:
                try:
                    if a != b:
                        return False
                except ValueError:
                    if np.any(np.asarray(a) != np.asarray(b)):
                        return False
        return True

    def memoize(self, *keys):
        """Store a copy of model variable(s) to be restored later.

        Parameters
        ----------
        *args : str | ~typing.Any
            Save the given variables or variable names.

        Raises
        ------
        ValueError
            If no variables or variable names are provided, or if the
            given variables or variable names are disallowed.


        .. seealso::

            :py:meth:`restore`, :py:meth:`unmemoize`

        Examples
        --------

            .. code-block:: python

                memoize("synaptic_weights")
                memoize(snn.synaptic_weights)
        """
        if not keys:
            raise ValueError("No objects to memoize.")
        for key in keys:
            self._memoize(key)

    def _memoize(self, key):
        key = self._get_self_eqvar(key)
        obj = getattr(self, key)
        if isinstance(obj, np.ndarray):
            self.memoized[key] = obj.copy()
        else:
            self.memoized[key] = copy.deepcopy(getattr(self, key))

    def _get_self_eqvar(self, key):
        msg = "Invalid key: {}"
        if isinstance(key, str):
            if key not in self.eqvars:
                raise ValueError(msg.format(key))
        else:
            for name in self.eqvars:
                if key is getattr(self, name):
                    key = name
                    break
            else:
                raise ValueError(msg.format(key))
        return key

    def unmemoize(self, *keys):
        """Delete stored copies of model variables.

        Parameters
        ----------
        *args : str | ~typing.Any
            Delete the saved copy of the given variables or variable names.

        It is safe to pass in variables or names which were not memoized.

        Raises
        ------
        ValueError
            If no variables or variable names are provided, or if the
            given variables or variable names are disallowed.


        .. seealso::

            :py:meth:`memoize()`, :py:meth:`restore()`, :py:meth:`clear_memos()`
        """
        if not keys:
            raise ValueError("No objects to unmemoize.")
        for key in keys:
            self._unmemoize(key)

    def _unmemoize(self, key):
        key = self._get_self_eqvar(key)
        if key in self.memoized:
            del self.memoized[key]

    def clear_memos(self):
        """Delete all stored copies of model variables.


        .. seealso::

            :py:meth:`memoize()`, :py:meth:`unmemoize()`
        """
        del self.memoized
        self.memoized = {}
