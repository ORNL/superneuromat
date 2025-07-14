"""Spiking Neural Network model implementing LIF and STDP using matrix representations."""

from __future__ import annotations
import sys
import math
import copy
import warnings
import numpy as np
from numpy import typing as npt
from textwrap import dedent
from scipy.sparse import csc_array  # scipy is also used for BLAS + numpy (dense matrix)
from .util import getenv, getenvbool, is_intlike_catch, pretty_spike_train, int_err, float_err
from . import json
from .accessor_classes import Neuron, Synapse, NeuronList, SynapseList

from typing import Any, TYPE_CHECKING

try:
    import numba
    from numba import cuda
    from .numba_jit import lif_jit, stdp_update_jit
    from .numba_jit import stdp_update_jit_apos, stdp_update_jit_aneg
except ImportError:
    numba = None
if TYPE_CHECKING:
    from numba import cuda  # stops pyright from analyzing the ^^^ except code path
    from .numba_jit import lif_jit, stdp_update_jit
    from .numba_jit import stdp_update_jit_apos, stdp_update_jit_aneg

try:
    # cuda.is_available() may give TypeError on Win64 w/ no CUDA. @ numba-cuda 0.9.0
    GPU_AVAILABLE = numba and cuda.is_available()
except Exception:
    GPU_AVAILABLE = False


if TYPE_CHECKING:
    _arr_boollike_T = np.dtype[np.bool]
    _arr_floatlike_T = np.dtype[np.floating]
else:
    _arr_boollike_T = _arr_floatlike_T = None


def check_numba():
    msg = dedent("""\
        Numba is not installed. Please install Numba to use this feature.
        You can install JIT support for SuperNeuroMAT with `pip install superneuromat[jit]`,
        or install Numba manually with `pip install numba`.
        See https://ORNL.github.io/superneuromat/guide/install.html#snm-install-numba for more information.
    """)
    global numba
    if numba is None:
        try:
            import numba
        except ImportError as err:
            raise ImportError(msg) from err


def check_gpu():
    msg = dedent("""\
        GPU support is not installed. Please install Numba to use this feature.
        You can install JIT support for SuperNeuroMAT with `pip install superneuromat[cuda]`,
        or see the install instructions for GPU support at https://ORNL.github.io/superneuromat/guide/install.html#snm-install-numba.
    """)
    global numba
    if numba is None:
        try:
            import numba
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
        self.default_bool_dtype: type | np.dtype = np.bool_

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
        self._backend = getenv('SNMAT_BACKEND', default='auto')
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
            Returns 'cpu' | 'jit' | 'gpu' for the backend that was used during
            :py:meth:`simulate`, or ``None`` if no simulation has been run yet.


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
            The backend to use. Can be ``'auto'``, ``'cpu'``, ``'jit'``, or ``'gpu'``.

        Raises
        ------
        ValueError
            If ``use`` is not one of ``'auto'``, ``'cpu'``, ``'jit'``, or ``'gpu'``.


        ``'auto'`` is the default value. This will choose a backend at :py:meth:`simulate()` time
        based on the network size and time steps, as chosen by :py:meth:`recommend()`.

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
        """The user-requested sparsity setting for the SNN

        When creating an :py:class:`SNN`\\ , the ``sparse`` parameter is ``'auto'`` by default.

        Parameters
        ----------
        sparse : bool | str | Any
            If one of ``1``, ``'1'``, ``True``, ``'true'``, or ``'sparse'``,
            the SNN will be internally represented using a sparse representation.

            If one of ``0``, ``'0'``, ``False``, ``'false'``, or ``'dense'``,
            the SNN will be internally represented using a dense representation.

            If ``'auto'``, the sparsity will be determined at setup-time via :py:meth:`recommend_sparsity()`.

        Returns
        -------
        bool | str
            Returns ``True``, ``False``, or ``'auto'``.

        """
        return self._sparse

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
        """Sets the requested sparsity setting.
        """
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

    def get_synaptic_ids_by_pre(self, pre_id: int | Neuron) -> list[int]:
        """Returns a list of synapse ids with the given pre-synaptic neuron.

        Parameters
        ----------
        pre_id : int | Neuron, required
            The ID of the pre-synaptic neuron.

        Raises
        ------
        TypeError
            If ``pre_id`` is not an int or Neuron.

        Returns
        -------
        list
            A list of synapse ids with the given pre-synaptic neuron. May be empty.
        """
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if not is_intlike_catch(pre_id):
            raise TypeError("pre_id must be int or Neuron.")
        return [idx for idx, pre in enumerate(self.pre_synaptic_neuron_ids) if pre == pre_id]

    def get_synapses_by_pre(self, pre_id: int | Neuron) -> list[Synapse]:
        """Returns a list of synapses with the given pre-synaptic neuron.

        Parameters
        ----------
        pre_id : int | Neuron, required
            The ID of the pre-synaptic neuron.

        Raises
        ------
        TypeError
            If ``pre_id`` is not an int or Neuron.

        Returns
        -------
        list
            A list of :py:class:`Synapse`\\ s with the given pre-synaptic neuron. May be empty.
        """
        return [self.synapses[i] for i in self.get_synaptic_ids_by_pre(pre_id)]

    def get_synaptic_ids_by_post(self, post_id: int | Neuron) -> list[int]:
        """Returns a list of synapse ids with the given post-synaptic neuron.

        Parameters
        ----------
        post_id : int | Neuron, required
            The ID of the post-synaptic neuron.

        Raises
        ------
        TypeError
            If ``post_id`` is not an int or Neuron.

        Returns
        -------
        list
            A list of synapse ids with the given post-synaptic neuron. May be empty.
        """
        if isinstance(post_id, Neuron):
            post_id = post_id.idx
        if not is_intlike_catch(post_id):
            raise TypeError("post_id must be int or Neuron.")
        return [idx for idx, post in enumerate(self.post_synaptic_neuron_ids) if post == post_id]

    def get_synapses_by_post(self, post_id: int | Neuron) -> list[Synapse]:
        """Returns the synapses that connect the given post-synaptic neuron.

        Parameters
        ----------
        post_id : int | Neuron, required
            The ID of the post-synaptic neuron.

        Raises
        ------
        TypeError
            If ``post_id`` is not an int or Neuron.

        Returns
        -------
        list
            A list of :py:class:`Synapse`\\ s with the given post-synaptic neuron. May be empty.
        """
        return [self.synapses[i] for i in self.get_synaptic_ids_by_post(post_id)]

    def get_synapse(self, pre_id: int | Neuron, post_id: int | Neuron) -> Synapse:
        """Returns the synapse that connects the given pre- and post-synaptic neurons.

        Parameters
        ----------
        pre_id : int | Neuron, required
        post_id : int | Neuron, required

        Returns
        -------
        Synapse

        Raises
        ------
        IndexError
            If no matching synapse is found.
        TypeError
            When `pre_id` or `post_id` is not a Neuron or neuron ID (int).
        """
        if (idx := self.get_synapse_id(pre_id, post_id)) is not None:
            return self.synapses[idx]

        # synapse not found, raise error.
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if isinstance(post_id, Neuron):
            post_id = post_id.idx
        msg = f"Synapse not found between neurons {pre_id} and {post_id}."
        raise IndexError(msg)

    def get_synapse_id(self, pre_id: int | Neuron, post_id: int | Neuron) -> int | None:
        """Returns the id of the synapse connecting the given pre- and post-synaptic neurons.

        Parameters
        ----------
        pre_id : int | Neuron, required
        post_id : int | Neuron, required

        Returns
        -------
        int | None
            The id of the synapse connecting the pre-synaptic -> post-synaptic neurons.
            If no matching synapse exists, returns ``None``.

        Raises
        ------
        TypeError
            If `pre_id` or `post_id` is not a Neuron or neuron ID (int).
        """
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if isinstance(post_id, Neuron):
            post_id = post_id.idx
        if not (is_intlike_catch(pre_id) and is_intlike_catch(post_id)):
            raise TypeError("pre_id and post_id must be int or Neuron.")
        return self.connection_ids.get((pre_id, post_id), None)

    @property
    def stdp_time_steps(self) -> int:
        """Returns the number of time steps over which STDP updates will be made.

        This is the effective number of time steps that Spike-Timing Dependent Plasticity (STDP)
        will be applied over. This depends on the length of the :py:attr:`apos` and :py:attr:`aneg`
        lists, as well whether :py:attr:`stdp_positive_update` or :py:attr:`stdp_negative_update`
        are enabled.

        If STDP is not enabled, returns ``0``.

        Raises
        ------
        RuntimeError
            If both positive and negative updates are enabled,
            but :py:attr:`apos` and :py:attr:`aneg` are not the same length.
        """
        apos, aneg = len(self.apos), len(self.aneg)

        if not self.stdp:
            return 0
        if self.stdp_positive_update and self.stdp_negative_update:
            if not apos or not aneg:
                return max(apos, aneg)
            else:
                if apos != aneg:
                    msg = "positive and negative updates are enabled, but apos and aneg are not the same length."
                    msg += " Refusing to report STDP time steps."
                    raise RuntimeError(msg)
                return apos
        if self.stdp_positive_update:
            return apos
        elif self.stdp_negative_update:
            return aneg
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
    def ispikes(self) -> np.ndarray[(int, int), _arr_boollike_T]:
        """Convert the output spike train to a dense binary :py:class:`numpy.ndarray`.

        This is useful for slicing and analyzing the spike train.

        Index with ``snn.ispikes[time, neuron]``.

        Note that this is converted from the output spike train, which may be cleared
        with :py:meth:`clear_spike_train` or :py:meth:`reset`.

        Returns
        -------
        numpy.ndarray[(int, int), bool]
            The ``dtype`` of the array will be :py:attr:`default_bool_dtype`.

        Examples
        --------
        >>> # Here's an example spike train:
        >>> snn.ispikes
        array([[False, False, False, False,  True],
               [False,  True, False,  True,  True],
               [ True,  True,  True,  True,  True],
               [ True,  True, False, False, False]])
        >>> snn.print_spike_train()
        t|id0 1 2 3 4
        0: [│ │ │ │ ├─]
        1: [│ ├─│ ├─├─]
        2: [├─├─├─├─├─]
        3: [├─├─│ │ │ ]

        >>> # single neuron
        >>> snn.ispikes[0, 0]  # True if neuron 0 spiked at time 0
        np.False_

        >>> # multiple neurons
        >>> snn.ispikes[0, :]  # Whether a neuron spiked at time step 0
        array([False, False, False, False,  True])
        >>> snn.ispikes[:, 0]  # Whether neuron 0 spiked for a particular time step
        array([False, False,  True,  True])
        >>> snn.ispikes[-1]  # Whether a neuron spiked at the last time step
        array([ True,  True, False, False, False])
        >>> snn.ispikes[-3:].sum()  # Number of spikes emitted in the last 3 time steps
        np.int64(10)
        >>> snn.ispikes[-3:].sum(0)  # Number of spikes emitted for each neuron in the last 3 time steps
        array([2, 3, 1, 2, 2])
        >>> snn.ispikes[:, 2:].sum(0)  # Number of spikes emitted by the first 2 neurons over all time
        array([2, 3])
        >>> snn.ispikes.sum(1)  # number of spikes emitted per neuron over all time
        array([1, 3, 5, 2])
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

    def pretty_print(self, n=20, **kwargs):
        """Pretty-prints the model.

        You can also do ``print(snn)``.

        Parameters
        ----------
        n : int, default=20
            Limits the size each section of the printout.

        .. seealso::

            * :py:meth:`SNN.pretty`
            * :py:meth:`SNN.short`
            * :py:meth:`SNN.print_spike_train`
        """
        print(self.pretty(n), **kwargs)

    def short(self):
        """Return a 1-line summary of the SNN.

        Examples
        --------

        .. code-block::

            SNN with 100 neurons and 1000 synapses @ 0x7f9c0a0c5d10

        .. seealso::

            * :py:meth:`SNN.pretty`
        """
        # this will be grammatically incorrect for n=1, but this makes it easier to parse
        return f"SNN with {self.num_neurons} neurons and {self.num_synapses} synapses @ {hex(id(self))}"

    def stdp_info(self):
        """Generate a description of the current global STDP settings.

        Returns
        -------
        list[str]

        Examples
        --------
        >>> print(snn.stdp_info())
        STDP is globally enabled over 3 time steps.
        apos: [1.0, 0.5, 0.25]
        aneg: [-0.1, -0.05, -0.025]
        5 synapses have STDP enabled.

        .. seealso::

            * :py:meth:`SNN.pretty`
        """
        extra = f" over {self.stdp_time_steps} time steps." if self.stdp_time_steps else ""
        return '\n'.join([
            f"STDP is globally {'enabled' if self.stdp else 'disabled.'}" + extra,
            f"apos: {self.apos}",
            f"aneg: {self.aneg}",
            f"{self.stdp_enabled_mat().sum()} synapses have STDP enabled.",
        ])

    def neuron_info(self, max_neurons=None):
        """Generate a description of the neurons in the SNN.

        Generates a list of strings which can be printed, which shows a table of the neurons in the SNN.
        The first line also shows the number of neurons in the SNN.

        Here are the headers and columns which will be shown:

        +-----------+--------------+-----------+------+------------------+-------------------+-------------+
        | idx       | state        | thresh    | leak | ref_state        | ref_period        | spikes      |
        +===========+==============+===========+======+==================+===================+=============+
        | Neuron ID | Charge State | Threshold | Leak | Refractory State | Refractory Period | Spike Train |
        +-----------+--------------+-----------+------+------------------+-------------------+-------------+

        Parameters
        ----------
        max_neurons : int, optional
            If more than ``max_neurons`` neurons are in the SNN, only the
            first and last ``max_neurons // 2`` neurons will be printed.

        Returns
        -------
        list[str]

        Examples
        --------
        >>> print(snn.neuron_info())
        Neuron Info (2):
           idx       state      thresh        leak  ref_state  ref_period spikes
             0          -2          -1           2          0           3 [----⋯------]
             1          -2           0           1          0           1 [-┴--⋯-┴----]

        .. seealso::

            * :py:meth:`SNN.pretty`
            * :py:meth:`Neuron.info`

        These functions generate parts of the table above:
        :py:meth:`Neuron.row_header`, :py:meth:`Neuron.info_row`, :py:meth:`Neuron.row_cont`
        """
        if max_neurons is None or self.num_neurons <= max_neurons:
            rows = (neuron.info_row() for neuron in self.neurons)
        else:
            fi = max_neurons // 2
            first = [neuron.info_row() for neuron in self.neurons[:fi]]
            last = [neuron.info_row() for neuron in self.neurons[-fi:]]
            rows = first + [Neuron.row_cont()] + last
        return '\n'.join([
            f"Neuron Info ({self.num_neurons}):",
            Neuron.row_header(),
            '\n'.join(rows),
        ])

    def synapse_info(self, max_synapses=None):
        """Generate a description of the synapses in the SNN.

        Generates a list of strings which can be printed, which shows a table of the synapses in the SNN.
        The first line also shows the number of synapses in the SNN.

        Here are the headers and columns which will be shown:

        +------------+------------------------+-------------------------+--------+-------+--------------+
        | idx        | pre                    | post                    | weight | delay | stdp_enabled |
        +============+========================+=========================+========+=======+==============+
        | Synapse ID | Pre-synaptic Neuron ID | Post-synaptic Neuron ID | Weight | Delay | STDP Enabled |
        +------------+------------------------+-------------------------+--------+-------+--------------+

        The stdp_enabled column will contain either ``Y`` or ``-``
        if the synapse allows STDP updates or not, respectively.

        Parameters
        ----------
        max_synapses : int, optional
            If more than ``max_synapses`` synapses are in the SNN, only the
            first and last ``max_synapses // 2`` synapses will be printed.

        Returns
        -------
        list[str]

        Examples
        --------
        >>> print(snn.synapse_info())
        Synapse Info (2):
          idx     pre   post         weight     delay   stdp_enabled
            0       0      1              1         1   -
            1       1      0              4         1   Y

        .. seealso::

            * :py:meth:`SNN.pretty`
            * :py:meth:`Synapse.info`

        These functions generate parts of the table above:
        :py:meth:`Synapse.row_header`, :py:meth:`Synapse.info_row`, :py:meth:`Synapse.row_cont`
        """
        if max_synapses is None or self.num_synapses <= max_synapses:
            rows = (synapse.info_row() for synapse in self.synapses)
        else:
            fi = max_synapses // 2
            first = [synapse.info_row() for synapse in self.synapses[:fi]]
            last = [synapse.info_row() for synapse in self.synapses[-fi:]]
            rows = first + [Synapse.row_cont()] + last
        return '\n'.join([
            f"Synapse Info ({self.num_synapses}):",
            Synapse.row_header(),
            '\n'.join(rows),
        ])

    def input_spikes_info(self, max_entries=None):
        """Generate a description of the synapses in the SNN.

        Generates a list of strings which can be printed, which shows a
        table of the individual input spikes queued to be sent to the SNN.
        The first line also shows the number of queued spikes.

        Here are the headers for the information which will be shown:

        +------+-------------+-------------+
        | Time | Spike-value | Destination |
        +------+-------------+-------------+

        1. Time: The number of time steps to wait before sending the spike.
        2. Spike-value: The amplitude or value of the spike.
        3. Destination: The ID of the neuron to which the spike will be sent.

        Parameters
        ----------
        max_entries : int, optional
            If more than ``max_entries`` input spikes are queued, only the
            first and last ``max_entries // 2`` spikes will be printed.

        Returns
        -------
        list[str]

        Examples
        --------
        >>> print(snn.input_spikes_info())
        Input Spikes (1):
        Time:  Spike-value    Destination
            0:          2.1 -> <Virtual Neuron 4 on model at 0x290f4bb8800>

        .. seealso::

            * :py:meth:`SNN.pretty`
        """

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
            f"Input Spikes ({len(self.input_spikes)}):",
            f" Time:  {'Spike-value':>11s}    Destination",
            '\n'.join(rows),
        ])

    def pretty(self, n=20):
        """Generates a text description of the model.

        .. seealso::

            Used in:

            * :py:meth:`pretty_print`
            * :py:meth:`SNN.__str__`

            Generated by:

            * :py:meth:`short`
            * :py:meth:`stdp_info`
            * :py:meth:`neuron_info`
            * :py:meth:`synapse_info`
            * :py:meth:`input_spikes_info`
            * :py:meth:`print_spike_train`

        Parameters
        ----------
        n : int, default=20
            Limits the size each section of the printout.
        """
        lines = [
            self.short(),
            self.stdp_info(),
            '',
            self.neuron_info(n),
            '',
            self.synapse_info(n),
            '',
            self.input_spikes_info(n),
            '',
            "Spike Train:",
            self.pretty_spike_train(max_steps=n),
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
        # Input validation
        fname = 'create_neuron()'

        leak = float_err(leak, 'leak', fname)
        if not self.allow_signed_leak and leak < 0.0:
            raise ValueError("leak must be grater than or equal to zero.")

        refractory_period = int_err(refractory_period, 'refractory_period', fname)
        if refractory_period < 0:
            raise ValueError("refractory_period must be greater than or equal to zero.")

        refractory_state = int_err(refractory_state, 'refractory_state', fname)
        if refractory_state < 0:
            raise ValueError("refractory_state must be greater than or equal to zero.")

        # Add neurons to SNN
        self.neuron_thresholds.append(float_err(threshold, 'threshold', fname))
        self.neuron_leaks.append(leak)
        self.neuron_reset_states.append(float_err(reset_state, 'reset_state', fname))
        self.neuron_refractory_periods.append(refractory_period)
        self.neuron_refractory_periods_state.append(refractory_state)
        self.neuron_states.append(reset_state if initial_state is None else float_err(initial_state, 'initial_state', fname))

        # Return neuron ID
        return Neuron(self, self.num_neurons - 1)

    def create_synapse(
        self,
        pre_id: int | Neuron,
        post_id: int | Neuron,
        weight: float = 1.0,
        delay: int = 1,
        stdp_enabled: bool | Any = False,
        exist: str = "error",
        **kwargs,
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
            Synaptic delay; number of time steps by which the outgoing signal of the synapse is delayed by.
        stdp_enabled : bool | Any, default=False
            If True, stdp will be enabled on the synapse, allowing the weight of this synapse to be updated.
        exist : str, default='error'
            Action if synapse  already exists with the exact pre- and post-synaptic neurons.
            Should be one of ['error', 'overwrite', 'dontadd'].


        If a delay is specified, a chain of neurons and synapses will automatically be added to the model
        to represent the delay, and this function will return the last synapse of the chain.
        The other neurons and synapses in the chain can be accessed via the :py:attr:`delay_chain` and
        :py:attr:`delay_chain_synapses` properties of the synapse, respectively.

        While only positive delay values are supported due to temporal consistency and causality requirements,
        The delay will be stored as ``delay * -1`` in the model to represent that it is a chained delay.
        This does not affect the effective delay value, as the delay will still be applied via the delay chain.

        Note that delays of delay chains cannot be modified after creation.

        Raises
        ------
        TypeError

            * ``pre_id`` or ``post_id`` is not neuron or neuron ID (``int``).
            * ``weight`` is not a ``float``.
            * ``delay`` cannot be cast to ``int``.
            * ``exist`` is not a ``str``.

        ValueError

            * ``pre_id`` or ``post_id`` is not a valid neuron or neuron ID.
            * ``delay`` is less than or equal to ``0``
            * ``exist`` is not one of ``'error', 'overwrite', 'dontadd'``.
            * Synapse with the given pre- and post-synaptic neurons already exists, ``exist='overwrite'``, and ``delay != 1``.

        RuntimeError

            * Synapse with the given pre- and post-synaptic neurons already exists and ``exist='error'``.

        Returns
        -------
        Synapse


        .. seealso::

           :py:meth:`Neuron.connect_child`, :py:meth:`Neuron.connect_parent`
        """

        # TODO: make delay chaining an SNN option
        # TODO: ensure created hidden synapses are not flagged as newdelay

        # Ensure we work with neuron ids
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if isinstance(post_id, Neuron):
            post_id = post_id.idx

        # input validation
        fname = 'create_synapse()'

        if not is_intlike_catch(pre_id):
            raise TypeError("pre_id must be int or Neuron.")
        pre_id = int(pre_id)

        if not is_intlike_catch(post_id):
            raise TypeError("post_id must be int or Neuron.")
        post_id = int(post_id)

        weight = float_err(weight, 'weight', fname)
        delay = int_err(delay, 'delay', fname)

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

        if (enable_stdp := kwargs.pop('enable_stdp', None)) is not None:
            warnings.warn("create_synapse kwarg 'enable_stdp' is deprecated. Use 'stdp_enabled' instead.",
                          DeprecationWarning, stacklevel=2)
            stdp_enabled = enable_stdp

        ambiguous = ('true', '1', 'y', 'yes', 'on', 'f', 'false', '0', 'n', 'no', 'off')
        if isinstance(stdp_enabled, str) and stdp_enabled.lower() in ambiguous:
            msg = f"{fname} argument stdp_enabled received {stdp_enabled!r}"
            msg += " which has ambiguous truthiness. Consider using an explicit boolean value instead."
            warnings.warn(msg, stacklevel=2)

        last_in_chain = kwargs.pop('_is_last_chained_synapse', False)
        if delay <= 0 and not last_in_chain:
            raise ValueError("delay must be greater than or equal to 1")

        if kwargs:
            msg = f"create_synapse() received unexpected keyword arguments: {list(kwargs.keys())}"
            raise TypeError(msg)

        if (idx := self.get_synapse_id(pre_id, post_id)) is not None:  # if synapse already exists
            if not isinstance(exist, str):
                raise TypeError("exist must be a string")
            exist = exist.lower()
            if exist == "error":
                msg = f"Synapse already exists: {self.synapses[idx]!s}"
                msg += "If this was intentional, choose arg exist=<'dontadd', 'overwrite'>."
                raise RuntimeError(msg)
            elif exist == "overwrite":
                # check if delay has changed
                if delay != self.synaptic_delays[idx]:
                    raise ValueError("create_synapse() tried to overwrite chained synapse with different delay.")
                # overwrite old synapse params
                self.pre_synaptic_neuron_ids[idx] = pre_id
                self.post_synaptic_neuron_ids[idx] = post_id
                self.synaptic_weights[idx] = weight
                self.synaptic_delays[idx] = delay
                self.enable_stdp[idx] = stdp_enabled
            elif exist == "dontadd":
                return self.synapses[idx]
            else:
                msg = f"Invalid value for exist: {exist}. Expected 'error', 'overwrite', or 'dontadd'."
                raise ValueError(msg)
            return self.synapses[idx]  # prevent fall-through if user catches the error

        # Set new synapse parameters
        if delay == 1 or last_in_chain:
            self.pre_synaptic_neuron_ids.append(pre_id)
            self.post_synaptic_neuron_ids.append(post_id)
            self.synaptic_weights.append(weight)
            self.synaptic_delays.append(delay)
            self.enable_stdp.append(stdp_enabled)
            self.connection_ids[(pre_id, post_id)] = self.num_synapses - 1
        else:
            for _d in range(int(delay) - 1):  # delay by stringing together hidden synapses
                temp_id = self.create_neuron()
                self.create_synapse(pre_id, temp_id)
                pre_id = temp_id
            # place weight on last hidden synapse
            self.create_synapse(pre_id, post_id, weight=weight, stdp_enabled=stdp_enabled,
                                delay=-delay, _is_last_chained_synapse=True)  # , chained_neuron_delay=True)

        # Return synapse ID
        return Synapse(self, self.num_synapses - 1)

    def add_spike(
        self,
        time: int,
        neuron_id: int | Neuron,
        value: float = 1.0,
        exist: str = "error",
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
        exist : str
            action for existing spikes on a neuron at a given time step.
            Should be one of ['error', 'overwrite', 'add', 'dontadd']. (default: 'error')

            if exist='add', the existing spike value is added to the new value.

        Raises
        ------
        TypeError
            if:

            * time cannot be precisely cast to int
            * neuron_id is not a Neuron or neuron ID (int)
            * value is not an int or float

        ValueError
            if spike already exists at that neuron and timestep and exist='error',
            or if exist is an invalid setting.


        .. seealso::

           :py:meth:`Neuron.add_spike`
        """

        # input validation
        fname = 'add_spike()'
        time = int_err(time, 'time', fname)
        value = float_err(value, 'value', fname)

        if isinstance(neuron_id, Neuron):
            neuron_id = neuron_id.idx
        if not is_intlike_catch(neuron_id):
            raise TypeError("neuron_id must be int or Neuron.")
        neuron_id = int(neuron_id)

        if time < 0:
            raise ValueError("time must be greater than or equal to zero.")

        if neuron_id < 0:
            raise ValueError("neuron_id must be greater than or equal to zero")
        elif neuron_id >= self.num_neurons:
            msg = f"Added spike to non-existent Neuron {neuron_id} at time {time}."
            raise warnings.warn(msg, stacklevel=2)

        # Ensure data structure exists for the requested time
        if time not in self.input_spikes:
            self.input_spikes[time] = {"nids": [], "values": []}

        if neuron_id in self.input_spikes[time]["nids"]:  # queued spike already exists
            if not isinstance(exist, str):
                raise TypeError("duplicate must be a string")
            exist = exist.lower()
            if exist == "error":
                msg = f"add_spike() encountered existing spike at time {time} for Neuron {neuron_id}. "
                msg += "If this was intentional, pass arg exist='add', 'dontadd', 'overwrite'."
                raise ValueError(msg)
            elif exist == "overwrite":
                idx = self.input_spikes[time]["nids"].index(neuron_id)
                self.input_spikes[time]["values"][idx] = value
            elif exist == "add":
                idx = self.input_spikes[time]["nids"].index(neuron_id)
                self.input_spikes[time]["values"][idx] += value
            elif exist == "dontadd":
                return
            else:
                msg = f"Invalid value for exist: {exist}. Expected 'error', 'overwrite', 'add', or 'dontadd'."
                raise ValueError(msg)
            return  # prevent fall-through if user catches the error

        # Add spike to input spikes
        self.input_spikes[time]["nids"].append(neuron_id)
        self.input_spikes[time]["values"].append(value)

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

        msg = "{} received a string which has ambiguous truthiness. Use an explicit boolean value instead."
        if isinstance(positive_update, str):
            warnings.warn(msg.format("positive_update"), stacklevel=2)
        if isinstance(negative_update, str):
            warnings.warn(msg.format("negative_update"), stacklevel=2)

        if positive_update or Apos is not None:
            if not isinstance(Apos, (list, np.ndarray)):
                raise TypeError("Apos should be a list")
            Apos: list
            if isinstance(Apos, list) and not all([isinstance(x, (int, float)) for x in Apos]):
                raise ValueError("All elements in Apos should be int or float")
            if positive_update is None:
                positive_update = True
            self.apos = Apos

        if negative_update or Aneg is not None:
            if not isinstance(Aneg, (list, np.ndarray)):
                raise TypeError("Aneg should be a list")
            Aneg: list
            if isinstance(Aneg, list) and not all([isinstance(x, (int, float)) for x in Aneg]):
                raise ValueError("All elements in Aneg should be int or float.")
            if negative_update is None:
                negative_update = True
            self.aneg = Aneg

        self.stdp_positive_update = bool(positive_update)
        self.stdp_negative_update = bool(negative_update)

        if positive_update and negative_update:
            if len(Apos) != len(Aneg):  # pyright: ignore[reportArgumentType]
                raise ValueError("Apos and Aneg should have the same size.")

        if not self.allow_incorrect_stdp_sign:
            if positive_update and not all([x >= 0.0 for x in Apos]):  # pyright:ignore[reportOptionalIterable]
                raise ValueError("All elements in Apos should be positive. To ignore this, set snn.allow_incorrect_stdp_sign=True .")  # noqa
            if negative_update and not all([x <= 0.0 for x in Aneg]):  # pyright:ignore[reportOptionalIterable]
                raise ValueError("All elements in Aneg should be negative. To ignore this, set snn.allow_incorrect_stdp_sign=True .")  # noqa

        if not any(self.enable_stdp):
            warnings.warn("STDP is not enabled on any synapse.", RuntimeWarning, stacklevel=2)
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

    def set_weights_from_mat(self, mat: np.ndarray[(int, int), _arr_floatlike_T] | np.ndarray | csc_array):
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

    def set_stdp_enabled_from_mat(self, mat: np.ndarray[(int, int), np.dtype[Any]] | np.ndarray | csc_array):
        self.enable_stdp = list(mat[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])

    def stdp_enabled_sparse(self, dtype=None):
        dtype = self.dbin if dtype is None else dtype
        return csc_array(
            (self.enable_stdp, (self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids)),
            shape=[self.num_neurons, self.num_neurons], dtype=dtype
        )

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
                if np.any(self._stdp_Apos < 0.0):
                    raise ValueError("All elements in Apos should be positive. To ignore this, set snn.allow_incorrect_stdp_sign=True .")  # noqa
            if self._do_negative_update:
                self._stdp_Aneg = np.asarray(self.aneg, self.dd)
                if np.any(self._stdp_Aneg > 0.0):
                    raise ValueError("All elements in Aneg should be negative. To ignore this, set snn.allow_incorrect_stdp_sign=True .")  # noqa

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

    def shorten_spike_train(self, time_steps: int | None = None):
        """Remove the oldest spikes from the spike train.

        If no arguments are provided, only keep what is necessary to resume the simulation.
        This will either be :py:attr:`stdp_time_steps` or 1 spike, whichever is larger.

        Parameters
        ----------
        time_steps : int, optional
            The number of time steps to keep at the end of the spike train.
            If ``None``, only keep what is necessary to resume the simulation.
        """
        if time_steps is None:
            time_steps = max(bool(self.spike_train), self.stdp_time_steps)
        self.spike_train = self.spike_train[-time_steps:]

    _internal_vars = [
        "_neuron_thresholds", "_neuron_leaks", "_neuron_reset_states", "_internal_states",
        "_neuron_refractory_periods", "_neuron_refractory_periods_original", "_weights",
        "_stdp_enabled_synapses", "_stdp_Apos", "_stdp_Aneg",
        "_spikes", "_input_spikes",
    ]
    """NumPy representation of the internal state variables used during simulation.

    When :py:meth:`SNN.release_mem()` is called, these variables are dereferenced.
    """

    def release_mem(self):
        """Delete internal variables created during computation. Doesn't delete model.

        This is a low-level function and has few safeguards. Use with caution.
        It will call :ref:`del` on all numpy internal state variables in :py:attr:`_internal_vars`.

        .. warning::

            ``release_mem()`` should be called at most once after :py:meth:`setup()` or :py:meth:`simulate()`.
            Do not call it before :py:meth:`setup()` or twice in a row.

        This function may alleviate memory leaks in some cases.
        """
        for var in self._internal_vars:
            if hasattr(self, var):
                delattr(self, var)
        self._stdp_Apos = np.array([])
        self._stdp_Aneg = np.array([])
        self._do_stdp = False
        self._is_sparse = False

    def recommend(self, time_steps: int):
        """Recommend a backend to use based on network size and continuous sim time steps."""
        score = self.num_neurons ** 2 * time_steps / 1e6

        if self.gpu and score > self.gpu_threshold and self.weight_sparsity() > 0.0005:
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
        if not is_intlike_catch(time_steps):
            raise TypeError("time_steps must be int.")
        time_steps = int(time_steps)

        # Value errors
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than zero.")

        # explicit_use = use = use.lower() if isinstance(use, str) else use
        if use is None:
            use = self.backend

        # is the user asking for sparsity explicitly?
        # first, canonicalize sparse arg
        sparse = self._parse_sparsity(sparse) if sparse is not None else None
        # sparse may be None, 'auto', True, or False, with None meaning unset
        # self.sparse may be 'auto', True, or False
        explicitly_sparse = (sparse is True
                             or self.sparse is True and sparse is not False)

        if not self.manual_setup:
            if use == 'auto':
                use = self.recommend(time_steps)
                if explicitly_sparse:
                    use = 'cpu'
            elif (explicitly_sparse and use != 'cpu'):
                msg = "simulate() received explicit request to use sparsity with a "
                msg += "non-cpu backend. Sparsity is not supported on 'jit' or 'gpu' yet."
                raise ValueError(msg)
            if use == 'gpu':
                sparse = False

            self._setup(sparse=sparse)
            self.setup_input_spikes(time_steps)
        elif sparse is not None:
            msg = "simulate() received sparsity argument in manual_setup mode."
            msg += " Pass sparse to setup() instead."
            raise ValueError(msg)

        if not use:
            use = 'cpu'

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

            gpu.post_synaptic[v_blocks, v_tpb](  # pyright: ignore[reportIndexIssue]
                weights,
                output_spikes,
                post_synapse,
            )

            gpu.lif[v_blocks, v_tpb](  # pyright: ignore[reportIndexIssue]
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
                    gpu.stdp_update[m_blocks, m_tpb](weights, prev_spikes, output_spikes,  # pyright: ignore[reportIndexIssue]
                                    stdp_enabled, apos[i], aneg[i])  # pyright: ignore[reportPossiblyUnboundVariable]

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

    eqvars = [
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
        'connection_ids',
        'default_dtype',
        'enable_stdp',
        'spike_train',
        'input_spikes',
        'stdp', 'apos', 'aneg',
        'stdp_positive_update', 'stdp_negative_update',
        'allow_incorrect_stdp_sign',
        'allow_signed_leak',
        '_sparse',
        '_backend',
        'manual_setup',
    ]
    """The list of public variables which define the SNN model.


    For equality comparisons (:py:meth:`SNN.__eq__()`) between two SNNs,
    this list is used to check that variables within the SNN are equal.

    For memoization (:py:meth:`SNN.memoize()`), this list gives the
    names of variables which can be memoized.

    .. versionchanged:: v3.2.0

        Added ``'connection_ids'`` to the list of public variables.

    """

    def __eq__(self, other, ignore_vars=None, mismatch='ignore'):
        """Checks if two SNNs are equal."""

        ignore_vars = ignore_vars if ignore_vars is not None else []

        def raise_mismatch(key, a, b):
            if mismatch == 'raise':
                msg = f"SNNs are not equal. Attribute {key} has different values: {a} != {b}"
                raise ValueError(msg)

        if mismatch not in ['ignore', 'raise']:
            msg = f"Invalid value for mismatch: {mismatch}"
            raise ValueError(msg)

        if not isinstance(other, SNN):
            return False
        for var in self.eqvars:
            a = getattr(self, var)
            b = getattr(other, var)
            if isinstance(a, (np.ndarray, csc_array)):
                if not np.array_equal(a, b):  # pyright: ignore[reportArgumentType]
                    raise_mismatch(var, a, b)
                    return False
            else:
                try:
                    if a != b:
                        raise_mismatch(var, a, b)
                        return False
                except ValueError as err:
                    try:
                        if np.any(np.asarray(a) != np.asarray(b)):
                            raise_mismatch(var, a, b)
                            return False
                    except ValueError as err:
                        raise
                    except Exception as err:
                        msg = f"Failed to compare {var}"
                        raise RuntimeError(msg) from err
        # must ensure all other code paths are covered by return False.
        # this way we can return False early if any of the above comparisons fail.
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

    def _to_json_dict(self, array_representation="json-native", skipkeys=None,
                      net_name=None, extra=None):
        """Exports the SNN to a dictionary.

        Parameters
        ----------
        array_representation : str, default="json-native"
            The representation to use for arrays.
            Can be "json-native", "base64", or "base85".
        skipkeys : list[str] | None, default=None
            The names of variables to omit from the export.
            Can contain any key from :py:attr:`eqvars`.
            Can also be None or empty list ``[]`` to export all variables.
        net_name : str | None, default=None
            The name of the network to export.
            If None, the resulting JSON will not have a ``"name"`` key.
        extra : dict | None, default=None
            User-defined data to include in the exported JSON.
            If None, the resulting JSON will not have an ``"extra"`` key.

        Returns
        -------
        dict
            The SNN as a dictionary.
        """
        from . import __version__ as snm_version
        skipkeys = [] if skipkeys is None else skipkeys
        skipkeys += ["connection_ids"]
        varnames = set(self.eqvars) - set(skipkeys)
        arep = array_representation

        def is_numeric_array(o):
            is_np = isinstance(o, np.ndarray) and o.dtype.kind in 'iuf'
            is_py = isinstance(o, (tuple, list)) and all([isinstance(x, (int, float, bool)) for x in o])
            return is_np or is_py

        def is_bool_array(o):
            if isinstance(o, np.ndarray):
                try:
                    return np.issubdtype(o.dtype, np.bool)
                except AttributeError:
                    return issubclass(o.dtype.type, np.bool_)
            else:
                return all([isinstance(x, bool) for x in o])

        def default(self, o):
            if isinstance(o, np.ndarray):
                if is_bool_array(o):
                    return o.astype(np.int_).tolist()
                return o.tolist()
            if issubclass(o, np.generic):
                return o.__name__
            msg = f'Object of type {o.__class__.__name__} is not JSON serializable'
            raise TypeError(msg)

        def get_dtype(o):
            byteorder = o.dtype.byteorder
            if byteorder == '=':
                if sys.byteorder == 'little':
                    byteorder = '<'
                elif sys.byteorder == 'big':
                    byteorder = '>'
                else:
                    byteorder = sys.byteorder
            if byteorder not in ['<', '>', '|']:
                msg = f"Unknown byteorder {byteorder}"
                raise RuntimeError(msg)
            return byteorder + o.dtype.char

        if arep == "json-native":
            data = {var: getattr(self, var) for var in varnames}

            for k, v in data.items():
                if is_numeric_array(v) and is_bool_array(v):
                    data[k] = np.asarray(v, dtype=np.int_).tolist()
        elif arep in ["base85", "base64"]:
            from base64 import b85encode, b64encode
            encode = b85encode if arep == "base85" else b64encode
            data = {var: getattr(self, var) for var in varnames}
            for k, v in data.items():
                if is_numeric_array(v):
                    dtype = self.dbin if is_bool_array(v) else self.dd
                    arr = np.asarray(v, dtype=dtype)
                    data[k] = {
                        "dtype": get_dtype(arr),
                        "original_type": v.__class__.__name__,
                        arep: encode(arr.tobytes()).decode('utf-8')
                    }
        else:
            raise ValueError("array_representation must be 'json-native' or 'base85'.")

        json.JSONEncoder.default = default

        d = {
            "$schema": "https://ornl.github.io/superneuromat/schema/0.1/snn.json",
            "version": "0.1",
            "networks": [],
        }
        networkd = {
            "meta": {
                "array_representation": arep,
                "from": {  # We don't validate this right now.
                    "module": "superneuromat",
                    "version": snm_version,
                },
                "format": "snm",
                "format_version": "0.1",  # Not currently validating this
                "type": self.__class__.__name__,  # differentiate from other snm network types
            },
            "data": data,
        }
        if net_name is not None:
            networkd["name"] = net_name
        if extra is not None and isinstance(extra, dict):
            d["extra"] = extra
        d["networks"].append(networkd)
        return d

    def to_json(self, array_representation="json-native", skipkeys: list[str] | None = None,
                net_name: str | None = None, extra: dict | None = None, indent=2, **kwargs):
        """Exports the SNN to a JSON string.

        Parameters
        ----------
        array_representation : str, default="json-native"
            The representation to use for arrays.
            Can be "json-native", "base64", or "base85".
        skipkeys : list[str] | None, default=None
            The names of variables to omit from the export.
            Can contain any key from :py:attr:`eqvars`.
            Can also be None or empty list ``[]`` to export all variables.
        net_name : str | None, default=None
            The name of the network to export.
            If None, the resulting JSON will not have a ``"name"`` key.
        extra : dict | None, default=None
            User-defined data to include in the exported JSON.
            If None, the resulting JSON will not have an ``"extra"`` key.
        indent : int, default=2
            The indentation to use for the JSON.
            Set to ``None`` to get a compact JSON string.


        This function additionally accepts the same arguments as :py:meth:`json.dumps`.

        Returns
        -------
        str
            The SNN as a JSON string.

        Examples
        --------
        >>> snn.to_json(net_name="My SNN", indent=None)
        {"$schema": "https://ornl.github.io/superneuromat/schema/0.1/snn.json", "version": "0.1", "networks": [{"meta": {"array_representation": "json-native", "from": {"module": "superneuromat", "version": "3.1.0"}, "format": "snm", "format_version": "0.1", "type": "SNN"}, "data": {"neuron_refractory_periods": [0, 0], "neuron_states": [0.0, 0.0], "num_synapses": 2, "post_synaptic_neuron_ids": [1, 0], "aneg": [], "enable_stdp": [0, 0], "pre_synaptic_neuron_ids": [0, 1], "neuron_thresholds": [3.141592653589793115997963468544185161590576171875, 0.0], "neuron_refractory_periods_state": [0.0, 0.0], "neuron_reset_states": [0.0, 0.0], "stdp_positive_update": true, "input_spikes": {"3": {"nids": [1], "values": [1.0]}}, "synaptic_delays": [1, 1], "allow_signed_leak": false, "num_neurons": 2, "_sparse": "auto", "_backend": "auto", "apos": [], "manual_setup": false, "spike_train": [[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], "neuron_leaks": [Infinity, Infinity], "allow_incorrect_stdp_sign": false, "stdp": true, "synaptic_weights": [1.0, 1.0], "default_dtype": "float64", "stdp_negative_update": true}}]}
        """  # noqa: E501 (line length)
        d = self._to_json_dict(array_representation, skipkeys=skipkeys, net_name=net_name, extra=extra)
        return json.dumps(d, indent=indent, **kwargs)

    def saveas_json(self, fp,
                    array_representation="json-native", skipkeys: list[str] | None = None,
                    net_name: str | None = None, extra: dict | None = None, indent=2, **kwargs):
        """Exports the SNN to a JSON file.

        Parameters
        ----------
        array_representation : str, default="json-native"
            The representation to use for arrays.
            Can be "json-native", "base64", or "base85".
        skipkeys : list[str] | None, default=None
            The names of variables to omit from the export.
            Can contain any key from :py:attr:`eqvars`.
            Can also be None or empty list ``[]`` to export all variables.
        net_name : str | None, default=None
            The name of the network to export.
            If None, the resulting JSON will not have a ``"name"`` key.
        extra : dict | None, default=None
            User-defined data to include in the exported JSON.
            If None, the resulting JSON will not have an ``"extra"`` key.
        indent : int, default=2
            The indentation to use for the JSON.
            Set to ``None`` to get a compact JSON string.


        This function additionally accepts the same arguments as :py:meth:`json.dump`.

        Examples
        --------
        >>> with open('my_model.snn.json', 'w') as f:
        >>>     snn.saveas_json(f, net_name="My SNN")
        """
        d = self._to_json_dict(array_representation, skipkeys=skipkeys, net_name=net_name, extra=extra)
        return json.dump(d, fp, indent=indent, **kwargs)

    def from_json_network(self, net_dict: dict, skipkeys: list[str] | tuple[str] | None = None):
        from base64 import b85decode, b64decode
        if net_dict["meta"]["format"] != "snm" or net_dict["meta"]["type"] != self.__class__.__name__:
            msg = f"Could not import network with format {net_dict['meta']['format']}. "
            msg += "Only networks in the snm format are supported."
            raise NotImplementedError(msg)

        def is_encoded_dict(d) -> bool | str:
            # return the encoding type if it's a dict with a base85 or base64 key
            if isinstance(d, dict):
                for k in ("base85", "base64"):
                    if k in d:
                        return k
            return False

        data = net_dict["data"]

        skipkeys = set(skipkeys) if skipkeys is not None else set()
        should_modify = set(data.keys()) - skipkeys  # self variables to modify

        # set our default dtype before setting the arrays
        if "default_dtype" in should_modify:
            np.dtype(data["default_dtype"])  # test if dtype is valid
            self.default_dtype = getattr(np, data["default_dtype"])

        # variables which we'll take care of outside of the loop
        special_vars = {"default_dtype", "spike_train", "connection_ids", "num_neurons", "num_synapses"}

        # set most of our properties
        for key in should_modify - special_vars:
            value = data[key]
            if (arep := is_encoded_dict(value)):
                bdict = {"dtype": np.dtype(self.dd), "original_type": "list"}
                bdict.update(value)
                decode = b85decode if arep == "base85" else b64decode
                value = np.frombuffer(decode(value[arep]), dtype=bdict["dtype"])
                if bdict["original_type"] == "list":
                    value = value.tolist()
            try:
                setattr(self, key, value)
            except AttributeError as err:
                if key not in self.eqvars:
                    msg = f"Invalid key: {key}"
                    raise ValueError(msg) from err

        # deal with importing special variables
        if "spike_train" in should_modify:
            self.spike_train = np.asarray(data["spike_train"], dtype=self.dbin)
        if "input_spikes" in should_modify:
            self.input_spikes = {int(k): v for k, v in self.input_spikes.items()}
        if "enable_stdp" in should_modify and not is_encoded_dict(data["enable_stdp"]):
            self.enable_stdp = np.asarray(self.enable_stdp, dtype=self.dbin).tolist()

        # deal with variables which derive from other variables (not set from the JSON)
        if "connection_ids" not in skipkeys:
            self.rebuild_connection_ids()
            if "num_synapses" in should_modify:  # if num_synapses is in the JSON, verify it
                if data["num_synapses"] != len(self.connection_ids):
                    msg = f"num_synapses ({data['num_synapses']}) does not match the number of synapses"
                    msg += f"in the network ({len(self.connection_ids)})."
                    raise RuntimeError(msg)

        return self

    def rebuild_connection_ids(self):
        self.connection_ids = {(a, b): i for i, (a, b) in
                               enumerate(zip(self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids))}

    def from_jsons(self, json_str: str, net_id: int | str = 0,
                   skipkeys: list[str] | tuple[str] | None = None):
        """Update this SNN from a SuperNeuroMat JSON string.

        Parameters
        ----------
        json_str : str
            _description_
        net_id : int | str (default: 0)
            ID of the network to load. Can be an integer, which represents its index in the JSON network list,
            or a string, which represents the name of the network stored in ``json_dict["networks"]["meta"]["name"]``.
        skipkeys : list[str] | tuple[str] | None (default: None)
            Keys from ``json_dict["networks"]["data"]`` to skip when loading the network.

        Returns
        -------
        self
            This SNN will be updated with the specified network data from the JSON string.

        Raises
        ------
        ValueError
            If network with given ID not found, or if multiple networks with the given ID are found.

        Examples
        --------
        >>> snn = SNN()
        >>> with open('my_model.snn.json', 'r') as f:
        >>>     snn.from_jsons(f.read(), net_id="My SNN")
        """
        from . import json

        j = json.loads(json_str)

        skipkeys = list(skipkeys) if skipkeys is not None else []

        nets = j["networks"]
        if isinstance(net_id, int):
            net_dict = nets[net_id]
        elif isinstance(net_id, str):
            net_dicts = [net for net in nets if net.get("name", None) == net_id]
            if len(net_dicts) == 0:
                msg = f"No network with name {net_id} found."
                raise ValueError(msg)
            elif len(net_dicts) > 1:
                msg = f"Multiple networks with name {net_id} found."
                raise ValueError(msg)
            net_dict = net_dicts[0]
        else:
            raise NotImplementedError("net_id must be int or str. Importing multiple networks is not supported yet.")

        return self.from_json_network(net_dict, skipkeys=skipkeys)
