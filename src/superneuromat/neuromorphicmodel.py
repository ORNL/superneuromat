import math
import numpy as np
import pandas as pd
from numba import jit, cuda
import ctypes


"""

TODO:

1. Input spikes [DONE]
2. Remove threshold and leak from the weight matrix? [DONE]
3. Implement synaptic delays by adding proxy neurons [DONE, BUT IMPEDES PERFORMANCE]
4. Have Neuron and Synapse classes? [NOPE - NOT EFFICIENT]
5. Reset state [DONE]
6. Spike monitoring [DONE]
7. Leak should push towards zero regardless of positive or negative internal state [DONE]
8. Refractory period [DONE]
9. STDP (outer product) [DONE]
10. Function arguments in a way that includes the data type [DONE]
11. Type, value and runtime errors: Test thoroughly using unittest [DONE]
12. Print/display function to list all variables, all neuron parameters and all synapse parameters [DONE]
13. Docstrings everywhere [DONE]
14. create_neurons()
15. create_synapses()
16. Tutorials: one neuron, two neurons
17. Leak equations should not check for spikes [DONE]

"""


"""

FEATURE REQUESTS:

1. Visualize spike raster
2. Monitor STDP synapses
3. Reset neuromorphic model

"""


@jit(nopython=True)
def lif_jit(
    tick: int,
    input_spikes,
    prev_spikes,
    states,
    thresholds,
    leaks,
    reset_states,
    refractory_periods,
    refractory_periods_original,
    weights,
):
    # CAUTION: This function has side-effects (not a pure function)
    # prev_spikes and states are modified in-place
    # ___________     ______
    # DO NOT ASSIGN THESE VARIABLES WITHIN THIS FUNCTION or things will break
    # DO NOT states = something

    # Leak: internal state > reset state
    indices = states > reset_states
    states[indices] = np.maximum(
        states[indices] - leaks[indices], reset_states[indices]
    )

    # Leak: internal state < reset state
    indices = states < reset_states
    states[indices] = np.minimum(
        states[indices] + leaks[indices], reset_states[indices]
    )

    # Internal state (in-place)
    states += input_spikes[tick] + (weights.T @ prev_spikes)

    # Compute spikes (in-place) into prev_spikes (numba doesn't support keyword 'out')
    spikes = np.greater(states, thresholds, prev_spikes)

    # Refractory period: Compute indices of neuron which are in their refractory period
    indices = refractory_periods > 0

    # For neurons in their refractory period, zero out their spikes and decrement refractory period by one
    spikes[indices] = 0
    refractory_periods[indices] -= 1

    # For spiking neurons, turn on refractory period
    mask = spikes.astype(np.bool)
    refractory_periods[mask] = refractory_periods_original[mask]

    # Reset internal states (in-place)
    states[spikes == 1.0] = reset_states[spikes == 1.0]

    prev_spikes[:] = spikes

    # states, prev_spikes were modified in-place
    # everything else is local


@jit(nopython=True)
def stdp_update_jit(n_steps, spike_train, weights_pointer, apos, aneg, stdp_enabled):
    t = min(len(spike_train) - 1, n_steps)
    shape = spike_train[-1].shape

    if t > 0:
        update_synapses = np.outer(
            spike_train[-t - 1:-1],
            spike_train[-1]
        ).reshape((-1, *shape))

        if np.any(apos):
            weights_pointer += (
                (update_synapses.T * apos[0:t][::-1]).T
            ).sum(axis=0) * stdp_enabled

        if np.any(aneg):
            weights_pointer += (
                ((1 - update_synapses).T * aneg[0:t][::-1]).T
            ).sum(axis=0) * stdp_enabled


def resize_vec(a, len, dtype=np.float64):
    a = np.asarray(a, dtype=dtype)
    if len(a) < len:
        return a[:len].copy()
    elif len(a) > len:
        return np.pad(a, (0, len - len(a)), 'constant')
    else:
        return a


class NeuromorphicModel:
    """Defines a neuromorphic model with neurons and synapses

    Attributes:
            num_neurons (int): Number of neurons in the neuromorphic model
            neuron_thresholds (list): List of neuron thresholds
            neuron_leaks (list): List of neuron leaks, defined as the amount by which the internal states of the neurons are pushed towards the neurons' reset states
            neuron_reset_states (list): List of neuron reset states
            neuron_refractory_periods (list): List of neuron refractory periods

            num_synapses (int): Number of synapses in the neuromorphic model
            pre_synaptic_neuron_ids (list): List of pre-synaptic neuron IDs
            post_synaptic_neuron_ids (list): List of post-synaptic neuron IDs
            synaptic_weights (list): List of synaptic weights
            synaptic_delays (list): List of synaptic delays
            enable_stdp (list): List of Boolean values denoting whether STDP learning is enabled on each synapse

            input_spikes (dict): Dictionary of input spikes indexed by time
            spike_train (list): List of spike trains for each time step

            stdp (bool): Boolean parameter that denotes whether STDP learning has been enabled in the neuromorphic model
            stdp_time_steps (int): Number of time steps over which STDP updates are made
            stdp_Apos (list): List of STDP parameters per time step for excitatory update of weights
            stdp_Aneg (list): List of STDP parameters per time step for inhibitory update of weights


    Methods:
            create_neuron: Creates a neuron in the neuromorphic model
            create_synapse: Creates a synapse in the neuromorphic model
            add_spike: Add an external spike at a particular time step for a given neuron with a given value
            stdp_setup: Setup the STDP parameters
            setup: Setup the neuromorphic model and prepare for simulation
            simulate: Simulate the neuromorphic model for a given number of time steps
            print_spike_train: Print the spike train


    CAUTION:
            1. Delay is implemented by adding a chain of proxy neurons. A delay of 10 between neuron A and neuron B would add 9 proxy neurons between A and B.
            2. Leak brings the internal state of the neuron back to the reset state. The leak value is the amount by which the internal state of the neuron is pushed towards its reset state.
            3. Deletion of neurons is not permitted
            4. Input spikes can have a value
            5. All neurons are monitored by default

    """

    gpu_threshold = 10000
    jit_threshold = 1000
    disable_performance_warnings = True

    def __init__(self):
        """Initialize the neuromorphic model"""

        # Neuron parameters
        self.num_neurons = 0
        self.neuron_thresholds = []
        self.neuron_leaks = []
        self.neuron_reset_states = []
        self.neuron_refractory_periods = []

        # Synapse parameters
        self.num_synapses = 0
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
        self.stdp = False
        self._stdp_Apos = []
        self._stdp_Aneg = []
        self.stdp_positive_update = False
        self.stdp_negative_update = False

        self.gpu = cuda.is_available()

    @property
    def stdp_time_steps(self):
        assert len(self._stdp_Apos) == len(self._stdp_Aneg)
        return len(self._stdp_Apos)
        """Display the neuromorphic class in a legible format"""

    @property
    def neuron_df(self):
        """Returns a DataFrame containing information about neurons."""
        return pd.DataFrame({
            "Neuron ID": list(range(self.num_neurons)),
            "Threshold": self.neuron_thresholds,
            "Leak": self.neuron_leaks,
            "Reset State": self.neuron_reset_states,
            "Refractory Period": self.neuron_refractory_periods,
        })

    @property
    def synapse_df(self):
        """Returns a DataFrame containing information about synapses."""
        return pd.DataFrame({
            "Pre Neuron ID": self.pre_synaptic_neuron_ids,
            "Post Neuron ID": self.post_synaptic_neuron_ids,
            "Weight": self.synaptic_weights,
            "Delay": self.synaptic_delays,
            "STDP Enabled": self.enable_stdp,
        })

    @property
    def stdp_info(self):
        """Returns a string containing information about STDP."""
        return (
            f"STDP Enabled: {self.stdp} \n"
            + f"STDP Time Steps: {self.stdp_time_steps} \n"
            + f"STDP A positive: {self._stdp_Apos} \n"
            + f"STDP A negative: {self._stdp_Aneg}"
        )

    def __repr__(self):

        # Input Spikes
        times = []
        nids = []
        values = []

        num_neurons_str = f"Number of neurons: {self.num_neurons}"
        num_synapses_str = f"Number of synapses: {self.num_synapses}"

        for time in self.input_spikes:
            for nid, value in zip(self.input_spikes[time]["nids"], self.input_spikes[time]["values"]):
                times.append(time)
                nids.append(nid)
                values.append(value)

        input_spikes_df = pd.DataFrame({"Time": times, "Neuron ID": nids, "Value": values})

        # Spike train
        spike_train = ""
        for time, spikes in enumerate(self.spike_train):
            spike_train += f"Time: {time}, Spikes: {spikes}\n"

        return (
            num_neurons_str + "\n" + num_synapses_str + "\n"
            "\nNeuron Info: \n"
            + self.neuron_df.to_string(index=False)
            + "\n"
            + "\nSynapse Info: \n"
            + self.synapse_df.to_string(index=False)
            + "\n"
            + "\nSTDP Info: \n"
            + self.stdp_info
            + "\n"
            + "\nInput Spikes: \n"
            + input_spikes_df.to_string(index=False)
            + "\n"
            + "\nSpike Train: \n"
            + spike_train
        )

    def create_neuron(
        self, threshold: float = 0.0, leak: float = np.inf, reset_state: float = 0.0, refractory_period: int = 0
    ) -> int:
        """Create a neuron

        Args:
                threshold (float): Neuron threshold; the neuron spikes if its internal state is strictly greater than the neuron threshold (default: 0.0)
                leak (float): Neuron leak; the amount by which by which the internal state of the neuron is pushed towards its reset state (default: np.inf)
                reset_state (float): Reset state of the neuron; the value assigned to the internal state of the neuron after spiking (default: 0.0)
                refractory_period (int): Refractory period of the neuron; the number of time steps for which the neuron remains in a dormant state after spiking

        Returns:
                Returns the neuron ID

        Raises:
                TypeError if:
                        1. threshold is not an int or a float
                        2. leak is not an int or a float
                        3. reset_state is not an int or a float
                        4. refractory_period is not an int

                ValueError if:
                        1. leak is less than 0.0
                        2. refractory_period is less than 0

        """

        # Type errors
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be int or float")

        if not isinstance(leak, (int, float)):
            raise TypeError("leak must be int or float")

        if not isinstance(reset_state, (int, float)):
            raise TypeError("reset_state must be int or float")

        if not isinstance(refractory_period, int):
            raise TypeError("refractory_period must be int")

        # Value errors
        if leak < 0.0:
            raise ValueError("leak must be grater than or equal to zero")

        if refractory_period < 0:
            raise ValueError("refractory_period must be greater than or equal to zero")

        # Collect neuron parameters
        self.neuron_thresholds.append(threshold)
        self.neuron_leaks.append(leak)
        self.neuron_reset_states.append(reset_state)
        self.neuron_refractory_periods.append(refractory_period)
        self.num_neurons += 1

        # Return neuron ID
        return self.num_neurons - 1

    def create_synapse(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 1.0,
        delay: int = 1,
        stdp_enable: bool = False
    ) -> None:
        """Creates a synapse in the neuromorphic model from a pre-synaptic neuron to a post-synaptic neuron with a given set of synaptic parameters (weight, delay and enable_stdp)

        Args:
                pre_id (int): ID of the pre-synaptic neuron
                post_id (int): ID of the post-synaptic neuron
                weight (float): Synaptic weight; weight is multiplied to the incoming spike (default: 1.0)
                delay (int): Synaptic delay; number of time steps by which the outgoing signal of the syanpse is delayed by (default: 1)
                enable_stdp (bool): Boolean value that denotes whether or not STDP learning is enabled on the synapse (default: False)

        Raises:
                TypeError if:
                        1. pre_id is not an int
                        2. post_id is not an int
                        3. weight is not a float
                        4. delay is not an int
                        5. enable_stdp is not a bool

            ValueError if:
                1. pre_id is less than 0
                2. post_id is less than 0
                3. delay is less than or equal to 0

        """
        # TODO: deprecate enable_stdp

        # Type errors
        if not isinstance(pre_id, int):
            raise TypeError("pre_id must be int")

        if not isinstance(post_id, int):
            raise TypeError("post_id must be int")

        if not isinstance(weight, (int, float)):
            raise TypeError("weight must be a float")

        if not isinstance(delay, int):
            raise TypeError("delay must be an integer")

        if not isinstance(stdp_enable, bool):
            raise TypeError("stdp_enable must be a bool")

        # Value errors
        if pre_id < 0:
            raise ValueError("pre_id must be greater than or equal to zero")

        if post_id < 0:
            raise ValueError("post_id must be greater than or equal to zero")

        if delay <= 0:
            raise ValueError("delay must be greater than or equal to 1")

        # Collect synapse parameters
        if delay == 1:
            self.pre_synaptic_neuron_ids.append(pre_id)
            self.post_synaptic_neuron_ids.append(post_id)
            self.synaptic_weights.append(weight)
            self.synaptic_delays.append(delay)
            self.enable_stdp.append(stdp_enable)
            self.num_synapses += 1

        else:
            for _d in range(int(delay) - 1):
                temp_id = self.create_neuron()
                self.create_synapse(pre_id, temp_id)
                pre_id = temp_id

            self.create_synapse(pre_id, post_id, weight=weight, stdp_enable=stdp_enable)

        # Return synapse ID
        return self.num_synapses - 1

    def add_spike(self, time: int, neuron_id: int, value: float = 1.0) -> None:
        """Adds an external spike in the neuromorphic model

        Args:
                time (int): The time step at which the external spike is added
                neuron_id (int): The neuron for which the external spike is added
                value (float): The value of the external spike (default: 1.0)

        Raises:
                TypeError if:
                        1. time is not an int
                        2. neuron_id is not an int
                        3. value is not an int or float

        """

        # Type errors
        if not isinstance(time, int):
            raise TypeError("time must be int")

        if not isinstance(neuron_id, int):
            raise TypeError("neuron_id must be int")

        if not isinstance(value, (int, float)):
            raise TypeError("value must be int or float")

        # Value errors
        if time < 0:
            raise ValueError("time must be greater than or equal to zero")

        if neuron_id < 0:
            raise ValueError("neuron_id must be greater than or equal to zero")

        # Add spikes
        if time in self.input_spikes:
            self.input_spikes[time]["nids"].append(neuron_id)
            self.input_spikes[time]["values"].append(value)

        else:
            self.input_spikes[time] = {}
            self.input_spikes[time]["nids"] = [neuron_id]
            self.input_spikes[time]["values"] = [value]

    @property
    def apos(self):
        return self._stdp_Apos

    @apos.setter
    def apos(self, value):
        value = np.asarray(value, np.float64)
        if len(value) != len(self._stdp_Aneg):
            n = max(len(value), len(self._stdp_Aneg))
            self._stdp_Aneg = resize_vec(self._stdp_Aneg, n)
            value = resize_vec(value, n)
        self._stdp_Apos = value

    @property
    def aneg(self):
        return self._stdp_Aneg

    @aneg.setter
    def aneg(self, value):
        value = np.asarray(value, np.float64)
        if len(value) != len(self._stdp_Apos):
            n = max(len(value), len(self._stdp_Apos))
            self._stdp_Apos = resize_vec(self._stdp_Apos, n)
            value = resize_vec(value, n)
        self._stdp_Aneg = value

    def stdp_setup(
        self,
        time_steps: int = 3,
        Apos: list | None = None,
        Aneg: list | None = None,
        positive_update: bool = True,
        negative_update: bool = True,
    ) -> None:
        """Setup the Spike-Time-Dependent Plasticity (STDP) parameters

        Args:
                time_steps (int): Number of time steps over which STDP learning occurs (default: 3)
                Apos (list): List of parameters for excitatory STDP updates (default: [1.0, 0.5, 0.25]); number of elements in the list must be equal to time_steps
                Aneg (list): List of parameters for inhibitory STDP updates (default: [1.0, 0.5, 0.25]); number of elements in the list must be equal to time_steps
                positive_update (bool): Boolean parameter indicating whether excitatory STDP update should be enabled
                negative_update (bool): Boolean parameter indicating whether inhibitory STDP update should be enabled

        Raises:
                TypeError if:
                        1. time_steps is not an int
                        2. Apos is not a list
                        3. Aneg is not a list
                        4. positive_update is not a bool
                        5. negative_update is not a bool

                ValueError if:
                        1. time_steps is less than or equal to zero
                        2. Number of elements in Apos is not equal to the time_steps
                        3. Number of elements in Aneg is not equal to the time_steps
                        4. The elements of Apos are not int or float
                        5. The elements of Aneg are not int or float
                        6. The elements of Apos are not greater than or equal to 0.0
                        7. The elements of Apos are not greater than or equal to 0.0

                RuntimeError if:
                        1. enable_stdp is not set to True on any of the synapses

        """

        # Type errors
        if not isinstance(time_steps, int):
            raise TypeError("time_steps should be int")

        if Apos is None and Aneg is None:
            Apos = [1.0, 0.5, 0.25]
            Aneg = [1.0, 0.5, 0.25]

        if not isinstance(Apos, list):
            raise TypeError("Apos should be a list")
        Apos: list

        if not isinstance(Aneg, list):
            raise TypeError("Aneg should be a list")
        Aneg: list

        if not isinstance(positive_update, bool):
            raise TypeError("positive_update must be a bool")

        if not isinstance(negative_update, bool):
            raise TypeError("negative_update must be a bool")

        # Value error
        if time_steps <= 0:
            raise ValueError("time_steps should be greater than zero")

        if positive_update and len(Apos) != time_steps:
            msg = f"Length of Apos should be {time_steps}"
            raise ValueError(msg)

        if negative_update and len(Aneg) != time_steps:
            msg = f"Length of Aneg should be {time_steps}"
            raise ValueError(msg)

        if positive_update and not all([isinstance(x, (int, float)) for x in Apos]):
            raise ValueError("All elements in Apos should be int or float")

        if negative_update and not all([isinstance(x, (int, float)) for x in Aneg]):
            raise ValueError("All elements in Aneg should be int or float")

        # if positive_update and not all([x >= 0.0 for x in Apos]):
        #     raise ValueError("All elements in Apos should be positive")

        # if negative_update and not all([x >= 0.0 for x in Aneg]):
        #     raise ValueError("All elements in Aneg should be positive")

        # Runtime error
        if not any(self.enable_stdp):
            raise RuntimeError("STDP is not enabled on any synapse")

        # Collect STDP parameters
        self.stdp = True
        self._stdp_Apos = Apos
        self._stdp_Aneg = Aneg
        self.stdp_positive_update = positive_update
        self.stdp_negative_update = negative_update

    def setup(self):
        """Setup the neuromorphic model for simulation"""

        # Create numpy arrays for neuron state variables
        self._neuron_thresholds = np.asarray(self.neuron_thresholds, np.float64)
        self._neuron_leaks = np.asarray(self.neuron_leaks, np.float64)
        self._neuron_reset_states = np.asarray(self.neuron_reset_states, np.float64)
        self._neuron_refractory_periods_original = np.asarray(self.neuron_refractory_periods, np.float64)
        self._internal_states = np.asarray(self.neuron_reset_states, np.float64)

        self._neuron_refractory_periods = np.zeros(self.num_neurons, np.float64)
        self._spikes = np.zeros(self.num_neurons, np.float64)

        # Create numpy arrays for synapse state variables
        self._weights = np.zeros((self.num_neurons, self.num_neurons), np.float64)
        self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.synaptic_weights

        # Create numpy arrays for STDP state variables
        if self.stdp:
            self._stdp_enabled_synapses = np.zeros((self.num_neurons, self.num_neurons), np.int64)
            self._stdp_enabled_synapses[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.enable_stdp

            self._stdp_Apos = np.asarray(self._stdp_Apos, np.float64)
            self._stdp_Aneg = np.asarray(self._stdp_Aneg, np.float64)

        # Create numpy array for input spikes
        self._input_spikes = np.zeros((1, self.num_neurons), np.float64)

    def recommend(self, time_steps: int):
        """Recommend a backend to use based on network size and continuous sim time steps."""
        score = 0
        score += self.num_neurons * 1
        score += time_steps * 100

        if score > self.gpu_threshold and self.gpu:
            return 'gpu'
        elif score > self.jit_threshold:
            return 'jit'
        return 'cpu'

    def simulate(self, time_steps: int = 1000, callback=None, use='auto') -> None:
        """Simulate the neuromorphic spiking neural network

        Parameters
        ----------
        time_steps : int
            Number of time steps for which the neuromorphic circuit is to be simulated
        callback : function, optional
            Function to be called after each time step, by default None
        use : str, default='auto'
            Which backend to use. Can be 'cpu', 'jit', or 'gpu'.
            'auto' will choose a backend based on the network size and time steps.

        Raises
        ------
        TypeError
            If ``time_steps`` is not an int.
        ValueError
            If ``time_steps`` is less than or equal to zero.
        """

        # Type errors
        if not isinstance(time_steps, int):
            raise TypeError("time_steps must be int")

        # Value errors
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than zero")

        if use == 'auto':
            use = self.recommend(time_steps)
        if use == 'jit':
            self.simulate_cpu_jit(time_steps, callback)
        elif use == 'gpu':
            self.simulate_gpu(time_steps, callback)
        else:
            self.simulate_cpu(time_steps, callback)

    def simulate_cpu_jit(self, time_steps: int = 1000, callback=None) -> None:
        # print("Using CPU with Numba JIT optimizations")
        max_timestep = max(list(self.input_spikes.keys()) + [time_steps]) + 1
        self._input_spikes = np.zeros((max_timestep, self.num_neurons), np.float64)
        for t, spikes_dict in self.input_spikes.items():
            for neuron_id, amplitude in zip(spikes_dict["nids"], spikes_dict["values"]):
                self._input_spikes[t][neuron_id] = amplitude

        self._output_spikes = np.zeros((time_steps, len(self._internal_states)), np.float64)

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

            self.spike_train.append(self._spikes)

            if self.stdp:
                stdp_update_jit(
                    tick,
                    self.stdp_time_steps,
                    self._spikes,
                    self._weights,
                    self.apos,
                    self.aneg,
                    self._stdp_enabled_synapses,
                )

        # Update weights if STDP was enabled
        if self.stdp:
            self.synaptic_weights = list(self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])

    def simulate_cpu(self, time_steps: int = 1000, callback=None) -> None:
        max_timestep = max(list(self.input_spikes.keys()) + [time_steps]) + 1
        self._input_spikes = np.zeros((max_timestep, len(self._spikes)))
        for t, spikes_dict in self.input_spikes.items():
            for neuron_id, amplitude in zip(spikes_dict["nids"], spikes_dict["values"]):
                self._input_spikes[t][neuron_id] = amplitude

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
            self._internal_states = self._internal_states + self._input_spikes[tick] + (self._weights.T @ self._spikes)

            # Compute spikes
            self._spikes = np.greater(self._internal_states, self._neuron_thresholds).astype(int)

            # Refractory period: Compute indices of neuron which are in their refractory period
            indices = np.greater(self._neuron_refractory_periods, 0)

            # For neurons in their refractory period, zero out their spikes and decrement refractory period by one
            self._spikes[indices] = 0
            self._neuron_refractory_periods[indices] -= 1

            # For spiking neurons, turn on refractory period
            self._neuron_refractory_periods[self._spikes.astype(bool)] = self._neuron_refractory_periods_original[
                self._spikes.astype(bool)
            ]

            # Reset internal states
            self._internal_states[self._spikes == 1.0] = self._neuron_reset_states[self._spikes == 1.0]

            # Append spike train
            self.spike_train.append(self._spikes)

            # STDP Operations
            t = min(self.stdp_time_steps, len(self.spike_train) - 1)

            if t > 0:
                update_synapses = np.outer(
                    np.array(self.spike_train[-t - 1:-1]),
                    np.array(self.spike_train[-1])).reshape([-1, self.num_neurons, self.num_neurons]
                )

                if self.stdp_positive_update:
                    self._weights += (
                        (update_synapses.T * self._stdp_Apos[0:t][::-1]).T
                    ).sum(axis=0) * self._stdp_enabled_synapses

                if self.stdp_negative_update:
                    self._weights += (
                        ((1 - update_synapses).T * self._stdp_Aneg[0:t][::-1]).T
                    ).sum(axis=0) * self._stdp_enabled_synapses

        # Update weights if STDP was enabled
        if self.stdp:
            self.synaptic_weights = list(self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])

    def simulate_gpu(self, time_steps: int = 1000, callback=None) -> None:
        """Simulate the neuromorphic circuit using the GPU backend.

        Parameters
        ----------
        time_steps : int, optional
        callback : _type_, optional
            WARNING: If using the GPU backend, this callback will
            not be able to modify the neuromorphic model via ``self`` .
        """
        # print("Using CUDA GPU via Numba")
        from .gpu import cuda as gpu
        if self.disable_performance_warnings:
            gpu.disable_numba_performance_warnings()
        # print("Using CPU with Numba JIT optimizations")
        max_timestep = max(list(self.input_spikes.keys()) + [time_steps]) + 1
        self._input_spikes = np.zeros((max_timestep, self.num_neurons), np.float64)
        for t, spikes_dict in self.input_spikes.items():
            for neuron_id, amplitude in zip(spikes_dict["nids"], spikes_dict["values"]):
                self._input_spikes[t][neuron_id] = amplitude

        self._output_spikes = np.zeros((time_steps, len(self._internal_states)), np.float64)

        # self.stdp_Apos = np.asarray(self._stdp_Apos, np.float64) if self.stdp_positive_update else np.zeros(1)
        # self.stdp_Aneg = np.asarray(self._stdp_Aneg, np.float64) if self.stdp_negative_update else np.zeros(1)

        post_synapse = cuda.to_device(np.zeros(self.num_neurons, np.float64))
        output_spikes = cuda.to_device(self._output_spikes[-1].astype(np.int8))
        states = cuda.to_device(self._internal_states)
        thresholds = cuda.to_device(self._neuron_thresholds)
        leaks = cuda.to_device(self._neuron_leaks)
        reset_states = cuda.to_device(self._neuron_reset_states)
        refractory_periods = cuda.to_device(self._neuron_refractory_periods)
        refractory_periods_original = cuda.to_device(self._neuron_refractory_periods_original)
        weights = cuda.to_device(self._weights)
        if self.stdp:
            stdp_enabled = cuda.to_device(self._stdp_enabled_synapses)

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

            self._output_spikes[tick] = output_spikes.copy_to_host()

            if self.stdp:
                for i in range(min(tick, self.stdp_time_steps)):
                    old_spikes = cuda.to_device(self._output_spikes[tick - i - 1].astype(np.int8))
                    gpu.stdp_update[m_blocks, m_tpb](weights, old_spikes, output_spikes,
                                    stdp_enabled, self._stdp_Apos[i], self._stdp_Aneg[i])

        self.spike_train.extend(self._output_spikes)
        self._weights = weights.copy_to_host()
        self._neuron_refractory_periods = refractory_periods.copy_to_host()
        self._internal_states = states.copy_to_host()

        # Update weights if STDP was enabled
        if self.stdp:
            self.synaptic_weights = list(self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])

    def _simulate_frontier(self, time_steps):
        """ Simulates the neuromorphic SNN on the Frontier supercomputer
        """

        # Define argument and return types
        self.c_frontier_library.argtypes = [ctypes.c_int]
        self.c_frontier_library.restype = ctypes.c_int

        # Call the C function
        data = 10
        result = self.c_frontier_library.simulate_frontier(data)

        print(f"[Python Source] Result from C: {result}")

    def print_spike_train(self):
        """Prints the spike train."""

        for time, spike_train in enumerate(self.spike_train):
            print(f"Time: {time}, Spikes: {spike_train}")
