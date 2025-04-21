import os
import subprocess
import pathlib as pl

import numpy as np

from ..neuromorphicmodel import NeuromorphicModel as BaseModel


curpath = pl.Path(__file__).resolve().parent


class FrontierModel(BaseModel):
    """Defines a neuromorphic model with neurons and synapses for use on Frontier"""

    def __init__(self, backend='auto', num_mpi_ranks=1, initialize=True):
        super().__init__()

        self.num_mpi_ranks = num_mpi_ranks
        self.fifo_python = None
        self.fifo_c = None
        self.current_directory = None

        self.set_word_size(64)

        if initialize:
            self._setup_frontier_communication()

    def set_word_size(self, word_size):
        if word_size == 64:
            self.dtype_int = np.int64
            self.dtype_float = np.float64
        else:
            self.dtype_int = np.int32
            self.dtype_float = np.float32

    def _setup_frontier_communication(self):
        """ Setup the communication
        """

        # i'd highly recommend using python build tools to do this instead of trying to
        # resolve build issues at runtime like this --kz apr'25

        # Current directory
        self.current_directory = curpath
        cwd = curpath

        # Define the FIFO path
        self.fifo_python = self.current_directory + "/fifo_python"
        self.fifo_c = self.current_directory + "/fifo_c"

        # Compile and run C program
        compile_command = ["mpicc", "-Wall", str(cwd / "frontier.c"), "-o", str(cwd / "frontier.o")]
        run_command = ["mpiexec", "-np", str(self.num_mpi_ranks), str(cwd / "frontier.o"), self.fifo_python, self.fifo_c]

        subprocess.run(compile_command)
        print("[Python _setup_frontier_communication] frontier.c compiled")

        subprocess.Popen(run_command)
        print("[Python _setup_frontier_communication] frontier.o execution started")
        print()

    def _stdp_setup_frontier(self):

        """ Setup the Spike-Time-Dependent Plasticity (STDP) parameters for Frontier backend

        """

        print("[Python _stdp_setup_frontier] Entered the _stdp_setup_frontier function")

        if not os.path.exists(self.fifo_python):
            os.mkfifo(self.fifo_python)

        if not os.path.exists(self.fifo_c):
            os.mkfifo(self.fifo_c)

        print("[Python _stdp_setup_frontier] fifo_python and fifo_c created")

        # Send data over the pipe
        with open(self.fifo_python, "wb") as fp:
            fp.write(self.dtype_int(self.stdp_time_steps))
            print("[Python _stdp_setup_frontier] Wrote stdp_time_steps to fifo_python")
            fp.write(np.array(self.stdp_Apos).astype(self.dtype_float))
            print("[Python _stdp_setup_frontier] Wrote stdp_Apos to fifo_python")
            fp.write(np.array(self.stdp_Aneg).astype(self.dtype_float))
            print("[Python _stdp_setup_frontier] Wrote stdp_Aneg to fifo_python")
            fp.write(self.dtype_int(self.stdp_positive_update))
            fp.write(self.dtype_int(self.stdp_negative_update))

            # with open(self.fifo_c, "rb") as fc:
            # 	data = fc.read()

        print("[Python _stdp_setup_frontier] STDP data sent")

        # Initialize the data
        # first_data = 1.0
        # first_data = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Setup the C shared library
        # c_executable_path = os.path.dirname(os.path.abspath(__file__)) + "/frontier.o"
        # lib = ctypes.CDLL(lib_path)

        # Setup the ctypes datatypes
        # c_float_array_type = ctypes.c_float * time_steps

        # Setup the data
        # time_steps = ctypes.c_int(time_steps)
        # Apos = c_float_array_type(*Apos)
        # Aneg = c_float_array_type(*Aneg)
        # positive_update = ctypes.c_int(positive_update)
        # negative_update = ctypes.c_int(negative_update)

        # Setup the C function
        # c_stdp_setup_frontier = c_lib.stdp_setup_frontier
        # c_stdp_setup_frontier.argtypes = [ctypes.c_int, c_float_array_type, c_float_array_type, ctypes.c_bool, ctypes.c_bool]
        # c_stdp_setup_frontier.restype = ctypes.c_int

        # lib.stdp_setup_frontier.argtypes = [ctypes.c_int, c_float_array_type, c_float_array_type, ctypes.c_bool, ctypes.c_bool]
        # lib.stdp_setup_frontier.restype = ctypes.c_int

        # Call the function
        # val = lib.stdp_setup_frontier(time_steps, Apos, Aneg, positive_update, negative_update)

        # Call the function
        # c_lib.stdp_setup_frontier(ctypes.c_float_p(first_data))

        # print(f"[Python _stdp_setup_frontier] Function returned with value {val}")

        # Shared memory will contain:
        # 5 neuron properties: number of neurons, thresholds, leaks, reset states, and refractory periods,
        # 4 synapse properties: weights, delays, and stdp enabled, and
        # 5 STDP properties: STDP time steps, Apos, Aneg, positive update, and negative update
        # shared_memory_size = 0
        # shared_memory_size += size(np.int32) + size(self.dtype) * self.num_neurons * 4
        # shared_memory_size += size(np.int32) + size(self.dtype) * self.num_synapses * 3
        # shared_memory_size += size(np.int32) * 3 + size(self.dtype) * self.stdp_time_steps * 2

        # process = subprocess.Popen(current_directory + "reader", stdin=subprocess.PIPE, text=True)
        # process.communicate()

    def _setup_frontier(self):
        """ Setup the neuromorphic model for simulation on Frontier backend

        """

        print("[Python _setup_frontier] Entered the _setup_frontier function")

        # Make fifos
        if not os.path.exists(self.fifo_python):
            os.mkfifo(self.fifo_python)

        # if not os.path.exists(self.fifo_c):
        # 	os.mkfifo(self.fifo_c)

        print("[Python _setup_frontier] fifo_python created")

        # Send neuron and synapse data over the pipe
        with open(self.fifo_python, "wb") as fp:
            # Neuron parameters: num_neurons, neuron_thresholds, neuron_leaks, neuron_reset_states, and neuron_refractory_periods
            fp.write(self.dtype_int(self.num_neurons))
            fp.write(np.array(self.neuron_thresholds).astype(self.dtype_float))
            fp.write(np.array(self.neuron_leaks).astype(self.dtype_float))
            fp.write(np.array(self.neuron_reset_states).astype(self.dtype_float))
            fp.write(np.array(self.neuron_refractory_periods).astype(self.dtype_int))

            # Synapse parameters: num_synapses, pre_synaptic_neuron_ids, post_synaptic_neuron_ids, synaptic_weights, stdp_enabled
            fp.write(self.dtype_int(self.num_synapses))
            fp.write(np.array(self.pre_synaptic_neuron_ids).astype(self.dtype_int))
            fp.write(np.array(self.post_synaptic_neuron_ids).astype(self.dtype_int))
            fp.write(np.array(self.synaptic_weights).astype(self.dtype_float))
            fp.write(np.array(self.enable_stdp).astype(self.dtype_int))

        print("[Python _setup_frontier] Neuron and synapse data sent")

        # with open(self.fifo_c, "rb") as fc:
        # 	data = fc.readline()
        # 	print(f"Data received from C: {str(data)}")

        # Initialize the data
        # second_data = 5.0
        # second_data = [6.0, 7.0, 8.0, 9.0, 10.0]

        # # Setup the C shared library
        # c_lib_path = os.path.dirname(os.path.abspath(__file__)) + "/frontier.so"
        # c_lib = ctypes.CDLL(c_lib_path)

        # # Setup the function
        # c_setup_frontier = c_lib.setup_frontier
        # c_setup_frontier.argtypes = [ctypes.c_float]
        # c_setup_frontier.restype = ctypes.c_int

        # Setup the C shared library
        # lib_path = os.path.dirname(os.path.abspath(__file__)) + "/frontier.o"
        # lib = ctypes.CDLL(lib_path)

        # # Setup the ctypes datatypes and data
        # c_float_array_type = ctypes.c_float * len(second_data)

        # # Setup the C function
        # c_setup_frontier = lib.setup_frontier
        # c_setup_frontier.argtypes = [ctypes.c_int, c_float_array_type]
        # c_setup_frontier.restype = ctypes.c_int

        # Call the functions
        # lib.initialize_mpi();
        # val = c_setup_frontier(len(second_data), c_float_array_type(*second_data))

        # num_processes = 4

        # print(f"[Python _setup_frontier] Function returned with value {result}")

    def _simulate_frontier(self, time_steps):
        """ Simulates the neuromorphic SNN on the Frontier supercomputer

        Args:
            time_steps (int): Number of time steps for which the neuromorphic circuit is to be simulated
            backend (string): Backend is either cpu or frontier

        """

        print("[Python _simulate_frontier] Entered the _simulate_frontier function")

        # Setup the C shared library
        # c_lib_path = os.path.dirname(os.path.abspath(__file__)) + "/frontier.so"
        # c_lib = ctypes.CDLL(c_lib_path)

        # Setup the ctypes datatypes and data
        # c_float_pointer_type = ctypes.POINTER(ctypes.c_float)

        # Setup the function
        # c_simulate_frontier = c_lib.simulate_frontier
        # c_simulate_frontier.argtypes = []
        # c_simulate_frontier.restype = c_float_pointer_type

        # Call the function
        # val = c_simulate_frontier()
        # val = np.frombuffer(val, dtype=np.float32)

        # if val:
        # 	val = [val[i] for i in range(5)]
        # 	print(f"[Python _simulate_frontier] Function returned {val}")
        # else:
        # 	print("Function did not return anything good")
