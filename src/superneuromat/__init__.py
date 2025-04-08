import numpy as np
import pandas as pd
# import ctypes
import os
# import multiprocessing as mp
import subprocess
# import mmap


"""

TODO:

1. create_neurons() 
2. create_synapses()
3. create_neurons_from_file()
4. create_synapses_from_file() 
5. Tutorials: one neuron, two neurons 

"""



"""

FEATURE REQUESTS: 

1. Visualize spike raster
2. Monitor STDP synapses
3. Reset neuromorphic model

"""




class NeuromorphicModel:
	""" Defines a neuromorphic model with neurons and synapses

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


	def __init__(self, backend="cpu", num_mpi_ranks=1, dtype=32):
		""" Initialize the neuromorphic model

		Args:
			backend (string): Backend is either 'cpu' or 'frontier'

		Raises: 
			TypeError if:
				1. backend is not a string

			ValueError if: 
				2. backend is not one of the following values: cpu, frontier

		"""

		# Type errors
		if not isinstance(backend, str):
			raise TypeError("backend must be either 'cpu', or 'frontier'")

		# Value errors
		if backend not in {"cpu", "frontier"}:
			raise ValueError("backend must be either 'cpu', or 'frontier'")

		if dtype not in {32, 64}:
			raise ValueError("dtype must be 32 or 64")


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
		self.stdp_time_steps = 0
		self.stdp_Apos = []
		self.stdp_Aneg = []
		self.stdp_positive_update = False
		self.stdp_negative_update = False

		# Hardware backend parameters
		self.backend = backend
		self.num_mpi_ranks = num_mpi_ranks
		self.fifo_python = None 
		self.fifo_c = None
		self.current_directory = None

		if backend == "frontier":
			self._setup_frontier_communication()

		if dtype == 64:
			self.dtype_int = np.int64
			self.dtype_float = np.float64
		else:
			self.dtype_int = np.int32
			self.dtype_float = np.float32

		# self.shared_memory_name = None

		# Simulation parameters
		self.num_spikes = 0




	def __repr__(self):
		""" Display the neuromorphic class in a legible format

		"""

		num_neurons_str = f"Number of neurons: {self.num_neurons}"
		num_synapses_str = f"Number of synapses: {self.num_synapses}"
		
		# Neurons 
		neuron_df = pd.DataFrame({	"Neuron ID": list(range(self.num_neurons)),
									"Threshold": self.neuron_thresholds, 
									"Leak": self.neuron_leaks, 
									"Reset State": self.neuron_reset_states,
									"Refractory Period": self.neuron_refractory_periods
								})

		# Synapses
		synapse_df = pd.DataFrame({	"Synapse ID": list(range(self.num_synapses)),
									"Pre Neuron ID": self.pre_synaptic_neuron_ids,
									"Post Neuron ID": self.post_synaptic_neuron_ids,
									"Weight": self.synaptic_weights,
									"Delay": self.synaptic_delays,
									"STDP Enabled": self.enable_stdp
								 })

		# STDP
		stdp_info = f"STDP Enabled: {self.stdp} \n" + \
					f"STDP Time Steps: {self.stdp_time_steps} \n" + \
					f"STDP A positive: {self.stdp_Apos} \n" + \
					f"STDP A negative: {self.stdp_Aneg}"

		# Input Spikes
		times = []
		nids = []
		values = []

		for time in self.input_spikes:
			for nid, value in zip(self.input_spikes[time]["nids"], self.input_spikes[time]["values"]):
				times.append(time)
				nids.append(nid)
				values.append(value)

		input_spikes_df = pd.DataFrame(	{ "Time": times,
										  "Neuron ID": nids,
										  "Value": values
										})

		# Spike train
		spike_train = ""
		for time, spikes in enumerate(self.spike_train):
			spike_train += f"Time: {time}, Spikes: {spikes}\n"
				

		return 	num_neurons_str + "\n" + \
				num_synapses_str + "\n" \
				"\nNeuron Info: \n" + \
				neuron_df.to_string(index=False) + "\n" + \
				"\nSynapse Info: \n" + \
				synapse_df.to_string(index=False) + "\n" + \
				"\nSTDP Info: \n" + \
				stdp_info + "\n" + \
				"\nInput Spikes: \n" + \
				input_spikes_df.to_string(index=False) + "\n" + \
				"\nSpike Train: \n" + \
				spike_train + \
				f"\nNumber of spikes: {self.num_spikes}\n"




	def create_neuron(	
		self, 
		threshold: float=0.0, 
		leak: float=np.inf, 
		reset_state: float=0.0, 
		refractory_period: int=0
	) -> int:

		""" Create a neuron

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
		if not isinstance(threshold, (int, float)) :
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
		delay: int=1, 
		stdp_enabled: bool=False
	) -> None:

		""" Creates a synapse in the neuromorphic model from a pre-synaptic neuron to a post-synaptic neuron with a given set of synaptic parameters (weight, delay and enable_stdp)

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

		# Type errors
		if not isinstance(pre_id, int):
			raise TypeError("pre_id must be int")

		if not isinstance(post_id, int):
			raise TypeError("post_id must be int")

		if not isinstance(weight, (int, float)):
			raise TypeError("weight must be a float")

		if not isinstance(delay, int):
			raise TypeError("delay must be an integer")

		if not isinstance(stdp_enabled, bool):
			raise TypeError("enable_stdp must be a bool")
		

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
			self.enable_stdp.append(stdp_enabled)
			self.num_synapses += 1

		else:
			for d in range(int(delay) - 1):
				temp_id = self.create_neuron()
				self.create_synapse(pre_id, temp_id)
				pre_id = temp_id

			self.create_synapse(pre_id, post_id, weight=weight, stdp_enabled=stdp_enabled)


		# Return synapse ID
		return self.num_synapses - 1




	def add_spike(	
		self, 
		time: int, 
		neuron_id: int, 
		value: float=1.0
	) -> None:
		
		""" Adds an external spike in the neuromorphic model 

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




	def stdp_setup(	
		self, 
		time_steps: int=3, 
		Apos: list=[1.0, 0.5, 0.25], 
		Aneg: list=[1.0, 0.5, 0.25], 
		positive_update: bool=True,
		negative_update: bool=True
	) -> None:

		""" Choose the appropriate STDP setup function based on backend

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

		if not isinstance(Apos, list):
			raise TypeError("Apos should be a list")

		if not isinstance(Aneg, list):
			raise TypeError("Aneg should be a list")

		if not isinstance(positive_update, bool):
			raise TypeError("positive_update must be a bool")

		if not isinstance(negative_update, bool):
			raise TypeError("negative_update must be a bool")


		# Value error
		if time_steps <= 0:
			raise ValueError("time_steps should be greater than zero")

		if positive_update and len(Apos) != time_steps:
			raise ValueError(f"Length of Apos should be {time_steps}")

		if negative_update and len(Aneg) != time_steps:
			raise ValueError(f"Length of Aneg should be {time_steps}")

		if positive_update and not all([isinstance(x, (int, float)) for x in Apos]):
			raise ValueError("All elements in Apos should be int or float")

		if negative_update and not all([isinstance(x, (int, float)) for x in Aneg]):
			raise ValueError("All elements in Aneg should be int or float")

		if positive_update and not all([x >= 0.0 for x in Apos]):
			raise ValueError("All elements in Apos should be positive")

		if negative_update and not all([x >= 0.0 for x in Aneg]):
			raise ValueError("All elements in Aneg should be positive")		

		
		# Runtime error
		if not any(self.enable_stdp):
			raise RuntimeError("STDP is not enabled on any synapse, might want to skip stdp_setup()")


		# Set STDP flag
		self.stdp = True 


		# Choose the appropriate STDP setup function based on backend
		self._stdp_setup_cpu(time_steps, Apos, Aneg, positive_update, negative_update)

		if self.backend == "frontier":
			self._stdp_setup_frontier()




	def setup(self):
		""" Choose the appropriate setup function based on backend

		"""

		if self.backend == "cpu":
			self._setup_cpu()

		elif self.backend == "frontier":
			self._setup_frontier()

		else:
			raise RuntimeError(f"Backend {self.backend} not supported currently, only backends supported are 'cpu' and 'frontier'")




	def simulate(self, time_steps: int=1000) -> None:
		""" Simulate the neuromorphic spiking neural network 

		Args:
			time_steps (int): Number of time steps for which the neuromorphic circuit is to be simulated
			backend (string): Backend is either cpu or frontier

		Raises: 
			TypeError if:
				1. time_steps is not an int
				2. backend is not a string

			ValueError if: 
				1. time_steps is less than or equal to zero
				2. backend is not one of the following values: cpu, frontier

		""" 

		# Type errors
		if not isinstance(time_steps, int):
			raise TypeError("time_steps must be int")


		# Value errors
		if time_steps <= 0:
			raise ValueError("time_steps must be greater than zero")


		# Select appropriate simulation function
		if self.backend == "cpu":
			self._simulate_cpu(time_steps=time_steps)

		elif self.backend == "frontier":
			self._simulate_frontier(time_steps=time_steps)

		else:
			raise RuntimeError("Unsupported backend:", backend)




	def print_spike_train(self):
		""" Prints the spike train

		"""

		for time, spike_train in enumerate(self.spike_train):
			print(f"Time: {time}, Spikes: {spike_train}")

		print(f"\nNumber of spikes: {self.num_spikes}\n")




	def _stdp_setup_cpu(	
		self, 
		time_steps: int=3, 
		Apos: list=[1.0, 0.5, 0.25], 
		Aneg: list=[1.0, 0.5, 0.25], 
		positive_update: bool=True,
		negative_update: bool=True
	) -> None:

		""" Setup the Spike-Time-Dependent Plasticity (STDP) parameters on CPU backend

		Args:
			time_steps (int): Number of time steps over which STDP learning occurs (default: 3)
			Apos (list): List of parameters for excitatory STDP updates (default: [1.0, 0.5, 0.25]); number of elements in the list must be equal to time_steps
			Aneg (list): List of parameters for inhibitory STDP updates (default: [1.0, 0.5, 0.25]); number of elements in the list must be equal to time_steps
			positive_update (bool): Boolean parameter indicating whether excitatory STDP update should be enabled
			negative_update (bool): Boolean parameter indicating whether inhibitory STDP update should be enabled

		"""

		# Collect STDP parameters
		self.stdp_time_steps = time_steps
		self.stdp_Apos = Apos 
		self.stdp_Aneg = Aneg 
		self.stdp_positive_update = positive_update
		self.stdp_negative_update = negative_update



	def _setup_cpu(self):
		""" Setup the neuromorphic model for simulation on CPU backend

		"""

		# Create numpy arrays for neuron state variables
		self._neuron_thresholds = np.array(self.neuron_thresholds)
		self._neuron_leaks = np.array(self.neuron_leaks)
		self._neuron_reset_states = np.array(self.neuron_reset_states)
		self._neuron_refractory_periods_original = np.array(self.neuron_refractory_periods)
		self._neuron_refractory_periods = np.zeros(self.num_neurons)
		self._internal_states = np.array(self.neuron_reset_states)
		self._spikes = np.zeros(self.num_neurons)
		
		# Create numpy arrays for synapse state variables
		self._weights = np.zeros((self.num_neurons, self.num_neurons))
		self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.synaptic_weights

		# Create numpy arrays for STDP state variables
		if self.stdp:
			self._stdp_enabled_synapses = np.zeros((self.num_neurons, self.num_neurons))
			self._stdp_enabled_synapses[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.enable_stdp

		# Create numpy array for input spikes state variable
		self._input_spikes = np.zeros(self.num_neurons)




	def _simulate_cpu(self, time_steps: int=1000):
		""" Simulates the neuromorphic SNN on CPU backend

		Args:
			time_steps (int): Number of time steps for which the neuromorphic circuit is to be simulated
			backend (string): Backend is either cpu or frontier

		"""

		# Simulate
		for time_step in range(time_steps):
			# Leak: internal state > reset state
			indices = (self._internal_states > self._neuron_reset_states)
			self._internal_states[indices] = np.maximum(self._internal_states[indices] - self._neuron_leaks[indices], self._neuron_reset_states[indices])

			
			# Leak: internal state < reset state
			indices = (self._internal_states < self._neuron_reset_states)
			self._internal_states[indices] = np.minimum(self._internal_states[indices] + self._neuron_leaks[indices], self._neuron_reset_states[indices])


			# Zero out _input_spikes to prepare them for input spikes in current time step
			self._input_spikes -= self._input_spikes 	


			# Include input spikes for current time step
			if time_step in self.input_spikes:
				self._input_spikes[self.input_spikes[time_step]["nids"]] = self.input_spikes[time_step]["values"] 


			# Internal state
			self._internal_states = self._internal_states + self._input_spikes + (self._weights.T @ self._spikes)


			# Compute spikes
			self._spikes = np.greater(self._internal_states, self._neuron_thresholds).astype(int)
			

			# Refractory period: Compute indices of neuron which are in their refractory period
			indices = np.greater(self._neuron_refractory_periods, 0)
			

			# For neurons in their refractory period, zero out their spikes and decrement refractory period by one
			self._spikes[indices] = 0
			self._neuron_refractory_periods[indices] -= 1


			# For spiking neurons, turn on refractory period
			self._neuron_refractory_periods[self._spikes.astype(bool)] = self._neuron_refractory_periods_original[self._spikes.astype(bool)]

			
			# Reset internal states
			self._internal_states[self._spikes == 1.0] = self._neuron_reset_states[self._spikes == 1.0]


			# Update spike count
			self.num_spikes += sum(self._spikes)


			# Append spike train
			self.spike_train.append(self._spikes)


			# STDP Operations
			t = min(self.stdp_time_steps, len(self.spike_train)-1)

			if t > 0:
				_update_synapses = np.outer(np.array(self.spike_train[-t-1:-1]), np.array(self.spike_train[-1])).reshape([-1, self.num_neurons, self.num_neurons])

				if self.stdp_positive_update:
					self._weights += ((_update_synapses.T * self.stdp_Apos[0:t][::-1]).T).sum(axis=0) * self._stdp_enabled_synapses

				if self.stdp_negative_update:
					self._weights += (((1 - _update_synapses).T * self.stdp_Aneg[0:t][::-1]).T).sum(axis=0) * self._stdp_enabled_synapses


		# Update weights if STDP was enabled
		if self.stdp:
			self.synaptic_weights = list(self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])



	def _setup_frontier_communication(self):
		""" Setup the communication 
		"""

		# Current directory
		self.current_directory = os.path.dirname(os.path.abspath(__file__))

		
		# Define the FIFO path
		self.fifo_python = self.current_directory + "/fifo_python"
		self.fifo_c = self.current_directory + "/fifo_c"

		
		# Compile and run C program
		compile_command = ["mpicc", "-Wall", self.current_directory + "/frontier.c", "-o", self.current_directory + "/frontier.o"]
		run_command = ["mpiexec", "-np", str(self.num_mpi_ranks), self.current_directory + "/frontier.o", self.fifo_python, self.fifo_c]

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





	

			



