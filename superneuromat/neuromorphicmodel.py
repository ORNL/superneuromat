import numpy as np
import pandas as pd


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
13. Docstrings everywhere
14. create_neurons()
15. create_synapses()
16. Tutorials: one neuron, two neurons
17. Leak equations should not check for spikes [DONE]

"""




"""

FEATURE REQS: 

1. Visualize spike raster
2. Monitor STDP synapses
3. Reset neuromorphic model

"""




"""

CAUTION:

1. Delay is implemented by adding a chain of proxy neurons. A delay of 10 between neuron A and neuron B would add 9 proxy neurons between A and B.
2. The notion of leak: Leak tries to bring the internal state of the neuron back to the reset state. The leak value is the amount by which the internal state of the neuron is pushed towards its reset state.
3. Deletion of neurons is not permitted
4. Careful with STDP parameters.
5. Input spikes can have a value
6. Monitoring all neurons by default

"""




class NeuromorphicModel:
	"""
	"""

	def __init__(self):
		"""
		"""

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
		synapse_df = pd.DataFrame({	"Pre Neuron ID": self.pre_synaptic_neuron_ids,
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
				spike_train




	def create_neuron(	
		self, 
		threshold: float=0.0, 
		leak: float=0.0, 
		reset_state: float=0.0, 
		refractory_period: int=0
	) -> int:

		"""
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
		enable_stdp: bool=False
	) -> None:

		""" 
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

		if not isinstance(enable_stdp, bool):
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
			self.enable_stdp.append(enable_stdp)
			self.num_synapses += 1

		else:
			for d in range(int(delay) - 1):
				temp_id = self.create_neuron()
				self.create_synapse(pre_id, temp_id)
				pre_id = temp_id

			self.create_synapse(pre_id, post_id, weight=weight, enable_stdp=enable_stdp)




	def add_spike(	
		self, 
		time: int, 
		neuron_id: int, 
		value: float=1.0
	) -> None:
		
		"""
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

		"""
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
			raise RuntimeError("STDP is not enabled on any synapse")


		# Collect STDP parameters
		self.stdp = True 
		self.stdp_time_steps = time_steps
		self.stdp_Apos = Apos 
		self.stdp_Aneg = Aneg 
		self.stdp_negative_update = negative_update




	def setup(self):
		"""
		"""

		# Create numpy arrays for neuron state variables
		self._neuron_thresholds = np.array(self.neuron_thresholds)
		self._neuron_leaks = np.array(self.neuron_leaks)
		self._neuron_reset_states = np.array(self.neuron_reset_states)
		self._neuron_refractory_periods_original = np.array(self.neuron_refractory_periods)
		self._neuron_refractory_periods = np.zeros(self.num_neurons)
		self._internal_states = np.array(self.neuron_reset_states)
		self._spikes = np.zeros(self.num_neurons)
		
		# Create numpy arrays for synapses
		self._weights = np.zeros((self.num_neurons, self.num_neurons))
		self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.synaptic_weights

		# Create numpy arrays for STDP operations
		if self.stdp:
			self._stdp_enabled_synapses = np.zeros((self.num_neurons, self.num_neurons))
			self._stdp_enabled_synapses[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.enable_stdp

		# Create numpy array for input spikes
		self._input_spikes = np.zeros(self.num_neurons)




	def simulate(self, time_steps: int=1000) -> None:
		""" 
		""" 

		# Type errors
		if not isinstance(time_steps, int):
			raise TypeError("time_steps must be int")


		# Value errors
		if time_steps <= 0:
			raise ValueError("time_steps must be greater than zero")


		# Simulate
		for tick in range(time_steps):
			# Zero out _input_spikes
			self._input_spikes -= self._input_spikes 	


			# Include input spikes for current tick
			if tick in self.input_spikes:
				self._input_spikes[self.input_spikes[tick]["nids"]] = self.input_spikes[tick]["values"] 


			# Internal state
			self._internal_states = self._internal_states + self._input_spikes + (self._weights.T @ self._spikes)


			# Leak: internal state > reset state
			indices = (self._internal_states > self._neuron_reset_states)
			self._internal_states[indices] = np.maximum(self._internal_states[indices] - self._neuron_leaks[indices], self._neuron_reset_states[indices])

			
			# Leak: internal state < reset state
			indices = (self._internal_states < self._neuron_reset_states)
			self._internal_states[indices] = np.minimum(self._internal_states[indices] + self._neuron_leaks[indices], self._neuron_reset_states[indices])


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


			# Append spike train
			self.spike_train.append(self._spikes)


			# STDP Operations
			if self.stdp:
				for i in range(self.stdp_time_steps):
					if len(self.spike_train) >= i + 2:
						update_synapses = np.outer(np.array(self.spike_train[-i-2]), np.array(self.spike_train[-1]))
						self._weights += self.stdp_Apos[i] * update_synapses * self._stdp_enabled_synapses

						if self.stdp_negative_update:
							self._weights -= self.stdp_Aneg[i] * (1 - update_synapses) * self._stdp_enabled_synapses




	def print_spike_train(self):
		"""
		"""

		for time, spike_train in enumerate(self.spike_train):
			print(f"Time: {time}, Spikes: {spike_train}")

			



