import unittest
import numpy as np 

import sys 
sys.path.insert(0, "../src/")

from superneuromat import SNN


class DtypeTest(unittest.TestCase):
	""" Test if the datatypes are working properly

	"""

	def test_dtype_64(self):
		""" Test if data types are working properly for ping-pong SNN with double precision

		"""

		# Create SNN, neurons, and synapses
		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		snn.create_synapse(a, b, stdp_enabled=True)
		snn.create_synapse(b, a)


		# Add spikes
		snn.add_spike(0, a, 1)


		# Setup and simulate
		snn.stdp_setup(Aneg=[0.1, 0.05, 0.025])
		snn.setup()
		snn.simulate(10)


		# Print SNN after simulation
		print(snn)


		# Assertions
		assert (isinstance(snn._neuron_thresholds[0], np.float64))
		assert (isinstance(snn._neuron_leaks[0], np.float64))
		assert (isinstance(snn._neuron_reset_states[0], np.float64))
		assert (isinstance(snn._neuron_refractory_periods_original[0], np.float64))
		assert (isinstance(snn._neuron_refractory_periods[0], np.float64))
		assert (isinstance(snn._internal_states[0], np.float64))
		assert (isinstance(snn._spikes[0], np.int64))		
		assert (isinstance(snn._weights[0,0], np.float64))
		assert (isinstance(snn._stdp_enabled_synapses[0,0], np.float64))
		assert (isinstance(snn._input_spikes[0], np.float64))

		
		print("test_dtype64 completed successfully")




	def test_dtype_32(self):
		""" Test if data types are working properly for ping-pong SNN with single precision

		"""

		# Create SNN, neurons, and synapses
		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		snn.create_synapse(a, b, stdp_enabled=True)
		snn.create_synapse(b, a)


		# Add spikes
		snn.add_spike(0, a, 1)


		# Setup and simulate
		snn.stdp_setup(Aneg=[0.1, 0.05, 0.025])
		snn.setup(dtype=32)
		snn.simulate(10)


		# Print SNN after simulation
		print(snn)


		# Assertions
		assert (isinstance(snn._neuron_thresholds[0], np.float32))
		assert (isinstance(snn._neuron_leaks[0], np.float32))
		assert (isinstance(snn._neuron_reset_states[0], np.float32))
		assert (isinstance(snn._neuron_refractory_periods_original[0], np.float32))
		assert (isinstance(snn._neuron_refractory_periods[0], np.float32))
		assert (isinstance(snn._internal_states[0], np.float32))
		assert (isinstance(snn._spikes[0], np.int32))		
		assert (isinstance(snn._weights[0,0], np.float32))
		assert (isinstance(snn._stdp_enabled_synapses[0,0], np.float32))
		assert (isinstance(snn._input_spikes[0], np.float32))

		
		print("test_dtype32 completed successfully")




	def test_dtype_32_with_reset(self):
		""" Test if data types are working properly for ping-pong SNN with single precision with reset

		"""

		# Create SNN, neurons, and synapses
		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		snn.create_synapse(a, b, stdp_enabled=True)
		snn.create_synapse(b, a)


		# Add spikes
		snn.add_spike(0, a, 1)


		# Setup and simulate
		snn.stdp_setup(Aneg=[0.1, 0.05, 0.025])
		snn.setup(dtype=32)
		snn.simulate(10)


		# Reset
		snn.reset()


		# Print SNN after simulation
		print(snn)


		# Assertions
		assert (isinstance(snn._neuron_thresholds[0], np.float32))
		assert (isinstance(snn._neuron_leaks[0], np.float32))
		assert (isinstance(snn._neuron_reset_states[0], np.float32))
		assert (isinstance(snn._neuron_refractory_periods_original[0], np.float32))
		assert (isinstance(snn._neuron_refractory_periods[0], np.float32))
		assert (isinstance(snn._internal_states[0], np.float32))
		assert (isinstance(snn._spikes[0], np.float32))		
		assert (isinstance(snn._weights[0,0], np.float32))
		assert (isinstance(snn._stdp_enabled_synapses[0,0], np.float32))
		assert (isinstance(snn._input_spikes[0], np.float32))

		
		print("test_dtype32_with_reset completed successfully")




	def test_dtype_32_sparse(self):
		""" Test if data types are working properly for ping-pong SNN with single precision with sparse operations enabled

		"""

		# Create SNN, neurons, and synapses
		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		snn.create_synapse(a, b, stdp_enabled=True)
		snn.create_synapse(b, a)


		# Add spikes
		snn.add_spike(0, a, 1)


		# Setup and simulate
		snn.stdp_setup(Aneg=[0.1, 0.05, 0.025])
		snn.setup(dtype=32, sparse=True)
		snn.simulate(10)


		# Print SNN after simulation
		print(snn)


		# Assertions
		assert (isinstance(snn._neuron_thresholds[0], np.float32))
		assert (isinstance(snn._neuron_leaks[0], np.float32))
		assert (isinstance(snn._neuron_reset_states[0], np.float32))
		assert (isinstance(snn._neuron_refractory_periods_original[0], np.float32))
		assert (isinstance(snn._neuron_refractory_periods[0], np.float32))
		assert (isinstance(snn._internal_states[0], np.float32))
		assert (isinstance(snn._spikes[0], np.int32))		
		assert (isinstance(snn._weights[0,0], np.float32))
		assert (isinstance(snn._stdp_enabled_synapses[0,0], np.float32))
		assert (isinstance(snn._input_spikes[0], np.float32))

		
		print("test_dtype32_sparse completed successfully")





if __name__ == "__main__":
	unittest.main()