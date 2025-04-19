import unittest
import numpy as np 

import sys 
sys.path.insert(0, "../src/")

from superneuromat import SNN


class ResetTest(unittest.TestCase):
	""" Test the reset function

	"""

	def test_reset_1(self):
		""" Test reset function for ping-pong SNN

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


		# Print SNN before reset
		print("Before reset:")
		print(snn)


		# Reset 
		snn.reset()

		
		# Print SNN after reset
		print("After reset:")
		print(snn)


		# Assertions
		assert (np.array_equal(snn._internal_states, snn.neuron_reset_states))
		assert (not np.any(snn._refractory_periods))
		assert (not np.any(snn._spikes))
		assert (snn._weights[0,1] == 1.0) and (snn._weights[1,0] == 1.0)
		assert (not snn.spike_train)
		assert (snn.count_spikes() == 0)
		assert (not np.any(snn._input_spikes))

		print("test_reset_1 completed successfully")





if __name__ == "__main__":
	unittest.main()