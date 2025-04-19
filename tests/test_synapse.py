import unittest
import numpy as np 

import sys 
sys.path.insert(0, "../src/")

from superneuromat import SNN


class SynapseTest(unittest.TestCase):
	""" Test if the create_synapse functionality is working properly

	"""

	def test_multiple_synapses(self):
		""" Test if multiple synapses from a neuron to another neuron are possible.
			This test shoud throw an error as multiple synapses between the same 2 neurons are not allowed.

		"""

		# Create SNN, neurons, and synapses
		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		snn.create_synapse(a, b, delay=1, stdp_enabled=True)

		try:
			snn.create_synapse(a, b, delay=2)

		except RuntimeError:
			print("test_multiple_synapses completed successfully")




if __name__ == "__main__":
	unittest.main()