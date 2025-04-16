import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import NeuromorphicModel


class CountSpikeTest(unittest.TestCase):
	""" Test the count_spike function

	"""

	def test_count_spike(self):
		""" Test the count spike function for a ping-pong SNN

		"""

		snn = NeuromorphicModel()

		a = snn.create_neuron()
		b = snn.create_neuron()

		snn.create_synapse(a, b)
		snn.create_synapse(b, a)

		snn.add_spike(0, a, 1)

		snn.setup()

		snn.simulate(10)

		assert (snn.count_spikes() == 10)

		print(snn)

		# print("Expected spike count: 10")
		# print(f"Received spike count: {snn.count_spikes()}")
		# print("test_count_spike completed successfully")





if __name__ == "__main__":
	unittest.main()