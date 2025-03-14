import unittest
import numpy as np

import sys 
sys.path.insert(0,"../")

from superneuromat.neuromorphicmodel import NeuromorphicModel



class RefractoryTest(unittest.TestCase):
	""" Test refractory period

	"""

	def test_refractory_one(self):
		print("One neuron refractory period test")

		model = NeuromorphicModel()

		n_id = model.create_neuron(refractory_period=2)

		model.add_spike(1, n_id, 1)
		model.add_spike(2, n_id, 3)
		model.add_spike(3, n_id, 4)
		model.add_spike(4, n_id, 1)


		model.setup()
		model.simulate(10)

		model.print_spike_train()
		print()


		""" Expected Output:
		Time: 0, Spikes: [0]
		Time: 1, Spikes: [1]
		Time: 2, Spikes: [0]
		Time: 3, Spikes: [0]
		Time: 4, Spikes: [1]
		Time: 5, Spikes: [0]
		Time: 6, Spikes: [0]
		Time: 7, Spikes: [0]
		Time: 8, Spikes: [0]
		Time: 9, Spikes: [0]
		"""




	def test_refractory_two(self):
		print("Two neuron refractory period test")

		model = NeuromorphicModel()

		n1 = model.create_neuron(threshold=-1.0, reset_state=-1.0, refractory_period=2)
		n2 = model.create_neuron(refractory_period=1000000)

		model.create_synapse(n1, n2, weight=2.0, delay=2)

		model.add_spike(1, n2, -1.0)
		model.add_spike(2, n1, 10.0)
		model.add_spike(3, n1, 10.0)
		model.add_spike(5, n1, 10.0)

		model.setup()
		model.simulate(10)

		model.print_spike_train()


		""" Expected Output: 
		Time: 0, Spikes: [0 0 0]
		Time: 1, Spikes: [0 0 0]
		Time: 2, Spikes: [1 0 0]
		Time: 3, Spikes: [0 0 1]
		Time: 4, Spikes: [0 1 0]
		Time: 5, Spikes: [1 0 0]
		Time: 6, Spikes: [0 0 1]
		Time: 7, Spikes: [0 0 0]
		Time: 8, Spikes: [0 0 0]
		Time: 9, Spikes: [0 0 0]
		"""


	


if __name__ == "__main__":
	unittest.main()





# model = NeuromorphicModel()

# n_id = model.create_neuron(refractory_period=2)

# model.add_spike(1, n_id, 1)
# model.add_spike(2, n_id, 3)
# model.add_spike(3, n_id, 4)


# model.setup()
# model.simulate(5)

# for spike_train in model.spike_train:
# 	print(spike_train)