import unittest
import numpy as np

import sys 
sys.path.insert(0,"../src/")

from superneuromat import SNN



class RefractoryTest(unittest.TestCase):
	""" Test refractory period

	"""

	def test_refractory_one(self):
		""" Test refractory period for 1 neuron

		"""

		print("One neuron refractory period test")

		snn = SNN()

		n = snn.create_neuron(refractory_period=2)

		snn.add_spike(1, n, 1)
		snn.add_spike(2, n, 3)
		snn.add_spike(3, n, 4)
		snn.add_spike(4, n, 1)


		snn.setup()
		snn.simulate(10)

		snn.print_spike_train()

		assert (np.array_equal(np.array(snn.spike_train), np.array([[0],[1],[0],[0],[1],[0],[0],[0],[0],[0]])))

		print("test_refractory_one completed successfully")




	def test_refractory_two(self):
		""" Test refractory period for 2 neurons
		
		"""

		print("Two neuron refractory period test")

		snn = SNN()

		n0 = snn.create_neuron(threshold=-1.0, reset_state=-1.0, refractory_period=2)
		n1 = snn.create_neuron(refractory_period=1000000)

		snn.create_synapse(n0, n1, weight=2.0, delay=2)

		snn.add_spike(1, n1, -1.0)
		snn.add_spike(2, n0, 10.0)
		snn.add_spike(3, n0, 10.0)
		snn.add_spike(5, n0, 10.0)

		snn.setup()
		snn.simulate(10)

		snn.print_spike_train()

		assert (np.array_equal(np.array(snn.spike_train), np.array([[0,0,0],[0,0,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,0,1],[0,0,0],[0,0,0],[0,0,0]])))

		print("test_refractory_one completed successfully")


	


if __name__ == "__main__":
	unittest.main()

