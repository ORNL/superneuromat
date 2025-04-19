import numpy as np
import unittest

import sys 
sys.path.insert(0,"../src/")

from superneuromat import SNN


class DisplayTest(unittest.TestCase):
	""" Test display

	"""

	def test_display(self):
		snn = SNN()

		n0 = snn.create_neuron(threshold=-1.0, leak=2.0, refractory_period=3, reset_state=-2.0)
		n1 = snn.create_neuron(threshold=0.0, leak=1.0, refractory_period=1, reset_state=-2.0)
		n2 = snn.create_neuron(threshold=2.0, leak=0.0, refractory_period=0, reset_state=-1.0)
		n3 = snn.create_neuron(threshold=5.0, leak=np.inf, refractory_period=2, reset_state=-2.0)
		n4 = snn.create_neuron(threshold=-2.0, leak=5.0, refractory_period=1, reset_state=-2.0)

		snn.create_synapse(n0, n1)
		snn.create_synapse(n0, n2)
		snn.create_synapse(n0, n3, weight=4.0, delay=3, stdp_enabled=True)
		snn.create_synapse(n4, n2, weight=2.0, delay=2, stdp_enabled=False)
		snn.create_synapse(n2, n1, weight=30.0, delay=4, stdp_enabled=True)

		snn.add_spike(0, n2, 4.0)
		snn.add_spike(1, n1, 3.0)
		snn.add_spike(0, n3, 2.0)

		# snn.stdp_setup()
		snn.setup()
		snn.simulate(20)

		print(snn)

		snn.print_spike_train()

		assert (snn.spike_train[0][2] == 1)
		assert (snn.spike_train[1][1] == 1)
		assert (snn.spike_train[4][1] == 1)
		assert (snn.num_spikes == 6)

		print("test_display completed successfully")




if __name__ == "__main__":
	unittest.main()