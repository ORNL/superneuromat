import numpy as np
import unittest
from superneuromat.neuromorphicmodel import NeuromorphicModel


class DisplayTest(unittest.TestCase):
	""" Test display

	"""

	def test_display(self):
		model = NeuromorphicModel()

		n0 = model.create_neuron(threshold=-1.0, leak=2.0, refractory_period=3, reset_state=-2.0)
		n1 = model.create_neuron(threshold=0.0, leak=1.0, refractory_period=1, reset_state=-2.0)
		n2 = model.create_neuron(threshold=2.0, leak=0.0, refractory_period=0, reset_state=-1.0)
		n3 = model.create_neuron(threshold=5.0, leak=np.inf, refractory_period=2, reset_state=-2.0)
		n4 = model.create_neuron(threshold=-2.0, leak=5.0, refractory_period=1, reset_state=-2.0)

		model.create_synapse(n0, n1)
		model.create_synapse(n0, n2)
		model.create_synapse(n0, n3, weight=4.0, delay=3, enable_stdp=True)
		model.create_synapse(n4, n2, weight=2.0, delay=2, enable_stdp=False)
		model.create_synapse(n2, n1, weight=30.0, delay=4, enable_stdp=True)

		model.add_spike(0, n2, 4.0)
		model.add_spike(1, n1, 3.0)
		model.add_spike(0, n3, 2.0)

		# model.stdp_setup()
		model.setup()
		model.simulate(20)

		print(model)

		# model.print_spike_train()




if __name__ == "__main__":
	unittest.main()