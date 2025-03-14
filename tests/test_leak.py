import unittest
import numpy as np
from superneuromat.neuromorphicmodel import NeuromorphicModel



class LeakTest(unittest.TestCase):
	""" Test refractory period

	"""

	def int_state_greater_than_reset_state(self):
		model = NeuromorphicModel()

		n1 = model.create_neuron(threshold=10.0, leak=1.0, reset_state=-2.0)

		model.add_spike(1, n1, 2.0)
		model.add_spike(2, n1, 4.0)
		model.add_spike(3, n1, 3.0)
		model.add_spike(4, n1, 10.0)

		model.setup()
		model.simulate(5)

		model.print_spike_train()




	def int_state_less_than_reset_state(self):
		model = NeuromorphicModel()

		n1 = model.create_neuron(threshold=10.0, leak=5.0, reset_state=-2.0)

		model.add_spike(1, n1, -2.0)
		model.add_spike(2, n1, -4.0)
		model.add_spike(3, n1, -6.0)
		model.add_spike(4, n1, -10.0)

		model.setup()
		model.simulate(5)

		model.print_spike_train()



	def infinite_leak(self):
		model = NeuromorphicModel()

		n1 = model.create_neuron(threshold=0.0, leak=np.inf, reset_state=0.0)
		n2 = model.create_neuron(threshold=10.0, leak=np.inf, reset_state=0.0)

		model.add_spike(1, n1, -2.0)
		model.add_spike(2, n1, -4.0)
		model.add_spike(3, n1, -6.0)
		model.add_spike(4, n1, -10.0)

		model.add_spike(1, n2, 2.0)
		model.add_spike(2, n2, 4.0)
		model.add_spike(3, n2, 6.0)
		model.add_spike(4, n2, 10.0)

		model.setup()
		model.simulate(5)

		model.print_spike_train()



	def zero_leak(self):
		model = NeuromorphicModel()

		n1 = model.create_neuron(threshold=0.0, leak=0.0, reset_state=0.0)
		n2 = model.create_neuron(threshold=10.0, leak=0.0, reset_state=0.0)

		model.add_spike(1, n1, -2.0)
		model.add_spike(2, n1, -4.0)
		model.add_spike(3, n1, -6.0)
		model.add_spike(4, n1, -10.0)

		model.add_spike(1, n2, 2.0)
		model.add_spike(2, n2, 4.0)
		model.add_spike(3, n2, 6.0)
		model.add_spike(4, n2, 10.0)

		model.setup()
		model.simulate(5)

		model.print_spike_train()



	def leak_before_spike(self):
		model = NeuromorphicModel()

		n0 = model.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)
		n1 = model.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)

		model.add_spike(0, n0, 3.0)
		model.add_spike(0, n1, 10.0)
		model.add_spike(1, n0, 10.0)
		model.add_spike(1, n1, 12.0)

		model.setup()
		model.simulate(10)

		model.print_spike_train()
	


if __name__ == "__main__":
	unittest.main()

