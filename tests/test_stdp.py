import unittest
import numpy as np
from superneuromat.neuromorphicmodel import NeuromorphicModel



class StdpTest(unittest.TestCase):
	""" Test refractory period

	"""

	def stdp_test(self):
		"""
		"""

		model = NeuromorphicModel()

		n0 = model.create_neuron()
		n1 = model.create_neuron()
		n2 = model.create_neuron()
		n3 = model.create_neuron()
		n4 = model.create_neuron()

		model.create_synapse(n0, n0, weight=-1.0, enable_stdp=True)
		model.create_synapse(n0, n1, weight=0.0, enable_stdp=True)
		model.create_synapse(n0, n2, weight=0.0, enable_stdp=True)
		model.create_synapse(n0, n3, weight=0.0, enable_stdp=True)
		model.create_synapse(n0, n4, weight=0.0, enable_stdp=True)

		model.add_spike(0, n0, 1.0)
		model.add_spike(1, n0, 1.0)
		model.add_spike(1, n1, 1.0)
		model.add_spike(2, n2, 1.0)
		model.add_spike(3, n3, 1.0)
		model.add_spike(4, n4, 1.0)

		model.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], Aneg=[0.01, 0.005, 0.0025], negative_update=True)

		model.setup()
		
		print("Synaptic weights before:")
		print(model._weights)

		model.simulate(6)

		print("Synaptic weights after:")
		print(model._weights)

		model.print_spike_train()

		print(model)






if __name__ == "__main__":
	unittest.main()

