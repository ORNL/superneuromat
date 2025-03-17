import unittest
import numpy as np
import time 

import sys 
sys.path.insert(0,"../")

from superneuromat.neuromorphicmodel import NeuromorphicModel



class StdpTest(unittest.TestCase):
	""" Test refractory period

	"""

	def test_stdp_1(self):
		"""
		"""

		start = time.time()

		model = NeuromorphicModel()

		n0 = model.create_neuron()
		n1 = model.create_neuron()
		n2 = model.create_neuron()
		n3 = model.create_neuron()
		n4 = model.create_neuron()

		model.create_synapse(n0, n0, weight=-1.5, stdp_enabled=True)
		model.create_synapse(n0, n1, weight=0.1, stdp_enabled=True)
		model.create_synapse(n2, n3, weight=0.01, stdp_enabled=True)
		model.create_synapse(n3, n2, weight=0.25, stdp_enabled=True)
		model.create_synapse(n0, n3, weight=-0.73, stdp_enabled=True)
		model.create_synapse(n0, n4, weight=10.0, stdp_enabled=True)

		model.add_spike(0, n0, 1.0)
		model.add_spike(1, n0, 2.0)
		model.add_spike(1, n1, -0.3)
		model.add_spike(2, n2, 10.0)
		model.add_spike(3, n3, 21.1)
		model.add_spike(4, n4, 12.0)

		model.stdp_setup(time_steps=20, Apos=[1.0]*20, Aneg=[0.1]*20, positive_update=True, negative_update=True)

		model.setup()
		
		print("Synaptic weights before:")
		print(model._weights)

		print("\nSTDP enabled synapses before:")
		print(model._stdp_enabled_synapses)

		model.simulate(100000)

		print("Synaptic weights after:")
		print(model._weights)

		# model.print_spike_train()

		# print(model)

		end = time.time()

		print("test_stdp_1 finished in", end - start, "seconds")
		print()




	def test_stdp_2(self):
		"""
		"""

		model = NeuromorphicModel()

		n0 = model.create_neuron()
		n1 = model.create_neuron()
		n2 = model.create_neuron()
		n3 = model.create_neuron()
		n4 = model.create_neuron()

		model.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
		model.create_synapse(n0, n1, weight=0.0, stdp_enabled=True)
		model.create_synapse(n0, n2, weight=0.0, stdp_enabled=True)
		model.create_synapse(n0, n3, weight=0.0, stdp_enabled=True)
		model.create_synapse(n0, n4, weight=0.0, stdp_enabled=True)

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
		print()



	def test_stdp_3(self):
		"""
		"""

		model = NeuromorphicModel()

		n0 = model.create_neuron()
		n1 = model.create_neuron()
		n2 = model.create_neuron()
		n3 = model.create_neuron()
		n4 = model.create_neuron()

		model.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
		model.create_synapse(n0, n1, weight=0.0, stdp_enabled=True)
		model.create_synapse(n0, n2, weight=0.0, stdp_enabled=True)
		model.create_synapse(n0, n3, weight=0.0, stdp_enabled=True)
		model.create_synapse(n0, n4, weight=0.0, stdp_enabled=True)

		model.add_spike(0, n0, 1.0)
		model.add_spike(1, n0, 1.0)
		model.add_spike(1, n1, 1.0)
		model.add_spike(2, n2, 1.0)
		model.add_spike(3, n3, 1.0)
		model.add_spike(4, n4, 1.0)

		model.stdp_setup(time_steps=2, Apos=[1.0, 0.5], Aneg=[0.01, 0.005], positive_update=True, negative_update=True)

		model.setup()
		
		print("Synaptic weights before:")
		print(model._weights)

		model.simulate(6)

		print("Synaptic weights after:")
		print(model._weights)

		model.print_spike_train()
		print()






if __name__ == "__main__":
	unittest.main()

