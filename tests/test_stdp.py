import unittest
import numpy as np
import time 

import sys 
sys.path.insert(0, "../")

from src.superneuromat import NeuromorphicModel


class StdpTest(unittest.TestCase):
	""" Test refractory period

	"""

	def test_positive_update(self):
		""" 2 neuron STDP positive update
		"""

		model = NeuromorphicModel()

		a = model.create_neuron()
		b = model.create_neuron()

		s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		model.add_spike(0, a, 10.0)
		model.add_spike(1, a, 10.0)
		model.add_spike(2, a, 10.0)

		model.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
		model.setup()
		model.simulate(4)

		print(model)

		assert (model.synaptic_weights[0] == 5.25) 
		assert (model.synaptic_weights[1] == 3.5) 

		print("test_positive_update completed successfully")



	def test_negative_update(self):
		""" 2 neuron STDP negative update
		"""

		model = NeuromorphicModel()

		a = model.create_neuron()
		b = model.create_neuron()

		s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		model.add_spike(0, a, 10.0)
		model.add_spike(1, a, 10.0)
		model.add_spike(2, a, 10.0)

		model.stdp_setup(time_steps=3, Aneg=[0.1, 0.05, 0.025], positive_update=False, negative_update=True)
		model.setup()
		model.simulate(4)

		print(model)

		assert (model.synaptic_weights[0] == 1.0) 
		assert (model.synaptic_weights[1] == 0.825) 

		print("test_positive_update completed successfully")



	def test_positive_update_after_stdp_time_steps(self):
		""" 2 neuron STDP negative update 2
		"""

		model = NeuromorphicModel()

		a = model.create_neuron()
		b = model.create_neuron()

		s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		model.add_spike(3, a, 10.0)
		model.add_spike(4, a, 10.0)
		model.add_spike(5, a, 10.0)

		model.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
		model.setup()
		model.simulate(7)

		print(model)

		assert (model.synaptic_weights[0] == 5.25) 
		assert (model.synaptic_weights[1] == 3.5) 

		print("test_positive_update completed successfully")



	def test_negative_update_after_stdp_time_steps(self):
		""" 2 neuron STDP negative update
		"""

		model = NeuromorphicModel()

		a = model.create_neuron()
		b = model.create_neuron()

		s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		# model.add_spike(0, a, 10.0)
		# model.add_spike(0, b, 10.0)
		# model.add_spike(1, a, 10.0)
		# model.add_spike(1, b, 10.0)
		# model.add_spike(2, a, 10.0)
		# model.add_spike(2, b, 10.0)
		model.add_spike(3, a, 10.0)
		model.add_spike(4, a, 10.0)
		model.add_spike(5, a, 10.0)

		model.stdp_setup(time_steps=3, Aneg=[0.1, 0.05, 0.025], positive_update=False, negative_update=True)
		model.setup()
		model.simulate(7)

		print(model)

		# assert (model.synaptic_weights[0] == 1.0) 
		# assert (model.synaptic_weights[1] == 0.825) 

		print("test_positive_update completed successfully")



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

		model.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], Aneg=[0.01, 0.005, 0.0025], positive_update=False, negative_update=True)

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
		model.create_synapse(n0, n1, weight=0.0001, stdp_enabled=True)
		model.create_synapse(n0, n2, weight=0.0001, stdp_enabled=True)
		model.create_synapse(n0, n3, weight=0.0001, stdp_enabled=True)
		model.create_synapse(n0, n4, weight=0.0001, stdp_enabled=True)

		model.add_spike(2, n0, 1.0)
		model.add_spike(3, n0, 1.0)
		model.add_spike(3, n1, 1.0)
		model.add_spike(4, n2, 1.0)
		model.add_spike(5, n3, 1.0)
		model.add_spike(6, n4, 1.0)

		model.stdp_setup(time_steps=2, Apos=[1.0, 0.5], Aneg=[0.01, 0.005], positive_update=True, negative_update=True)

		model.setup()
		
		print("Synaptic weights before:")
		print(model._weights)

		model.simulate(8)

		print("Synaptic weights after:")
		print(model._weights)

		print(model._internal_states)

		model.print_spike_train()
		print()



	def test_stdp_4(self):
		"""
		"""

		model = NeuromorphicModel()

		n0 = model.create_neuron()
		n1 = model.create_neuron()
		n2 = model.create_neuron()
		n3 = model.create_neuron()
		n4 = model.create_neuron()

		model.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
		model.create_synapse(n0, n1, weight=0.0001, stdp_enabled=True)
		model.create_synapse(n0, n2, weight=0.0001, stdp_enabled=True)
		model.create_synapse(n0, n3, weight=0.0001, stdp_enabled=True)
		model.create_synapse(n0, n4, weight=0.0001, stdp_enabled=True)

		model.add_spike(2, n0, 1.0)
		model.add_spike(3, n0, 1.0)
		model.add_spike(3, n1, 1.0)
		model.add_spike(4, n2, 1.0)
		model.add_spike(5, n3, 1.0)
		model.add_spike(6, n4, 1.0)

		model.stdp_setup(time_steps=2, Apos=[1.0, 0.5], Aneg=[0.01, 0.005], positive_update=True, negative_update=True)

		model.setup(sparse=True)
		
		print("Synaptic weights before:")
		print(model._weights)

		model.simulate(8)

		print("Synaptic weights after:")
		print(model._weights)

		print(model._internal_states)

		print(model)
		
		print()






if __name__ == "__main__":
	unittest.main()

