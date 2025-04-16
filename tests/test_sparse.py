import unittest
import numpy as np
import time 

import sys 
sys.path.insert(0, "../")

from src.superneuromat import NeuromorphicModel


class SparseTest(unittest.TestCase):
	""" Test sparse operations

	"""

	def test_sparse_1(self):
		""" Less than 200 neurons, should default to model.sparse = False
		"""

		start = time.time()

		num_neurons = 2

		model = NeuromorphicModel()

		a = model.create_neuron()
		b = model.create_neuron()

		s = model.create_synapse(a, b, stdp_enabled=True)

		model.add_spike(0, a, 50.0)
		model.add_spike(3, b, 23.5)
		model.add_spike(1, a, 0.02)
		model.add_spike(4, b, 0.6)

		model.stdp_setup()
		model.setup()
		model.simulate(10)

		print(f"model.sparse: {model.sparse}")
		print(model)


		end = time.time() 

		print(f"Test 1 completed in {end - start} sec")



	def test_sparse_2(self):
		""" Less than 200 neurons, explicitly making model.sparse = True
		"""

		start = time.time()

		num_neurons = 2

		model = NeuromorphicModel()

		a = model.create_neuron()
		b = model.create_neuron()

		s = model.create_synapse(a, b, stdp_enabled=True)

		model.add_spike(0, a, 50.0)
		model.add_spike(3, b, 23.5)
		model.add_spike(1, a, 0.02)
		model.add_spike(4, b, 0.6)

		model.stdp_setup()
		
		print(model)

		model.setup(sparse = True)
		model.simulate(10)

		print(f"model.sparse: {model.sparse}")
		print(model)

		end = time.time() 

		print(f"Test 1 completed in {end - start} sec")



	def test_sparse_3(self):
		""" More than 200 neurons, within sparsity threshold, explicitly making model.sparse = False
		"""

		start = time.time()

		num_neurons = 12000
		sparsity = 0.005
		num_spikes = 100
		num_simulation_time_steps = 20
		np.random.seed(42)


		model = NeuromorphicModel()

		for i in range(num_neurons):
			model.create_neuron(refractory_period=2)

		for i in range(int(num_neurons * num_neurons * sparsity)):
			model.create_synapse(np.random.randint(num_neurons), np.random.randint(num_neurons), stdp_enabled=True)

		for i in range(num_spikes):
			model.add_spike(np.random.randint(num_simulation_time_steps), np.random.randint(num_neurons), np.random.random() * 10)

		model.stdp_setup()
		
		# print(model)

		model.setup(sparse = False)
		model.simulate(num_simulation_time_steps)

		print(f"model.sparse: {model.sparse}")
		# print(model)

		end = time.time() 

		print(f"Test 1 completed in {end - start} sec")



	def test_sparse_4(self):
		""" More than 200 neurons, within sparsity threshold, explicitly making model.sparse = True
		"""

		start = time.time()

		num_neurons = 12000
		sparsity = 0.005
		num_spikes = 100
		num_simulation_time_steps = 20
		np.random.seed(42)


		model = NeuromorphicModel()

		for i in range(num_neurons):
			model.create_neuron(refractory_period=2)

		for i in range(int(num_neurons * num_neurons * sparsity)):
			model.create_synapse(np.random.randint(num_neurons), np.random.randint(num_neurons), stdp_enabled=True)

		for i in range(num_spikes):
			model.add_spike(np.random.randint(num_simulation_time_steps), np.random.randint(num_neurons), np.random.random() * 10)

		model.stdp_setup()
		
		# print(model)

		model.setup(sparse = True)
		model.simulate(num_simulation_time_steps)

		print(f"model.sparse: {model.sparse}")
		# print(model)

		end = time.time() 

		print(f"Test 1 completed in {end - start} sec")






if __name__ == "__main__":
	unittest.main()

