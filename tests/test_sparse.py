import unittest
import numpy as np
from scipy.sparse import csc_array
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





	def test_sparse_5(self):
		""" Test sparsity during STDP updates

		"""

		t = 3 # STDP time steps
		n = 4 # Num neurons

		Apos = [1, 0.5, 0.25]
		Aneg = [0.1, 0.05, 0.025]

		dtype_int = np.int32

		spike_train = [	[1,0,1,0],
						[0,1,1,0],
						[0,0,0,1],
						[1,0,0,0],
						[0,0,1,0]
					  ]

		# Create spike vector and (sparse) matrix
		# vector = np.array(spike_train[-1], dtype=dtype_int)
		# sparse_matrix = csc_array(spike_train[-t-1:-1], dtype=dtype_int)
		# dense_matrix = np.array(spike_train[-t-1:-1], dtype=dtype_int)


		# # Einsum for outer product
		# # sparse_outer = np.outer(sparse_matrix, vector)
		# sparse_outer = np.einsum("ij,k->ijk", sparse_matrix.todense(), vector)
		# # sparse_outer = (sparse_matrix.transpose() @ vector.reshape([-1,n])).transpose()
		# dense_outer = np.outer(dense_matrix, vector).reshape([-1, n, n])

		# # Print spike vector and matrix
		# print(type(vector[0]), type(vector), vector)
		# print(type(sparse_matrix[0,0]), type(sparse_matrix), sparse_matrix)
		# print()

		# print(sparse_matrix.shape)
		# print(type(sparse_outer[0,0]), type(sparse_outer))
		# print()
		# print(sparse_outer)
		# print()

		# print(dense_matrix.shape, vector.shape, dense_outer.shape)
		# print(dense_matrix)
		# print()
		# print(dense_outer)




		Spre = csc_array(spike_train[-t-1:-1])
		Snow = csc_array(spike_train[-1]*t)









if __name__ == "__main__":
	unittest.main()

