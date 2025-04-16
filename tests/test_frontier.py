import unittest
import numpy as np
import time 

import sys 
sys.path.insert(0,"../")

from src.superneuromat import NeuromorphicModel


class TestFrontier(unittest.TestCase):
	""" Tests the SNN simulation on the Frontier supercomputer
	"""


	def test_basic(self):
		"""
		"""

		# Create neuromorphic model
		model = NeuromorphicModel(backend="frontier", num_mpi_ranks=4, dtype=32)

		# Create dummy neurons and a synapse
		a = model.create_neuron(threshold=24.3, reset_state=5.0, refractory_period=2)
		b = model.create_neuron(threshold=-4.5, reset_state=-5.0, refractory_period=1)
		s = model.create_synapse(a, b, weight=0.00324, stdp_enabled=True)

		# Send the first array into C
		model.stdp_setup()

		# Send the second array into C
		model.setup()

		# Add the two arrays in C and get the solution back in Python
		# model.simulate()

		# model._test_openmp(1024) 



	def test_and_frontier(self):
		""" Tests the AND gate with Frontier backend
		"""

		# Create neuromorphic model
		model = NeuromorphicModel(backend="frontier")


		# Add neurons
		a = model.create_neuron()
		b = model.create_neuron()
		c = model.create_neuron(threshold=1.0)


		# Add synapses
		a = model.create_synapse(a, c, weight=1.0, delay=1, stdp_enabled=False)
		b = model.create_synapse(b, c, weight=1.0, delay=1, stdp_enabled=True)


		# Add spikes
		# Input: (0, 0)
		model.add_spike(0, a, 0.0)
		model.add_spike(0, b, 0.0)

		# Input: (0, 1)
		model.add_spike(2, a, 0.0)
		model.add_spike(2, b, 1.0)

		# Input: (1, 0)
		model.add_spike(4, a, 1.0)
		model.add_spike(4, b, 0.0)

		# Input: (1, 1)
		model.add_spike(6, a, 1.0)
		model.add_spike(6, b, 1.0)


		# Setup and simulate
		model.setup()
		model.simulate(10)



if __name__ == "__main__":
	unittest.main()