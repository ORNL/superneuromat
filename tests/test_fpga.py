import unittest
import numpy as np
import time 
import pickle

import sys 
sys.path.insert(0, "../")

# from src.superneuromat import NeuromorphicModel
import superneuromat as snm


class FpgaTest(unittest.TestCase):
	""" Test small network for FPGA

	"""
	def test_fpga(self):

		# Simulation parameters
		num_neurons = 90 
		num_synapses = 5 * num_neurons
		num_input_spikes = 3
		num_sim_time_steps = 10


		# Create the neuromorphic model
		# model = NeuromorphicModel()
		model = snm.NeuromorphicModel()

		for i in range(num_neurons):
			model.create_neuron(	threshold=1.0,
									leak=0.0,
									reset_state=0.0,
									refractory_period=1
								)

		for i in range(num_synapses):
			pre = np.random.randint(num_neurons)
			post = np.random.randint(num_neurons)

			model.create_synapse(pre, post, weight=1.0, stdp_enabled=True)

		for i in range(num_input_spikes):
			n = np.random.randint(num_neurons)
			t = np.random.randint(num_sim_time_steps)

			model.add_spike(t, n, 10.0)


		# STDP setup
		model.stdp_setup(time_steps=10, Apos=[0.1]*10, Aneg=[0.1]*10)


		# Setup
		model.setup()


		# Pickle initial state
		with open("initial.pkl", "wb") as file:
			pickle.dump(model, file)


		# Simulate
		model.simulate(num_sim_time_steps)


		# Print
		print(model)


		# Pickle initial state
		# with open("final.pkl", "wb") as file:
			# pickle.dump(model, file)




	def test_fpga_read(self):
		""" Test reading of the pickle file
		"""

		# Pickle file paths
		initial_path = "initial.pkl"
		final_path = "final.pkl"


		# Load initial model
		with open(initial_path, "rb") as file:
			model_initial = pickle.load(file)


		# Load final model
		with open(final_path, "rb") as file:
			model_final = pickle.load(file)


		# Print both initial and final models
		print(model_initial)
		print(model_final)


		# Print model weights
		print(model_initial._weights)










	



if __name__ == "__main__":
	unittest.main()

