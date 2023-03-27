import numpy as np
import time 
from superneuromat.neuromorphicmodel import NeuromorphicModel


total_time_start = time.time()


# NUM_NEURONS = [10, 100, 1000, 10000]
NUM_NEURONS = [1000]
# CONNECTION_PROBABILITY = [0.25, 0.5, 0.75, 1.0]
CONNECTION_PROBABILITY = [1.0]
INPUT_SPIKE_PROBABILITY = 0.3
TIME_STEPS = 1000
NUM_REPEATS = 10


file_name = "superneuromat_results.csv"

with open(file_name, 'w') as file:
	file.write("num_neurons, connection_probability, num_repeat, setup_time, simulate_time\n")



for num_neurons in NUM_NEURONS:
	for connection_probability in CONNECTION_PROBABILITY:
		for num_repeat in range(NUM_REPEATS):

			# Create NeuromorphicModel
			# setup_start = time.time()
			model = NeuromorphicModel()


			# Create neurons
			for n in range(num_neurons):
				model.create_neuron(threshold=np.random.rand(), leak=np.random.rand())

			# print("Neurons created...")


			# Create synapses
			for i in range(num_neurons):
				for j in range(num_neurons):
					if np.random.rand() <= connection_probability:
						model.create_synapse(i, j, weight=np.random.rand(), delay=1)

			# print("Synapses created...")


			# Add spikes
			for neuron_id in range(num_neurons):
				if np.random.rand() <= INPUT_SPIKE_PROBABILITY:
					model.add_spike(0, neuron_id, 1.0)

			# print("Spikes added...")	


			# Setup 
			setup_start = time.time()
			model.setup()
			setup_end = time.time()

			# print()
			# print("Setup complete...")


			# print(model._internal_states)

			# print(f"# neurons: {model.num_neurons}")
			# print(f"# synapses: {model.num_synapses}")

			# print("Setup time:", setup_end - setup_start)


			# Simulate
			simulate_start = time.time()
			model.simulate(time_steps=TIME_STEPS)
			simulate_end = time.time()

			# print()
			# print("Simulation complete!")


			# print(model._internal_states)


			# for spike_train in model.spike_train:
			# 	print(spike_train)




			# print("Simulation time:", simulate_end - simulate_start)


			print([num_neurons, connection_probability, num_repeat, setup_end - setup_start, simulate_end - simulate_start])

			with open(file_name, 'a') as file:
				file.write(f"{num_neurons}, {connection_probability}, {num_repeat}, {setup_end - setup_start}, {simulate_end - simulate_start}\n")

		print()




total_time_end = time.time()

print()
print("Total time:", total_time_end - total_time_start)






