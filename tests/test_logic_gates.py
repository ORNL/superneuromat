import unittest
import numpy as np

import sys 
sys.path.insert(0,"../src/")

from superneuromat import SNN



class LogicGatesTest(unittest.TestCase):
	""" Test SNNs for AND and OR gate

	"""

	def test_and(self):
		""" AND gate

		"""

		print("\nAND GATE")
		and_gate = SNN()

		# Create neurons
		a_id = and_gate.create_neuron(threshold=0.0)
		b_id = and_gate.create_neuron(threshold=0.0)
		c_id = and_gate.create_neuron(threshold=1.0)

		# Create synapses
		and_gate.create_synapse(a_id, c_id, weight=1.0)
		and_gate.create_synapse(b_id, c_id, weight=1.0)

		# Add spikes: [0,0]
		and_gate.add_spike(0, a_id, 0.0)
		and_gate.add_spike(0, b_id, 0.0)

		# Add spikes: [0,1]
		and_gate.add_spike(2, a_id, 0.0)
		and_gate.add_spike(2, b_id, 1.0)

		# Add spikes: [1,0]
		and_gate.add_spike(4, a_id, 1.0)
		and_gate.add_spike(4, b_id, 0.0)

		# # Add spikes: [1,1]
		and_gate.add_spike(6, a_id, 1.0)
		and_gate.add_spike(6, b_id, 1.0)

		# Setup and simulate
		and_gate.setup()
		and_gate.simulate(8)

		# Print spike train and neuromorphic model
		and_gate.print_spike_train()

		assert (np.array_equal(np.array(and_gate.spike_train), np.array([	[0,0,0],
																			[0,0,0],
																			[0,1,0],
																			[0,0,0],
																			[1,0,0],
																			[0,0,0],
																			[1,1,0],
																			[0,0,1]
																		])))

		print("test_and completed successfully")


	def test_or(self):
		""" OR gate

		"""

		print("\nOR GATE")
		or_gate = SNN()

		# Create neurons
		a_id = or_gate.create_neuron()
		b_id = or_gate.create_neuron()
		c_id = or_gate.create_neuron()

		# Create synapses
		or_gate.create_synapse(a_id, c_id, weight=1.0)
		or_gate.create_synapse(b_id, c_id, weight=1.0)

		# Add spikes: [0,0]
		or_gate.add_spike(0, a_id, 0.0)
		or_gate.add_spike(0, b_id, 0.0)

		# # Add spikes: [0,11]
		or_gate.add_spike(2, a_id, 0.0)
		or_gate.add_spike(2, b_id, 1.0)

		# Add spikes: [1,0]
		or_gate.add_spike(4, a_id, 1.0)
		or_gate.add_spike(4, b_id, 0.0)

		# Add spikes: [1,1]
		or_gate.add_spike(6, a_id, 1.0)
		or_gate.add_spike(6, b_id, 1.0)

		# Setup and simulate
		or_gate.setup()
		or_gate.simulate(8)

		# Print spike train and neuromorphic model
		or_gate.print_spike_train()
		
		assert (np.array_equal(np.array(or_gate.spike_train), np.array([	[0,0,0],
																			[0,0,0],
																			[0,1,0],
																			[0,0,1],
																			[1,0,0],
																			[0,0,1],
																			[1,1,0],
																			[0,0,1]
																		])))

		print("test_or completed successfully")



if __name__ == "__main__":
	unittest.main()