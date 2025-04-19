import unittest
import numpy as np
import time 

import sys 
sys.path.insert(0, "../src/")

from superneuromat import SNN


class StdpTest(unittest.TestCase):
	""" Test refractory period

	"""

	def test_positive_update(self):
		""" 2 neuron STDP positive update
		"""

		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		snn.add_spike(0, a, 10.0)
		snn.add_spike(1, a, 10.0)
		snn.add_spike(2, a, 10.0)

		snn.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
		snn.setup()
		snn.simulate(4)

		print(snn)

		assert (snn.synaptic_weights[0] == 5.25) 
		assert (snn.synaptic_weights[1] == 3.5) 

		print("test_positive_update completed successfully")



	def test_negative_update(self):
		""" 2 neuron STDP negative update
		"""

		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		snn.add_spike(0, a, 10.0)
		snn.add_spike(1, a, 10.0)
		snn.add_spike(2, a, 10.0)

		snn.stdp_setup(time_steps=3, Aneg=[0.1, 0.05, 0.025], positive_update=False, negative_update=True)
		snn.setup()
		snn.simulate(4)

		print(snn)

		assert (snn.synaptic_weights[0] == 1.0) 
		assert (snn.synaptic_weights[1] == 0.825) 

		print("test_negative_update completed successfully")



	def test_positive_update_after_stdp_time_steps(self):
		""" 2 neuron STDP positive update but after simulation has run for more than STDP time steps
		"""

		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		snn.add_spike(3, a, 10.0)
		snn.add_spike(4, a, 10.0)
		snn.add_spike(5, a, 10.0)

		snn.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
		snn.setup()
		snn.simulate(7)

		print(snn)

		assert (snn.synaptic_weights[0] == 5.25) 
		assert (snn.synaptic_weights[1] == 3.5) 

		print("test_positive_update_after_stdp_time_steps completed successfully")



	def test_negative_update_after_stdp_time_steps(self):
		""" 2 neuron STDP negative update but after simulation has run for more than STDP time steps
		"""

		snn = SNN()

		a = snn.create_neuron()
		b = snn.create_neuron()

		s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
		s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

		# snn.add_spike(0, a, 10.0)
		# snn.add_spike(0, b, 10.0)
		# snn.add_spike(1, a, 10.0)
		# snn.add_spike(1, b, 10.0)
		# snn.add_spike(2, a, 10.0)
		# snn.add_spike(2, b, 10.0)
		snn.add_spike(3, a, 10.0)
		snn.add_spike(4, a, 10.0)
		snn.add_spike(5, a, 10.0)

		snn.stdp_setup(time_steps=3, Aneg=[0.1, 0.05, 0.025], positive_update=False, negative_update=True)
		snn.setup()
		snn.simulate(7)

		print(snn)

		assert (snn.synaptic_weights[0] == 0.475) 
		assert (snn.synaptic_weights[1] == 0.300) 

		print("test_negative_update_after_stdp_time_steps completed successfully")



	def test_stdp_1(self):
		"""
		"""

		start = time.time()

		snn = SNN()

		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		n2 = snn.create_neuron()
		n3 = snn.create_neuron()
		n4 = snn.create_neuron()

		snn.create_synapse(n0, n0, weight=-1.5, stdp_enabled=True)
		snn.create_synapse(n0, n1, weight=0.1, stdp_enabled=True)
		snn.create_synapse(n2, n3, weight=0.01, stdp_enabled=True)
		snn.create_synapse(n3, n2, weight=0.25, stdp_enabled=True)
		snn.create_synapse(n0, n3, weight=-0.73, stdp_enabled=True)
		snn.create_synapse(n0, n4, weight=10.0, stdp_enabled=True)

		snn.add_spike(0, n0, 1.0)
		snn.add_spike(1, n0, 2.0)
		snn.add_spike(1, n1, -0.3)
		snn.add_spike(2, n2, 10.0)
		snn.add_spike(3, n3, 21.1)
		snn.add_spike(4, n4, 12.0)

		snn.stdp_setup(time_steps=20, Apos=[1.0]*20, Aneg=[0.1]*20, positive_update=True, negative_update=True)

		snn.setup()
		
		print("Synaptic weights before:")
		print(snn._weights)

		print("\nSTDP enabled synapses before:")
		print(snn._stdp_enabled_synapses)

		snn.simulate(100000)

		print("Synaptic weights after:")
		print(snn._weights)

		end = time.time()

		assert (np.allclose(snn._weights, np.array([	[-199979.4,  -199978.9,        0.,   -199977.53, -199963.5 ],
																[      0.,         0.,         0.,         0.,         0.  ],
																[      0.,         0.,         0.,   -199977.89,       0.  ],
																[      0.,         0.,   -199978.75,       0.,         0.  ],
																[      0.,         0.,         0.,         0.,         0.  ]])))

		print("test_stdp_1 finished in", end - start, "seconds")




	def test_stdp_2(self):
		"""
		"""

		snn = SNN()

		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		n2 = snn.create_neuron()
		n3 = snn.create_neuron()
		n4 = snn.create_neuron()

		snn.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
		snn.create_synapse(n0, n1, weight=0.0, stdp_enabled=True)
		snn.create_synapse(n0, n2, weight=0.0, stdp_enabled=True)
		snn.create_synapse(n0, n3, weight=0.0, stdp_enabled=True)
		snn.create_synapse(n0, n4, weight=0.0, stdp_enabled=True)

		snn.add_spike(0, n0, 1.0)
		snn.add_spike(1, n0, 1.0)
		snn.add_spike(1, n1, 1.0)
		snn.add_spike(2, n2, 1.0)
		snn.add_spike(3, n3, 1.0)
		snn.add_spike(4, n4, 1.0)

		snn.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], Aneg=[0.01, 0.005, 0.0025], positive_update=False, negative_update=True)

		snn.setup(sparse=False)

		
		print("Synaptic weights before:")
		print(snn._weights)

		snn.simulate(6)

		print("Synaptic weights after:")
		print(snn._weights)

		snn.print_spike_train()

		assert (np.allclose(snn._weights, np.array([	[-1.0775, -0.0675, -0.0725, -0.075,  -0.0775],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ] 
													])))

		print("test_stdp_2 completed successfully")



	def test_stdp_3(self):
		"""
		"""

		snn = SNN()

		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		n2 = snn.create_neuron()
		n3 = snn.create_neuron()
		n4 = snn.create_neuron()

		snn.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
		snn.create_synapse(n0, n1, weight=0.0001, stdp_enabled=True)
		snn.create_synapse(n0, n2, weight=0.0001, stdp_enabled=True)
		snn.create_synapse(n0, n3, weight=0.0001, stdp_enabled=True)
		snn.create_synapse(n0, n4, weight=0.0001, stdp_enabled=True)

		snn.add_spike(2, n0, 1.0)
		snn.add_spike(3, n0, 1.0)
		snn.add_spike(3, n1, 1.0)
		snn.add_spike(4, n2, 1.0)
		snn.add_spike(5, n3, 1.0)
		snn.add_spike(6, n4, 1.0)

		snn.stdp_setup(time_steps=2, Apos=[1.0, 0.5], Aneg=[0.01, 0.005], positive_update=True, negative_update=True)

		snn.setup()
		
		print("Synaptic weights before:")
		print(snn._weights)

		snn.simulate(8)

		print("Synaptic weights after:")
		print(snn._weights)

		print(snn._internal_states)

		snn.print_spike_train()

		assert (np.allclose(snn._weights, np.array([	[-1.1,     0.9101,  0.4051, -0.0999, -0.0999],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ] 
													])))
		
		print("test_stdp_3 completed successfully")



	def test_stdp_4(self):
		"""
		"""

		snn = SNN()

		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		n2 = snn.create_neuron()
		n3 = snn.create_neuron()
		n4 = snn.create_neuron()

		snn.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
		snn.create_synapse(n0, n1, weight=0.0001, stdp_enabled=True)
		snn.create_synapse(n0, n2, weight=0.0001, stdp_enabled=True)
		snn.create_synapse(n0, n3, weight=0.0001, stdp_enabled=True)
		snn.create_synapse(n0, n4, weight=0.0001, stdp_enabled=True)

		snn.add_spike(2, n0, 1.0)
		snn.add_spike(3, n0, 1.0)
		snn.add_spike(3, n1, 1.0)
		snn.add_spike(4, n2, 1.0)
		snn.add_spike(5, n3, 1.0)
		snn.add_spike(6, n4, 1.0)

		snn.stdp_setup(time_steps=2, Apos=[1.0, 0.5], Aneg=[0.01, 0.005], positive_update=True, negative_update=True)

		snn.setup(sparse=True)
		
		print("Synaptic weights before:")
		print(snn._weights)

		snn.simulate(8)

		print("Synaptic weights after:")
		print(snn._weights)

		print(snn._internal_states)

		print(snn)

		assert (np.allclose(snn._weights.todense(), np.array([	[-1.1,     0.9101,  0.4051, -0.0999, -0.0999],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ],
														[ 0.,      0.,      0.,      0.,      0.    ] 
													])))
		
		print("test_stdp_4 completed successfully")






if __name__ == "__main__":
	unittest.main()

