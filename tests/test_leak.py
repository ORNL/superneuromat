import unittest
import numpy as np

import sys 
sys.path.insert(0,"../src/")

from superneuromat import SNN



class LeakTest(unittest.TestCase):
	""" Test leak

	"""

	def test_int_state_greater_than_reset_state(self):
		""" Test leak when internal state is greater than reset state

		"""

		print("Internal state greater than reset state")

		snn = SNN()

		n0 = snn.create_neuron(threshold=10.0, leak=1.0, reset_state=-2.0)

		snn.add_spike(1, n0, 2.0)
		snn.add_spike(2, n0, 4.0)
		snn.add_spike(3, n0, 3.0)
		snn.add_spike(4, n0, 10.0)

		snn.setup()
		snn.simulate(5)

		snn.print_spike_train()

		assert (snn._internal_states[0] == -2.0)
		assert (snn.spike_train == [[0],[0],[0],[0],[1]])

		print("test_int_state_greater_than_reset_state completed successfully")




	def test_int_state_less_than_reset_state(self):
		""" Test leak when internal state is less than reset state

		"""

		print("Internal state lesss than reset state")

		snn = SNN()

		n0 = snn.create_neuron(threshold=10.0, leak=5.0, reset_state=-2.0)

		snn.add_spike(1, n0, -2.0)
		snn.add_spike(2, n0, -4.0)
		snn.add_spike(3, n0, -6.0)
		snn.add_spike(4, n0, -10.0)

		snn.setup()
		snn.simulate(5)

		snn.print_spike_train()

		assert (snn.spike_train == [[0],[0],[0],[0],[0]])
		assert (snn._internal_states[0] == -13.0)

		print("test_int_state_less_than_reset_state completed successfully")



	def test_infinite_leak(self):
		""" Test infinite leak

		"""

		print("Infinite leak")

		snn = SNN()

		n0 = snn.create_neuron(threshold=0.0, leak=np.inf, reset_state=0.0)
		n1 = snn.create_neuron(threshold=10.0, leak=np.inf, reset_state=0.0)

		snn.add_spike(1, n0, -2.0)
		snn.add_spike(2, n0, -4.0)
		snn.add_spike(3, n0, -6.0)
		snn.add_spike(4, n0, -10.0)

		snn.add_spike(1, n1, 2.0)
		snn.add_spike(2, n1, 4.0)
		snn.add_spike(3, n1, 6.0)
		snn.add_spike(4, n1, 10.0)

		snn.setup()
		snn.simulate(5)

		snn.print_spike_train()

		assert (np.array_equal(np.array(snn.spike_train), np.array([[0,0],[0,0],[0,0],[0,0],[0,0]])))
		assert (np.array_equal(snn._internal_states, np.array([-10.0, 10.0])))
		print("test_infinite_leak completed successfully")



	def test_zero_leak(self):
		""" Test zero leak

		"""

		print("Zero leak")
		snn = SNN()

		n0 = snn.create_neuron(threshold=0.0, leak=0.0, reset_state=0.0)
		n1 = snn.create_neuron(threshold=10.0, leak=0.0, reset_state=0.0)

		snn.add_spike(1, n0, -2.0)
		snn.add_spike(2, n0, -4.0)
		snn.add_spike(3, n0, -6.0)
		snn.add_spike(4, n0, -10.0)

		snn.add_spike(1, n1, 2.0)
		snn.add_spike(2, n1, 4.0)
		snn.add_spike(3, n1, 6.0)
		snn.add_spike(4, n1, 10.0)

		snn.setup()
		snn.simulate(5)

		snn.print_spike_train()

		assert (np.array_equal(snn._internal_states, np.array([-22.0, 10.0])))
		assert (np.array_equal(np.array(snn.spike_train), np.array([[0,0],[0,0],[0,0],[0,1],[0,0]])))
		print("test_zero_leak completed successfully")



	def test_leak_before_spike(self):
		""" Test leak before spike

		"""

		print("Leak before spike")

		snn = SNN()

		n0 = snn.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)
		n1 = snn.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)

		snn.add_spike(0, n0, 3.0)
		snn.add_spike(0, n1, 10.0)
		snn.add_spike(1, n0, 10.0)
		snn.add_spike(1, n1, 12.0)

		snn.setup()
		snn.simulate(10)

		snn.print_spike_train()

		assert (np.array_equal(snn._internal_states, np.array([0.0, 0.0]))) 
		assert (np.array_equal(np.array(snn.spike_train), np.array([[1,1],[0,0],[0,0],[0,0],[0,0],[0,0],[0,1],[0,0],[0,0],[0,0]]))) 

		print("test_leak_before_spike completed successfully")
	


if __name__ == "__main__":
	unittest.main()

