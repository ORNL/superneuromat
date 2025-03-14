import numpy as np
import unittest

import sys 
sys.path.insert(0,"../")

from superneuromat.neuromorphicmodel import NeuromorphicModel


class TestTypeErrors(unittest.TestCase):
	""" Test all type errors

	"""

	def test_neuron_threshold(self):
		model = NeuromorphicModel()
		model.create_neuron(threshold="five")


	def test_neuron_leak(self):
		model = NeuromorphicModel()
		model.create_neuron(leak="two")


	def test_neuron_reset_state(self):
		model = NeuromorphicModel()
		model.create_neuron(reset_state="alpha")


	def test_neuron_refractory_period(self):
		model = NeuromorphicModel()
		model.create_neuron(refractory_period={"beta": 1})


	def test_synapse_pre_id(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(-1.0, n1)


	def test_synapse_post_id(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, [])


	def test_synapse_weight(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, weight="something")


	def test_synapse_delay(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, weight=1.0, delay=-5.4)


	def test_synapse_stdp(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, stdp_enabled=1)


	def test_spike_time(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		model.add_spike("zero", n0, 1.0)


	def test_spike_id(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		model.add_spike(0, float(n0), 1.0)


	def test_spike_value(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		model.add_spike(0, n0, "one")


	def test_stdp_time_steps(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, stdp_enabled=True)
		model.stdp_setup(time_steps="forty three")


	def test_stdp_apos(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, stdp_enabled=True)
		model.stdp_setup(time_steps=3, Apos=1.0)


	def test_stdp_aneg(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, stdp_enabled=True)
		model.stdp_setup(time_steps=3, Aneg=1.0)


	def test_stdp_pos_update(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, stdp_enabled=True)
		model.stdp_setup(time_steps=3, positive_update="False")


	def test_stdp_neg_update(self):
		model=NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, stdp_enabled=True)
		model.stdp_setup(time_steps=3, negative_update="False")


	def test_simulate(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1)

		model.setup()
		model.simulate([500])





class TestValueErrors(unittest.TestCase):
	""" Test all value errors

	"""

	def test_neuron_leak(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron(leak=-1.0)


	def test_neuron_refractory_period(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron(refractory_period=-1)


	def test_synapse_pre_id(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(-1, n1)


	def test_synapse_post_id(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, -1)


	def test_synapse_delay(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=-2)


	def test_add_spike_time(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2)
		model.add_spike(-1, n0)


	def test_add_spike_time(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2)
		model.add_spike(1, -1)


	def test_stdp_time_steps(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		model.add_spike(1, n0)
		model.stdp_setup(-1)


	def test_stdp_apos_len(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		model.add_spike(1, n0)
		model.stdp_setup(3, [])


	def test_stdp_aneg_len(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		model.add_spike(1, n0)
		model.stdp_setup(2, [1.0, 0.5], [1.0, 0.5, 0.25])


	def test_stdp_apos_element_type(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		model.add_spike(1, n0)
		model.stdp_setup(3, Apos=["a", "b", "c"])


	def test_stdp_aneg_element_type(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		model.add_spike(1, n0)
		model.stdp_setup(3, Aneg=["a", "b", "c"])


	def test_stdp_apos_element_value(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		model.add_spike(1, n0)
		model.stdp_setup(2, Apos=[-1, -1], negative_update=False)


	def test_stdp_aneg_element_value(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		model.add_spike(1, n0)
		model.stdp_setup(1, Apos=[1.0], Aneg=[-5.0])


	def test_simulate(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		model.create_synapse(n0, n1)
		model.add_spike(0, n0)
		model.setup()
		model.simulate(-1)





class TestRuntimeErrors(unittest.TestCase):
	""" Test runtime errors

	"""

	def test_stdp_runtime(self):
		model = NeuromorphicModel()
		n0 = model.create_neuron()
		n1 = model.create_neuron()
		n2 = model.create_neuron()
		n3 = model.create_neuron()
		model.create_synapse(n0, n1)
		model.create_synapse(n0, n2)
		model.create_synapse(n2, n3)

		model.add_spike(0, n0)
		model.add_spike(1, n1)
		model.add_spike(2, n1)
		model.add_spike(0, n2)

		model.stdp_setup()
		model.setup()
		model.simulate(10)




	






if __name__ == "__main__":
	unittest.main()