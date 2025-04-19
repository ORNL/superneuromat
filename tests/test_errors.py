import numpy as np
import unittest

import sys 
sys.path.insert(0,"../src/")

from superneuromat import SNN


class TypeErrorTest(unittest.TestCase):
	""" Test all type errors

	"""

	def test_neuron_threshold(self):
		""" Test type error for neuron threshold

		"""

		snn = SNN()

		try:
			snn.create_neuron(threshold="five")

		except TypeError:
			print("test_neuron_threshold completed successfully")




	def test_neuron_leak(self):
		""" Test type error for neuron leak

		"""

		snn = SNN()

		try:
			snn.create_neuron(leak="two")
			
		except TypeError:
			print("test_neuron_leak completed successfully")




	def test_neuron_reset_state(self):
		""" Test type error for neuron reset state
		
		"""

		snn = SNN()
		
		try:
			snn.create_neuron(reset_state="alpha")

		except TypeError:
			print("test_neuron_reset_state completed successfully")




	def test_neuron_refractory_period(self):
		""" Test type error for neuron refractory period
		
		"""

		snn = SNN()
		
		try:
			snn.create_neuron(refractory_period={"beta": 1})

		except TypeError:
			print("test_neuron_refractory_period completed successfully")




	def test_synapse_pre_id(self):
		""" Test type error for synapse pre ID
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(-1.0, n1)

		except TypeError:
			print("test_synapse_pre_id completed successfully")




	def test_synapse_post_id(self):
		""" Test type error for synapse post ID
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(n0, [])

		except TypeError:
			print("test_synapse_post_id completed successfully")




	def test_synapse_weight(self):
		""" Test type error for synapse weight
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(n0, n1, weight="something")

		except TypeError:
			print("test_synapse_weight completed successfully")




	def test_synapse_delay(self):
		""" Test type error for synapse delay
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(n0, n1, weight=1.0, delay=-5.4)

		except TypeError:
			print("test_synapse_delay completed successfully")




	def test_synapse_stdp(self):
		""" Test type error for synapse STDP
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(n0, n1, stdp_enabled=1)

		except TypeError:
			print("test_synapse_stdp completed successfully")




	def test_spike_time(self):
		""" Test type error for spike time
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		
		try:
			snn.add_spike("zero", n0, 1.0)

		except TypeError:
			print("test_spike_time completed successfully")




	def test_spike_id(self):
		""" Test type error for spike ID
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		
		try:
			snn.add_spike(0, float(n0), 1.0)

		except TypeError:
			print("test_spike_id completed successfully")




	def test_spike_value(self):
		""" Test type error for spike value
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		
		try:
			snn.add_spike(0, n0, "one")

		except TypeError:
			print("test_spike_value completed successfully")




	def test_stdp_time_steps(self):
		""" Test type error for STDP time steps
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, stdp_enabled=True)
		
		try:
			snn.stdp_setup(time_steps="forty three")

		except TypeError:
			print("test_stdp_time_steps completed successfully")




	def test_stdp_apos(self):
		""" Test type error for STDP Apos
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, stdp_enabled=True)
		
		try:
			snn.stdp_setup(time_steps=3, Apos=1.0)

		except TypeError:
			print("test_stdp_apos completed successfully")




	def test_stdp_aneg(self):
		""" Test type error for STDP Aneg
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, stdp_enabled=True)
		
		try:
			snn.stdp_setup(time_steps=3, Aneg=1.0)

		except TypeError:
			print("test_stdp_aneg completed successfully")




	def test_stdp_pos_update(self):
		""" Test type error for STDP positive update
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, stdp_enabled=True)
		
		try:
			snn.stdp_setup(time_steps=3, positive_update="False")

		except TypeError:
			print("test_stdp_pos_update completed successfully")




	def test_stdp_neg_update(self):
		""" Test type error for STDP negative update
		
		"""

		snn=SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, stdp_enabled=True)
		
		try:
			snn.stdp_setup(time_steps=3, negative_update="False")

		except TypeError:
			print("test_stdp_neg_update completed successfully")




	def test_simulate(self):
		""" Test type error for simulate
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1)

		snn.setup()
		
		try:
			snn.simulate([500])

		except TypeError:
			print("test_simulate completed successfully")





class ValueErrorTest(unittest.TestCase):
	""" Test all value errors

	"""

	def test_neuron_leak(self):
		""" Test value error for neuron leak
		
		"""

		snn = SNN()
		
		try:
			n0 = snn.create_neuron(leak=-1.0)

		except ValueError:
			print("test_neuron_leak completed successfully")




	def test_neuron_refractory_period(self):
		""" Test value error for neuron refractory period
		
		"""

		snn = SNN()
		
		try:
			n0 = snn.create_neuron(refractory_period=-1)

		except ValueError:
			print("test_neuron_refractory_period completed successfully")




	def test_synapse_pre_id(self):
		""" Test value error for synapse pre ID
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(-1, n1)

		except ValueError:
			print("test_synapse_pre_id completed successfully")




	def test_synapse_post_id(self):
		""" Test value error for synapse post ID
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(n0, -1)

		except ValueError:
			print("test_synapse_post_id completed successfully")




	def test_synapse_delay(self):
		""" Test value error for synapse delay
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		
		try:
			snn.create_synapse(n0, n1, delay=-2)

		except ValueError:
			print("test_synapse_delay completed successfully")




	def test_add_spike_time(self):
		""" Test value error for add_spike time
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2)
		
		try:
			snn.add_spike(-1, n0)

		except ValueError:
			print("test_add_spike_time completed successfully")




	def test_add_spike_neuron_id(self):
		""" Test value error for add_spike neuron ID
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2)
		
		try:
			snn.add_spike(1, -1)

		except ValueError:
			print("test_add_spike_neuron_id completed successfully")




	def test_stdp_time_steps(self):
		""" Test value error for STDP time steps
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		snn.add_spike(1, n0)
		
		try:
			snn.stdp_setup(-1)

		except ValueError:
			print("test_stdp_time_steps completed successfully")




	def test_stdp_apos_len(self):
		""" Test value error for STDP Apos
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		snn.add_spike(1, n0)
		
		try:
			snn.stdp_setup(3, [])

		except ValueError:
			print("test_stdp_apos_len completed successfully")




	def test_stdp_aneg_len(self):
		""" Test value error for STDP Aneg
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		snn.add_spike(1, n0)
		
		try:
			snn.stdp_setup(2, [1.0, 0.5], [1.0, 0.5, 0.25])

		except ValueError:
			print("test_stdp_aneg_len completed successfully")




	def test_stdp_apos_element_type(self):
		""" Test value error for Apos element type
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		snn.add_spike(1, n0)
		
		try:
			snn.stdp_setup(3, Apos=["a", "b", "c"])

		except ValueError:
			print("test_stdp_apos_element_type completed successfully")




	def test_stdp_aneg_element_type(self):
		""" Test value error for Aneg element type
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		snn.add_spike(1, n0)
		
		try:
			snn.stdp_setup(3, Aneg=["a", "b", "c"])

		except ValueError:
			print("test_stdp_aneg_element_type completed successfully")




	def test_stdp_apos_element_value(self):
		""" Test value error for STDP Apos element value
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		snn.add_spike(1, n0)
		
		try:
			snn.stdp_setup(2, Apos=[-1, -1], negative_update=False)

		except ValueError:
			print("test_stdp_apos_element_value completed successfully")




	def test_stdp_aneg_element_value(self):
		""" Test value error for Aneg element value
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
		snn.add_spike(1, n0)
		
		try:
			snn.stdp_setup(1, Apos=[1.0], Aneg=[-5.0])

		except ValueError:
			print("test_stdp_aneg_element_value completed successfully")




	def test_simulate(self):
		""" Test value error for simulate
		
		"""

		snn = SNN()
		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		snn.create_synapse(n0, n1)
		snn.add_spike(0, n0)
		snn.setup()
		
		try:
			snn.simulate(-1)

		except ValueError:
			print("test_simulate completed successfully")





class RuntimeErrorTest(unittest.TestCase):
	""" Test runtime errors

	"""

	def test_stdp_runtime(self):
		""" Test runtime error for STDP runtime
		
		"""

		snn = SNN()

		n0 = snn.create_neuron()
		n1 = snn.create_neuron()
		n2 = snn.create_neuron()
		n3 = snn.create_neuron()

		snn.create_synapse(n0, n1)
		snn.create_synapse(n0, n2)
		snn.create_synapse(n2, n3)

		snn.add_spike(0, n0)
		snn.add_spike(1, n1)
		snn.add_spike(2, n1)
		snn.add_spike(0, n2)

		try:
			snn.stdp_setup()

		except RuntimeError:
			print("test_stdp_runtime completed successfully")
		




	






if __name__ == "__main__":
	unittest.main()