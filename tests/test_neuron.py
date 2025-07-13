import numpy as np
import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class NeuronTest(unittest.TestCase):
    """ Test all type errors

    """

    def test_create_neuron_errors(self):
        """ Test input validation for create_neuron()

        """

        snn = SNN()

        with self.assertRaises(ValueError):
            snn.create_neuron(threshold="five")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.create_neuron(leak="two")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.create_neuron(reset_state="alpha")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(TypeError):
            snn.create_neuron(refractory_period={"beta": 1})  # pyright: ignore[reportArgumentType]

    def test_accessor_create_synapse(self):
        """ Test if the create_parent, create_child functions are working properly.

        """
        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        ab = a.connect_child(b, weight=1.0, delay=1, stdp_enabled=True)
        ba = a.connect_parent(b, weight=1.0, delay=1, stdp_enabled=True)

        assert snn.num_synapses == 2
        assert (ab.pre_id, ab.post_id) == (0, 1)
        assert (ba.pre_id, ba.post_id) == (1, 0)

        print("test_accessor_create_synapse completed successfully")


if __name__ == "__main__":
    unittest.main()
