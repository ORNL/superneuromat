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

    def test_neuronlist(self):
        """ Test if NeuronList works as expected """
        print("begin test_neuronlist")
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()

        assert len(snn.neurons) == 3
        assert snn.neurons[0] == a
        assert snn.neurons[1] == b
        assert snn.neurons[2] == c
        with self.assertRaises(IndexError):
            snn.neurons[3]
        assert snn.neurons[2:4]
        assert not snn.neurons[3:4]
        assert snn.neurons == [a, b, c]
        assert snn.neurons.tolist() == [a, b, c]
        assert snn.neurons[:] == [a, b, c]
        assert snn.neurons[:0:-1] == [c, b]
        assert snn.neurons[:].indices == snn.neurons.indices
        with self.assertRaises(ValueError):
            snn.neurons[0.001]  # pyright: ignore[reportArgumentType]
        with self.assertRaises(TypeError):
            snn.neurons[None]  # pyright: ignore[reportArgumentType]

    def test_neuronlistview(self):
        """ Test if NeuronListView works as expected """
        print("begin test_neuronlistview")

        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()
        d = snn.create_neuron()
        e = snn.create_neuron()

        print(snn.neurons[1:2])
        assert len(snn.neurons[1:2]) == 1
        assert snn.neurons[1:2][0] == b
        assert snn.neurons[-3:].indices == [c.idx, d.idx, e.idx]
        print(snn.neurons[a:b].indices)
        assert snn.neurons[a:b] == [a]
        assert a in snn.neurons
        assert b in snn.neurons[0:3]
        print(snn.neurons[0::2][-2:-1])
        with self.assertRaises(IndexError):
            snn.neurons[6]




if __name__ == "__main__":
    unittest.main()
