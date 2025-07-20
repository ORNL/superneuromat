import numpy as np
import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN, mlist, asmlist


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

    def test_neuronlistview_modify(self):
        """ Test if NeuronListView works as expected """
        print("begin test_neuronlistview_modify")

        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()
        d = snn.create_neuron()
        e = snn.create_neuron()

        vl = snn.neurons[:2]
        vl.append(c)
        assert vl == [a, b, c]
        vl.extend([d, e])
        assert vl == [a, b, c, d, e]
        vl.insert(1, b)
        assert vl == [a, b, b, c, d, e]
        vl.remove(b)
        assert vl == [a, b, c, d, e]
        vl.remove(b)
        vl.pop()
        assert vl == [a, c, d]
        vl.pop(0)
        assert vl == [c, d]
        vl.reverse()
        assert vl == [d, c]
        vl.sort()
        assert vl == [c, d]
        vl.clear()
        assert vl == []
        vl[:] = [a, b]
        assert vl == [a, b]
        vl = snn.neurons[:]
        vl[1:2] = [b]
        assert vl == snn.neurons[:]
        vl[::2] = [b, d, e]
        print(vl)
        assert vl == [b, b, d, d, e]
        vl[0, 2] = [a, c]
        assert vl == snn.neurons[:]
        vl[:6] = []
        assert vl == []

    def test_create_listview(self):
        """ Test if creating listview works as expected """
        print("begin test_create_listview")

        empty = mlist([])
        assert empty == []

        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()

        vl = mlist([a, b])

        assert snn.neurons == [a, b]
        assert snn.neurons[:] == [a, b]

        vl2 = asmlist(vl)

        assert vl2 is vl


if __name__ == "__main__":
    unittest.main()
