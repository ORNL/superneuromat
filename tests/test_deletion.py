import unittest
import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class DeletionTest(unittest.TestCase):

    def test_nocache(self):
        snn = SNN()

        snn.create_neuron(threshold=0.0)
        snn.create_neuron(threshold=1.0)
        snn.create_neuron(threshold=2.0)
        snn.create_synapse(snn.neurons[0], snn.neurons[1])

        assert not snn._neuron_cache
        assert not snn._synapse_cache

    def setup_3_3_network(self):
        snn = SNN()

        a = snn.create_neuron(threshold=0.0)
        b = snn.create_neuron(threshold=1.0)
        c = snn.create_neuron(threshold=2.0)

        ab = a.connect_child(b, weight=1.0)
        ba = b.connect_child(a, weight=2.0)
        ac = a.connect_child(c, weight=3.0)

        return snn, a, b, c, ab, ba, ac

    def test_synapse_deletion(self):

        snn, a, b, c, ab, ba, ac = self.setup_3_3_network()

        assert len(snn._neuron_cache) == 3
        assert len(snn._synapse_cache) == 3

        # print(snn)

        assert ba.idx == 1
        assert ac.idx == 2

        sl = snn.synapses[:]
        # print(sl.info())
        assert snn._synapselist_cache

        mapping = snn.delete_synapse(ab)
        # print(mapping)
        assert mapping == {1: 0, 2: 1}
        assert len(snn._synapse_cache) == 2
        assert 2 not in snn._synapse_cache
        assert ab.idx is None
        assert ba.idx == 0
        assert ac.idx == 1
        assert sl == [ba, ac]
        assert snn.synaptic_weights == [2.0, 3.0]

    def test_synapse_arr_deletion(self):

        snn, a, b, c, ab, ba, ac = self.setup_3_3_network()

        assert len(snn._neuron_cache) == 3
        assert len(snn._synapse_cache) == 3

        # print(snn)

        assert ba.idx == 1
        assert ac.idx == 2

        sl = snn.synapses[:]
        # print(sl.info())
        assert snn._synapselist_cache

        mapping = snn.delete_synapses([ab])
        # print(mapping)
        assert mapping == {1: 0, 2: 1}
        assert len(snn._synapse_cache) == 2
        assert ab.idx is None
        assert ba.idx == 0
        assert ac.idx == 1
        assert sl == [ba, ac]
        assert snn.synaptic_weights == [2.0, 3.0]

    def test_neuron_deletion(self):

        snn, a, b, c, ab, ba, ac = self.setup_3_3_network()

        neuron_mapping, synapse_mapping = snn.delete_neuron(b)

        assert len(snn._neuron_cache) == 2
        assert len(snn._synapse_cache) == 1

        assert b.idx is None

        assert a.idx == 0
        assert c.idx == 1

        assert neuron_mapping == {0: 0, 2: 1}
        assert synapse_mapping == {2: 0}

        print(snn)

        d = {neuron: neuron.idx for neuron in snn.neurons}

        snn.neurons[0].m = None

        with self.assertRaises(KeyError):
            d[snn.neurons[0]]

        assert snn.synaptic_weights == [3.0]
        assert snn.neuron_thresholds == [0.0, 2.0]

    def test_neuron_arr_deletion(self):

        snn, a, b, c, ab, ba, ac = self.setup_3_3_network()

        neuron_mapping, synapse_mapping = snn.delete_neurons([b])

        assert len(snn._neuron_cache) == 2
        assert len(snn._synapse_cache) == 1

        assert b.idx is None

        assert a.idx == 0
        assert c.idx == 1

        assert neuron_mapping == {0: 0, 2: 1}
        assert synapse_mapping == {2: 0}

        assert snn.synaptic_weights == [3.0]
        assert snn.neuron_thresholds == [0.0, 2.0]

    def test_neuron_arr_deletion_last(self):

        snn, a, b, c, ab, ba, ac = self.setup_3_3_network()

        neuron_mapping, synapse_mapping = snn.delete_neurons([c])

        assert len(snn._neuron_cache) == 2
        assert len(snn._synapse_cache) == 2

        assert neuron_mapping == {0: 0, 1: 1}
        assert synapse_mapping == {0: 0, 1: 1}

        assert snn.synaptic_weights == [1.0, 2.0]
        assert snn.neuron_thresholds == [0.0, 1.0]


if __name__ == "__main__":
    unittest.main()
