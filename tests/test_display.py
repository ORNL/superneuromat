import numpy as np
import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class DisplayTest(unittest.TestCase):
    """ Test display

    """

    use = 'cpu'
    sparse = False

    def setUp(self):
        self.snn = SNN()
        self.snn.backend = self.use
        self.snn.sparse = self.sparse

    def test_display(self):
        snn = self.snn

        print(snn)

        n0 = snn.create_neuron(threshold=-1.0, leak=2.0, refractory_period=3, reset_state=-2.0)
        n1 = snn.create_neuron(threshold=0.0, leak=1.0, refractory_period=1, reset_state=-2.0)
        n2 = snn.create_neuron(threshold=2.0, leak=0.0, refractory_period=0, reset_state=-1.0)
        n3 = snn.create_neuron(threshold=5.0, leak=np.inf, refractory_period=2, reset_state=-2.0)
        n4 = snn.create_neuron(threshold=-2.0, leak=5.0, refractory_period=1, reset_state=-2.0)

        snn.create_synapse(n0, n1)
        snn.create_synapse(n0, n2)
        snn.create_synapse(n0, n3, weight=4.0, delay=3, stdp_enabled=True)
        snn.create_synapse(n4, n2, weight=2.0, delay=2, stdp_enabled=False)
        snn.create_synapse(n2, n1, weight=30.0, delay=4, stdp_enabled=True)

        snn.add_spike(0, n2, 4.0)
        snn.add_spike(1, n1, 3.0)
        snn.add_spike(0, n3, 2.0)
        snn.add_spike(15, n3, 7.1)
        snn.add_spike(16, n1, 2.1)
        snn.add_spike(20, n4, 2.1)

        print(snn.input_spikes_info())

        print(snn)

        snn.simulate(20)

        print(snn)

        snn.simulate(1)

        print(snn)

        snn.print_spike_train()

        print('   ', snn.neuron_spike_totals())

        expected_spikes = [
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ]

        assert np.array_equal(snn.ispikes, expected_spikes)
        assert (snn.ispikes.sum() == np.sum(expected_spikes))

        # TODO: test slicing and printing of neurons and synapses lists
        # print(snn.neurons[-3:].info())
        # print(snn.synapses[-3:].info())


if __name__ == "__main__":
    unittest.main()
