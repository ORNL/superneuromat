import unittest

import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN

use = 'cpu'


class CountSpikeTest(unittest.TestCase):
    """ Test the count_spike function

    """

    def test_count_spike(self):
        """Test the count spike function for a ping-pong SNN"""

        print("begin test_count_spike")

        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b)
        snn.create_synapse(b, a)

        snn.add_spike(0, a, 1)

        snn.simulate(10, use=use)

        assert (snn.ispikes.sum() == 10)

        print(snn)

        print("test_count_spike completed successfully")

    def test_send_spikes_vec(self):
        """Test the add_spikes function with a time-series of spikes"""

        print("begin test_send_spikes_vec")

        snn = SNN()

        a = snn.create_neuron()

        a.add_spikes(np.ones(10))

        print(snn.input_spikes_info())

        snn.simulate(10, use=use)

        assert (snn.ispikes.sum() == 10)

        snn.print_spike_train()

    def test_send_spikes_arr(self):
        """Test the add_spikes function with a time-series of spikes"""

        print("begin test_send_spikes_arr")

        snn = SNN()

        a = snn.create_neuron()

        a.add_spikes([
            (5, 1.0),
            (6, 1.0),
            (7, 1.0),
            (8, 1.0),
            (9, 1.0),
        ])

        print(snn.input_spikes_info())

        snn.simulate(10, use=use)

        assert (snn.ispikes.sum() == 5)

        snn.print_spike_train()

    def test_send_spikes_dup(self):
        """Test the add_spikes duplicate parameter"""

        print("begin test_send_spikes_dup")

        snn = SNN()

        a = snn.create_neuron()

        a.add_spikes([
            (5, 1.0),
            (6, 1.0),
            (7, 1.0),
            (8, 1.0),
        ])

        a.add_spikes([
            (5, 0.0),
            (6, 0.0),
        ], duplicate='overwrite')

        def raises_valueerror():
            a.add_spike(8, 1.0, duplicate='error')

        a.add_spike(7, 0.0, duplicate='dontadd')

        print(snn.input_spikes_info())

        self.assertRaises(ValueError, raises_valueerror)

        snn.simulate(10, use=use)

        snn.print_spike_train()
        expected_spike_train = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        assert np.array_equal(a.spikes, expected_spike_train)

        snn.print_spike_train()


if __name__ == "__main__":
    unittest.main()