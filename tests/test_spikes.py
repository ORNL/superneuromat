import unittest

import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN

use = 'cpu'


class SpikeTest(unittest.TestCase):
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

    def test_add_spike_errors(self):
        """ Test input validation for snn.add_spike()

        """

        snn = SNN()
        n0 = snn.create_neuron()

        with self.assertRaises(TypeError):
            snn.add_spike(0, float(n0), 1.0)  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.add_spike(0, n0, "one")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.add_spike(1, n0, 'heck')  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.add_spike("zero", n0, 1.0)  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.add_spike(-1, n0)

        with self.assertRaises(ValueError):
            snn.add_spike(1, -1)

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
        """Test the add_spikes exist parameter"""

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
        ], exist='overwrite')

        def raises_valueerror():
            a.add_spike(8, 1.0, exist='error')

        a.add_spike(7, 0.0, exist='dontadd')

        print(snn.input_spikes_info())

        self.assertRaises(ValueError, raises_valueerror)

        snn.simulate(10, use=use)

        snn.print_spike_train()
        expected_spike_train = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        assert np.array_equal(a.spikes, expected_spike_train)

        snn.print_spike_train()

    def test_delete_spikes(self):
        """Test the clear_input_spikes function"""

        print("begin test_delete_spikes")

        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        b.add_spikes([1.0, 1.0])
        a.add_spikes([
            (5, 1.0),
            (6, 1.0),
            (7, 1.0),
            (8, 1.0),
            (9, 1.0),
        ])
        print(snn.input_spikes_info())
        assert b.idx in snn.input_spikes[0]["nids"]

        snn.clear_input_spikes(t=slice(3, 6))
        snn.clear_input_spikes(t=8, destination=b)
        snn.clear_input_spikes(t=[7])
        snn.clear_input_spikes(t=np.array([6]), destination=[a])
        snn.clear_input_spikes(destination=b)

        def remove_empty(d: dict):
            return {k: v for k, v in d.items() if v['nids'] and v['values']}

        print(snn.input_spikes_info())
        print(snn.input_spikes)
        assert snn.input_spikes == {
            8: {"nids": [0], "values": [1.0]},
            9: {"nids": [0], "values": [1.0]},
        }

if __name__ == "__main__":
    unittest.main()
