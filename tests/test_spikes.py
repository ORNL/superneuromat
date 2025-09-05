import unittest

import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN, mlist

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

    def test_neuronlist_ispikes(self):
        """Test the ispikes property of NeuronList"""
        print("begin test_neuronlist_ispikes")
        snn = SNN()
        inputs = mlist([snn.create_neuron() for _ in range(2)])
        outputs = mlist([snn.create_neuron() for _ in range(2)])
        inputs[0].connect_child(outputs[0], delay=3)
        inputs[0].add_spike(3, 9)
        snn.simulate(8)
        assert outputs.ispikes[6, 0]
        t, v = np.nonzero(outputs.ispikes)
        assert t[0], v[0] == (6, 0)
        expected = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        assert np.array_equal(snn.neurons.ispikes, expected)

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

    def test_neuronlist_clear_input_spikes(self):
        """Test the clear_input_spikes function"""
        print("begin test_neuronlist_clear_input_spikes")
        snn = SNN()
        for _i in range(4):
            snn.create_neuron()

        inputs = snn.neurons[2:]
        a, b = inputs
        a.add_spikes([
            (0, 1.0),
            (1, 1.0),
            (2, 1.0),
            (3, 1.0),
        ])
        b.add_spikes([1, 1])
        snn.neurons[0].add_spike(4, 9.0)
        print(snn.input_spikes_info())
        assert snn.input_spikes == {
            0: {"nids": [2, 3], "values": [1.0, 1.0]},
            1: {"nids": [2, 3], "values": [1.0, 1.0]},
            2: {"nids": [2], "values": [1.0]},
            3: {"nids": [2], "values": [1.0]},
            4: {"nids": [0], "values": [9.0]},
        }
        inputs.clear_input_spikes(destination=1)
        inputs.clear_input_spikes(t=slice(1, 3))
        assert snn.input_spikes == {
            0: {"nids": [2], "values": [1.0]},
            3: {"nids": [2], "values": [1.0]},
            4: {"nids": [0], "values": [9.0]},
        }
        inputs.clear_input_spikes(destination=[0])
        assert snn.input_spikes == {
            4: {"nids": [0], "values": [9.0]},
        }
        snn.input_spikes = {
            0: {"nids": [1, 2, 3], "values": [2.0, 5.0, 1.0]},
            1: {"nids": [2, 3], "values": [1.0, 1.0]},
            2: {"nids": [2], "values": [1.0]},
            3: {"nids": [2], "values": [1.0]},
            4: {"nids": [0], "values": [9.0]},
        }
        inputs.clear_input_spikes(destination=slice(0, 1))
        assert snn.input_spikes == {
            0: {"nids": [1, 3], "values": [2.0, 1.0]},
            1: {"nids": [3], "values": [1.0]},
            4: {"nids": [0], "values": [9.0]},
        }

    def test_neuronlist_add_spike(self):
        """Test the add_spike function"""
        print("begin test_neuronlist_add_spike")
        snn = SNN()
        for _i in range(4):
            snn.create_neuron()

        inputs = snn.neurons[2:]
        inputs.add_spike(0, 1, 2.0)
        assert snn.input_spikes == {0: {"nids": [3], "values": [2.0]}}

    def test_neuronlist_add_input_spikes(self):
        """Test the add_input_spikes function"""
        print("begin test_neuronlist_add_input_spikes")
        snn = SNN()
        for _i in range(4):
            snn.create_neuron()

        inputs = snn.neurons[2:]
        inputs.add_spikes([1, 2])
        assert snn.input_spikes == {0: {"nids": [2, 3], "values": [1.0, 2.0]}}
        inputs.add_spikes([[0, 2], [3, 4]], exist='overwrite')
        inputs.add_spikes([[0, 0], [0, 0]], exist='ignore')
        assert snn.input_spikes == {
            0: {"nids": [2, 3], "values": [0.0, 2.0]},
            1: {"nids": [2, 3], "values": [3.0, 4.0]}
        }
        snn.clear_input_spikes()
        assert snn.input_spikes == {}
        inputs.add_spikes(2, time_offset=5)
        assert snn.input_spikes == {5: {"nids": [2, 3], "values": [2.0, 2.0]}}
        with self.assertRaises(ValueError):
            inputs.add_spikes([0, 1, 2])
        with self.assertRaises(ValueError):
            inputs.add_spikes([[0, 1, 2], [3, 4, 5]], time_offset=5)


if __name__ == "__main__":
    unittest.main()
