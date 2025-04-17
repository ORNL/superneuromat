import unittest
import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import NeuromorphicModel

use = 'cpu'  # 'cpu' or 'jit' or 'gpu'


class RefractoryTest(unittest.TestCase):
    """ Test refractory period

    """

    def test_refractory_one(self):
        print("One neuron refractory period test")

        model = NeuromorphicModel()

        n_id = model.create_neuron(refractory_period=2).idx

        model.add_spike(1, n_id, 1)
        model.add_spike(2, n_id, 3)
        model.add_spike(3, n_id, 4)
        model.add_spike(4, n_id, 1)

        model.simulate(10, use=use)

        model.print_spike_train()
        print()

        expected_spike_train = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        expected_spike_train = np.reshape(expected_spike_train, (-1, 1)).tolist()
        assert model.ispikes.tolist() == expected_spike_train

    def test_refractory_two(self):
        print("Two neuron refractory period test")

        model = NeuromorphicModel()

        n1 = model.create_neuron(threshold=-1.0, reset_state=-1.0, refractory_period=2)
        n2 = model.create_neuron(refractory_period=1000000)

        model.create_synapse(n1, n2, weight=2.0, delay=2)

        model.add_spike(1, n2, -1.0)
        model.add_spike(2, n1, 10.0)
        model.add_spike(3, n1, 10.0)
        model.add_spike(5, n1, 10.0)

        model.simulate(10, use=use)

        model.print_spike_train()

        expected_spike_train = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        assert model.ispikes.tolist() == expected_spike_train


def print_testingwith(use):
    print()
    print('#' * 24)
    print('#' * 24)
    print(f"    TESTING WITH {use.upper()}   ")
    print('#' * 24)
    print('#' * 24)
    print()


if __name__ == "__main__":
    print_testingwith(use)
    unittest.main()


# model = NeuromorphicModel()

# n_id = model.create_neuron(refractory_period=2)

# model.add_spike(1, n_id, 1)
# model.add_spike(2, n_id, 3)
# model.add_spike(3, n_id, 4)


# model.setup()
# model.simulate(5)

# for spike_train in model.spike_train:
# 	print(spike_train)
