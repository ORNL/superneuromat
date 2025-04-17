import unittest
import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import NeuromorphicModel

use = 'jit'  # 'cpu' or 'jit' or 'gpu'


class LeakTest(unittest.TestCase):
    """ Test refractory period

    """

    def test_int_state_greater_than_reset_state(self):
        print("Internal state greater than reset state")

        model = NeuromorphicModel()
        model.backend = use

        n1 = model.create_neuron(threshold=10.0, leak=1.0, reset_state=-3.0)

        n1.add_spike(4, 2.0)
        n1.add_spike(5, 2)
        n1.add_spike(6, 5)
        n1.add_spike(8, 10.0)

        result = []
        for _i in range(10):
            model.simulate()
            result.append(model.neuron_states)

        expected_charge_states = [[-1.0], [-2.0], [-3.0], [-3.0], [-1.0], [0.0], [4.0], [3.0], [-3.0], [-3.0]]
        print('expect', expected_charge_states)
        print('actual', result)
        assert result == expected_charge_states
        # model.print_spike_train()
        print()

    def test_int_state_less_than_reset_state(self):
        print("Internal state lesss than reset state")

        model = NeuromorphicModel()
        model.backend = use

        n1 = model.create_neuron(threshold=10.0, leak=5.0, reset_state=-2.0)

        n1.add_spike(1, -2.0)
        n1.add_spike(2, -4.0)
        n1.add_spike(3, -6.0)
        n1.add_spike(4, -10.0)

        result = []
        for _i in range(5):
            model.simulate()
            result.append(model.neuron_states)

        expected_charge_states = [[-2.0], [-4.0], [-6.0], [-8.0], [-13.0]]
        print('expect', expected_charge_states)
        print('actual', result)
        assert result == expected_charge_states
        # model.print_spike_train()
        # print()

    def test_infinite_leak(self):
        print("Infinite leak")

        model = NeuromorphicModel()
        model.backend = use

        n1 = model.create_neuron(threshold=0.0, leak=np.inf, reset_state=0.0)
        n2 = model.create_neuron(threshold=10.0, leak=np.inf, reset_state=0.0)

        model.add_spike(1, n1, -2.0)
        model.add_spike(2, n1, -4.0)
        model.add_spike(3, n1, -6.0)
        model.add_spike(4, n1, -10.0)

        model.add_spike(1, n2, 2.0)
        model.add_spike(2, n2, 4.0)
        model.add_spike(3, n2, 6.0)
        model.add_spike(4, n2, 10.0)

        result = []
        for _i in range(5):
            model.simulate()
            result.append(model.neuron_states)

        expected_charge_states = [[0.0, 0.0], [-2.0, 2.0], [-4.0, 4.0], [-6.0, 6.0], [-10.0, 10.0]]
        print('expect', expected_charge_states)
        print('actual', result)
        assert result == expected_charge_states
        # model.print_spike_train()
        print()

    def test_zero_leak(self):
        print("Zero leak")
        model = NeuromorphicModel()

        n1 = model.create_neuron(threshold=0.0, leak=0.0, reset_state=0.0)
        n2 = model.create_neuron(threshold=10.0, leak=0.0, reset_state=0.0)

        n1.add_spike(1, -2.0)
        n1.add_spike(2, -4.0)
        n1.add_spike(3, -6.0)
        n1.add_spike(4, -10.0)

        n2.add_spike(1, 2.0)
        n2.add_spike(2, 4.0)
        n2.add_spike(3, 6.0)
        n2.add_spike(4, 10.0)

        result = []
        for _i in range(5):
            model.simulate()
            result.append(model.neuron_states)

        expected_charge_states = [[0.0, 0.0], [-2.0, 2.0], [-6.0, 6.0], [-12.0, 0.0], [-22.0, 10.0]]
        print('expect', expected_charge_states)
        print('actual', result)
        assert result == expected_charge_states
        # model.print_spike_train()
        print()

    def test_leak_before_spike(self):
        print("Leak before spike")

        model = NeuromorphicModel()
        model.backend = use

        n0 = model.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)
        n1 = model.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)

        n0.add_spike(0, 3.0)
        n1.add_spike(0, 10.0)
        n0.add_spike(1, 10.0)
        n1.add_spike(1, 12.0)

        result = []
        for _i in range(10):
            model.simulate()
            result.append(model.neuron_states)

        expected_charge_states = [
            [0.0, 0.0],
            [10.0, 12.0],
            [8.0, 10.0],
            [6.0, 8.0],
            [4.0, 6.0],
            [2.0, 4.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
        print('expect', expected_charge_states)
        print('actual', result)
        assert result == expected_charge_states
        # model.print_spike_train()
        print()


if __name__ == "__main__":
    unittest.main()
