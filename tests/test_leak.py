import unittest
import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN, print_spike_train


class LeakTest(unittest.TestCase):
    """ Test leak

    """

    use = 'cpu'
    sparse = False

    def setUp(self):
        self.snn = SNN()
        self.snn.backend = self.use
        self.snn.sparse = self.sparse

    def test_int_state_greater_than_reset_state(self):
        """Test leak when internal state is greater than reset state"""
        print("Internal state greater than reset state")

        snn = self.snn

        n1 = snn.create_neuron(threshold=10.0, leak=1.0, reset_state=-3.0)

        n1.add_spike(4, 2.0)
        n1.add_spike(5, 2)
        n1.add_spike(6, 5)
        n1.add_spike(8, 10.0)

        result = []
        for _i in range(10):
            snn.simulate()
            result.append(snn.neuron_states)

        expected_charge_states = [[-1.0], [-2.0], [-3.0], [-3.0], [-1.0], [0.0], [4.0], [3.0], [-3.0], [-3.0]]
        print('expect', expected_charge_states)
        print('actual', result)
        assert result == expected_charge_states
        snn.print_spike_train()
        print()

    def test_int_state_less_than_reset_state(self):
        """Test leak when internal state is less than reset state"""
        print("Internal state lesss than reset state")

        snn = self.snn

        n1 = snn.create_neuron(threshold=10.0, leak=5.0, reset_state=-2.0)

        n1.add_spike(1, -2.0)
        n1.add_spike(2, -4.0)
        n1.add_spike(3, -6.0)
        n1.add_spike(4, -10.0)

        result = []
        for _i in range(5):
            snn.simulate()
            result.append(snn.neuron_states)

        expected_charge_states = [[-2.0], [-4.0], [-6.0], [-8.0], [-13.0]]
        print('expect', expected_charge_states)
        print('actual', result)
        print("Spike train:")
        snn.print_spike_train()
        assert result == expected_charge_states
        assert snn.ispikes.sum() == 0
        print()

    def test_infinite_leak(self):
        """Test infinite leak"""
        print("Infinite leak")

        snn = self.snn

        n1 = snn.create_neuron(threshold=0.0, leak=np.inf, reset_state=0.0)
        n2 = snn.create_neuron(threshold=10.0, leak=np.inf, reset_state=0.0)

        snn.add_spike(1, n1, -2.0)
        snn.add_spike(2, n1, -4.0)
        snn.add_spike(3, n1, -6.0)
        snn.add_spike(4, n1, -10.0)

        snn.add_spike(1, n2, 2.0)
        snn.add_spike(2, n2, 4.0)
        snn.add_spike(3, n2, 6.0)
        snn.add_spike(4, n2, 10.0)

        result = []
        for _i in range(5):
            snn.simulate()
            result.append(snn.neuron_states)

        expected_charge_states = [[0.0, 0.0], [-2.0, 2.0], [-4.0, 4.0], [-6.0, 6.0], [-10.0, 10.0]]
        print('expect', expected_charge_states)
        print('actual', result)
        print("Spike train:")
        snn.print_spike_train()
        assert result == expected_charge_states
        assert snn.ispikes.sum() == 0
        print()

    def test_zero_leak(self):
        """Test zero leak"""
        print("Zero leak")
        snn = self.snn

        n1 = snn.create_neuron(threshold=0.0, leak=0.0, reset_state=0.0)
        n2 = snn.create_neuron(threshold=10.0, leak=0.0, reset_state=0.0)

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
            snn.simulate()
            result.append(snn.neuron_states)

        expected_charge_states = [[0.0, 0.0], [-2.0, 2.0], [-6.0, 6.0], [-12.0, 0.0], [-22.0, 10.0]]
        expected_spike_train = [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]
        print('expect', expected_charge_states)
        print('actual', result)
        print("Expected spike train:")
        print_spike_train(snn.spike_train)
        print("Actual spike train:")
        snn.print_spike_train()
        assert result == expected_charge_states
        assert snn.ispikes.tolist() == expected_spike_train
        print()

    def test_leak_before_spike(self):
        """Test leak before spike"""
        print("Leak before spike")

        snn = self.snn

        n0 = snn.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)
        n1 = snn.create_neuron(threshold=0.0, leak=2.0, refractory_period=5)

        n0.add_spike(0, 3.0)
        n1.add_spike(0, 10.0)
        n0.add_spike(1, 10.0)
        n1.add_spike(1, 12.0)

        result = []
        for _i in range(10):
            snn.simulate()
            result.append(snn.neuron_states)

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
        expected_spike_train = [
            [1, 1],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 0]
        ]
        print('expect', expected_charge_states)
        print('actual', result)
        print("Expected spike train:")
        print_spike_train(snn.spike_train)
        print("Actual spike train:")
        snn.print_spike_train()
        assert result == expected_charge_states
        assert snn.ispikes.tolist() == expected_spike_train
        print()
        print("test_leak_before_spike completed successfully")


if __name__ == "__main__":
    unittest.main()
