import unittest
from copy import copy

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class ResetTest(unittest.TestCase):
    """ Test the reset function

    """

    def setUp(self):
        # Create SNN, neurons, and synapses
        snn = self.snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b, stdp_enabled=True)
        snn.create_synapse(b, a)

        snn.stdp_setup(Aneg=[-0.1, -0.05, -0.025])

    def test_reset_1(self):
        """Test reset function for ping-pong SNN"""
        snn = self.snn

        original = snn.copy()

        # Add spikes
        snn.neurons[0].add_spike(0, 1)

        # Setup and simulate
        snn.simulate(10)

        assert snn != original
        # Print SNN before reset
        print("Before reset:")
        print(snn)

        # Reset
        snn.reset()

        # Print SNN after reset
        print("After reset:")
        print(snn)

        assert snn != original
        snn.synaptic_weights = copy(original.synaptic_weights)
        assert snn == original

    def test_memoize_weights_1(self):
        """Test reset function with memoization by object reference"""
        snn = self.snn

        original = snn.copy()

        snn.memoize(snn.synaptic_weights)

        # Add spikes
        snn.neurons[0].add_spike(0, 1)

        # Setup and simulate
        snn.simulate(10)

        assert snn != original
        snn.reset()
        assert snn == original

    def test_memoize_weights_2(self):
        """Test reset function with memoization by string"""
        snn = self.snn

        original = snn.copy()

        snn.memoize("synaptic_weights")

        # Add spikes
        snn.neurons[0].add_spike(0, 1)

        # Setup and simulate
        snn.simulate(10)

        assert snn != original
        snn.reset()
        assert snn == original

    def test_memoize_weights_3(self):
        """Test reset function and clearing memoization"""
        snn = self.snn

        original = snn.copy()

        snn.memoize(snn.synaptic_weights)

        # Add spikes
        snn.neurons[0].add_spike(0, 1)

        # Setup and simulate
        snn.simulate(10)

        assert snn != original
        snn.clear_memos()
        snn.reset()
        assert snn != original

    def test_memoize_weights_4(self):
        """Test reset function and memoization of input_spikes"""
        snn = self.snn

        # Add spikes
        snn.neurons[0].add_spike(0, 1)

        snn.memoize(snn.input_spikes, snn.synaptic_weights)

        original = snn.copy()

        # Setup and simulate
        snn.simulate(10)

        assert snn != original
        snn.reset()
        assert snn == original

    def test_memoize_weights_5(self):
        """Test reset function and memoization of output spikes"""
        snn = self.snn

        original = snn.copy()

        # Add spikes
        snn.neurons[0].add_spike(0, 1)

        snn.memoize(snn.spike_train, snn.synaptic_weights)

        # Setup and simulate
        snn.simulate(10)

        assert snn != original
        snn.reset()
        assert snn == original

    def test_memoize_weights_6(self):
        """Test memoization input validation"""
        snn = self.snn

        snn.memoize(snn.synaptic_weights)
        snn.memoize(snn.enable_stdp)
        snn.unmemoize(snn.synaptic_weights)

        self.assertRaises(ValueError, snn.memoize, snn.weight_mat)

        snn.restore(snn.enable_stdp)
        self.assertRaises(ValueError, snn.restore, snn.weight_mat)
        self.assertRaises(ValueError, snn.restore, snn.synaptic_weights)
        snn.unmemoize(snn.enable_stdp)
        assert snn.memoized == {}


if __name__ == "__main__":
    unittest.main()
