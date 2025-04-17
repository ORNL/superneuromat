import numpy as np
import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import NeuromorphicModel

use = 'cpu'  # 'cpu' or 'jit' or 'gpu'


class LogicGatesTest(unittest.TestCase):
    """ Test SNNs for AND and OR gate

    """

    def test_and(self):
        # AND GATE
        print("\nAND GATE")
        and_gate = NeuromorphicModel()

        # Create neurons
        a = and_gate.create_neuron(threshold=0.0)
        b = and_gate.create_neuron(threshold=0.0)
        c = and_gate.create_neuron(threshold=1.0)

        # Create synapses
        and_gate.create_synapse(a, c, weight=1.0)
        and_gate.create_synapse(b, c, weight=1.0)

        # Add spikes: [0,0]
        and_gate.add_spike(0, a, 0.0)
        and_gate.add_spike(0, b, 0.0)

        # Add spikes: [0,1]
        and_gate.add_spike(2, a, 0.0)
        and_gate.add_spike(2, b, 1.0)

        # Add spikes: [1,0]
        and_gate.add_spike(4, a, 1.0)
        and_gate.add_spike(4, b, 0.0)

        # # Add spikes: [1,1]
        and_gate.add_spike(6, a, 1.0)
        and_gate.add_spike(6, b, 1.0)

        # Setup and simulate
        and_gate.simulate(8, use=use)

        # Print spike train and neuromorphic model
        and_gate.print_spike_train()
        print(and_gate)

        expected_spike_train = [
            [0, 0, 0],  # in:  0┬0
            [0, 0, 0],  # out:  0
            [0, 1, 0],  # in:  0┬1
            [0, 0, 0],  # out:  0
            [1, 0, 0],  # in:  1┬0
            [0, 0, 0],  # out:  0
            [1, 1, 0],  # in:  1┬1
            [0, 0, 1],  # out:  1
        ]
        assert and_gate.ispikes.astype(int).tolist() == expected_spike_train

    def test_or(self):
        # OR GATE
        print("\nOR GATE")
        or_gate = NeuromorphicModel()

        # Create neurons
        a = or_gate.create_neuron()
        b = or_gate.create_neuron()
        c = or_gate.create_neuron()

        # Create synapses
        or_gate.create_synapse(a, c, weight=1.0)
        or_gate.create_synapse(b, c, weight=1.0)

        # Add spikes: [0,0]
        or_gate.add_spike(0, a, 0.0)
        or_gate.add_spike(0, b, 0.0)

        # # Add spikes: [0,11]
        or_gate.add_spike(2, a, 0.0)
        or_gate.add_spike(2, b, 1.0)

        # Add spikes: [1,0]
        or_gate.add_spike(4, a, 1.0)
        or_gate.add_spike(4, b, 0.0)

        # Add spikes: [1,1]
        or_gate.add_spike(6, a, 1.0)
        or_gate.add_spike(6, b, 1.0)

        # Setup and simulate
        # or_gate.setup()
        or_gate.simulate(8, use=use)

        # Print spike train and neuromorphic model
        or_gate.print_spike_train()
        print(or_gate)

        expected_spike_train = [
            [0, 0, 0],  # in:  0┬0
            [0, 0, 0],  # out:  0
            [0, 1, 0],  # in:  0┬1
            [0, 0, 1],  # out:  1
            [1, 0, 0],  # in:  1┬0
            [0, 0, 1],  # out:  1
            [1, 1, 0],  # in:  1┬1
            [0, 0, 1],  # out:  1
        ]
        assert or_gate.ispikes.astype(int).tolist() == expected_spike_train


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
