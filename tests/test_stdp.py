import unittest
import numpy as np
import time

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN

use = 'cpu'  # 'cpu' or 'jit' or 'gpu'


def epsilon(a, b, tol=1e-12):
    return np.abs(a - b) < tol


class StdpTest(unittest.TestCase):
    """ Test refractory period

    """

    def test_positive_update(self):
        """ 2 neuron STDP positive update
        """
        print("## TEST_POSITIVE_UPDATE ##")
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        snn.add_spike(0, a, 10.0)
        snn.add_spike(1, a, 10.0)
        snn.add_spike(2, a, 10.0)

        snn.stdp_setup(Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
        snn.simulate(4, use=use)

        print(snn)

        assert epsilon(s1.weight, 5.25)
        assert epsilon(s2.weight, 3.5)

        print("test_positive_update completed successfully")

    def test_negative_update(self):
        """ 2 neuron STDP negative update
        """
        print("## TEST_NEGATIVE_UPDATE ##")
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        snn.add_spike(0, a, 10.0)
        snn.add_spike(1, a, 10.0)
        snn.add_spike(2, a, 10.0)

        snn.stdp_setup(Aneg=[-0.1, -0.05, -0.025], positive_update=False, negative_update=True)
        snn.simulate(4, use=use)

        print(snn)

        assert epsilon(s1.weight, 1.0)
        assert epsilon(s2.weight, 0.825)

        print("test_negative_update completed successfully")

    def test_positive_update_after_stdp_time_steps(self):
        """ 2 neuron STDP positive update but after simulation has run for more than STDP time steps
        """
        print("## TEST_POSITIVE_UPDATE_AFTER_STDP_TIME_STEPS ##")
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        snn.add_spike(3, a, 10.0)
        snn.add_spike(4, a, 10.0)
        snn.add_spike(5, a, 10.0)

        snn.stdp_setup(Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
        snn.simulate(7, use=use)

        print(snn)

        assert epsilon(s1.weight, 5.25)
        assert epsilon(s2.weight, 3.5)

        print("test_positive_update_after_stdp_time_steps completed successfully")

    def test_negative_update_after_stdp_time_steps(self):
        """ 2 neuron STDP negative update but after simulation has run for more than STDP time steps
        """
        print("## TEST_NEGATIVE_UPDATE_AFTER_STDP_TIME_STEPS ##")
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        # model.add_spike(0, a, 10.0)
        # model.add_spike(0, b, 10.0)
        # model.add_spike(1, a, 10.0)
        # model.add_spike(1, b, 10.0)
        # model.add_spike(2, a, 10.0)
        # model.add_spike(2, b, 10.0)
        snn.add_spike(3, a, 10.0)
        snn.add_spike(4, a, 10.0)
        snn.add_spike(5, a, 10.0)

        snn.stdp_setup(Aneg=[-0.1, -0.05, -0.025], positive_update=False, negative_update=True)
        snn.simulate(7, use=use)

        print(snn)

        assert epsilon(s1.weight, 0.475)
        assert epsilon(s2.weight, 0.300)

        print("test_negative_update_after_stdp_time_steps completed successfully")

    def test_stdp_1(self):
        """
        """
        print("## TEST_STDP_1 ##")
        start = time.time()

        snn = SNN()

        n0 = snn.create_neuron()
        n1 = snn.create_neuron()
        n2 = snn.create_neuron()
        n3 = snn.create_neuron()
        n4 = snn.create_neuron()

        snn.create_synapse(n0, n0, weight=-1.5, stdp_enabled=True)
        snn.create_synapse(n0, n1, weight=0.1, stdp_enabled=True)
        snn.create_synapse(n2, n3, weight=0.01, stdp_enabled=True)
        snn.create_synapse(n3, n2, weight=0.25, stdp_enabled=True)
        snn.create_synapse(n0, n3, weight=-0.73, stdp_enabled=True)
        snn.create_synapse(n0, n4, weight=10.0, stdp_enabled=True)

        snn.add_spike(0, n0, 1.0)
        snn.add_spike(1, n0, 2.0)
        snn.add_spike(1, n1, -0.3)
        snn.add_spike(2, n2, 10.0)
        snn.add_spike(3, n3, 21.1)
        snn.add_spike(4, n4, 12.0)

        snn.stdp_setup(Apos=[1.0] * 20, Aneg=[-0.1] * 20, positive_update=True, negative_update=True)

        print("Synaptic weights before:")
        print(snn.weight_mat())

        print("STDP enabled synapses before:")
        print(snn.stdp_enabled_mat())

        if use == 'gpu':
            for _i in range(10):
                snn.simulate(10000, use=use)
        else:
            snn.simulate(100000, use=use)

        print("Synaptic weights after:")
        print(snn.weight_mat())

        end = time.time()

        expected_weights = [
            [-199979.4, -199978.9, 0., -199977.53, -199963.5],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., -199977.89, 0.],
            [0., 0., -199978.75, 0., 0.],
            [0., 0., 0., 0., 0.],
        ]
        assert np.allclose(snn.weight_mat(), np.array(expected_weights), rtol=1e-3)
        print()

        print("test_stdp_1 finished in", end - start, "seconds")

    def test_stdp_2(self):
        """
        """
        print("## TEST_STDP_2 ##")
        snn = SNN()

        n0 = snn.create_neuron()
        n1 = snn.create_neuron()
        n2 = snn.create_neuron()
        n3 = snn.create_neuron()
        n4 = snn.create_neuron()

        snn.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
        snn.create_synapse(n0, n1, weight=0.0, stdp_enabled=True)
        snn.create_synapse(n0, n2, weight=0.0, stdp_enabled=True)
        snn.create_synapse(n0, n3, weight=0.0, stdp_enabled=True)
        snn.create_synapse(n0, n4, weight=0.0, stdp_enabled=True)

        snn.add_spike(0, n0, 1.0)
        snn.add_spike(1, n0, 1.0)
        snn.add_spike(1, n1, 1.0)
        snn.add_spike(2, n2, 1.0)
        snn.add_spike(3, n3, 1.0)
        snn.add_spike(4, n4, 1.0)

        snn.stdp_setup(Apos=[1.0, 0.5, 0.25], Aneg=[-0.01, -0.005, -0.0025],
                         positive_update=True, negative_update=True)

        # model.setup()

        print("Synaptic weights before:")
        print(snn.weight_mat())

        snn.simulate(6, use=use)

        print("Synaptic weights after:")
        print(snn.weight_mat())

        expected_weights = [
            [-1.0775, -0.0675, -0.0725, -0.075, -0.0775],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ]
        assert np.allclose(snn.weight_mat(), np.array(expected_weights), rtol=1e-3)

        snn.print_spike_train()
        print()

        print("test_stdp_2 completed successfully")

    def test_stdp_3(self):
        """
        """
        print("## TEST_STDP_3 ##")
        snn = SNN()

        n0 = snn.create_neuron()
        n1 = snn.create_neuron()
        n2 = snn.create_neuron()
        n3 = snn.create_neuron()
        n4 = snn.create_neuron()

        snn.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
        snn.create_synapse(n0, n1, weight=0.0001, stdp_enabled=True)
        snn.create_synapse(n0, n2, weight=0.0001, stdp_enabled=True)
        snn.create_synapse(n0, n3, weight=0.0001, stdp_enabled=True)
        snn.create_synapse(n0, n4, weight=0.0001, stdp_enabled=True)

        snn.add_spike(2, n0, 1.0)
        snn.add_spike(3, n0, 1.0)
        snn.add_spike(3, n1, 1.0)
        snn.add_spike(4, n2, 1.0)
        snn.add_spike(5, n3, 1.0)
        snn.add_spike(6, n4, 1.0)

        snn.stdp_setup(Apos=[1.0, 0.5], Aneg=[-0.01, -0.005], positive_update=True, negative_update=True)

        print("Neuron states before:")
        print(snn.neuron_states)

        print("Synaptic weights before:")
        print(snn.weight_mat())

        snn.simulate(8, use=use)

        print("Neuron states before:")
        print(snn.neuron_states)

        print("Synaptic weights after:")
        print(snn.weight_mat())

        expected_weights = [
            [-1.1, 0.9101, 0.4051, -0.0999, -0.0999],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ]

        snn.print_spike_train()
        print()

        print("test_stdp_3 completed successfully")


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
