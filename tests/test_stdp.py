import unittest
import numpy as np
import time

import sys
sys.path.insert(0, "../")

from superneuromat import NeuromorphicModel

use = 'jit'


class StdpTest(unittest.TestCase):
    """ Test refractory period

    """

    def test_positive_update(self):
        """ 2 neuron STDP positive update
        """
        print("## TEST_POSITIVE_UPDATE ##")
        model = NeuromorphicModel()

        a = model.create_neuron()
        b = model.create_neuron()

        s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        model.add_spike(0, a, 10.0)
        model.add_spike(1, a, 10.0)
        model.add_spike(2, a, 10.0)

        model.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
        model.simulate(4, use=use)

        print(model)

        assert (s1.weight == 5.25)
        assert (s2.weight == 3.5)

        print("test_positive_update completed successfully")

    def test_negative_update(self):
        """ 2 neuron STDP negative update
        """
        print("## TEST_NEGATIVE_UPDATE ##")
        model = NeuromorphicModel()

        a = model.create_neuron()
        b = model.create_neuron()

        s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        model.add_spike(0, a, 10.0)
        model.add_spike(1, a, 10.0)
        model.add_spike(2, a, 10.0)

        model.stdp_setup(time_steps=3, Aneg=[-0.1, -0.05, -0.025], positive_update=False, negative_update=True)
        model.simulate(4)

        print(model)

        assert (s1.weight == 1.0)
        assert (s2.weight == 0.825)

        print("test_positive_update completed successfully")

    def test_positive_update_after_stdp_time_steps(self):
        """ 2 neuron STDP negative update 2
        """
        print("## TEST_POSITIVE_UPDATE_AFTER_STDP_TIME_STEPS ##")
        model = NeuromorphicModel()

        a = model.create_neuron()
        b = model.create_neuron()

        s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        model.add_spike(3, a, 10.0)
        model.add_spike(4, a, 10.0)
        model.add_spike(5, a, 10.0)

        model.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
        model.simulate(7, use=use)

        print(model)

        assert (s1.weight == 5.25)
        assert (s2.weight == 3.5)

        print("test_positive_update completed successfully")

    def test_negative_update_after_stdp_time_steps(self):
        """ 2 neuron STDP negative update
        """
        print("## TEST_NEGATIVE_UPDATE_AFTER_STDP_TIME_STEPS ##")
        model = NeuromorphicModel()

        a = model.create_neuron()
        b = model.create_neuron()

        s1 = model.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = model.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        # model.add_spike(0, a, 10.0)
        # model.add_spike(0, b, 10.0)
        # model.add_spike(1, a, 10.0)
        # model.add_spike(1, b, 10.0)
        # model.add_spike(2, a, 10.0)
        # model.add_spike(2, b, 10.0)
        model.add_spike(3, a, 10.0)
        model.add_spike(4, a, 10.0)
        model.add_spike(5, a, 10.0)

        model.stdp_setup(time_steps=3, Aneg=[-0.1, -0.05, -0.025], positive_update=False, negative_update=True)
        model.simulate(7, use=use)

        print(model)

        assert (s1.weight == 0.475)
        assert (s2.weight == 0.300)

        print("test_positive_update completed successfully")

    def test_stdp_1(self):
        """
        """
        print("## TEST_STDP_1 ##")
        start = time.time()

        model = NeuromorphicModel()

        n0 = model.create_neuron()
        n1 = model.create_neuron()
        n2 = model.create_neuron()
        n3 = model.create_neuron()
        n4 = model.create_neuron()

        model.create_synapse(n0, n0, weight=-1.5, stdp_enabled=True)
        model.create_synapse(n0, n1, weight=0.1, stdp_enabled=True)
        model.create_synapse(n2, n3, weight=0.01, stdp_enabled=True)
        model.create_synapse(n3, n2, weight=0.25, stdp_enabled=True)
        model.create_synapse(n0, n3, weight=-0.73, stdp_enabled=True)
        model.create_synapse(n0, n4, weight=10.0, stdp_enabled=True)

        model.add_spike(0, n0, 1.0)
        model.add_spike(1, n0, 2.0)
        model.add_spike(1, n1, -0.3)
        model.add_spike(2, n2, 10.0)
        model.add_spike(3, n3, 21.1)
        model.add_spike(4, n4, 12.0)

        model.stdp_setup(time_steps=20, Apos=[1.0] * 20, Aneg=[0.1] * 20, positive_update=True, negative_update=True)

        print("Synaptic weights before:")
        print(model.weight_mat())

        print("STDP enabled synapses before:")
        print(model.stdp_enabled_mat())

        if use == 'gpu':
            for _i in range(10):
                model.simulate(10000, use=use)
        else:
            model.simulate(100000, use=use)

        print("Synaptic weights after:")
        print(model.weight_mat())

        end = time.time()

        print("test_stdp_1 finished in", end - start, "seconds")
        print()

    def test_stdp_2(self):
        """
        """
        print("## TEST_STDP_2 ##")
        model = NeuromorphicModel()

        n0 = model.create_neuron()
        n1 = model.create_neuron()
        n2 = model.create_neuron()
        n3 = model.create_neuron()
        n4 = model.create_neuron()

        model.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
        model.create_synapse(n0, n1, weight=0.0, stdp_enabled=True)
        model.create_synapse(n0, n2, weight=0.0, stdp_enabled=True)
        model.create_synapse(n0, n3, weight=0.0, stdp_enabled=True)
        model.create_synapse(n0, n4, weight=0.0, stdp_enabled=True)

        model.add_spike(0, n0, 1.0)
        model.add_spike(1, n0, 1.0)
        model.add_spike(1, n1, 1.0)
        model.add_spike(2, n2, 1.0)
        model.add_spike(3, n3, 1.0)
        model.add_spike(4, n4, 1.0)

        model.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], Aneg=[-0.01, -0.005, -0.0025],
                         positive_update=True, negative_update=True)

        # model.setup()

        print("Synaptic weights before:")
        print(model.weight_mat())

        model.simulate(6, use=use)

        print("Synaptic weights after:")
        print(model.weight_mat())

        model.print_spike_train()
        print()

    def test_stdp_3(self):
        """
        """
        print("## TEST_STDP_3 ##")
        model = NeuromorphicModel()

        n0 = model.create_neuron()
        n1 = model.create_neuron()
        n2 = model.create_neuron()
        n3 = model.create_neuron()
        n4 = model.create_neuron()

        model.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
        model.create_synapse(n0, n1, weight=0.0001, stdp_enabled=True)
        model.create_synapse(n0, n2, weight=0.0001, stdp_enabled=True)
        model.create_synapse(n0, n3, weight=0.0001, stdp_enabled=True)
        model.create_synapse(n0, n4, weight=0.0001, stdp_enabled=True)

        model.add_spike(2, n0, 1.0)
        model.add_spike(3, n0, 1.0)
        model.add_spike(3, n1, 1.0)
        model.add_spike(4, n2, 1.0)
        model.add_spike(5, n3, 1.0)
        model.add_spike(6, n4, 1.0)

        model.stdp_setup(time_steps=2, Apos=[1.0, 0.5], Aneg=[-0.01, -0.005], positive_update=True, negative_update=True)

        print("Neuron states before:")
        print(model.neuron_states)

        print("Synaptic weights before:")
        print(model.weight_mat())

        model.simulate(8, use=use)

        print("Neuron states before:")
        print(model.neuron_states)

        print("Synaptic weights after:")
        print(model.weight_mat())

        model.print_spike_train()
        print()

    # def test_stdp_4(self):
    #     """
    #     """
    #     print("## TEST_STDP_4 ##")
    #     model = NeuromorphicModel()

    #     n0 = model.create_neuron()
    #     n1 = model.create_neuron()
    #     n2 = model.create_neuron()
    #     n3 = model.create_neuron()
    #     n4 = model.create_neuron()

    #     model.create_synapse(n0, n0, weight=-1.0, stdp_enabled=True)
    #     model.create_synapse(n0, n1, weight=0.0001, stdp_enabled=True)
    #     model.create_synapse(n0, n2, weight=0.0001, stdp_enabled=True)
    #     model.create_synapse(n0, n3, weight=0.0001, stdp_enabled=True)
    #     model.create_synapse(n0, n4, weight=0.0001, stdp_enabled=True)

    #     model.add_spike(2, n0, 1.0)
    #     model.add_spike(3, n0, 1.0)
    #     model.add_spike(3, n1, 1.0)
    #     model.add_spike(4, n2, 1.0)
    #     model.add_spike(5, n3, 1.0)
    #     model.add_spike(6, n4, 1.0)

    #     model.stdp_setup(time_steps=2, Apos=[1.0, 0.5], Aneg=[0.01, 0.005], positive_update=True, negative_update=True)

    #     # model.setup(sparse=True)

    #     print("Neuron states before:")
    #     print(model.neuron_states)

    #     print("Synaptic weights before:")
    #     print(model.weight_mat())

    #     model.simulate(8, use=use)

    #     print("Neuron states before:")
    #     print(model.neuron_states)

    #     print("Synaptic weights after:")
    #     print(model.weight_mat())

    #     model.print_spike_train()
    #     print()


def print_testingwith(use):
    print()
    print('#' * 24)
    print('#' * 24)
    print(f"    TESTING WITH {use.upper()}   ")
    print('#' * 24)
    print('#' * 24)
    print()


if __name__ == "__main__":
    if use != 'all':
        unittest.main()
    else:
        use = 'cpu'
        print(print_testingwith(use))
        unittest.main()

        use = 'jit'
        print(print_testingwith(use))
        unittest.main()

        use = 'gpu'
        print(print_testingwith(use))
        unittest.main()
