import unittest
import numpy as np
import time
import base

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


def epsilon(a, b, tol=1e-12):
    return np.abs(a - b) < tol


class StdpTest(base.BaseTest):
    """ Test refractory period

    """

    use = 'cpu'
    sparse = False

    def setUp(self):
        self.snn = SNN()
        self.snn.backend = self.use
        self.snn.sparse = self.sparse

    def test_positive_update(self):
        """ 2 neuron STDP positive update
        """
        print("## TEST_POSITIVE_UPDATE ##")
        snn = self.snn

        a = snn.create_neuron()
        b = snn.create_neuron()

        s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        snn.add_spike(0, a, 10.0)
        snn.add_spike(1, a, 10.0)
        snn.add_spike(2, a, 10.0)

        snn.stdp_setup(Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
        snn.simulate(4)

        print(snn)

        assert epsilon(s1.weight, 5.25)
        assert epsilon(s2.weight, 3.5)

        print("test_positive_update completed successfully")

    def test_negative_update(self):
        """ 2 neuron STDP negative update
        """
        print("## TEST_NEGATIVE_UPDATE ##")
        snn = self.snn

        a = snn.create_neuron()
        b = snn.create_neuron()

        s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        snn.add_spike(0, a, 10.0)
        snn.add_spike(1, a, 10.0)
        snn.add_spike(2, a, 10.0)

        snn.stdp_setup(Aneg=[-0.1, -0.05, -0.025], positive_update=False, negative_update=True)
        snn.simulate(4)

        print(snn)

        assert epsilon(s1.weight, 1.0)
        assert epsilon(s2.weight, 0.825)

        print("test_negative_update completed successfully")

    def test_positive_update_after_stdp_time_steps(self):
        """ 2 neuron STDP positive update but after simulation has run for more than STDP time steps
        """
        print("## TEST_POSITIVE_UPDATE_AFTER_STDP_TIME_STEPS ##")
        snn = self.snn

        a = snn.create_neuron()
        b = snn.create_neuron()

        s1 = snn.create_synapse(a, b, weight=1.0, stdp_enabled=True)
        s2 = snn.create_synapse(b, a, weight=1.0, stdp_enabled=True)

        snn.add_spike(3, a, 10.0)
        snn.add_spike(4, a, 10.0)
        snn.add_spike(5, a, 10.0)

        snn.stdp_setup(Apos=[1.0, 0.5, 0.25], positive_update=True, negative_update=False)
        snn.simulate(7)

        print(snn)

        assert epsilon(s1.weight, 5.25)
        assert epsilon(s2.weight, 3.5)

        print("test_positive_update_after_stdp_time_steps completed successfully")

    def test_negative_update_after_stdp_time_steps(self):
        """ 2 neuron STDP negative update but after simulation has run for more than STDP time steps
        """
        print("## TEST_NEGATIVE_UPDATE_AFTER_STDP_TIME_STEPS ##")
        snn = self.snn

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
        snn.simulate(7)

        print(snn)

        assert epsilon(s1.weight, 0.475)
        assert epsilon(s2.weight, 0.300)

        print("test_negative_update_after_stdp_time_steps completed successfully")

    def test_stdp_1(self):
        """
        """
        print("## TEST_STDP_1 ##")
        start = time.time()

        snn = self.snn
        if snn.backend == 'gpu':
            # snn._last_used_backend = 'gpu'  # hack to beat this check on GPU test
            self.cheat_teardown(snn)
            raise unittest.SkipTest("Skipping long test on GPU")
        if snn.sparse:
            # snn._last_used_backend = 'cpu'  # hack to beat this check on sparse
            self.cheat_teardown(snn)
            raise unittest.SkipTest("Skipping long test on sparse")

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

        if snn.backend == 'gpu':
            for _i in range(10):
                snn.simulate(10000, use=self.use)
        else:
            snn.simulate(100000, use=self.use)

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
        assert np.allclose(snn.weight_mat(), np.array(expected_weights), rtol=1e-6)
        print()

        print("test_stdp_1 finished in", end - start, "seconds")

    def test_stdp_2(self):
        """
        """
        print("## TEST_STDP_2 ##")
        snn = self.snn

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
                         positive_update=False, negative_update=True)

        # model.setup()

        print("Synaptic weights before:")
        print(snn.weight_mat())

        snn.simulate(6)

        print("Synaptic weights after:")
        print(snn.weight_mat())

        expected_weights = [
            [-1.0775, -0.0675, -0.0725, -0.075, -0.0775],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ]
        assert np.allclose(snn.weight_mat(), np.array(expected_weights), rtol=1e-6)

        snn.print_spike_train()
        print()

        print("test_stdp_2 completed successfully")

    def test_stdp_3(self):
        """
        """
        print("## TEST_STDP_3 ##")
        snn = self.snn

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

        snn.simulate(8)

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

        assert np.allclose(snn.weight_mat(), expected_weights, rtol=1e-6)
        print()

        print("test_stdp_3 completed successfully")

    def test_alt_stdp_setup(self):
        """ 2 neuron STDP alt setup
        """
        print("## TEST_ALT_STDP_SETUP ##")
        snns = [self.snn.copy() for _ in range(4)]

        for i, snn in enumerate(snns):
            a = snn.create_neuron()
            b = snn.create_neuron()

            snn.create_synapse(a, b, weight=1.0, stdp_enabled=False)
            snn.create_synapse(b, a, weight=1.0, stdp_enabled=False)

            snn.synapses[0].stdp_enabled = 1
            snn.synapses[1].stdp_enabled = 1

            snn.add_spike(0, a, 10.0)
            snn.add_spike(1, a, 10.0)
            snn.add_spike(2, a, 10.0)

            if i & 1:
                if i == 1:
                    snn.stdp_setup()
                    snn.stdp_negative_update = False
                    print("Checking stdp_setup() works correctly")
                snn.apos = [1.0, 0.5, 0.25]
            if i & 2:
                snn.aneg = [-0.1, -0.05, -0.025]
            snn.simulate(4)

        assert epsilon(snns[0].synapses[0].weight, 1.0)
        assert epsilon(snns[0].synapses[1].weight, 1.0)

        assert epsilon(snns[1].synapses[0].weight, 5.25)
        assert epsilon(snns[1].synapses[1].weight, 3.5)

        assert epsilon(snns[2].synapses[0].weight, 1.0)
        assert epsilon(snns[2].synapses[1].weight, 0.825)

        assert epsilon(snns[3].synapses[0].weight, 5.25)
        assert epsilon(snns[3].synapses[1].weight, 3.325)

        for snn in snns:
            assert snn.last_used_backend() == self.snn.backend
            assert snn._is_sparse == self.snn.sparse
        self.snn._last_used_backend = snns[0].backend
        self.snn._is_sparse = snns[0]._is_sparse

    def test_stdp_setup_errors(self):
        """ Test input validation for stdp_setup()

        """

        snn = self.snn
        n0 = snn.create_neuron()
        n1 = snn.create_neuron()
        snn.create_synapse(n0, n1, delay=2, stdp_enabled=True)
        snn.add_spike(1, n0)

        with self.assertRaises(TypeError):
            snn.stdp_setup(-1)  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.stdp_setup([1.0, 0.5], [1.0, 0.5, 0.25])

        with self.assertRaises(ValueError):
            snn.stdp_setup(Apos=["a", "b", "c"])

        with self.assertRaises(ValueError):
            snn.stdp_setup(Aneg=["a", "b", "c"])

        with self.assertRaises(ValueError):
            snn.stdp_setup(Apos=[-1, -1], negative_update=False)

        with self.assertRaises(ValueError):
            snn.stdp_setup(Apos=[1.0], Aneg=[5.0])

        with self.assertRaises(TypeError):
            snn.stdp_setup(Apos=1.0)  # pyright: ignore[reportArgumentType]

        with self.assertRaises(TypeError):
            snn.stdp_setup(Aneg=1.0)  # pyright: ignore[reportArgumentType]

        with self.assertWarns(UserWarning):
            snn.stdp_setup(positive_update="False")

        with self.assertWarns(UserWarning):
            snn.stdp_setup(negative_update="False")

        self.cheat_teardown(snn)

    def test_stdp_no_synapses_enabled(self):
        """ Test runtime error for STDP runtime

        """

        snn = self.snn

        n0 = snn.create_neuron()
        n1 = snn.create_neuron()
        n2 = snn.create_neuron()
        n3 = snn.create_neuron()

        snn.create_synapse(n0, n1)
        snn.create_synapse(n0, n2)
        snn.create_synapse(n2, n3)

        snn.add_spike(0, n0)
        snn.add_spike(1, n1)
        snn.add_spike(2, n1)
        snn.add_spike(0, n2)

        with self.assertWarns(RuntimeWarning):
            snn.stdp_setup()

        self.cheat_teardown(snn)


if __name__ == "__main__":
    unittest.main()
