import unittest
import numpy as np
from scipy.sparse import csc_array
import time

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class SparseTest(unittest.TestCase):
    """ Test sparse operations

    """

    def test_sparse_1(self):
        """ Less than 200 neurons, should default to snn.sparse = False

        """

        start = time.time()

        num_neurons = 2

        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        s = snn.create_synapse(a, b, stdp_enabled=True)

        snn.add_spike(0, a, 50.0)
        snn.add_spike(3, b, 23.5)
        snn.add_spike(1, a, 0.02)
        snn.add_spike(4, b, 0.6)

        snn.stdp_setup()
        snn.setup()
        snn.simulate(10)

        print(snn)

        end = time.time()

        assert (snn.sparse == False)

        print(f"test_sparse_1 completed in {end - start} sec")

    def test_sparse_2(self):
        """ Less than 200 neurons, explicitly making snn.sparse = True
        """

        start = time.time()

        num_neurons = 2

        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        s = snn.create_synapse(a, b, stdp_enabled=True)

        snn.add_spike(0, a, 50.0)
        snn.add_spike(3, b, 23.5)
        snn.add_spike(1, a, 0.02)
        snn.add_spike(4, b, 0.6)

        snn.stdp_setup()

        print(snn)

        snn.setup(sparse=True)
        snn.simulate(10)

        print(snn)

        end = time.time()

        assert (snn.sparse == True)

        print(f"test_sparse_2 completed in {end - start} sec")

    def test_sparse_vs_dense(self):
        """ More than 200 neurons, within sparsity threshold, explicitly making snn.sparse = False
        """

        num_neurons, sparsity = 2715, 0.006347			# Cora
        # num_neurons, sparsity = 3318, 0.004348		# Citeseer
        # num_neurons, sparsity = 19720, 0.000532 		# Pubmed

        num_spikes = num_neurons // 10
        num_simulation_time_steps = 10

        np.random.seed(42)

        synapse_ids = set()

        sparse = True
        dense = True

        # Create snn
        snn_sparse = SNN()
        snn_dense = SNN()

        # Create neurons
        for i in range(num_neurons):
            if sparse:
                snn_sparse.create_neuron(refractory_period=2)

            if dense:
                snn_dense.create_neuron(refractory_period=2)

        print("Neurons created")

        # Create synapses
        for i in range(int(num_neurons * num_neurons * sparsity)):
            pre = np.random.randint(num_neurons)
            post = np.random.randint(num_neurons)

            if (pre, post) not in synapse_ids:
                if sparse:
                    snn_sparse.create_synapse(pre, post, stdp_enabled=True)

                if dense:
                    snn_dense.create_synapse(pre, post, stdp_enabled=True)

                synapse_ids.add((pre, post))

        print("Synapses created")

        # Add spikes
        for i in range(num_spikes):
            t = np.random.randint(num_simulation_time_steps)
            n = np.random.randint(num_neurons)

            if sparse:
                snn_sparse.add_spike(t, n, 10)

            if dense:
                snn_dense.add_spike(t, n, 10)

        print("Spikes added")

        # Setup
        if sparse:
            snn_sparse.stdp_setup(positive_update=True, negative_update=True)
            snn_sparse.setup(sparse=True, dtype=64)

        if dense:
            snn_dense.stdp_setup(positive_update=True, negative_update=True)
            snn_dense.setup(sparse=False, dtype=64)

        print("Setup complete")

        # Simulate
        t1 = time.time()

        if sparse:
            snn_sparse.simulate(num_simulation_time_steps)

        t2 = time.time()

        if dense:
            snn_dense.simulate(num_simulation_time_steps)

        t3 = time.time()

        # Print
        if dense and sparse:
            print(f"Sparse simulation time: {t2 - t1} sec")
            print(f"Dense simulation time: {t3 - t2} sec")
            print(f"Sparse speedup: {(t3 - t2) / (t2 - t1)}")
            print(f"# neurons: {snn_dense.num_neurons}, #synapses: {snn_dense.num_synapses}")

            assert (np.array_equal(snn_sparse._weights.todense(), snn_dense._weights))

        elif dense:
            print(f"Dense simulation time: {t3 - t2} sec")
            print(f"# neurons: {snn_dense.num_neurons}, #synapses: {snn_dense.num_synapses}")

        else:
            print(f"Sparse simulation time: {t2 - t1} sec")
            print(f"# neurons: {snn_sparse.num_neurons}, #synapses: {snn_sparse.num_synapses}")

        print("test_sparse_vs_dense completed successfully")

    def test_sparse_stdp(self):
        """ Test sparsity during STDP updates

        """

        num_neurons = 4
        sim_time = 10

        stdp_enabled_synapses = np.array([[0, 1, 1, 0],
                                                [1, 0, 1, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 1]
                                            ])

        snn_dense = SNN()
        snn_sparse = SNN()

        for i in range(num_neurons):
            snn_dense.create_neuron()
            snn_sparse.create_neuron()

        for i in range(num_neurons):
            for j in range(num_neurons):

                if stdp_enabled_synapses[i, j]:
                    snn_dense.create_synapse(i, j, weight=0.0, stdp_enabled=True)
                    snn_sparse.create_synapse(i, j, weight=0.0, stdp_enabled=True)

                else:
                    snn_dense.create_synapse(i, j, weight=0.0, stdp_enabled=False)
                    snn_sparse.create_synapse(i, j, weight=0.0, stdp_enabled=False)

        snn_dense.add_spike(0, 0, 1.0)
        snn_dense.add_spike(0, 2, 1.0)
        snn_dense.add_spike(1, 1, 1.0)
        snn_dense.add_spike(1, 2, 1.0)
        snn_dense.add_spike(2, 3, 1.0)
        snn_dense.add_spike(3, 0, 1.0)
        snn_dense.add_spike(4, 2, 1.0)

        snn_sparse.add_spike(0, 0, 1.0)
        snn_sparse.add_spike(0, 2, 1.0)
        snn_sparse.add_spike(1, 1, 1.0)
        snn_sparse.add_spike(1, 2, 1.0)
        snn_sparse.add_spike(2, 3, 1.0)
        snn_sparse.add_spike(3, 0, 1.0)
        snn_sparse.add_spike(4, 2, 1.0)

        snn_dense.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], Aneg=[0.1, 0.05, 0.025], positive_update=True, negative_update=True)
        snn_sparse.stdp_setup(time_steps=3, Apos=[1.0, 0.5, 0.25], Aneg=[0.1, 0.05, 0.025], positive_update=True, negative_update=True)

        snn_dense.setup(sparse=False, dtype=32)
        snn_sparse.setup(sparse=True, dtype=32)

        snn_dense.simulate(sim_time)
        snn_sparse.simulate(sim_time)

        print()
        print(f"Dense Weights: {snn_dense._weights.shape}, {type(snn_dense._weights)}, {type(snn_dense._weights[0, 1])} \n{snn_dense._weights}\n")
        print(f"Sparse Weights: {snn_sparse._weights.shape}, {type(snn_sparse._weights)}, {type(snn_sparse._weights[0, 1])} \n{snn_sparse._weights.todense()}\n")

        assert (np.array_equal(snn_dense._weights, snn_sparse._weights.todense()))

        print("test_sparse_stdp completed successfully")

    def test_sparse_stdp_2(self):
        """
        """
        print("## TEST_STDP_4 ##")
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

        snn.stdp_setup(Apos=[1.0, 0.5], Aneg=[0.01, 0.005], positive_update=True, negative_update=True)

        # model.setup(sparse=True)

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
        assert np.allclose(snn.weight_mat(), np.array(expected_weights), rtol=1e-3)

        snn.print_spike_train()
        print()

        print("test_stdp_4 completed successfully")


if __name__ == "__main__":
    unittest.main()
