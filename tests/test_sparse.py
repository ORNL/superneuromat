import unittest
import time

import numpy as np

import test_leak
import test_stdp
import test_refractory
import test_logic_gates
import base

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class SparseTest(unittest.TestCase):
    """ Test sparse operations

    """

    use = 'cpu'

    def test_sparse_1(self):
        """Test for
        Less than 200 neurons, should default to snn.sparse = False
        """

        time_taken = time.time()

        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b, stdp_enabled=True)

        snn.add_spike(0, a, 50.0)
        snn.add_spike(3, b, 23.5)
        snn.add_spike(1, a, 0.02)
        snn.add_spike(4, b, 0.6)

        snn.backend = self.use
        snn.stdp_setup()
        snn.simulate(10)

        print(snn)

        time_taken = time_taken - time.time()

        assert snn._is_sparse is False

        print(f"test_sparse_1 completed in {time_taken} sec")

    def test_sparse_2(self):
        """ Less than 200 neurons, explicitly making snn.sparse = True
        """

        time_taken = time.time()

        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b, stdp_enabled=True)

        snn.add_spike(0, a, 50.0)
        snn.add_spike(3, b, 23.5)
        snn.add_spike(1, a, 0.02)
        snn.add_spike(4, b, 0.6)

        snn.stdp_setup()
        snn.sparse = True
        snn.backend = self.use

        print(snn)

        snn.simulate(10)

        print(snn)

        time_taken = time_taken - time.time()

        assert snn._is_sparse is True

        print(f"test_sparse_2 completed in {time_taken} sec")

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

        snn_sparse.backend = self.use
        snn_dense.backend = self.use

        # Create neurons
        for _i in range(num_neurons):
            if sparse:
                snn_sparse.create_neuron(refractory_period=2)

            if dense:
                snn_dense.create_neuron(refractory_period=2)

        print("Neurons created")

        # Create synapses
        for _i in range(int(num_neurons * num_neurons * sparsity)):
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
        for _i in range(num_spikes):
            t = np.random.randint(num_simulation_time_steps)
            n = np.random.randint(num_neurons)

            if sparse:
                snn_sparse.add_spike(t, n, 10, exist='add')
                snn_sparse.sparse = True

            if dense:
                snn_dense.add_spike(t, n, 10, exist='add')
                snn_dense.sparse = False

        print("Spikes added")

        # Setup
        if sparse:
            snn_sparse.stdp_setup(positive_update=True, negative_update=True)

        if dense:
            snn_dense.stdp_setup(positive_update=True, negative_update=True)

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

        for _i in range(num_neurons):
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

        snn_dense.apos = snn_sparse.apos = [1.0, 0.5, 0.25]
        snn_dense.aneg = snn_sparse.aneg = [-0.1, -0.05, -0.025]

        snn_dense.backend = self.use
        snn_sparse.backend = self.use

        snn_dense.sparse = False
        snn_sparse.sparse = True

        snn_dense.simulate(sim_time)
        snn_sparse.simulate(sim_time)

        print()
        print(f"Dense Weights: {snn_dense._weights.shape}, {type(snn_dense._weights)}, {type(snn_dense._weights[0, 1])} \n{snn_dense._weights}\n")
        print(f"Sparse Weights: {snn_sparse._weights.shape}, {type(snn_sparse._weights)}, {type(snn_sparse._weights[0, 1])} \n{snn_sparse._weights.todense()}\n")

        assert snn_dense != snn_sparse

        snn_sparse.sparse = snn_sparse._is_sparse = False

        assert (np.array_equal(snn_dense._weights, snn_sparse._weights.todense()))
        assert snn_dense == snn_sparse

        print("test_sparse_stdp completed successfully")


class SparseBase(base.BaseTest):
    """Test JIT"""

    use = 'cpu'
    sparse = True


class SparseLogicGatesTest(SparseBase, test_logic_gates.LogicGatesTest):
    pass


class SparseRefractoryTest(SparseBase, test_refractory.RefractoryTest):
    pass


class SparseLeakTest(SparseBase, test_leak.LeakTest):
    pass


class SparseStdpTest(SparseBase, test_stdp.StdpTest):
    pass


if __name__ == "__main__":
    unittest.main()
