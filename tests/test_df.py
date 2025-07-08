import numpy as np
import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class DisplayPandasTest(unittest.TestCase):
    """ Test display

    """

    use = 'cpu'
    sparse = False

    def setUp(self):
        self.snn = SNN()
        self.snn.backend = self.use
        self.snn.sparse = self.sparse

    def test_display_pandas(self):
        snn = self.snn

        n0 = snn.create_neuron(threshold=-1.0, leak=2.0, refractory_period=3, reset_state=-2.0)
        n1 = snn.create_neuron(threshold=0.0, leak=1.0, refractory_period=1, reset_state=-2.0)
        n2 = snn.create_neuron(threshold=2.0, leak=0.0, refractory_period=0, reset_state=-1.0)
        n3 = snn.create_neuron(threshold=5.0, leak=np.inf, refractory_period=2, reset_state=-2.0)
        n4 = snn.create_neuron(threshold=-2.0, leak=5.0, refractory_period=1, reset_state=-2.0)

        snn.create_synapse(n0, n1)
        snn.create_synapse(n0, n2)
        snn.create_synapse(n0, n3, weight=4.0, delay=3, stdp_enabled=True)
        snn.create_synapse(n4, n2, weight=2.0, delay=2, stdp_enabled=False)
        snn.create_synapse(n2, n1, weight=30.0, delay=4, stdp_enabled=True)

        apos = [1.0, 0.5, 0.25]
        aneg = [-0.1, -0.05, -0.025]

        print("Input spikes before adding:")
        print(snn.get_input_spikes_df().to_string())

        snn.add_spike(0, n2, 4.0)
        snn.add_spike(1, n1, 3.0)
        snn.add_spike(0, n3, 2.0)
        snn.add_spike(15, n3, 7.1)
        snn.add_spike(16, n1, 2.1)
        snn.add_spike(20, n4, 2.1)

        print("Input spikes after adding:")
        print(snn.get_input_spikes_df().to_string())
        print("Neuron info:")
        print(snn.get_neuron_df().to_string())
        print("Synapse info:")
        print(snn.get_synapse_df().to_string())
        print("STDP enabled:")
        print(snn.get_stdp_enabled_df().to_string())
        print("Weights:")
        print(snn.get_weights_df().to_string())

        snn.simulate(21)

        print("Input spikes after simulation:")
        print(snn.get_input_spikes_df().to_string())
        print("Neuron info:")
        print(snn.get_neuron_df().to_string())
        print("Synapse info:")
        print(snn.get_synapse_df().to_string())
        print("Weights:")
        print(snn.get_weights_df().to_string())

if __name__ == "__main__":
    unittest.main()
