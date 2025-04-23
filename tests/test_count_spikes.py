import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN

use = 'cpu'


class CountSpikeTest(unittest.TestCase):
    """ Test the count_spike function

    """

    def test_count_spike(self):
        """ Test the count spike function for a ping-pong SNN

        """

        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b)
        snn.create_synapse(b, a)

        snn.add_spike(0, a, 1)

        snn.simulate(10, use=use)

        assert (snn.ispikes.sum() == 10)

        print(snn)

        print("test_count_spike completed successfully")


if __name__ == "__main__":
    unittest.main()