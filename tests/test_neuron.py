import numpy as np
import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class NeuronTest(unittest.TestCase):
    """ Test all type errors

    """

    def test_create_neuron_errors(self):
        """ Test input validation for create_neuron()

        """

        snn = SNN()

        with self.assertRaises(ValueError):
            snn.create_neuron(threshold="five")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.create_neuron(leak="two")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.create_neuron(reset_state="alpha")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(TypeError):
            snn.create_neuron(refractory_period={"beta": 1})  # pyright: ignore[reportArgumentType]


if __name__ == "__main__":
    unittest.main()
