import unittest
import test_leak
import test_stdp
import test_refractory
import test_logic_gates
from superneuromat import SNN
import base


class CPUTest(
    test_logic_gates.LogicGatesTest,
    test_refractory.RefractoryTest,
    test_leak.LeakTest,
    test_stdp.StdpTest,
    base.BaseTest,
):
    """Test CPU"""
    use = 'cpu'
    sparse = False

    def test_simulate(self):
        """ Test value error for simulate

        """

        snn = SNN()
        n0 = snn.create_neuron()
        n1 = snn.create_neuron()
        snn.create_synapse(n0, n1)
        snn.add_spike(0, n0)

        with self.assertRaises(ValueError):
            snn.simulate(-1)

        with self.assertRaises(TypeError):
            snn.simulate([500])  # pyright: ignore[reportArgumentType]

        self.cheat_teardown(self.snn)


if __name__ == "__main__":
    unittest.main()
