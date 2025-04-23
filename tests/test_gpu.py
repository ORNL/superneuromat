import unittest
# import numpy as np
import test_leak
import test_stdp
import test_refractory
import test_logic_gates

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class GPUTest(unittest.TestCase):
    """Test GPU"""

    use = 'gpu'
    sparse = False

    def tearDown(self):
        assert self.snn.last_used_backend() == 'gpu'
        return super().tearDown()


class GPULogicGatesTest(GPUTest, test_logic_gates.LogicGatesTest):
    pass


class GPURefractoryTest(GPUTest, test_refractory.RefractoryTest):
    pass


class GPULeakTest(GPUTest, test_leak.LeakTest):
    pass


class GPUStdpTest(GPUTest, test_stdp.StdpTest):
    pass


if __name__ == "__main__":
    unittest.main()
