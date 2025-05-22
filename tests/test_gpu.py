import unittest
# import numpy as np
import test_leak
import test_stdp
import test_refractory
import test_logic_gates
import base

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class GPUTest(base.BaseTest):
    """Test GPU"""

    use = 'gpu'
    sparse = False


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
