import unittest
# import numpy as np
import test_leak
import test_stdp
import test_refractory
import test_logic_gates
import base

import sys
sys.path.insert(0, "../src/")


class JITTest(base.BaseTest):
    """Test JIT"""

    use = 'jit'
    sparse = False


class JITLogicGatesTest(JITTest, test_logic_gates.LogicGatesTest):
    pass


class JITRefractoryTest(JITTest, test_refractory.RefractoryTest):
    pass


class JITLeakTest(JITTest, test_leak.LeakTest):
    pass


class JITStdpTest(JITTest, test_stdp.StdpTest):
    pass


if __name__ == "__main__":
    unittest.main()
