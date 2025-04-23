import unittest
# import numpy as np
import test_leak
import test_stdp
import test_refractory
import test_logic_gates

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class JITTest(unittest.TestCase):
    """Test JIT"""

    use = 'jit'
    sparse = False

    def tearDown(self):
        assert self.snn.last_used_backend() == 'jit'
        return super().tearDown()


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
