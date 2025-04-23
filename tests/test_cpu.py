import unittest
import test_leak
import test_stdp
import test_refractory
import test_logic_gates


class CPUTest(
    test_logic_gates.LogicGatesTest,
    test_refractory.RefractoryTest,
    test_leak.LeakTest,
    test_stdp.StdpTest,
    unittest.TestCase,
):
    """Test CPU"""
    use = 'cpu'
    sparse = False

    def tearDown(self):
        assert self.snn.last_used_backend() == 'cpu'
        assert self.snn._is_sparse is False
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
