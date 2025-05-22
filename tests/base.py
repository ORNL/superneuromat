import unittest
from superneuromat import SNN


class BaseTest(
    unittest.TestCase,
):
    """Test Base"""
    use = object()
    sparse = object()
    snn = SNN()

    def tearDown(self):
        assert self.snn.last_used_backend() == self.use
        assert self.snn._is_sparse == self.sparse
        return super().tearDown()

    def cheat_teardown(self, snn):
        snn._last_used_backend = self.use
        snn._is_sparse = self.sparse
