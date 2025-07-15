import unittest

import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN

use = 'cpu'


class JSONTest(unittest.TestCase):
    """ Test the JSON array export and import functions

    """

    def export_array(self, do_print=False, **kwargs):
        from superneuromat import json
        from math import pi

        a = [1.0, 2, 3.3]
        b = [True, False]
        d = {'a': a, 'b': b, 'c': [b, b, pi]}
        kwargs.setdefault('indent', 4)
        kwargs.setdefault('separators', (', ', ': '))
        j = json.dumps(d, **kwargs)
        if do_print:
            print()
            print(j)

        return d, j

    def test_export_array(self):
        """Test exporting a custom formatted array to JSON"""

        self.export_array(do_print=True)

    def test_import_array(self):
        """Test importing a custom formatted array from JSON

        verify round-trip capability, including full float precision
        """
        from superneuromat import json

        expected, j = self.export_array(do_print=False)

        d = json.loads(j)

        assert d == expected


class JSONSNNTest(unittest.TestCase):
    array_representation = "json-native"

    def export_snn(self, do_print=False, **kwargs):
        """Test exporting a SNN to JSON"""

        snn = SNN()
        import math

        a = snn.create_neuron(threshold=math.pi)
        b = snn.create_neuron()

        snn.create_synapse(a, b)
        snn.create_synapse(b, a)

        a.add_spike(0, 5)

        snn.simulate(10, use=use)

        b.add_spike(3, 1)

        s = snn.to_json(array_representation=self.array_representation, **kwargs)

        if do_print:
            print()
            print(s)

        return snn, s

    def test_export_snn_str(self):
        """Test exporting a SNN to JSON"""
        print()
        print("begin test_export_snn_str")
        self.export_snn(do_print=True)

    def test_export_snn_str_indent_none(self):
        """Test exporting a SNN to JSON with no indents"""
        print()
        print("begin test_export_snn_str_indent_none")
        self.export_snn(do_print=True, indent=None)

    def test_export_snn_str_skipkeys(self):
        """Test exporting a SNN to JSON while skipping keys"""
        print()
        print("begin test_export_snn_str_skipvars")
        _snn, s = self.export_snn(do_print=False)
        eqvars = set(SNN.eqvars)
        eqvars.discard('connection_ids')
        for key in eqvars:
            assert key in s
        _snn, s = self.export_snn(do_print=False, skipkeys=["synaptic_weights"])
        assert "synaptic_weights" not in s

    def test_export_snn_str_extra(self):
        """Test exporting a SNN to JSON with extra parameters."""
        print()
        print("begin test_export_snn_str_extra")
        _snn, s = self.export_snn(do_print=False)
        assert "foo" not in s
        assert "extra" not in s
        _snn, s = self.export_snn(do_print=False, extra={"foo": "bar"})
        print(s)
        assert "foo" in s
        assert "extra" in s

    def test_import_snn_str(self):
        """Test importing a SNN from JSON

        verify round-trip capability
        """
        print("begin test_import_snn_str")
        from superneuromat import json, SNN

        snn, s = self.export_snn(do_print=False, net_name="My SNN")

        _j = json.loads(s)

        print()
        print(_j)

        new = SNN()

        new = new.from_jsons(s)

        assert snn.__eq__(new, mismatch='raise')

        # test name
        new = SNN().from_jsons(s, net_id="My SNN")
        assert snn.__eq__(new, mismatch='raise')
        self.assertRaises(ValueError, SNN().from_jsons, s, net_id="other")
        self.assertRaises(IndexError, SNN().from_jsons, s, net_id=1)

    def test_import_snn_skipkeys(self):
        """Test importing a SNN from JSON

        verify skipping keys does not import them
        """
        print("begin test_import_snn_str")
        from superneuromat import json, SNN

        snn, s = self.export_snn(do_print=False)

        new = SNN()

        new = new.from_jsons(s, skipkeys=["synaptic_weights"])

        assert snn.synaptic_weights
        assert not new.synaptic_weights


class JSONBase85SNNTest(JSONSNNTest):
    array_representation = "base85"


class JSONBase64SNNTest(JSONSNNTest):
    array_representation = "base64"


if __name__ == "__main__":
    unittest.main()
