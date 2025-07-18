import unittest
import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class SynapseTest(unittest.TestCase):
    """ Test if the create_synapse functionality is working properly

    """

    def test_create_synapse_errors(self):
        """ Test input validation for create_synapse()

        """

        snn = SNN()
        n0 = snn.create_neuron()
        n1 = snn.create_neuron()

        with self.assertRaises(ValueError):
            snn.create_synapse(-1, n1)

        with self.assertRaises(ValueError):
            snn.create_synapse(n0, -1)

        with self.assertRaises(ValueError):
            snn.create_synapse(n0, n1, delay=-2)

        with self.assertRaises(ValueError):
            snn.create_synapse(-1.0, n1)  # pyright: ignore[reportArgumentType]

        with self.assertRaises(TypeError):
            snn.create_synapse(n0, [])  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.create_synapse(n0, n1, weight="something")  # pyright: ignore[reportArgumentType]

        with self.assertRaises(TypeError):
            snn.create_synapse(n0, n1, weight=snn)  # pyright: ignore[reportArgumentType]

        with self.assertRaises(ValueError):
            snn.create_synapse(n0, n1, weight=1.0, delay=-5.4)  # pyright: ignore[reportArgumentType]

        with self.assertWarns(UserWarning):
            snn.create_synapse(n0, n1, stdp_enabled='True')

    def test_multiple_synapses(self):
        """ Test if multiple synapses from a neuron to another neuron are possible.
            This test shoud throw an error as multiple synapses between the same 2 neurons are not allowed.
        """

        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b, delay=1, stdp_enabled=True)

        assert snn.num_synapses == 1

        # test exist options
        with self.assertRaises(RuntimeError):
            snn.create_synapse(a, b)

        s = snn.create_synapse(a, b, exist='dontadd')
        assert s.idx == 0
        assert snn.num_synapses == 1

        s = snn.create_synapse(a, b, weight=9, exist='overwrite')
        assert s.idx == 0
        assert snn.num_synapses == 1
        assert snn.synapses[0].weight == snn.synaptic_weights[0] == 9.0

    def test_synapse_exist_values(self):
        """ Test for input validation of create_synapse() exist parameter
        """

        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b, delay=1, stdp_enabled=True)

        # input validation
        with self.assertRaises(ValueError):
            snn.create_synapse(a, b, exist='oops')

        with self.assertRaises(ValueError):
            snn.create_synapse(a, b, delay=2, exist='overwrite')

        with self.assertRaises(TypeError):
            snn.create_synapse(a, b, exist=1)  # pyright: ignore[reportArgumentType]

    def test_get_synapses(self):
        """ Test if the get_synapses functions are working properly.

        """
        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()

        snn.create_synapse(a, b, delay=1, stdp_enabled=True)
        snn.create_synapse(a, c, delay=2, stdp_enabled=True)

        synapses = snn.get_synapses_by_pre(a)
        self.assertEqual(len(synapses), 2)

        synapses = snn.get_synapses_by_post(b)
        self.assertEqual(len(synapses), 1)

        assert snn.get_synaptic_ids_by_pre(a) == [0, 1]
        assert snn.get_synaptic_ids_by_post(b) == [0]

        assert snn.get_synaptic_ids_by_pre(c) == []

        assert snn.get_synaptic_ids_by_pre(0) == [0, 1]
        assert snn.get_synaptic_ids_by_post(1.0) == [0]  # pyright: ignore[reportArgumentType]
        self.assertRaises(TypeError, snn.get_synapses_by_pre, "I'm not an int or a Neuron")

        assert snn.get_synapse(a, b).idx == 0
        assert snn.get_synapse_id(a, b) == 0
        assert snn.get_synapse_id(0, 1) == 0
        assert snn.get_synapse_id(0.0, 1.0) == 0  # pyright: ignore[reportArgumentType]
        self.assertRaises(IndexError, snn.get_synapse, b, a)
        try:
            snn.get_synapse(b, a)
        except IndexError as e:
            print(e)
        self.assertRaises(TypeError, snn.get_synapse, a, "I'm not an int or a Neuron")
        self.assertRaises(TypeError, snn.get_synapse, "I'm not an int or a Neuron", a)

        print("test_get_synapses completed successfully")

    def test_get_synapse_accessors(self):
        """ Test if the get_synapses and SynapseListView are working properly. """
        print("begin test_get_synapse_accessors")
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()

        ab = snn.create_synapse(a, b, delay=1, stdp_enabled=True)
        ac = snn.create_synapse(a, c, delay=2, stdp_enabled=True)

        print("testing neuron.get_synapse accessors")
        assert a.incoming_synapses == []
        assert all([x == y for x, y in zip(a.incoming_synapses, snn.get_synapses_by_post(a))])
        assert all([x == y for x, y in zip(a.outgoing_synapses, snn.get_synapses_by_pre(a))])
        assert a.incoming_synapses == snn.get_synapses_by_post(a) == []
        assert a.outgoing_synapses == snn.get_synapses_by_pre(a) == [ab, ac.delay_chain_synapses[0]]
        assert b.parents == [a]
        assert a.children == [b, ac.delay_chain[1]]
        assert a.get_synapse_to(b) == ab
        assert b.get_synapse_from(a) == ab

        print("testing SynapseListView")
        print(snn.synapses)
        assert snn.synapses[:] == [ab] + ac.delay_chain_synapses
        assert snn.synapses[0:2] == [ab, ac.delay_chain_synapses[0]]
        assert snn.synapses[2:4]
        assert not snn.synapses[3:4]
        assert snn.synapses[-2:] == ac.delay_chain_synapses
        assert snn.synapses[:0:-1] == ac.delay_chain_synapses[::-1]
        assert snn.synapses[:].indices == snn.synapses.indices
        with self.assertRaises(IndexError):
            snn.synapses[6]
        with self.assertRaises(ValueError):
            snn.synapses[0.001]  # pyright: ignore[reportArgumentType]
        with self.assertRaises(TypeError):
            snn.synapses[None]  # pyright: ignore[reportArgumentType]

    def test_synapse_change_chained_delay(self):
        """ Test if error raised when changing delay on chained synapse

        """
        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        syn = snn.create_synapse(a, b, delay=2, stdp_enabled=True)  # a -> _ -> b  (hidden neuron created)
        snn.create_synapse(a, b, delay=1, stdp_enabled=True)  # this is fine because a -> b doesn't exist yet
        with self.assertRaises(ValueError):
            snn.create_synapse(syn.pre, syn.post, delay=1, stdp_enabled=True, exist='overwrite')
        with self.assertRaises(ValueError):
            syn.delay = 1

    def test_synapse_get_delay_chain(self):
        """ Test if synapse delay chain properties work as expected

        """
        print("begin test_synapse_get_delay_chain")
        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        syn = snn.create_synapse(a, b, delay=3, stdp_enabled=True)  # a -> _ -> b  (hidden neuron created)
        neuron_chain = syn.delay_chain
        synapse_chain = syn.delay_chain_synapses
        print(snn)
        print("neuron_chain: ", neuron_chain)
        # print(*(syn.info_row() for syn in synapse_chain), sep='\n')
        print("synapse_chain: ", synapse_chain)
        assert [int(n) for n in neuron_chain] == [0, 2, 3, 1]
        assert [int(s) for s in synapse_chain] == [0, 1, 2]
        assert snn.synapses[0].delay_chain == []


if __name__ == "__main__":
    unittest.main()
