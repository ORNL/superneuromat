from collections import deque
import numpy as np
import unittest

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN, mlist, asmlist, NeuronListView
from superneuromat.accessor_classes import NeuronIterator, NeuronViewIterator


class NeuronTest(unittest.TestCase):
    """ Test all type errors

    """

    def test_neuron_accessor(self):
        """ Test if Neuron works as expected """
        print("begin test_neuron_accessor")
        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()

        assert a.idx == 0
        assert not a == 0
        assert a != b

        print(a)
        print(repr(a))

    def test_neurons_iter(self):
        """ Test if Neuron iterator works as expected """
        print("begin test_neurons_iter")
        snn = SNN()

        _neurons = mlist([snn.create_neuron(threshold=i) for i in range(3)])

        it = NeuronIterator(snn)
        li = list(it)
        assert len(li) == 3

    def test_accessor_change_model(self):
        """ Test if Neuron model change works as expected """

        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()

        snn2 = SNN()
        a2 = snn2.create_neuron()
        b2 = snn2.create_neuron()

        assert a in snn.neurons
        assert b in snn.neurons
        assert a2 in snn2.neurons
        assert b2 in snn2.neurons

        a.m = snn2
        c.m = snn2

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

    def test_accessor_create_synapse(self):
        """ Test if the create_parent, create_child functions are working properly.

        """
        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        ab = a.connect_child(b, weight=1.0, delay=1, stdp_enabled=True)
        ba = a.connect_parent(b, weight=1.0, delay=1, stdp_enabled=True)

        assert snn.num_synapses == 2
        assert (ab.pre_id, ab.post_id) == (0, 1)
        assert (ba.pre_id, ba.post_id) == (1, 0)

        print("test_accessor_create_synapse completed successfully")

    def test_neuron_properties(self):
        """ Test if neuron properties work as expected """
        print("begin test_neuron_properties")
        snn = SNN()
        a = snn.create_neuron(
            initial_state=1.2,
            threshold=10,
            refractory_state=2,
            refractory_period=3,
            reset_state=0.0,
            leak=0.05,
        )
        assert a.state == 1.2
        assert a.threshold == 10
        assert a.refractory_state == 2
        assert a.refractory_period == 3
        assert a.reset_state == 0.0
        assert a.leak == 0.05

        a.state = 0.1
        assert snn.neuron_states[0] == 0.1
        a.threshold = 0.2
        assert snn.neuron_thresholds[0] == 0.2
        a.leak = 0.3
        assert snn.neuron_leaks[0] == 0.3
        with self.assertRaises(ValueError):
            a.leak = -1.0
        snn.allow_signed_leak = True
        a.leak = -1.0
        assert snn.neuron_leaks[0] == -1.0
        a.reset_state = -0.4
        assert snn.neuron_reset_states[0] == -0.4
        a.zero_state()
        assert snn.neuron_states[0] == 0.0
        a.activate_state()
        assert snn.neuron_states[0] == a.reset_state == -0.4
        a.refractory_state = 100
        assert snn.neuron_refractory_periods_state[0] == 100
        with self.assertRaises(ValueError):
            a.refractory_state = -1
        with self.assertRaises(TypeError):
            a.refractory_state = 1.1  # pyright: ignore[reportAttributeAccessIssue]
        a.zero_refractory_period()
        assert snn.neuron_refractory_periods_state[0] == 0
        a.activate_refractory_period()
        assert snn.neuron_refractory_periods_state[0] == 3
        a.refractory_period = 10
        assert snn.neuron_refractory_periods[0] == 10
        with self.assertRaises(ValueError):
            a.refractory_period = -1
        with self.assertRaises(TypeError):
            a.refractory_period = 1.1  # pyright: ignore[reportAttributeAccessIssue]

    def test_neuronlist(self):
        """ Test if NeuronList works as expected """
        print("begin test_neuronlist")
        snn = SNN()

        assert snn.neurons == []
        assert not snn.neurons

        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()

        snn2 = SNN()
        a2 = snn2.create_neuron()

        assert snn.neurons
        assert len(snn.neurons) == 3
        assert snn.neurons[0] == a
        assert snn.neurons[1] == b
        assert snn.neurons[2] == c
        assert snn.neurons[a].idx == 0
        assert a in snn.neurons
        assert a2 not in snn.neurons
        assert 1 in snn.neurons
        assert 3 not in snn.neurons
        assert None not in snn.neurons
        with self.assertRaises(IndexError):
            snn.neurons[3]
        assert snn.neurons[2:4] == [c]
        assert snn.neurons[[1, 2]] == [b, c]  # pyright: ignore[reportArgumentType]
        assert not snn.neurons[3:4]
        assert snn.neurons != []  # make sure it's not empty (since we're subclassing list)
        assert snn.neurons is snn.neurons
        assert snn.neurons == snn.neurons
        assert snn.neurons != snn2.neurons
        assert snn.neurons == [a, b, c]
        assert snn.neurons[a] == a
        assert not (snn.neurons != [a, b, c])
        assert snn.neurons == deque([a, b, c])  # test funky non-list ordered iterable
        assert snn.neurons != {a, b, c, a2}
        assert snn.neurons != None  # noqa: E711  # note that this won't use __eq__
        assert snn.neurons.tolist() == [a, b, c]
        assert snn.neurons[:] == [a, b, c]
        assert snn.neurons == snn.neurons[:]
        assert snn.neurons[:0:-1] == [c, b]
        assert snn.neurons[:].indices == snn.neurons.indices
        with self.assertRaises(ValueError):
            snn.neurons[0.001]  # pyright: ignore[reportArgumentType]
        with self.assertRaises(TypeError):
            snn.neurons[None]  # pyright: ignore[reportArgumentType]

        print(snn.neurons)
        print(repr(snn.neurons))

    def test_neuronlistview(self):
        """ Test if NeuronListView works as expected """
        print("begin test_neuronlistview")

        snn = SNN()

        assert not snn.neurons[:]
        assert snn.neurons[:] == []
        print(snn.neurons[:])
        print(repr(snn.neurons[:]))

        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()
        d = snn.create_neuron()
        e = snn.create_neuron()

        assert snn.neurons[:]
        print(snn.neurons[1:2])
        print(snn.neurons[:].info(3))
        assert len(snn.neurons[1:2]) == 1
        assert snn.neurons[1:2][0] == b
        assert snn.neurons[-3:].indices == [c.idx, d.idx, e.idx]
        print(snn.neurons[a:b].indices)
        assert snn.neurons[a:b] == [a]
        assert a in snn.neurons
        assert b in snn.neurons[0:3]
        assert 1 in snn.neurons[0:3]
        assert 3 not in snn.neurons[0:3]
        assert None not in snn.neurons[0:3]
        print(snn.neurons[0::2][-2:-1])
        with self.assertRaises(IndexError):
            snn.neurons[6]
        assert NeuronListView(snn, [0, 4, 3, 3, 3, 1]).indices == [0, 4, 3, 3, 3, 1]
        with self.assertRaises(IndexError):
            NeuronListView(snn, [0, 1, 5])
        l1 = NeuronListView(snn, [0, 1, 2])
        l2 = mlist([a, b, c])
        l3 = asmlist([a, b, c])
        assert l1 == l2
        assert l1 == l3
        assert l1 is not l2
        assert l1 is not l3
        assert l1[[0, 1]] == [a, b]
        assert l1[[a, 1]] == [a, b]
        with self.assertRaises(TypeError):
            l1[[0, 'a']]
        assert l1[0:1] == [a]
        assert l1[0:1] != [None]
        assert l1[0:1] != None  # not iterable  # noqa: E711
        assert l1[0:1] != []
        assert l1.tolist() == [a, b, c]
        assert l1.index(a) == 0
        assert l1.index(b) == 1
        assert l1.index(2) == 2
        l4 = mlist([b, c, c, d, d, d])
        assert l4.index(c) == 1
        assert l4.index(c, start=2, stop=3) == 2
        with self.assertRaises(ValueError):
            l4.index(c, start=3)
        with self.assertRaises(ValueError):
            l4.index(d, stop=2)
        with self.assertRaises(ValueError):
            l4.index(1.2)
        with self.assertRaises(ValueError):
            a.connect_child(b)
            l1.index(snn.synapses[0])
        with self.assertRaises(ValueError):
            snn2 = SNN()
            a2 = snn2.create_neuron()
            l4.index(a2)
        assert l4.count(1) == 1
        assert l4.count(c) == 2
        assert l4.count(d) == 3
        assert l4.count(a) == 0
        assert l4.count(0) == 0
        assert l4.count(snn.synapses[0]) == 0
        assert l4.count(1.2) == 0

    def test_neuronlist_empty(self):
        """ Test if NeuronList works as expected """
        print("begin test_neuronlist_empty")
        snn = SNN()
        a = snn.create_neuron()
        empty = NeuronListView(None, [])
        with self.assertRaises(RuntimeError):
            empty.append(a)
        assert len(empty) == 0
        assert not empty
        with self.assertRaises(IndexError):
            assert not empty[0]
        print(empty)
        print(repr(empty))

    def test_neuronlistview_error(self):
        """ Test if NeuronListView works as expected """
        print("begin test_neuronlistview_error")

        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()
        li = snn.neurons[:]

        snn2 = SNN()
        b2 = snn2.create_neuron()
        li2 = snn2.neurons[:]

        assert li[0] == a
        with self.assertRaises(ValueError):
            li[0] = b2  # neuron wrong model
        with self.assertRaises(TypeError):
            li[0] = "I'm not a Neuron"
        with self.assertRaises(ValueError):
            li[0:1] = li2  # listview wrong model
        with self.assertRaises(ValueError):
            li[0:3:2] = [b]  # extended slice
        with self.assertRaises(ValueError):
            li[[0]] = []  # wrong size
        with self.assertRaises(ValueError):
            li[[3]] = [c]  # invalid index

        with self.assertRaises(ValueError):
            li.append(b2)  # neuron wrong model
        with self.assertRaises(ValueError):
            li.append("I'm not a Neuron")
        with self.assertRaises(ValueError):
            li.insert(0, b2)  # neuron wrong model
        with self.assertRaises(ValueError):
            li.insert(0, "I'm not a Neuron")
        with self.assertRaises(ValueError):
            li.extend(li2)  # listview wrong model

    def test_neuronlistview_modify(self):
        """ Test if NeuronListView works as expected """
        print("begin test_neuronlistview_modify")

        snn2 = SNN()
        a2 = snn2.create_neuron()

        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()
        c = snn.create_neuron()
        d = snn.create_neuron()
        e = snn.create_neuron()

        a.connect_child(b)

        vl = snn.neurons[:2]
        vl.append(c)
        assert vl == [a, b, c]
        vl.extend([d, e])
        assert vl == [a, b, c, d, e]
        vl.insert(1, b)
        assert vl == [a, b, b, c, d, e]
        vl.remove(b)
        assert vl == [a, b, c, d, e]
        vl.remove(b)
        with self.assertRaises(ValueError):
            vl.remove(None)
        with self.assertRaises(ValueError):
            vl.remove(a2)
        vl.pop()
        assert vl == [a, c, d]
        vl.pop(0)
        assert vl == [c, d]
        vl.reverse()
        assert vl == [d, c]
        vl.sort()
        assert vl == [c, d]
        vl.clear()
        assert vl == []
        vl[:] = [a, b]
        assert vl == [a, b]
        vl = snn.neurons[:]
        vl[1:2] = [b]
        assert vl == snn.neurons[:]
        vl[::2] = [b, d, e]
        print(vl)
        assert vl == [b, b, d, d, e]
        vl[0, 2] = [a, c]
        assert vl == snn.neurons[:]
        vl[:6] = []
        assert vl == []
        vl2 = snn.neurons[:2]
        vl[0:1] = vl2
        assert vl == vl2
        vl[0] = b
        assert vl == [b, b]
        vl[1:1] = [c]
        assert vl == [b, c, b]
        del vl[1]
        assert vl == [b, b]
        del vl[0:2]
        assert vl == []
        assert vl2 * 2 == [a, b, a, b]
        assert vl + vl2 + [c, d] == [a, b, c, d]
        assert [] + vl2 == [a, b]
        with self.assertRaises(TypeError):
            del vl[None]
        with self.assertRaises(IndexError):
            del vl[0]
        vl = snn.neurons[:4]
        vl2 = vl.copy()
        vl.sort(reverse=True)
        assert vl == [d, c, b, a]
        vl2.sort(key=lambda neuron: -neuron.idx)
        assert vl == vl2
        vl.sort()
        assert vl == [a, b, c, d]
        vl2.reverse()
        assert vl == vl2
        vl.extend(vl2)
        assert vl == vl2 + vl2
        with self.assertRaises(ValueError):
            vl.extend(snn2.neurons[:])
        with self.assertRaises(ValueError):
            vl.extend([a2])
        with self.assertRaises(ValueError):
            vl.extend(snn.synapses[:])

    def test_create_listview(self):
        """ Test if creating listview works as expected """
        print("begin test_create_listview")

        empty = mlist([])
        assert empty == []

        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()

        vl = mlist([a, b])

        assert snn.neurons == [a, b]
        assert snn.neurons[:] == [a, b]

        vl2 = asmlist(vl)

        assert vl2 is vl

        vl3 = NeuronListView(vl)
        assert vl == vl3
        assert vl is not vl3

        with self.assertRaises(ValueError):
            NeuronListView(None, [0, 1, 2])

    def test_neuron_cache(self):
        """ Test if neuron cache works as expected """
        print("begin test_neuron_cache")
        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()

        assert a is snn.neurons[0]
        assert b is snn.neurons[1]

    def test_neuron_hashing(self):
        """ Test if neuron hashing works as expected """
        print("begin test_neuron_hashing")
        snn = SNN()
        a = snn.create_neuron()
        b = snn.create_neuron()

        assert hash(a) == hash(snn.neurons[0])
        assert hash(b) == hash(snn.neurons[1])
        with self.assertRaises(IndexError):
            assert hash(a) != hash(snn.synapses[0])
        snn.create_synapse(a, b)
        assert hash(a) != hash(snn.synapses[0])

        assert a == snn.neurons[0]
        assert b == snn.neurons[1]

        assert a != b

        d = {a: 0, b: 1}
        assert d[a] == 0
        assert d[b] == 1
        assert a in d
        assert b in d

    def test_neuronlist_properties(self):
        """ Test if neuron properties work as expected """
        print("begin test_neuronlist_properties")
        snn = SNN()

        neurons = mlist([snn.create_neuron(threshold=i) for i in range(3)])
        assert neurons == snn.neurons
        assert all(neurons.thresholds == [0.0, 1.0, 2.0])

        assert all(neurons.thresholds[:2] == [0.0, 1.0])
        mask = neurons.thresholds > 1.0
        assert mask.tolist() == [False, False, True]
        with self.assertRaises(IndexError):
            neurons.thresholds[mask[:1]]

        assert str(neurons.thresholds) == str(snn.neuron_thresholds)
        assert repr(neurons.thresholds) == repr(snn.neuron_thresholds)
        assert neurons.thresholds.tolist() == snn.neuron_thresholds

    def test_neuronlist_properties_modify(self):
        """ Test if modifying neuron properties work as expected """
        print("begin test_neuronlist_properties_modify")
        snn = SNN()

        neurons = mlist([snn.create_neuron(threshold=i) for i in range(3)])
        assert all(neurons.thresholds == [0.0, 1.0, 2.0])
        neurons.thresholds -= 1
        assert all(neurons.thresholds == [-1.0, 0.0, 1.0])
        mask = neurons.thresholds > 0.0
        assert mask.tolist() == [False, False, True]
        assert neurons.thresholds[mask] == [1.0]
        assert np.array_equal(neurons.thresholds[neurons.thresholds + 1.0], [-1.0, 0.0, 1.0])

        neurons.thresholds[:2] *= 2
        assert all(neurons.thresholds == [-2.0, 0.0, 1.0])
        neurons.thresholds = np.arange(3)
        assert all(neurons.thresholds == [0.0, 1.0, 2.0])
        assert 2.0 in neurons.thresholds
        neurons.thresholds[0:9:2] = [-1, -1]
        assert all(neurons.thresholds == [-1.0, 1.0, -1.0])
        with self.assertRaises(ValueError):
            neurons.thresholds[0:9:2] = [(-1,), ()]  # broadcasting error
        neurons.thresholds[neurons.thresholds < 0.0] = [0.0, 9.0]
        assert all(neurons.thresholds == [0.0, 1.0, 9.0])
        neurons.thresholds[neurons.thresholds > 1.0] = [0.0]
        assert all(neurons.thresholds == [0.0, 1.0, 0.0])
        assert neurons.thresholds.count(0.0) == 2
        assert neurons.thresholds.index(1.0) == 1
        with self.assertRaises(ValueError):
            neurons.thresholds.index(8)
        neurons.thresholds[:] = [0, 0, 0]
        assert all(neurons.thresholds == [0.0, 0.0, 0.0])
        neurons.refractory_periods[0] = 1

        with self.assertRaises(ValueError):
            neurons.thresholds[0] = "alpha"
        with self.assertRaises(ValueError):
            neurons.refractory_periods_state[0] = -1
        with self.assertRaises(ValueError):
            neurons.refractory_periods[0] = -1
        with self.assertRaises(ValueError):
            neurons.leaks[0] = -1
        snn.allow_signed_leak = True
        neurons.leaks[0] = -1
        assert snn.neuron_leaks[0] == -1
        with self.assertRaises(TypeError):
            neurons.thresholds[0] = None

    def test_neuronlist_resets(self):
        """ Test if neuron resets work as expected """
        print("begin test_neuronlist_resets")
        snn = SNN()
        neurons = mlist([snn.create_neuron(
                initial_state=i * 2,
                reset_state=i + 1,
                refractory_state=i,
                refractory_period=i + 4,
            ) for i in range(3)])

        assert all(neurons.states == [0.0, 2.0, 4.0])
        neurons.reset_neuron_states()
        assert all(neurons.states == [1.0, 2.0, 3.0])
        neurons.zero_neuron_states()
        assert all(neurons.states == [0.0, 0.0, 0.0])

        assert all(neurons.refractory_periods_state == [0, 1, 2])
        neurons.activate_all_refractory_periods()
        assert all(neurons.refractory_periods_state == [4, 5, 6])
        neurons.zero_refractory_periods()
        assert all(neurons.refractory_periods_state == [0, 0, 0])

    def test_neuron_output_spikes(self):
        """ Test if neuron output spikes work as expected """
        print("begin test_neuron_output_spikes")
        snn = SNN()
        inputs = mlist([snn.create_neuron() for _ in range(3)])
        outputs = mlist([snn.create_neuron() for _ in range(2)])
        snn.create_synapse(inputs[0], outputs[0])
        snn.create_synapse(inputs[1], outputs[0])
        snn.create_synapse(inputs[2], outputs[1])

        inputs[0].add_spike(0, 9)
        snn.simulate(3)
        expected = [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        expected = np.asarray(expected)
        print(snn.ispikes)
        assert np.array_equal(snn.ispikes, expected)
        assert np.array_equal(outputs.ispikes, expected[:, 3:])


if __name__ == "__main__":
    unittest.main()
