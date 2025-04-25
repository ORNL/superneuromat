import unittest
import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


npfloat_attrs = [
        # post_synapse = cuda.to_device(np.zeros(self.num_neurons, self.dd))
        '_internal_states',
        '_neuron_thresholds',
        '_neuron_leaks',
        '_neuron_reset_states',
        '_neuron_refractory_periods',
        '_neuron_refractory_periods_original',
        '_weights',
        '_stdp_Apos',
        '_stdp_Aneg',
        '_input_spikes',
]

boollike_attrs = [
    '_stdp_enabled_synapses',
    '_spikes',
]


def verify_dtypes(snn, float_dtypes, boollike_dtypes):
    for attr in npfloat_attrs:
        arr = getattr(snn, attr)
        assert arr.dtype in float_dtypes, f"{attr}: is {arr.dtype} instead of {float_dtypes}"
    for attr in boollike_attrs:
        arr = getattr(snn, attr)
        assert arr.dtype in boollike_dtypes, f"{attr}: is {arr.dtype} instead of {boollike_dtypes}"


class DtypeTest(unittest.TestCase):
    """ Test if the datatypes are working properly

    """

    def setUp(self):
        # Create SNN, neurons, and synapses
        snn = SNN()

        a = snn.create_neuron()
        b = snn.create_neuron()

        snn.create_synapse(a, b, stdp_enabled=True)
        snn.create_synapse(b, a)

        # Add spikes
        snn.add_spike(0, a, 1)

        # Setup and simulate
        snn.stdp_setup(Aneg=[-0.1, -0.05, -0.025])

        self.snn = snn
        return super().setUp()

    def test_dtype_64(self):
        """ Test if data types are working properly for ping-pong SNN with double precision

        """

        # Create SNN, neurons, and synapses
        snn = self.snn

        snn.simulate(10)

        # Print SNN after simulation
        print(snn)

        # Assertions
        assert snn._do_stdp is True
        assert snn.default_dtype == snn.dd == np.float64
        assert snn.default_bool_dtype in [bool, np.bool_, np.int8]
        verify_dtypes(snn, [np.float64], [bool, np.bool_, np.int8])

        print("test_dtype64 completed successfully")

    def test_dtype_32(self):
        """ Test if data types are working properly for ping-pong SNN with single precision

        """

        # Create SNN, neurons, and synapses
        snn = self.snn

        snn.default_dtype = np.float32
        snn.default_bool_dtype = np.int32
        # Setup and simulate
        snn.simulate(10)

        # Print SNN after simulation
        print(snn)

        # Assertions
        assert snn.default_dtype == snn.dd == np.float32
        assert snn.default_bool_dtype == snn.dbin == np.int32
        verify_dtypes(snn, [np.float32], [np.int32])

        print("test_dtype32 completed successfully")

    def test_dtype_32_with_reset(self):
        """ Test if data types are working properly for ping-pong SNN with single precision with reset

        """

        # Create SNN, neurons, and synapses
        snn = self.snn

        snn.default_dtype = np.float32
        snn.default_bool_dtype = np.int32
        snn.simulate(10)

        # Reset
        snn.reset()

        # Print SNN after simulation
        print(snn)

        # Assertions
        assert snn.default_dtype == snn.dd == np.float32
        assert snn.default_bool_dtype == snn.dbin == np.int32
        verify_dtypes(snn, [np.float32], [np.int32])

        print("test_dtype32_with_reset completed successfully")

    def test_dtype_32_sparse(self):
        """ Test if data types are working properly for ping-pong SNN with single precision with sparse operations enabled

        """
        snn = self.snn

        snn.default_dtype = np.float32
        snn.default_bool_dtype = np.int32
        snn.sparse = True
        snn.simulate(10)

        # Print SNN after simulation
        print(snn)

        # Assertions
        assert snn.default_dtype == snn.dd == np.float32
        assert snn.default_bool_dtype == snn.dbin == np.int32
        verify_dtypes(snn, [np.float32], [np.int32])

        print("test_dtype32_sparse completed successfully")


if __name__ == "__main__":
    unittest.main()
