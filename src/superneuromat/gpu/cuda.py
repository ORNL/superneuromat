import numba
import numpy as np
from numba import cuda

# Learning resources:
# https://www.olcf.ornl.gov/cuda-training-series/
# https://nyu-cds.github.io/python-gpu/
# https://nyu-cds.github.io/python-numba/05-cuda/


def disable_numba_performance_warnings():
    import os
    os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "0"
    numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    # https://stackoverflow.com/questions/70289909/what-does-it-mean-by-say-gpu-under-ultilization-due-to-low-occupancy


@cuda.jit
def outer_shared(w, a, b, enabled, pos, neg):
    """Outer product of two vectors of equal length using shared memory

    .. warning::

        In testing this was far slower than the non-shared version.

    """
    i, j = cuda.grid(2)  # row, column
    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y
    bi, bj = cuda.blockDim.x * cuda.blockIdx.x, cuda.blockDim.y * cuda.blockIdx.y
    # print(i, j)
    m, n = w.shape[0], w.shape[1]

    sA = cuda.shared.array(shape=32, dtype=np.int8)
    sB = cuda.shared.array(shape=32, dtype=np.int8)

    if ti == 0:
        v = bj + tj
        if v < n:
            sB[tj] = b[v]  # let 0th row of threads load vector b
    if ti == 1 and j < m:
        v = bi + tj
        if v < m:
            sA[tj] = a[v]  # let 1st row of threads load vector a

    cuda.syncthreads()  # wait for all threads to finish loading

    if not (i < m and j < n):  # if outside array boundary...
        return  # quit

    if enabled[i, j]:
        w[i, j] = sA[ti] & sB[tj]
        # if sA[ti] & sB[tj]:
        #     w[i, j] += pos
        # else:
        #     w[i, j] += neg


# CUDA kernel
@cuda.jit
def outer(w, a, b):
    """Outer product of two vectors"""
    i, j = cuda.grid(2)
    m, n = w.shape[0], w.shape[1]
    if i < m and j < n:
        w[i, j] = a[i] * b[j]


@cuda.jit
def stdp_update(w, a, b, enabled, pos, neg):
    """Perform STDP weight update using two timesteps of spikes.

    Parameters
    ----------
    w : np.ndarray[(int, int), np.float64]
        A square n*n matrix of weights to be updated.
    a : np.ndarray[(int,), np.int8]
        A vector of length n of spikes from a previous timestep.
    b : np.ndarray[(int,), np.int8]
        A vector of length n of spikes from the current timestep.
    enabled : np.ndarray[(int, int), np.int8]
        A square n*n matrix which acts as a mask for the weight matrix.
        If a cell is 0, the corresponding weight is not updated.
    pos : float | np.float64
        Value to increment weights by if the neurons spiked at the same time.
    neg : float | np.float64
        Value to decrement weights by if the neurons did not spike at the same time.
    """
    i, j = cuda.grid(2)  # get current thread index in weight matrix
    m, n = w.shape[0], w.shape[1]  # size of weight matrix
    if i < m and j < n:  # only update if inside array boundary
        if enabled[i, j]:  # only update if that synapse is STDP enabled
            if a[i] & b[j]:  # if both spikes are present
                w[i, j] += pos  # update corresponding synapse weight
            else:
                w[i, j] += neg


@cuda.jit
def post_synaptic(weights, prev_spikes, out):
    """Perform weight multiplication on incoming spikes.

    The number of blocks necessary is ``math.ceil(len(out) / TPB)`` (\\ ``TPB`` is threads per block).

    We annotate the arrays as ``ndarray`` for readability, but actually
    all input arrays should be CUDA Device Arrays i.e. ``cuda.to_device(...)``\\ .

    Parameters
    ----------
    weights : np.ndarray[(int, int), np.float64]
        A square n*n matrix of weights
    prev_spikes : np.ndarray[(int,), np.int8]
        A vector of length n of spikes from the previous timestep.
    out : np.ndarray[(int,), np.float64]
        Output vector of length n. The result of (weights.T @ prev_spikes) will be stored here.
    """
    i = cuda.grid(1)  # index of current thread in output vector
    rows = weights.shape[0]
    cols = weights.shape[1]
    if not i < rows:
        return  # quit if outside vector boundary

    tmp = 0
    for j in range(cols):  # sum up the weighted spikes for each neuron. j is indexing prev_spikes
        tmp += weights[j, i] * prev_spikes[j]  # note that weights are transposed (weights.T)
    out[i] = tmp  # each thread calculates one element of the output vector


@cuda.jit
def lif(
    input_spikes,
    output_spikes,
    post_synapse,
    states,
    thresholds,
    leaks,
    reset_states,
    refractory_periods,
    refractory_periods_original,
):
    """Perform LIF update on network for a single timestep.

    The number of blocks necessary is ``math.ceil(len(states) / TPB)`` (\\ ``TPB`` is threads per block).

    We annotate the arrays as ``ndarray`` for readability, but actually
    all input arrays should be CUDA Device Arrays i.e. ``cuda.to_device(...)``\\ .

    All vectors should be the same length.

    Parameters
    ----------
    input_spikes : np.ndarray[(int,), np.int8]
        Binary vector of input spikes for current timestep.
    output_spikes : np.ndarray[(int,), np.int8]
        Vector for output spikes (binary) to be put into.
        This can be used as the ``prev_spikes`` input to ``post_synaptic()`` on the next timestep.
    post_synapse : np.ndarray[(int,), np.float64]
        The result of (weights.T @ prev_spikes) from ``post_synaptic()``\\ .
    states : np.ndarray[(int,), np.float64]
        Vector of neuron charge states.
    thresholds : np.ndarray[(int,), np.float64]
    leaks : np.ndarray[(int,), np.float64]
    reset_states : np.ndarray[(int,), np.float64]
        Neuron state will be reset to this value if it spiked in the previous timestep.
    refractory_periods : np.ndarray[(int,), np.float64]
        Per-neuron refractory state.
    refractory_periods_original : np.ndarray[(int,), np.float64]
        Time steps for which a neuron should be in its refractory state after spiking.
    """
    i = cuda.grid(1)  # index of current thread. Each thread deals with one neuron.
    n = states.shape[0]  # number of neurons
    if not i < n:  # quit if outside array boundary in case ``TPB`` or ``blockDim`` is not multiple of ``n``
        return

    if states[i] > reset_states[i]:
        leaked_state = states[i] - leaks[i]
        # apply negative leak without going below reset state
        if leaked_state < reset_states[i]:
            states[i] = reset_states[i]
        else:
            states[i] = leaked_state
    elif states[i] < reset_states[i]:
        leaked_state = states[i] + leaks[i]
        # apply positive leak without going above reset state
        if leaked_state > reset_states[i]:
            states[i] = reset_states[i]
        else:
            states[i] = leaked_state

    # on each neuron, add input spikes and post-synaptic spikes (multiplied by weights)
    states[i] += input_spikes[i] + post_synapse[i]

    if refractory_periods[i] <= 0:  # if not in refractory period, check if neuron spiked
        if states[i] > thresholds[i]:  # if neuron spiked, set refractory period
            refractory_periods[i] = refractory_periods_original[i]
            output_spikes[i] = 1  # and spike
            states[i] = reset_states[i]
        else:
            output_spikes[i] = 0
    else:  # if in refractory period, decrement refractory period by one and this neuron should not spike
        refractory_periods[i] -= 1
        output_spikes[i] = 0
