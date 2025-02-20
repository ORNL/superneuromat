import numpy as np
from numba import jit, cuda


@cuda.jit
def outer_shared(w, a, b, enabled, pos, neg):
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
    i, j = cuda.grid(2)
    m, n = w.shape[0], w.shape[1]
    if i < m and j < n:
        w[i, j] = a[i] & b[j]


@cuda.jit
def stdp_update(w, a, b, enabled, pos, neg):
    i, j = cuda.grid(2)
    m, n = w.shape[0], w.shape[1]
    if i < m and j < n:
        if enabled[i, j]:
            if a[i] & b[j]:
                w[i, j] += pos
            else:
                w[i, j] += neg


@cuda.jit
def post_synaptic(weights, prev_spikes, out):
    i = cuda.grid(1)
    rows = weights.shape[0]
    cols = weights.shape[1]
    if not i < rows:
        return

    tmp = 0
    for j in range(cols):
        tmp += weights[j, i] * prev_spikes[j]
    out[i] = tmp


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
    i = cuda.grid(1)
    n = states.shape[0]
    if not i < n:
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

    states[i] += input_spikes[i] + post_synapse[i]

    if refractory_periods[i] <= 0:
        if states[i] > thresholds[i]:
            refractory_periods[i] = refractory_periods_original[i]
            output_spikes[i] = True
        else:
            output_spikes[i] = False
    else:
        refractory_periods[i] -= 1
        output_spikes[i] = False
