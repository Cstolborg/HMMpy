# cython: language_level = 3
cimport cython
from cython cimport view

from libc.math cimport exp, log # 40x speedup using this instead of np.exp, np.log which result in python numpy calls

ctypedef double dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dtype_t lse_cython(dtype_t[:] a, int n_states) nogil:
    cdef int i
    cdef double result = 0.0
    cdef double largest_in_a = a[0]
    for i in range(1, n_states):
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    for i in range(n_states):
        result += exp(a[i] - largest_in_a)
    return largest_in_a + log(result)


def forward(int n_samples, int n_components,
             dtype_t[:] work_buffer,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] framelogprob,
             dtype_t[:, :] fwdlattice):

    cdef int t, i, j
    #cdef dtype_t[::view.contiguous] work_buffer = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            fwdlattice[0, i] = log_startprob[i] + framelogprob[0, i]

        for t in range(1, n_samples):
            for j in range(n_components):
                for i in range(n_components):
                    work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]

                fwdlattice[t, j] = lse_cython(work_buffer, n_components) + framelogprob[t, j]


def backward(int n_samples, int n_components,
              dtype_t[:] work_buffer,
              dtype_t[:] log_startprob,
              dtype_t[:, :] log_transmat,
              dtype_t[:, :] framelogprob,
              dtype_t[:, :] bwdlattice):

    cdef int t, i, j
    #cdef dtype_t[::view.contiguous] work_buffer = np.zeros(n_components)

    with nogil:
        for i in range(n_components):
            bwdlattice[n_samples - 1, i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[i, j]
                                      + framelogprob[t + 1, j]
                                      + bwdlattice[t + 1, j])

                bwdlattice[t, i] = lse_cython(work_buffer, n_components)