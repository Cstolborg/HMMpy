# cython: language_level = 3
cimport cython
from cython cimport view
from libc.math cimport exp, log # 40x speedup using this instead of np.exp, np.log which result in python numpy calls

import numpy as np

ctypedef double dtype_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dtype_t logsumexp_cython(dtype_t[:] a, int n_states) nogil:
    cdef int i
    cdef double result = 0.0
    cdef double largest_in_a = a[0]
    for i in range(1, n_states):
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    for i in range(n_states):
        result += exp(a[i] - largest_in_a)
    return largest_in_a + log(result)


def forward_proba(int n_obs, int n_states,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_tpm,
             dtype_t[:, :] log_emission_proba,
             dtype_t[:, :] log_alphas):

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] alpha_temp = np.zeros(n_states)  # vector to store each alpha_t before multiplied with emissions

    with nogil:
        # compute first forward
        for i in range(n_states):
            log_alphas[0, i] = log_startprob[i] + log_emission_proba[0, i]

        for t in range(1, n_obs):
            for j in range(n_states):
                for i in range(n_states):
                    alpha_temp[i] = log_alphas[t - 1, i] + log_tpm[i, j]

                log_alphas[t, j] = logsumexp_cython(alpha_temp, n_states) + log_emission_proba[t, j]


def backward_proba(int n_obs, int n_states,
              dtype_t[:] log_startprob,
              dtype_t[:, :] log_tpm,
              dtype_t[:, :] log_emission_proba,
              dtype_t[:, :] log_betas):

    cdef int t, i, j
    cdef dtype_t[::view.contiguous] beta_temp = np.zeros(n_states) # vector to store each beta_t before summing it up

    with nogil:
        for i in range(n_states):
            log_betas[n_obs - 1, i] = 0.0

        for t in range(n_obs - 2, -1, -1):
            for i in range(n_states):
                for j in range(n_states):
                    beta_temp[j] = (log_tpm[i, j]
                                      + log_emission_proba[t + 1, j]
                                      + log_betas[t + 1, j])

                log_betas[t, i] = logsumexp_cython(beta_temp, n_states)