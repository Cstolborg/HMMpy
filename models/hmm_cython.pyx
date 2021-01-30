# cython: language_level = 3
cimport cython
from cython cimport view
from libc.math cimport exp, log, INFINITY, isinf
#from numpy.math cimport expl, logl, isinf, INFINITY

import numpy as np

ctypedef double dtype_t

cdef inline int _argmax(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos

cdef inline dtype_t _max(dtype_t[:] X) nogil:
    return X[_argmax(X)]


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

def viterbi(int n_obs, int n_states,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_tpm,
             dtype_t[:, :] log_emission_proba):

    cdef int i, j, t, where_from

    cdef int[::view.contiguous] state_sequence = \
        np.empty(n_obs, dtype=np.int32)
    cdef dtype_t[:, ::view.contiguous] log_posteriors = \
        np.zeros((n_obs, n_states))
    cdef dtype_t[::view.contiguous] work_buffer = np.empty(n_states)

    with nogil:
        for i in range(n_states):
            log_posteriors[0, i] = log_startprob[i] + log_emission_proba[0, i]

        # Induction
        for t in range(1, n_obs):
            for i in range(n_states):
                for j in range(n_states):
                    work_buffer[j] = (log_tpm[j, i]
                                      + log_posteriors[t - 1, j])

                log_posteriors[t, i] = _max(work_buffer) + log_emission_proba[t, i]

        # Observation traceback
        state_sequence[n_obs - 1] = where_from = \
            _argmax(log_posteriors[n_obs - 1])

        for t in range(n_obs - 2, -1, -1):
            for i in range(n_states):
                work_buffer[i] = (log_posteriors[t, i]
                                  + log_tpm[i, where_from])

            state_sequence[t] = where_from = _argmax(work_buffer)

    return np.asarray(state_sequence)