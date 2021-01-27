from __future__ import print_function

import numpy as np

def fib(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b

    print()


import numpy as np

def _log_forward_proba_c(n_states, X, emission_probs, delta, TPM):  # TODO not working yet
    T = len(X)
    log_alphas = np.zeros((T, n_states))  # initialize matrix with zeros

    # a0, compute first forward as dot product of initial dist and state-dependent dist
    # Each element is scaled to sum to 1 in order to handle numerical underflow
    alpha_t = delta * emission_probs[0, :]
    sum_alpha_t = np.sum(alpha_t)
    alpha_t_scaled = alpha_t / sum_alpha_t
    llk = np.log(sum_alpha_t)  # Scalar to store the log likelihood
    log_alphas[0, :] = llk + np.log(alpha_t_scaled)

    # a1 to at, compute recursively
    for t in range(1, T):
        alpha_t = np.dot(alpha_t_scaled, TPM) * emission_probs[t, :]  # Dot product of previous forward_prob, transition matrix and emmission probablitites
        sum_alpha_t = np.sum(alpha_t)

        alpha_t_scaled = alpha_t / sum_alpha_t  # Scale forward_probs to sum to 1
        llk = llk + np.log(sum_alpha_t)  # Scalar to store likelihoods
        log_alphas[t, :] = llk + np.log(alpha_t_scaled)

    return log_alphas