import numpy as np
from scipy import stats

''' TO DO NEXT:

_log_forward_probs and _log_backward_probs currently returns alphas and betas as exponential and thus not log probs.

In the _e_step, the likelihood returns zero when evaluating on real data. Find a fix to this - probably set a_t dot b_t'
alpha_t dot beta_t' must equal likelihood for all t. Check that this condition is satisfied

Something is wrong wit the scaling of alphas and betas. They keep getting smaller as t increases.

'''




class BaseHMM:

    def __init__(self, n_states, m_observables, max_iter=10, random_state=42):
        self.n_states = n_states
        self.m_observables = m_observables

        self.delta = np.ones(n_states) * (1 / n_states)  # 1,N vector
        self.T = np.ones((n_states, n_states)) * (1 / n_states)  # N X N transmission matrix
        # self.B = np.ones((n_states, m_observables)) * (1/m_observables) # N X M Emission matrix

        np.random.seed(random_state)
        self.mu = np.random.rand(n_states)
        self.std = np.random.rand(n_states)

        self.max_iter = max_iter
        self.current_iter = 0
        self.old_log_prob = -np.inf

    def P(self, x: int):
        """ Function for computing diagonal prob matrix P(x).
         Change the function depending on the type of distribution you want to evaluate"""
        diag_probs = stats.norm.pdf(x, loc=self.mu, scale=self.std)
        diag_probs = np.diag(diag_probs)
        return diag_probs

    def _log_forward_probs(self, observations, likelihood=False):
        ''' Compute forward probabilities in scaled form. Follows the method by Zucchini A.1.8 p 334. '''
        N = len(observations)
        alphas = np.zeros((N, self.n_states))  # initialize matrix with zeros

        # a0, compute first forward as dot product of initial dist and state-dependent dist
        # Each element is scaled to sum to 1 in order to handle numerical underflow
        alpha_t = self.delta @ self.P(observations[0])
        sum_alpha_t = np.sum(alpha_t)
        alpha_t_scaled = alpha_t / sum_alpha_t
        llk = np.log(sum_alpha_t)  # Scalar to store the log likelihood
        alphas[0, :] = llk + np.log(alpha_t_scaled)

        # a1 to at, compute recursively
        for t in range(1, N):
            alpha_t = alpha_t_scaled @ self.T @ self.P(observations[t])
            sum_alpha_t = np.sum(alpha_t)
            alpha_t_scaled = alpha_t / sum_alpha_t
            llk = llk + np.log(sum_alpha_t) # Scalar to store likelihoods
            alphas[t, :] = llk + np.log(alpha_t_scaled)  # alpha_t = previous likelihoods plus logarithm of regular
                                                        # alpha - RESEARCH WHY YOU ADD THE PREVIOUS LIKELIHOOD

        return np.exp(alphas)


    def _log_backward_probs(self, observations):
        ''' Compute the log of backward probabilities in scaled form. Same procedure as forward probs.'''
        N = len(observations)
        betas = np.zeros((N, self.n_states))  # initialize matrix with zeros

        beta_t = np.ones(self.n_states)
        llk = np.log(np.sum(beta_t))
        betas[-1, :] = beta_t # Last result is 1

        for t in range(N-2, -1, -1):  # Count backwards
            beta_t = self.T @ self.P(observations[t + 1]) @ beta_t
            betas[t, :] = llk + np.log(beta_t)
            sum_beta_t = np.sum(beta_t)
            beta_t = beta_t / sum_beta_t
            llk = llk + np.log(sum_beta_t)

        return np.exp(betas)

    def _e_step(self, observations):
        alphas = self._log_forward_probs(observations)
        betas = self._log_backward_probs(observations)
        lk = alphas[0] @ betas[0].T #np.sum(alphas[-1, :])  # Likelihood

        u = (alphas * betas) / lk  # Expectation of being in state j at each time point
        print(lk)
        v = np.zeros(shape=(len(observations), self.n_states, self.n_states))  # Initialize 3D array of shape t X j X k
        for t in range(1, len(observations)):
            for j in range(self.n_states):
                for k in range(self.n_states):
                    v[t, j, k] = alphas[t - 1, j] * self.T[j, k] * np.diag(self.P(observations[t]))[k] * betas[t, k]
                    v[t, j, k] /= lk
        return u, v, lk

    def _m_step(self, observations, u, v):
        observations = np.array(observations)
        f_jk = np.sum(v, axis=0)
        self.T = f_jk / np.sum(f_jk, axis=1).reshape(2, 1)  # Check if this actually sums correct and to 1 on rows

        for j in range(self.n_states):
            self.delta[j] = u[0, j] / np.sum(u[0, :])
            self.mu[j] = np.sum(u[:, j] * observations / np.sum(u[:, j]))
            self.std[j] = np.sqrt(np.sum(u[:, j] * (observations - self.mu[j]) ** 2) / np.sum(u[:, j]))

    def em(self, observations, epochs):
        lk = 0
        for i in range(epochs):
            print(i)
            print('MEAN: ', self.mu)
            print('STD: ', self.std)
            print('DELTA', self.delta)
            print('likelihood', lk)

            print('.' * 40)
            u, v, lk = self._e_step(observations)
            self._m_step(observations, u, v)




if __name__ == '__main__':
    hmm = BaseHMM(2, 3)

    # Simulate data for bull and bear market
    import matplotlib.pyplot as plt
    N = 150
    bull_mean = 0.1
    bull_std = np.sqrt(0.1)
    bear_mean = -0.05
    bear_std = np.sqrt(0.2)

    np.random.seed(42)
    market_bull_1 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
    market_bear_2 = stats.norm.rvs(loc=bear_mean, scale=bear_std, size=N)
    market_bull_3 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)
    market_bear_4 = stats.norm.rvs(loc=bear_mean, scale=bear_std, size=N)
    market_bull_5 = stats.norm.rvs(loc=bull_mean, scale=bull_std, size=N)

    obs = np.array([market_bull_1]+ [market_bear_2]+ [market_bull_3]+ [market_bear_4]+ [market_bull_5]).flatten()
    plt.plot(obs)
    plt.show()
    print(np.shape(obs))
    print(obs[:10])

    #b = hmm._log_backward_probs(obs)
    #a = hmm._log_forward_probs(obs)
    hmm.em(obs, epochs=5)




else:  # Don't want to print this right now.....
    hmm = BaseHMM(2, 3)
    print('mu, std: ', hmm.mu, hmm.std)
    print('A matrix: ', hmm.T)
    # print('B matrix: ', hmm.B)
    print('delta vector: ', hmm.delta)

    x = 0
    print('P(X): ', hmm.P(x))
    print('Shape of P(X): ', np.shape(hmm.P(x)))

    obs = [0, 1, 2]
    print('-' * 50)
    print('Alphas: ', hmm._log_forward_probs(obs))
    # print('Likelihood: ', hmm._likelihood(obs))
    print('Likelihood: ', hmm._log_forward_probs(obs, likelihood=True))

    print('Betas: ', hmm._log_backward_probs(obs))


def backward_pass(Gamma, D, B):
    return np.dot(Gamma, D).dot(B)


def init_dist(u_j):
    ''' Compute the intial distribution delta_j '''
    return u_j


def trans_prob(v_jk):
    f_jk = np.sum(v_jk[1:])
    return f_jk / np.sum(f_jk)


def state_dist(x, u_j):
    mu = np.sum(u_j * x) / np.sum(u_j)  # Should perhaps be a dot product
    var = np.sum(u_j * (x - mu) ** 2) / np.sum(u_j)
    return mu, var

    def _likelihood(self, observations):
        alphas = self._log_forward_probs(observations)
        wt = np.sum(alphas, axis=1)
        print(wt)
        lt = []
        for t in range(1, len(observations)):
            lt.append(np.log(wt[t] / wt[t - 1]))

        return np.exp(np.sum(lt))  # alphas[-1].sum()