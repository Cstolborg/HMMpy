import numpy as np
from scipy import stats

''' TODO NEXT:

_log_forward_probs and _log_backward_probs currently returns alphas and betas as exponential and thus not log probs.

In the _e_step, the likelihood returns zero when evaluating on real data. Find a fix to this - probably set a_t dot b_t'
alpha_t dot beta_t' must equal likelihood for all t. Check that this condition is satisfied

Something is wrong wit the scaling of alphas and betas. They keep getting smaller as t increases.


Show why Zuchinni/Nystrups algorithm equals the scaling on p. 48 in the Zucchini book.
    - Derive in math how to compute log (alpha)

'''




class BaseHMM:

    def __init__(self, n_states, max_iter=10, random_state=42):
        self.n_states = n_states

        self.delta = np.array([0.2, 0.8])  # 1 X N vector
        self.T = self.trans_init()  # N X N transmission matrix

        # Random init of state distributions
        np.random.seed(random_state)
        self.mu = np.random.rand(n_states)
        self.std = np.random.rand(n_states)

        self.max_iter = max_iter
        self.current_iter = 0
        self.old_log_prob = -np.inf # TODO does this have a function?

    def trans_init(self):
        T = np.zeros((2,2))
        T[0, 0] = 0.7
        T[0, 1] = 0.3
        T[1, 0] = 0.2
        T[1, 1] = 0.8
        return T

    def P(self, x: int):
        """ Function for computing diagonal prob matrix P(x).
         Change the function depending on the type of distribution you want to evaluate"""
        diag_probs = stats.norm.pdf(x, loc=self.mu, scale=self.std)
        diag_probs = np.diag(diag_probs)
        return diag_probs

    def log_all_probs(self, x: int):
        ''' Compute all different probabilities p(x) given an observation sequence and n states '''
        N = len(x)
        log_probs = np.zeros((N, self.n_states))  # Init N X M matrix

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.norm.logpdf(x, loc=self.mu[j], scale=self.std[j])

        return log_probs


    def _log_forward_probs(self, observations):
        ''' Compute log forward probabilities in scaled form. P(M=i , X=x)
        Follows the method by Zucchini A.1.8 p 334. '''
        N = len(observations)
        log_alphas = np.zeros((N, self.n_states))  # initialize matrix with zeros
        norm = np.zeros(N)

        # a0, compute first forward as dot product of initial dist and state-dependent dist
        # Each element is scaled to sum to 1 in order to handle numerical underflow
        alpha_t = self.delta @ self.P(observations[0])
        sum_alpha_t = np.sum(alpha_t)
        norm[0] = 1 / sum_alpha_t

        # Scale a0
        alpha_t_scaled = alpha_t * norm[0]
        llk = np.log(sum_alpha_t)  # Scalar to store the log likelihood
        log_alphas[0, :] = np.log(alpha_t_scaled)

        # a1 to at, compute recursively
        for t in range(1, N):
            alpha_t = alpha_t_scaled @ self.T @ self.P(observations[t])
            sum_alpha_t = np.sum(alpha_t)
            norm[t] = 1 / sum_alpha_t

            alpha_t_scaled = alpha_t * norm[t]
            llk = llk + np.log(sum_alpha_t) # Scalar to store likelihoods
            log_alphas[t, :] = np.log(alpha_t_scaled)  # TODO RESEARCH WHY YOU ADD THE PREVIOUS LIKELIHOOD

        return log_alphas, norm, llk


    def _log_backward_probs(self, observations, norm):
        ''' Compute the log of backward probabilities in scaled form. Same procedure as forward probs.'''
        N = len(observations)
        log_betas = np.zeros((N, self.n_states))  # initialize matrix with zeros

        #beta_t = np.ones(self.n_states) *  1/self.n_states  # TODO CHECK WHY WE USE 1/M
        beta_t = np.ones(self.n_states)
        beta_t_scaled = beta_t * norm[-1]
        llk = np.log(self.n_states)
        log_betas[-1, :] = np.log(beta_t_scaled)  # Last result is 0 since log(1)=0

        for t in range(N-2, -1, -1):  # Count backwards
            beta_t = self.T @ self.P(observations[t + 1]) @ beta_t_scaled
            beta_t_scaled = beta_t * norm[t]
            log_betas[t, :] = np.log(beta_t_scaled)
            #log_betas[t, :] = np.log(beta_t)
            #sum_beta_t = np.sum(beta_t)
            #beta_t = beta_t / sum_beta_t
            #llk = llk + np.log(sum_beta_t)

        return log_betas

    def _e_step(self, observations):
        ''' Do a single e-step '''
        N = len(observations)
        log_alphas, norm, llk = self._log_forward_probs(observations)
        log_betas = self._log_backward_probs(observations, norm)
        log_all_probs = self.log_all_probs(observations)

        # TODO CHECK HOW THIS C SCALINg PARAMETER WORKS
        #c = np.max(log_alphas[-1, :]) # Max of the last vector in the matrix log_alpha
        #llk = c + np.log(np.sum(np.exp(log_alphas[-1, :] - c))) # Changed from earlier: alphas[0] @ betas[0].T #np.sum(alphas[-1, :])

        # Expectation of being in state j at each time point
        u = np.exp(log_alphas - log_betas - llk) # TODO FIND BETTER VARIABLE NAME


        # Initialize 2D array of shape j X j
        # We skip computing vhat and head straight to fhat
        f = np.zeros(shape=(self.n_states, self.n_states))  # TODO FIND BETTER VARIABLE NAME
        for j in range(self.n_states):
            for k in range(self.n_states):
                f[j, k] = self.T[j, k] * np.sum(np.exp(log_alphas[:-1, j] + log_betas[1:, k] + log_all_probs[1:, k] - llk))

        return u, f, llk

    def _m_step(self, observations, u, f):
        ''' Given u and f do a sigle m step '''
        observations = np.array(observations)
        self.T = f / np.sum(f, axis=1).reshape((2, 1))  # Check if this actually sums correct and to 1 on rows
        self.delta = u[0, :] / np.sum(u[0, :])

        for j in range(self.n_states):
            self.mu[j] = np.sum(u[:, j] * observations) / np.sum(u[:, j])
            self.std[j] = np.sqrt(np.sum(u[:, j] * (observations - self.mu[j]) ** 2) / np.sum(u[:, j]))


    def em(self, observations, epochs, print_output=False):
        llk = 0
        for i in range(epochs):
            if print_output:
                print(i)
                print('MEAN: ', self.mu)
                print('STD: ', self.std)
                print('DELTA', self.delta)
                print('loglikelihood', llk)

                print('.' * 40)
            u, f, llk = self._e_step(observations)
            self._m_step(observations, u, f)



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
    #plt.plot(obs)
    #plt.show()
    print(np.shape(obs))
    print(obs[:10])

    obs = obs[:10]
    a, norm, llk = hmm._log_forward_probs(obs)
    b = hmm._log_backward_probs(obs, norm)


    #u, f, llk = hmm._e_step(obs)
    #m = hmm._m_step(obs, u, f)
    em = hmm.em(obs, epochs=5, print_output=True)

    #hmm.em(obs, epochs=5)





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