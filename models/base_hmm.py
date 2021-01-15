import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from typing import List

from utils.simulate_returns import simulate_2state_gaussian

''' TODO NEXT:

If delta is misspecified it quickly converges to the wrong state.
    - I.e. if init as [0.2, 0.8] the 0.8 will quickly go to 1 despite it being the wrong starting state.

'''


class MLEHiddenMarkov(BaseEstimator):
    """ Class for computing HMM's using the EM algorithm.
    Scikit-learn api is used as Parent see --> https://scikit-learn.org/stable/developers/develop.html


    Parameters
    ----------
    n_states : Number of hidden states
    max_iter : Maximum number of iterations to perform during expectation-maximization
    tol : Criterion for early stopping

    Returns
    ----------
    Can be used to fit HMM parameters or to decode hidden states.

    """

    def __init__(self, n_states: int = 2, params_init: str = 'random', max_iter: int = 100, tol: int = 1e-4,
                 epochs: int = 1, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.max_iter = max_iter  # Max iterations to fit model
        self.epochs = epochs  # Set number of random inits used in model fitting
        self.tol = tol

        # Init parameters initial distribution, transition matrix and state-dependent distributions from function
        np.random.seed(self.random_state)
        self._init_params(init=params_init)  # Initializes model parameters.

    def _init_params(self, init: str = 'random', diag_uniform_dist: List[float] = [.7, .99]):
        """
        Function to initialize HMM parameters.

        Parameters
        ----------
        init: str
            Set to 'random' for random initialization.
            Set to None for deterministic init.

        diag_uniform_dist: 1D-array
            The lower and upper bounds of uniform distribution to sample init from.

        Returns
        -------
        self.T: N X N matrix of transition probabilities
        self.delta: 1 X N vector of initial probabilities
        self.mu: 1 X N vector of state dependent means
        self.std: 1 X N vector of state dependent STDs
        self.dof: int with degrees of freedom for t-distribution

        """

        if init == 'random':
            # Transition probabilities
            trans_prob = np.diag(np.random.uniform(low=diag_uniform_dist[0], high=diag_uniform_dist[1], size=self.n_states))  # Sample diag as uniform dist
            remaining_row_nums = (1 - np.diag(trans_prob)) / (self.n_states - 1)  # Spread the remaining mass evenly onto remaining row values
            trans_prob += remaining_row_nums.reshape(-1, 1)  # Add this to the uniform diagonal matrix
            np.fill_diagonal(trans_prob, trans_prob.diagonal() - remaining_row_nums)  # And subtract these numbers from the diagonal so it remains uniform

            # initial distribution
            #init_dist = np.random.uniform(low=0.4, high=0.6, size=self.n_states)
            #init_dist /= np.sum(init_dist)
            init_dist = np.ones(self.n_states) / np.sum(self.n_states)  # initial distribution 1 X N vector

            # State dependent distributions
            mu = np.random.uniform(low=-0.05, high=0.15, size=self.n_states)  # np.random.rand(self.n_states)
            std = np.random.uniform(low=0.01, high=0.3, size=self.n_states)  # np.random.rand(self.n_states) #

        else:
            trans_prob = np.zeros((2, 2))
            trans_prob[0, 0] = 0.7
            trans_prob[0, 1] = 0.3
            trans_prob[1, 0] = 0.2
            trans_prob[1, 1] = 0.8
            init_dist = np.array([0.2, 0.8])  # initial distribution 1 X N vector
            mu = [-0.05, 0.1]
            std = [0.2, 0.1]

        self.T = trans_prob
        self.delta = init_dist
        self.mu = mu
        self.std = std
        self.dof = 2

    def emission_probs(self, X: list):
        """ Compute all different log probabilities log(p(x)) given an observation sequence and n states

        Returns: T X N matrix
        """
        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init T X N matrix
        probs = np.zeros((T, self.n_states))

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])

        probs = np.exp(log_probs)

        return probs, log_probs

    def _log_forward_probs(self, X: List[float], emission_probs: np.ndarray):
        """ Compute log forward probabilities in scaled form.

        Forward probability is essentially the joint probability of observing
        a state = i and observation sequences x^t=x_1...x_t, i.e. P(St=i , X^t=x^t).
        Follows the method by Zucchini A.1.8 p 334.
        """
        T = len(X)
        log_alphas = np.zeros((T, self.n_states))  # initialize matrix with zeros

        # a0, compute first forward as dot product of initial dist and state-dependent dist
        # Each element is scaled to sum to 1 in order to handle numerical underflow
        alpha_t = self.delta * emission_probs[0, :]
        sum_alpha_t = np.sum(alpha_t)
        alpha_t_scaled = alpha_t / sum_alpha_t
        llk = np.log(sum_alpha_t)  # Scalar to store the log likelihood
        log_alphas[0, :] = llk + np.log(alpha_t_scaled)

        # a1 to at, compute recursively
        for t in range(1, T):
            alpha_t = (alpha_t_scaled @ self.T) * emission_probs[t, :]  # Dot product of previous forward_prob, transition matrix and emmission probablitites
            sum_alpha_t = np.sum(alpha_t)

            alpha_t_scaled = alpha_t / sum_alpha_t  # Scale forward_probs to sum to 1
            llk = llk + np.log(sum_alpha_t)  # Scalar to store likelihoods
            log_alphas[t, :] = llk + np.log(alpha_t_scaled)  # TODO RESEARCH WHY YOU ADD THE PREVIOUS LIKELIHOOD

        return log_alphas

    def _log_backward_probs(self, X: List[float], emission_probs: np.ndarray):
        """ Compute the log of backward probabilities in scaled form.
        Backward probabilities are the conditional probability of
        some observation at t+1 given the current state = i. Equivalent to P(X_t+1 = x_t+1 | S_t = i)
        """
        T = len(X)
        log_betas = np.zeros((T, self.n_states))  # initialize matrix with zeros

        beta_t = np.ones(self.n_states) * 1 / self.n_states  # TODO CHECK WHY WE USE 1/M rather than ones
        llk = np.log(self.n_states)
        log_betas[-1, :] = np.log(np.ones(self.n_states))  # Last result is 0 since log(1)=0

        for t in range(T - 2, -1, -1):  # Count backwards
            beta_t = (self.T * emission_probs[t+1, :]) @ beta_t
            log_betas[t, :] = llk + np.log(beta_t)
            sum_beta_t = np.sum(beta_t)
            beta_t = beta_t / sum_beta_t  # Scale rows to sum to 1
            llk = llk + np.log(sum_beta_t)

        return log_betas

    def _e_step(self, X: List[float]):
        ''' Do a single e-step in Baum-Welch algorithm

        '''
        T = len(X)
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)
        log_alphas = self._log_forward_probs(X, self.emission_probs_)
        log_betas = self._log_backward_probs(X, self.emission_probs_)

        # Compute scaled log-likelihood
        llk_scale_factor = np.max(log_alphas[-1, :])  # Max of the last vector in the matrix log_alpha
        llk = llk_scale_factor + np.log(
            np.sum(np.exp(log_alphas[-1, :] - llk_scale_factor)))  # Scale log-likelihood by c

        # Expectation of being in state j given a sequence x^T
        # P(S_t = j | x^T)
        u = np.exp(log_alphas + log_betas - llk)  # TODO FIND BETTER VARIABLE NAME

        # Initialize matrix of shape j X j
        # We skip computing vhat and head straight to fhat for computational reasons
        f = np.zeros(shape=(self.n_states, self.n_states))  # TODO FIND BETTER VARIABLE NAME
        for j in range(self.n_states):
            for k in range(self.n_states):
                f[j, k] = self.T[j, k] * np.sum(
                    np.exp(log_alphas[:-1, j] + log_betas[1:, k] + self.log_emission_probs_[1:, k] - llk))

        return u, f, llk

    def _m_step(self, X: List[float], u, f, iterations: int = None):
        ''' Given u and f do an m-step.

         Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        X = np.array(X)

        # Update transition matrix and initial probs
        self.T = f / np.sum(f, axis=1).reshape((-1, 1))  # Check if this actually sums correct and to 1 on rows
        self.delta = u[0, :] / np.sum(u[0, :])

        # Update state-dependent distributions
        for j in range(self.n_states):
            self.mu[j] = np.sum(u[:, j] * X) / np.sum(u[:, j])
            self.std[j] = np.sqrt(np.sum(u[:, j] * np.square(X - self.mu[j])) / np.sum(u[:, j]))

    def fit(self, X: List[float], verbose=0):
        """
        Iterates through the e-step and the m-step.
        Parameters
        ----------
        X
        verbose

        Returns
        -------

        """
        self.old_llk = -np.inf  # Used to check model convergence

        for epoch in range(self.epochs):
            # Do multiple random runs
            if epoch > 1: self._init_params(init='random')
            #print(f'Epoch {epoch} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')

            for iter in range(self.max_iter):
                # Do e- and m-step
                u, f, llk = self._e_step(X)
                self._m_step(X, u, f)

                # Check convergence criterion
                crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
                if crit < self.tol:
                    # Compute AIC and BIC and print model results
                    # AIC & BIC computed as shown on
                    # https://rdrr.io/cran/HMMpa/man/AIC_HMM.html
                    num_independent_params = self.n_states**2 + 2*self.n_states - 1  # True for normal distributions
                    self.aic_ = -2 * llk + 2 * num_independent_params
                    self.bic_ = -2 * llk + num_independent_params * np.log(len(X))

                    print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')
                    break

                elif iter == self.max_iter - 1:
                    print(f'No convergence after {iter} iterations')
                else:
                    self.old_llk = llk

                if verbose == 1: print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {np.diag(self.T)} - Delta {self.delta}')

    def predict(self, X):
        state_preds, posteriors = self._viterbi(X)
        return state_preds, posteriors

    def _viterbi(self, X):
        """ Compute the most likely sequence of states given the observations
         To reduce CPU time consider storing each sequence --> will save T*m function evaluations

         """
        T = len(X)
        posteriors = np.zeros((T, self.n_states))  # Init T X N matrix
        self.emission_probs_, self.log_emission_probs_ = self.emission_probs(X)

        # Initiate posterior at time 0 and scale it as:
        posterior_temp = self.delta * self.emission_probs_[0, :]  # posteriors at time 0
        posteriors[0, :] = posterior_temp / np.sum(posterior_temp)  # Scaled posteriors at time 0

        # Do a forward recursion to compute posteriors
        for t in range(1, T):
            posterior_temp = np.max(posteriors[t - 1, :] * self.T, axis=1) * self.emission_probs_[t, :]  # TODO double check the max function returns the correct values
            posteriors[t, :] = posterior_temp / np.sum(posterior_temp)  # Scale rows to sum to 1

        # From posteriors get the the most likeley sequence of states i
        state_preds = np.zeros(T).astype(int)  # Vector of length N
        state_preds[-1] = np.argmax(posteriors[-1, :])  # Last most likely state is the index position

        # Do a backward recursion to calculate most likely state sequence
        for t in range(T - 2, -1, -1):  # Count backwards
            state_preds[t] = np.argmax(posteriors[t, :] * self.T[:, state_preds[t + 1]])  # TODO double check the max function returns the correct values

        return state_preds, posteriors

    def sample(self, X):
        sample_len = len(X)

        state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=int)  # Array of possible states
        sample_states = np.zeros(sample_len).astype(int)  # Init sample vector
        sample_states[0] = np.random.choice(a=state_index, size=1, p=self.delta) # First state is determined by initial dist

        for t in range(1, sample_len):
            # Each new state is chosen using the transition probs corresponding to the previous state sojourn.
            sample_states[t] = np.random.choice(a=state_index, size=1, p=self.T[sample_states[t-1], :])

        samples = stats.norm.rvs(loc=self.mu[sample_states], scale = self.std[sample_states], size=sample_len)

        return samples, sample_states


if __name__ == '__main__':
    model = MLEHiddenMarkov(n_states=2, random_state=42)

    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some data from two normal distributions

    model.fit(returns, verbose=0)
    states, posteriors = model._viterbi(returns)


    plotting = False
    if plotting == True:
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].plot(posteriors[:, 0], label='Posteriors state 1', )
        ax[0].plot(posteriors[:, 1], label='Posteriors state 2', )
        ax[1].plot(states, label='Predicted states', ls='dotted')
        ax[1].plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()

    check_hmmlearn = False
    if check_hmmlearn == True:
        from hmmlearn import hmm

        hmmlearn = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(returns.reshape(-1, 1))

        print(hmmlearn.transmat_)
        print(hmmlearn.means_)
        print(hmmlearn.covars_)

        predictions = hmmlearn.predict(returns.reshape(-1, 1))
