import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

from abc import abstractmethod

from utils import simulate_2state_gaussian

''' TODO NEXT:

Improve parameter initialization
    - Multiple random initializations
    - Look into kmeans++ init


Show why Zuchinni/Nystrups algorithm equals the scaling on p. 48 in the Zucchini book.
    - Derive in math how to compute log (alpha)
'''

class HiddenMarkovModel(BaseEstimator):
    """ Class for computing HMM's using the EM algorithm.
    Scikit-learn api is used as Parent see --> https://scikit-learn.org/stable/developers/develop.html

    """
    def __init__(self, n_states, epochs=100, tol=1e-8, random_state=42):
        self.n_states = n_states

        # TODO improve parameter initialization
        self.delta = np.array([0.2, 0.8])  # initial distribution 1 X N vector
        self.T = self._init_params()  # N X N transition matrix

        # Random init of state distributions
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.mu = np.random.rand(n_states)
        self.std = np.random.rand(n_states)

        self.epochs = epochs
        self.tol = tol
        self.current_iter = 0
        self.old_llk = -np.inf #  Used to check model convergence

    @abstractmethod
    def _init_params(self):
        T = np.zeros((2,2))
        T[0, 0] = 0.7
        T[0, 1] = 0.3
        T[1, 0] = 0.2
        T[1, 1] = 0.8
        return T

    def P(self, x: int):
        """Function for computing diagonal prob matrix P(x).

         Returns: Diagonal matrix of size n_states X n_states
         """

        diag_probs = stats.norm.pdf(x, loc=self.mu, scale=self.std) # Evaluate x in every state
        diag_probs = np.diag(diag_probs)  # Transforms it into a diagonal matrix
        return diag_probs

    def log_all_probs(self, X: list):
        """ Compute all different log probabilities log(p(x)) given an observation sequence and n states

        Returns: T X N matrix
        """
        T = len(X)
        log_probs = np.zeros((T, self.n_states))  # Init T X N matrix

        # For all states evaluate the density function
        for j in range(self.n_states):
            log_probs[:, j] = stats.norm.logpdf(X, loc=self.mu[j], scale=self.std[j])

        return log_probs

    def _log_forward_probs(self, X):
        """ Compute log forward probabilities in scaled form.

        Forward probability is essentially the joint probability of observing
        a state = i and observation sequences x^t=x_1...x_t, i.e. P(St=i , X^t=x^t).
        Follows the method by Zucchini A.1.8 p 334.
        """
        T = len(X)
        log_alphas = np.zeros((T, self.n_states))  # initialize matrix with zeros

        # a0, compute first forward as dot product of initial dist and state-dependent dist
        # Each element is scaled to sum to 1 in order to handle numerical underflow
        alpha_t = self.delta @ self.P(X[0])
        sum_alpha_t = np.sum(alpha_t)
        alpha_t_scaled = alpha_t / sum_alpha_t
        llk = np.log(sum_alpha_t)  # Scalar to store the log likelihood
        log_alphas[0, :] = llk + np.log(alpha_t_scaled)

        # a1 to at, compute recursively
        for t in range(1, T):
            alpha_t = alpha_t_scaled @ self.T @ self.P(X[t])  # Dot product of previous forward_prob, transition matrix and P(X)
            sum_alpha_t = np.sum(alpha_t)

            alpha_t_scaled = alpha_t / sum_alpha_t  # Scale forward_probs to sum to 1
            llk = llk + np.log(sum_alpha_t) # Scalar to store likelihoods
            log_alphas[t, :] = llk + np.log(alpha_t_scaled)  # TODO RESEARCH WHY YOU ADD THE PREVIOUS LIKELIHOOD

        return log_alphas

    def _log_backward_probs(self, X):
        """ Compute the log of backward probabilities in scaled form.
        Backward probabilities are the conditional probability of
        some observation at t+1 given the current state = i. Equivalent to P(X_t+1 = x_t+1 | S_t = i)
        """
        T = len(X)
        log_betas = np.zeros((T, self.n_states))  # initialize matrix with zeros

        beta_t = np.ones(self.n_states) * 1/self.n_states  # TODO CHECK WHY WE USE 1/M rather than ones
        llk = np.log(self.n_states)
        log_betas[-1, :] = np.log(np.ones(self.n_states))  # Last result is 0 since log(1)=0

        for t in range(T-2, -1, -1):  # Count backwards
            beta_t = self.T @ self.P(X[t + 1]) @ beta_t
            log_betas[t, :] = llk + np.log(beta_t)
            sum_beta_t = np.sum(beta_t)
            beta_t = beta_t / sum_beta_t  # Scale rows to sum to 1
            llk = llk + np.log(sum_beta_t)

        return log_betas

    def _e_step(self, X):
        ''' Do a single e-step in Baum-Welch algorithm

        '''
        T = len(X)
        log_alphas = self._log_forward_probs(X)
        log_betas = self._log_backward_probs(X)
        log_all_probs = self.log_all_probs(X)

        # Compute scaled log-likelihood
        llk_scale_factor = np.max(log_alphas[-1, :]) # Max of the last vector in the matrix log_alpha
        llk = llk_scale_factor + np.log(np.sum(np.exp(log_alphas[-1, :] - llk_scale_factor)))  # Scale log-likelihood by c

        # Expectation of being in state j given a sequence x^T
        # P(S_t = j | x^T)
        u = np.exp(log_alphas + log_betas - llk) # TODO FIND BETTER VARIABLE NAME

        # Initialize matrix of shape j X j
        # We skip computing vhat and head straight to fhat for computational reasons
        f = np.zeros(shape=(self.n_states, self.n_states))  # TODO FIND BETTER VARIABLE NAME
        for j in range(self.n_states):
            for k in range(self.n_states):
                f[j, k] = self.T[j, k] * np.sum(np.exp(log_alphas[:-1, j] + log_betas[1:, k] + log_all_probs[1:, k] - llk))

        return u, f, llk

    def _m_step(self, X, u, f):
        ''' Given u and f do an m-step.

         Updates the model parameters delta, Transition matrix and state dependent distributions.
         '''
        X = np.array(X)

        # Update transition matrix and initial probs
        self.T = f / np.sum(f, axis=1).reshape((2, 1))  # Check if this actually sums correct and to 1 on rows
        self.delta = u[0, :] / np.sum(u[0, :])

        # Update state-dependent distributions
        for j in range(self.n_states):
            self.mu[j] = np.sum(u[:, j] * X) / np.sum(u[:, j])
            self.std[j] = np.sqrt(np.sum(u[:, j] * np.square(X - self.mu[j])) / np.sum(u[:, j]))

    def fit(self, X, verbose=0):
        """Iterates through the e-step and the m-step"""

        for iter in range(self.epochs):
            u, f, llk = self._e_step(X)
            self._m_step(X, u, f)

            # Check convergence criterion
            crit = np.abs(llk - self.old_llk)  # Improvement in log likelihood
            if crit < self.tol:
                #self.aic_ = -2 * np.log(llk) + 2 *
                print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {self.T.flatten()} - Delta {self.delta}')
                break

            elif iter == self.epochs-1:
                print(f'No convergence after {iter} iterations')

            else:
                self.old_llk = llk

            if verbose == 1:
                print(f'Iteration {iter} - LLK {llk} - Means: {self.mu} - STD {self.std} - Gamma {self.T.flatten()} - Delta {self.delta}')

    def predict(self, X):
        pass

    def _viterbi(self, X):
        """ Compute the most likely sequence of states given the observations
         To reduce CPU time consider storing each sequence --> will save T*m function evaluations

         """
        T = len(X)
        posteriors = np.zeros((T, self.n_states))  # Init T X N matrix

        # Initiate posterior at time 0 and scale it as:
        posterior_temp = self.delta @ self.P(X[0])  # posteriors at time 0
        posteriors[0, :] = posterior_temp / np.sum(posterior_temp)  # Scaled posteriors at time 0

        # Do a forward recursion to compute posteriors
        for t in range(1, T):
            posterior_temp = np.max(posteriors[t - 1, :] * self.T, axis=1) @ self.P(X[t])  # TODO double check the max function returns the correct values
            posteriors[t, :] = posterior_temp / np.sum(posterior_temp)  # Scale rows to sum to 1

        # From posteriors get the the most likeley sequence of states i
        state_preds = np.zeros(T).astype(int)  # Vector of length N
        state_preds[-1] = np.argmax(posteriors[-1, :])  # Last most likely state is the index position

        # Do a backward recursion to calculate most likely state sequence
        for t in range(T-2, -1, -1):  # Count backwards
            state_preds[t] = np.argmax(posteriors[t, :] * self.T[:, state_preds[t + 1]])  # TODO double check the max function returns the correct values

        return state_preds, posteriors



if __name__ == '__main__':
    X, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some X in two states from normal distributions
    print('Example data: ', X[:5])

    model = HiddenMarkovModel(n_states=2, epochs=500)
    print('N states', model.n_states)

    print('P(x) ', model.P(X[0]) )
    print("."*40)

    #model._log_forward_probs(X)

    print('Log All probs shape: ', np.shape(model.log_all_probs(X)))
    #print(model.log_all_probs(X))

    print('.'*50)
    Probability_state_i,f,llk = (model._e_step(X))
    #print(Probability_state_i)
    #print(f)

    #print('Initial delta: ', model.delta)
    #model._m_step(X, Probability_state_i, f)
    #print('Updated delta: ', model.delta)


    model.fit(X, verbose=0)

    states, preds = model._viterbi(X)






temp = False
if temp:
    hmm_model = HiddenMarkovModel(n_states=2)

    returns, true_regimes = simulate_2state_gaussian(plotting=False)  # Simulate some X in two states from normal distributions

    hmm_model.fit(returns, verbose=0)

    states, posteriors = hmm_model._viterbi(returns)
    #print(posteriors)

    plotting = False
    if plotting == True:
        plt.plot(posteriors[:, 0], label='Posteriors state 1', )
        plt.plot(posteriors[:, 1], label='Posteriors state 2', )
        #plt.plot(states, label='Predicted states', ls='dotted')
        #plt.plot(true_regimes, label='True states', ls='dashed')

        plt.legend()
        plt.show()

    check_hmmlearn = False
    if check_hmmlearn == True:
        from hmmlearn import hmm

        model = hmm.GaussianHMM(n_components=2, covariance_type="full",n_iter=1000).fit(returns.reshape(-1,1))

        print(model.transmat_)
        print(model.means_)
        print(model.covars_)

        predictions = model.predict(returns.reshape(-1,1))
        print(predictions)

