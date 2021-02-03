import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.cluster._kmeans import kmeans_plusplus
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from utils import plotting
from utils.hmm_sampler import SampleHMM
from models.hmm_base import BaseHiddenMarkov

import pyximport; pyximport.install()  # TODO can only be active during development -- must be done through setup.py
from models import hmm_cython


''' TODO:
 
Problems with fitting thetas due to no state changes. 
 
Z-score standardisation

Consider setting boundscheck = False for all cython code
'''

class JumpHMM(BaseHiddenMarkov):

    def __init__(self, n_states: int = 2, jump_penalty: float = .2, window_len: int = 6,
                 init: str = 'kmeans++', max_iter: int = 30, tol: int = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

        # Init parameters initial distribution, transition matrix and state-dependent distributions
        self.jump_penalty = jump_penalty
        self.window_len = window_len

        self.best_objective_likelihood = np.inf
        self.old_objective_likelihood = np.inf

    def _init_params(self, X=None, diag_uniform_dist = (.7, .99), output_hmm_params=False):
        super()._init_params()

        if self.init == "kmeans++":
            if not isinstance(X, np.ndarray):
                raise Exception("To initialize with kmeans++ a sequence of data must be provided in an ndarray")
            if X.ndim == 1:
                X = X.reshape(-1, 1)  # Compatible with sklearn

            # Theta
            centroids, _ = kmeans_plusplus(X, self.n_states)  # Use sklearns kmeans++ algorithm
            self.theta = centroids.T  # Transpose to get shape (n_features, n_states)

            # State sequence
            self._fit_state_seq(X)  # this function implicitly updates self.state_seq

            if output_hmm_params == True: # output all hmm parameters from state sequence
                self.get_params_from_seq(X, state_sequence=self.state_seq)

    def construct_features(self, X: ndarray, window_len: int):  # TODO remove forward-looking params and slice X accordingly
        N = len(X)
        df = pd.DataFrame(X)

        df['Left local mean'] = df.rolling(window_len).mean()
        df['Left local std'] = df[0].rolling(window_len).std(ddof=0)

        df['Right local mean'] = df[0].rolling(window_len).mean().shift(-window_len+1)
        df['Right local std'] = df[0].rolling(window_len).std(ddof=0).shift(-window_len + 1)

        look_ahead = df[0].rolling(window_len).sum().shift(-window_len)  # Looks forward with window_len (Helper 1)
        look_back = df[0].rolling(window_len).sum()  # Includes current position and looks window_len - 1 backward (Helper 2)
        df['Central local mean'] = (look_ahead + look_back) / (2 * window_len)
        df['Centered local std'] = df[0].rolling(window_len * 2).std(ddof=0).shift(-window_len)  # Rolls from 0 and 2x length iteratively, then shifts back 1x window length

        # Absolute changes
        df['left_abs_change'] = np.abs(df[0].diff()) #np.abs(np.diff(X))
        df['right_abs_change'] = df['left_abs_change'].shift(-1)

        Z = df.dropna().to_numpy()
        # Scale features
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z)

        self.n_features = Z.shape[1]

        return Z

    def _l2_norm_squared(self, z, theta):
        """
        Compute the squared l2 norm at each time step.

        Squared l2 norm is computed as ||z_t - theta ||_2^2.

        Parameters
        ----------
        z : ndarray of shape (n_samples, n_features)
            Data to be fitted
        theta : ndarray of shape (n_features, n_states)
            jump model parameters
        Returns
        -------
        norms: ndarray of shape (n_samples, n_states)
            Squared l2 norms conditional on latent states
        """
        norms = np.zeros(shape=(len(z), self.n_states))

        for j in range(self.n_states):
            diff = theta[:, j] - z  # ndarray of shape (n_samples, n_states) with differences
            norms[:, j] = np.square(np.linalg.norm(diff, axis=1))  # squared state conditional l2 norms

        return norms  # squared l2 norm.

    def _fit_theta(self, Z):
        """
        Fit theta, i.e minimize the squared L2 norm in each latent state.

        Computed analytically. See notebooks/math_overviews_algos for proof of solution.
        """
        for j in range(self.n_states):
            state_slicer = self.state_seq == j  # Boolean array of True/False
            N_j = sum(state_slicer)  # Number of terms in state j

            #assert N_j != 0, "No state changes detected"  # Check that state changes are detected
            if N_j != 0:
                z = Z[state_slicer].sum(axis=0)  # Sum each feature across time
                self.theta[:, j] = 1 / N_j * z
            else:
                self.theta[:, j] = 0.  # TODO is the best estimate of theta when the state is not present?

    def _fit_state_seq(self, X: ndarray):
        """
        Fit a state sequence based on current theta estimates.
        Used in jump model fitting. Uses a dynamic programming technique very
        similar to the viterbi algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Set of standardized times series features

        Returns
        -------
        fitted state sequence
        """
        l2_norms = self._l2_norm_squared(X, self.theta)  # Compute array of all squared l2 norms
        n_samples, _ = l2_norms.shape

        losses, state_preds = hmm_cython.jump_state_seq(n_samples, self.n_states,
                                                        self.n_features,
                                                        self.jump_penalty,
                                                        l2_norms)
        # Compute score of objective function
        all_likelihoods = losses[np.arange(len(losses)), state_preds].sum()
        state_changes = np.diff(state_preds) != 0  # True/False array showing state changes
        jump_penalty = (state_changes * self.jump_penalty).sum()  # Multiply all True values with penalty

        self.objective_likelihood = all_likelihoods + jump_penalty  # Float
        self.state_seq = state_preds

        return state_preds

    def fit(self, Z: ndarray, get_hmm_params=True, verbose=0):
        converged = False
        # Each epoch independently inits and fits a new model to the same data
        for epoch in range(self.epochs):
            # Do new init at each epoch
            self._init_params(Z, output_hmm_params=False)
            self.old_state_seq = self.state_seq

            for iter in range(self.max_iter):
                self._fit_theta(Z)
                self._fit_state_seq(Z)

                # Check convergence criterion
                crit1 = np.array_equal(self.state_seq, self.old_state_seq)  # No change in state sequence
                crit2 = np.abs(self.old_objective_likelihood - self.objective_likelihood)
                if crit1 == True or crit2 < self.tol:
                    # If model is converged check if current epoch is the best
                    # If current model is best all model params are updated
                    if self.objective_likelihood < self.best_objective_likelihood:
                        self.best_objective_likelihood = self.objective_likelihood
                        self.best_state_seq = self.state_seq
                        self.best_theta = self.theta
                        converged = True

                    if verbose == 1:
                        print(f'Epoch {epoch} -- Iter {iter} -- likelihood {self.objective_likelihood} -- Theta {self.theta[0]} ')
                    break

                else:
                    self.old_state_seq = self.state_seq
                    self.old_objective_likelihood = self.objective_likelihood

        self.state_seq = self.best_state_seq
        self.theta = self.best_theta
        if get_hmm_params == True:
            self.get_params_from_seq(Z, state_sequence=self.best_state_seq)

        if converged == False:
            print(f'No convergence in any epoch out of {self.epochs} epochs')

    def get_params_from_seq(self, X, state_sequence):  # TODO remove forward-looking params and slice X accordingly ofr X.ndim == 1
        """
        Stores and outputs the model parameters based on the input sequence.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Time series of data

        state_sequence : ndarray of shape (n_samples)
            State sequence for a given observation sequence
        Returns
        ----------
        Hmm parameters
        """

        # Slice data
        if X.ndim == 1:  # Makes function compatible on higher dimensions
            X = X[(self.window_len - 1): -self.window_len]
        elif X.ndim > 1:
            X = X[:, 0]

        # group by states
        diff = np.diff(state_sequence)
        df_states = pd.DataFrame({'state_seq': state_sequence,
                                  'X': X,
                                  'state_sojourns': np.append([False], diff == 0),
                                  'state_changes': np.append([False], diff != 0)})

        state_groupby = df_states.groupby('state_seq')

        # Transition probabilities
        # TODO only works for a 2-state HMM
        self.tpm = np.diag(state_groupby['state_sojourns'].sum())
        state_changes = state_groupby['state_changes'].sum()
        self.tpm[0, 1] = state_changes[0]
        self.tpm[1, 0] = state_changes[1]
        self.tpm = self.tpm / self.tpm.sum(axis=1).reshape(-1, 1)  # make rows sum to 1

        # init dist and stationary dist
        self.start_proba = np.zeros(self.n_states)
        self.start_proba[state_sequence[0]] = 1.
        self.stationary_dist = self.get_stationary_dist(tpm=self.tpm)

        # Conditional distributions
        self.mu = state_groupby['X'].mean().values.T  # transform mean back into 1darray
        self.std = state_groupby['X'].std().values.T

    def bac_score_1d(self, X, y_true, jump_penalty, window_len=6):
        self.jump_penalty = jump_penalty
        Z = self.construct_features(X, window_len=window_len)
        self.fit(Z, get_hmm_params=False)  # Updates self.state_seq
        y_pred = self.state_seq

        y_true = y_true[(self.window_len - 1): -self.window_len]  # slice y_true to have same dim as y_pred

        conf_matrix = confusion_matrix(y_true, y_pred)
        tp = np.diag(conf_matrix)
        fn = conf_matrix.sum(axis=1) - tp

        if y_true.all() == 0 or y_true.all() == 1:
            bac = tp[0] / (tp[0] + fn[0])  # If only one state is present bac == tpr
        else:
            tpr = tp / (tp + fn)
            bac = np.mean(tpr)

        return bac

    def bac_score_nd(self, X, y_true, jump_penalty, window_len=6):
        bac = []
        for seq in range(X.shape[1]):
            bac_temp = self.bac_score_1d(X[:, seq], y_true[:, seq], jump_penalty, window_len=6)
            bac.append(bac_temp)

        return np.mean(bac)

if __name__ == '__main__':
    model = JumpHMM(n_states=2, jump_penalty=2, random_state=42)
    sampler = SampleHMM(n_states=2, random_state=1)

    n_samples = 2000
    n_sequences = 2
    X, viterbi_states, true_states = sampler.sample_with_viterbi(n_samples, n_sequences)

    bac = model.bac_score_nd(X, true_states, jump_penalty=0.1)