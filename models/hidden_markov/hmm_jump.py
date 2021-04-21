import numpy as np
import pandas as pd
import pyximport
import tqdm
from numpy import ndarray
from sklearn.cluster._kmeans import kmeans_plusplus
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.hmm_sampler import SampleHMM
from models.hidden_markov.hmm_base import BaseHiddenMarkov
from models.hidden_markov import hmm_cython


pyximport.install()  # TODO can only be active during development -- must be done through setup.py

class JumpHMM(BaseHiddenMarkov):

    def __init__(self, n_states: int = 2, jump_penalty: float = 16, window_len: tuple = (6, 14),
                 init: str = 'kmeans++', max_iter: int = 30, tol: int = 1e-6,
                 epochs: int = 10, random_state: int = 42):
        super().__init__(n_states, init, max_iter, tol, epochs, random_state)

        # Init parameters initial distribution, transition matrix and state-dependent distributions
        self.type = 'jump'
        self.jump_penalty = jump_penalty
        self.window_len = window_len
        self.theta = None
        self.state_seq = None
        self.n_features = None

    def _init_params(self, Z, X=None, diag_uniform_dist=(.95, .99), output_hmm_params=False):
        super()._init_params()

        if self.init == "kmeans++":
            if not isinstance(Z, np.ndarray):
                raise Exception("To initialize with kmeans++ a sequence of data must be provided in an ndarray")
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)  # Compatible with sklearn

            # Theta
            centroids, _ = kmeans_plusplus(Z, self.n_states)  # Use sklearns kmeans++ algorithm
            theta = centroids.T  # Transpose to get shape (n_features, n_states)

            # State sequence
            state_seq, _ = self._fit_state_seq(Z, theta)

            if output_hmm_params == True:  # output all hmm parameters from state sequence
                if not X == None:
                    self.get_params_from_seq(X, state_sequence=self.state_seq)
                else:
                    raise Exception('''To output hmm-params from sequence, X must be provided as a 1D-array.
                                    X has to be non-standardized data otherwise it will return standardized results''')

            return state_seq, theta

    def construct_features(self, X: ndarray, window_len: tuple, feature_set='feature_set_2'):
        if not feature_set in ['feature_set_1', 'feature_set_2', 'feature_set_3']:
            raise Exception('Invalid feature set. Valid options are [feature_set_1, feature_set_2, feature_set_3]')

        df = pd.DataFrame({'raw_input': X})

        if feature_set == 'feature_set_1':
            # Absolute changes
            df['left_abs_change'] = np.abs(df['raw_input'].diff())  # np.abs(np.diff(X))
            df['prev_left_abs_change'] = df['left_abs_change'].shift(1)

            # Rolling features
            for i in range(len(window_len)):
                rolling_window = df['raw_input'].rolling(window_len[i])

                df['left_local_mean_'+str(i)] = rolling_window.mean()
                df['left_local_std_'+str(i)] = rolling_window.std(ddof=1)
                df['left_local_median_'+str(i)] = rolling_window.median() #
                df['min_max_diff_'+str(i)] = rolling_window.max() - rolling_window.min()

        elif feature_set == 'feature_set_2':
            for i in range(len(window_len)):
                rolling_window = df['raw_input'].rolling(window_len[i])
                #Bollinger bands (Remember middle band is equal to the rolling mean)
                df['upper_bollinger_band'] = rolling_window.mean() + (1.96 * rolling_window.std(ddof=1))
                df['lower_bollinger_band'] = rolling_window.mean() - (1.96 * rolling_window.std(ddof=1))
                df['band_width'] = df['upper_bollinger_band'] - df['lower_bollinger_band']
                df['bbands_%_bi'] = (rolling_window.mean() - df['lower_bollinger_band']) / (df['upper_bollinger_band'] - df['lower_bollinger_band']) # %B quantifies a security's average price relative to the upper and lower Bollinger Band.

            #Exponential moving average 10 days
            df['ema_10'] = df['raw_input'].ewm(span=10).mean()

            df.drop('raw_input', inplace=True, axis=1) #Active this for 2nd feature set!

        elif feature_set == 'feature_set_3':
            df['left_abs_change'] = np.abs(df['raw_input'].diff())  # np.abs(np.diff(X))
            df['prev_left_abs_change'] = df['left_abs_change'].shift(1)

            # Rolling features
            for i in range(len(window_len)):
                rolling_window = df['raw_input'].rolling(window_len[i])

                df['left_local_mean_' + str(i)] = rolling_window.mean()
                df['left_local_std_' + str(i)] = rolling_window.std(ddof=1)
                df['left_local_median_' + str(i)] = rolling_window.median()  #
                df['min_max_diff_' + str(i)] = rolling_window.max() - rolling_window.min()

                # Bollinger bands (Remember middle band is equal to the rolling mean)
                df['upper_bollinger_band'] = rolling_window.mean() + (1.96 * rolling_window.std(ddof=1))
                df['lower_bollinger_band'] = rolling_window.mean() - (1.96 * rolling_window.std(ddof=1))
                df['band_width'] = df['upper_bollinger_band'] - df['lower_bollinger_band']
                df['bbands_%_bi'] = (rolling_window.mean() - df['lower_bollinger_band']) / (
                df['upper_bollinger_band'] - df['lower_bollinger_band'])  # %B quantifies a security's average price relative to the upper and lower Bollinger Band.

            # Exponential moving average
            df['ema_10'] = df['raw_input'].ewm(span=10).mean()



        Z = df.dropna().to_numpy()
        # Scale features
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z)

        #Consider defining feature set from PCA?

        #pca = PCA(n_components=2)
        #pca_features = pca.fit_transform(Z)
        #Z['PCA_1'] = pca_features[:,0]
        #Z['PCA_2'] = pca_features[:,1]
        #Z = Z.loc[:,'PCA_1':'PCA_2']
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

    def _fit_theta(self, Z, state_seq):
        """
        Fit theta, i.e minimize the squared L2 norm in each latent state.
        Computed analytically. See notebooks/math_overviews_algos for proof of solution.

        Parameters
        ----------
        Z : ndarray of shape (n_samples, n_features)
            Set of standardized times series features
        state_seq : ndarray of shape (n_samples,)
            State sequence

        Returns
        -------
        theta : ndarray of shape (n_features, n_states)
            jump model parameters. Distances from state (cluster) centers.
        """
        theta = np.zeros(shape=(self.n_features, self.n_states))
        for j in range(self.n_states):
            state_slicer = np.array(state_seq == j)  # Boolean array of True/False
            N_j = np.sum(state_slicer)  # Number of terms in state j

            if N_j != 0:
                z = Z[state_slicer].sum(axis=0)  # Sum each feature across time
                theta[:, j] = z / N_j
            else:
                theta[:, j] = 0.  # TODO what is the best estimate of theta when the state is not present?

        return theta

    def _fit_state_seq(self, X, theta):
        """
        Fit a state sequence based on current theta estimates.
        Used in jump model fitting. Uses a dynamic programming technique very
        similar to the viterbi algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples)
            Standardized returns.
        theta : ndarray of shape (n_features, n_states)
            jump model parameters. Distances from state (cluster) centers.

        Returns
        -------
        state_seq : ndarray of shape (n_samples,)
            State sequence
        objective_score : float
            Objective score under the model
        """
        l2_norms = self._l2_norm_squared(X, theta)  # Compute array of all squared l2 norms
        n_samples, _ = l2_norms.shape

        losses, state_preds = hmm_cython.jump_state_seq(n_samples, self.n_states,
                                                        self.n_features,
                                                        self.jump_penalty,
                                                        l2_norms)
        # Compute score of objective function
        all_likelihoods = losses[np.arange(len(losses)), state_preds].sum()
        state_changes = np.diff(state_preds) != 0  # True/False array showing state changes
        jump_penalty = (state_changes * self.jump_penalty).sum()  # Multiply all True values with penalty

        objective_score = all_likelihoods + jump_penalty  # Float

        return state_preds, objective_score

    def _fit(self, X, Z, verbose=False):
        """ Container for the loop where fitting happens"""
        best_objective_score = np.inf

        # Each epoch independently inits and fits a new model to the same data
        for epoch in range(self.epochs):  # TODO consider parallel processing
            # Do new init at each epoch
            state_seq, theta = self._init_params(Z, output_hmm_params=False, X=X)
            old_state_seq = state_seq
            old_objective_score = np.inf

            for iter in range(self.max_iter):
                theta = self._fit_theta(Z, state_seq)
                state_seq, objective_score = self._fit_state_seq(Z, theta)

                # Check convergence criterion
                crit1 = np.array_equal(state_seq, old_state_seq)  # No change in state sequence
                crit2 = np.abs(old_objective_score - objective_score)
                if crit1 is True or crit2 < self.tol:
                    self.is_fitted = True
                    # If model is converged check if current epoch is the best
                    # If current model is best all model params are updated
                    if objective_score < best_objective_score:
                        best_objective_score = objective_score
                        self.theta = theta
                        self.state_seq = state_seq

                        if verbose == 2:
                            print(
                                f'Epoch {epoch} -- Iter {iter} -- likelihood {objective_score} -- Theta {self.theta[0]} ')

                    break  # Break out of inner loop and go to next epoch

                else:
                    old_state_seq = state_seq
                    old_objective_score = objective_score

    def fit(self, X, get_hmm_params=True, sort_state_seq=True, verbose=False, feature_set='feature_set_2'):
        self.is_fitted = False
        Z = self.construct_features(X, window_len=self.window_len, feature_set=feature_set)

        self._fit(X=X, Z=Z, verbose=verbose)  # Container for loop where fitting happens

        if self.is_fitted is False:
            max_iter = self.max_iter
            self.max_iter = max_iter * 2  # Double amount of iterations
            self._fit(X, Z)  # Try fitting again
            self.max_iter = max_iter  # Reset max_iter back to user-input
            if self.is_fitted == False and verbose == True:
                print(
                    f'Jump NOT FITTED -- epochs {self.epochs} -- iters {self.max_iter * 2}')

        if get_hmm_params is True and self.is_fitted is True:
            if sort_state_seq is True:
                self.state_seq = self._check_state_sort(X, self.state_seq)
            self.get_params_from_seq(X, state_sequence=self.state_seq)

    def get_params_from_seq(self, X, state_sequence):
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
            X = X[(self.window_len[-1] - 1):]
        else:
            raise Exception('''X must be one-dimensional. Pass raw data as 1-D array.''')

        # group by states
        diff = np.diff(state_sequence)
        df_states = pd.DataFrame({'state_seq': state_sequence,
                                  'X': X,
                                  'state_sojourns': np.append([False], diff == 0),
                                  'state_changes': np.append([False], diff != 0)})

        state_groupby = df_states.groupby('state_seq')

        # Conditional distributions
        self.mu = state_groupby['X'].mean().values.T  # transform mean back into 1darray
        self.std = state_groupby['X'].std(ddof=1).values.T

        # Transition probabilities
        # TODO only works for a 2-state HMM
        tpm = np.diag(state_groupby['state_sojourns'].sum())
        state_changes = state_groupby['state_changes'].sum()

        # if only 1 state is detected make the other state zero
        if state_changes.sum() > 0:
            tpm[0, 1] = state_changes[0]
            tpm[1, 0] = state_changes[1]

        else:
            tpm = np.array([[1., 0.], [1., 0.]])
            self.mu = np.append(self.mu, 0.)
            self.std = np.append(self.std, 0.)

        self.tpm = tpm / tpm.sum(axis=1).reshape(-1, 1)  # make rows sum to 1

        # init dist and stationary dist
        self.start_proba = np.zeros(self.n_states)
        self.start_proba[state_sequence[0]] = 1.
        self.stationary_dist = self.get_stationary_dist(tpm=self.tpm)

    def _check_state_sort(self, X, state_sequence):  # TODO give this same structure as in EM_HMM
        """
        Checks whether the low-variance state is the first state.

        Otherwise sorts state predictions accoridngly.

        Parameters
        ----------
        X : ndarray of shape (n_samples,)
            Data to be fitted.
        state_sequence : ndarray of shape (n_samples,)
            Predicted state sequence.

        Returns
        -------
        state_sequence : ndarray of shape (n_samples,)
            Predicted state sequence sorted in correct order.
        """
        # Slice data
        if X.ndim == 1:  # Makes function compatible on higher dimensions
            X = X[(self.window_len[-1] - 1):]
        else:
            raise Exception('''X must be one-dimensional. Pass raw data as 1-D array.''')

        df_states = pd.DataFrame({'state_seq': state_sequence,
                                  'X': X})

        state_groupby = df_states.groupby('state_seq')
        self.std = state_groupby['X'].std(ddof=1).values.T

        # Sort array ascending and check if order is changed
        # If the order is changed then states are reversed
        if np.sort(self.std)[0] != self.std[0]:
            state_sequence = np.where(state_sequence == 0, 1, 0)

        return state_sequence

    def bac_score_1d(self, X, y_true, jump_penalty, window_len=(6,14), verbose=False):
        self.jump_penalty = jump_penalty
        self.fit(X, get_hmm_params=False)  # Updates self.state_seq
        self.state_seq = self._check_state_sort(X, self.state_seq)

        y_pred = self.state_seq
        y_true = y_true[(self.window_len[-1] - 1):]  # slice y_true to have same dim as y_pred

        conf_matrix = confusion_matrix(y_true, y_pred)
        keep_idx = conf_matrix.sum(axis=1) != 0
        conf_matrix = conf_matrix[keep_idx]

        tp = np.diag(conf_matrix)
        fn = conf_matrix.sum(axis=1) - tp
        tpr = tp / (tp + fn)
        bac = np.mean(tpr)

        if verbose is True:
            logical_1 = bac < 0.5
            logical_2 = conf_matrix.ndim > 1 and \
                        (np.any(conf_matrix.sum(axis=1)==0) and \
                        jump_penalty < 100)

            if logical_1 or logical_2:
                print(f'bac {bac} -- tpr {tpr} -- jump_penalty {jump_penalty}')
                print(conf_matrix)

        return bac

    def bac_score_nd(self, X, y_true, jump_penalty, window_len=(6, 14)):
        bac = []
        for seq in tqdm.tqdm(range(X.shape[1])):
            bac_temp = self.bac_score_1d(X[:, seq], y_true[:, seq], jump_penalty, window_len=self.window_len)
            bac.append(bac_temp)

        return bac


if __name__ == '__main__':
    model = JumpHMM(n_states=2, jump_penalty=0.0025, window_len=(6, 14), random_state=2)
    sampler = SampleHMM(n_states=2, random_state=2)

    n_samples = 1000
    n_sequences = 1
    X, viterbi_states, true_states = sampler.sample_with_viterbi(n_samples, n_sequences)


    #plotting.plot_samples_states_viterbi(X[:,0], viterbi_states[:,0], true_states[:,0])

    #for i in range(50):
       # bac = model.bac_score_1d(X[:,i], viterbi_states[:, i] , 30)

    model.fit(X, verbose=True)




