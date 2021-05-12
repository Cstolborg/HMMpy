import numpy as np
import tqdm
from scipy import stats

from hmmpy.base import BaseHiddenMarkov

class SampleHMM(BaseHiddenMarkov):
    """
    Class to handle sampling from HMM hidden_markov with user parameters.
    

    Parameters
    ----------
    n_states : int, default=2
        Number of hidden states
    hmm_params: dict, default=None
        hmm model parameters to sample from.
        To set params, create a dict with 'mu', 'std' and 'tpm' as keys
        and their values in lists or numpy arrays.
    random_state : int, default = 42
        Parameter set to recreate output
 
    Attributes
    ----------
    mu : ndarray of shape (n_states,)
        means to sample from
    std : ndarray of shape (n_states,)
        STDs to sample from
    tpm : ndarray of shape (n_states, n_states)
        Transition probability matrix between states
    """
    
    def __init__(self, n_states=2, frequency='daily', hmm_params=None, random_state=42):

        if hmm_params == None or (hmm_params == None and frequency == "daily"):  # hmm params following Hardy (2001)
            # Convert from monthly time scale t=20 to daily t=1
            hmm_params = {'mu': np.array([0.0123, -0.0157]) / 20,
                          'std': np.array([0.0347, 0.0778]) /np.sqrt(20),
                          'tpm': np.array([[1-0.0021, 0.0021],  # TODO figure out powers of vectors in python
                                           [0.0120, 1-0.0120]])
                          }
        elif hmm_params == None and frequency == "monthly":
            hmm_params = {'mu': np.array([0.0123, -0.0157]),
                          'std': np.array([0.0347, 0.0778]),
                          'tpm': np.array([[0.9629, 0.0371],
                                           [0.2101, 0.7899]])
                          }

        self.type = 'sampler'
        self.is_fitted = True
        self.n_states = n_states
        self.mu = np.array(hmm_params['mu'])
        self.std = np.array(hmm_params['std'])
        self.tpm = np.array(hmm_params['tpm'])
        self.stationary_dist = super()._get_stationary_dist(self.tpm)
        self.start_proba = self.stationary_dist

        self.random_state = random_state
        np.random.seed(self.random_state)

    def sample(self, n_samples, n_sequences=1):
        '''
        Sample states from a fitted Hidden Markov Model.

        Parameters
        ----------
        n_samples : int
            Amount of samples to generate
        n_sequences : int, default=1
            Number of independent sequences to sample from, e.g. if n_samples=100 and n_sequences=3
            then 3 different sequences of length 100 are sampled

        Returns
        -------
        samples : ndarray of shape (n_samples, n_sequences)
            Outputs the generated samples of size n_samples
        sample_states : ndarray of shape (n_samples, n_sequences)
            Outputs sampled states
        '''
        mu = self.mu
        std = self.std
        tpm = self.tpm
        stationary_dist = self.stationary_dist

        state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=np.int32)  # Array of possible states
        sample_states = np.zeros(shape=(n_samples, n_sequences), dtype=np.int32) # Init sample vector
        samples = np.zeros(shape=(n_samples, n_sequences))  # Init sample vector

        print(f'Simulating {n_sequences} of lengths {n_samples}')
        for seq in tqdm.tqdm(range(n_sequences)):
            sample_states[0, seq] = np.random.choice(a=state_index, size=1, p=stationary_dist)

            for t in range(1, n_samples):
                # Each new state is chosen using the transition probs corresponding to the previous state sojourn.
                sample_states[t, seq] = np.random.choice(a=state_index, size=1, p=tpm[sample_states[t - 1, seq], :])

            samples[:, seq] = stats.norm.rvs(loc=mu[sample_states[:, seq]], scale=std[sample_states[:, seq]], size=n_samples)

        if n_sequences == 1:
            sample_states = sample_states[:, 0]
            samples = samples[:, 0]

        return samples, sample_states

    def sample_t(self, n_samples, n_sequences=1, dof=5):
        '''
        Sample states from a fitted Hidden Markov Model.

        Parameters
        ----------
        n_samples : int
            Amount of samples to generate
        n_sequences : int, default=1
            Number of independent sequences to sample from, e.g. if n_samples=100 and n_sequences=3
            then 3 different sequences of length 100 are sampled
        dof : int, default=5
            degrees of freedom in the conditional t-distributions.

        Returns
        -------
        samples : ndarray of shape (n_samples, n_sequences)
            Outputs the generated samples of size n_samples
        sample_states : ndarray of shape (n_samples, n_sequences)
            Outputs sampled states
        '''
        mu = self.mu
        std = self.std
        tpm = self.tpm
        stationary_dist = self.stationary_dist

        state_index = np.arange(start=0, stop=self.n_states, step=1, dtype=np.int32)  # Array of possible states
        sample_states = np.zeros(shape=(n_samples, n_sequences), dtype=np.int32) # Init sample vector
        samples = np.zeros(shape=(n_samples, n_sequences))  # Init sample vector

        for seq in tqdm.tqdm(range(n_sequences)):
            sample_states[0, seq] = np.random.choice(a=state_index, size=1, p=stationary_dist)

            for t in range(1, n_samples):
                # Each new state is chosen using the transition probs corresponding to the previous state sojourn.
                sample_states[t, seq] = np.random.choice(a=state_index, size=1, p=tpm[sample_states[t - 1, seq], :])

            samples[:, seq] = stats.t.rvs(loc=mu[sample_states[:, seq]],
                                          scale=std[sample_states[:, seq]]/np.sqrt(dof/(dof-2)),
                                          size=n_samples, df=dof)

        if n_sequences == 1:
            sample_states = sample_states[:, 0]
            samples = samples[:, 0]

        return samples, sample_states

    def sample_with_viterbi(self, n_samples, n_sequences=1):
        samples, true_states = self.sample(n_samples, n_sequences)

        viterbi_states = np.empty(shape=(n_samples, n_sequences), dtype=float)
        if n_sequences == 1:
            viterbi_states = self.decode(samples)
        else:
            for i in range(n_sequences):
                viterbi_states[:, i] = self.decode(samples[:, i])

        return samples, viterbi_states, true_states


if __name__ == "__main__":
    model = SampleHMM(n_states=2)
    print(model.mu)
    print(model.std)
    print(model.tpm)
    print(model.stationary_dist)



    n_samples = 1000
    n_sequences = 1000
    X, viterbi_states, true_states = model.sample_with_viterbi(n_samples, n_sequences)