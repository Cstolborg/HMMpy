import numpy as np

from utils.hmm_sampler import SampleHMM

if __name__ == '__main__':
    sampler = SampleHMM(n_states=2, random_state=42)

    n_sequences = 10000
    n_samples = 2000

    # Sample normal data
    X, viterbi_states, true_states = sampler.sample_with_viterbi(n_samples, n_sequences)

    path = '../../analysis/model_convergence/input_data/'
    np.save(path + 'sampled_returns_10k.npy', X)
    np.save(path + 'sampled_viterbi_states_10k.npy', viterbi_states)
    np.save(path + 'sampled_true_states_10k.npy', true_states)

    # Sample t-distributed data
    X, true_states = sampler.sample_t(n_samples, n_sequences, dof=5)
    np.save(path + 'sampled_t_returns_10k.npy', X)
    np.save(path + 'sampled_t_true_states_10k.npy', true_states)