import numpy as np
import tqdm

from utils.hmm_sampler import SampleHMM
from models.hidden_markov.hmm_gaussian_em import EMHiddenMarkov
from models.hidden_markov.hmm_jump import JumpHMM




if __name__ == '__main__':
    jump = JumpHMM(n_states=2, window_len=6)
    mle = EMHiddenMarkov(n_states=2)

    sampler = SampleHMM(n_states=2)
    #X, viterbi_states, true_states = sampler.sample_with_viterbi(2000, 1000)
    #np.save('../../analysis/model_convergence/output_data/sampled_returns.npy', X)
    #np.save('../../analysis/model_convergence/output_data/sampled_true_states.npy', true_states)
    #np.save('../../analysis/model_convergence/output_data/sampled_viterbi_states.npy', viterbi_states)

    path = '../../analysis/model_convergence/output_data/'
    X = np.load(path + 'sampled_returns.npy')
    true_states = np.load(path + 'sampled_true_states.npy')

    mu1_jump = []
    mu1_mle = []
    mu2_jump = []
    mu2_mle = []

    std1_jump = []
    std1_mle = []
    std2_jump = []
    std2_mle = []


    for seq in tqdm.trange(68, X.shape[1]):
        jump.fit(X[:, seq], sort_state_seq=True, get_hmm_params=True)
        mle.fit(X[:, seq], sort_state_seq=True)

        mu1_jump.append(jump.mu[0])
        mu1_mle.append(mle.mu[0])
        mu2_jump.append(jump.mu[1])
        mu2_mle.append(mle.mu[1])

        std1_jump.append(jump.std[0])
        std1_mle.append(mle.std[0])
        std2_jump.append(jump.std[1])
        std2_mle.append(mle.std[1])



