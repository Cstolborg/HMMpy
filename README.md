HMMpy
=====================
[![Build Status](https://travis-ci.com/cvxpy/cvxpy.png?branch=master)](https://travis-ci.com/github/cvxpy/cvxpy)
[![Build status](https://ci.appveyor.com/api/projects/status/jo7tkvc58c3hgfd7?svg=true)](https://ci.appveyor.com/project/StevenDiamond/cvxpy)

**The HMMpy documentation is at [insert link when website is ready](http://www.google.com/).**

- [Installation](#installation)
- [Getting started](#getting-started)

HMMpy is a Python-embedded modeling language for hidden markov models. It currently supports training of 2-state models using either maximum-likelihood or jump estimation, and uses and API that is very similar to scikit-learn.

HMMpy began as a University project at Copenhagen Business School, where it was used for financial times series forecasting in an asset allocation project. 


## Installation
HMMpy is available on TestPyPI, and can be installed with (only for windows)
```
pip install -i https://test.pypi.org/pypi/ --extra-index-url https://pypi.org/simple cstolborg==0.0.5
```

HMMpy has the following dependencies:

- Python >= 3.6
- NumPy >= 1.15
- SciPy >= 1.1.0

## Getting started
The following code samples some data, and then trains an hidden markov model using the JumpHMM class:

```python3
from hidden_markov.jump_hmm import JumpHMM
from hidden_markov.utils.sample_hmm import SampleHMM

# Instantiate the HMM model
hmm = JumpHMM(random_state=42)

# Instantiate the sampler using predefined HMM model parameters
hmm_params = {'mu': np.array([0.0123, -0.0157]) / 20,
                          'std': np.array([0.0347, 0.0778]) /np.sqrt(20),
                          'tpm': np.array([[1-0.0021, 0.0021],
                                           [0.0120, 1-0.0120]])
             }
sampler = SampleHMM(hmm_params=hmm_params, random_state=42)

# Simulate data
observations, state_sequence = sampler.sample(n_samples=1000, n_sequences=1)  # Outputs 1000 observations and the underlying states

# Fit the model
model.fit(observation)

# Inspect model parameters
print(model.mu)
print(model.std)
print(model.tpm)
```
