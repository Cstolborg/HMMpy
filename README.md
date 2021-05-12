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

- Python >= 3.8
- Cython >= 0.29
- NumPy >= 1.20.1
- Pandas >= 1.2.0
- SciPy >= 1.5.4
- tqdm


## Getting started
The following code samples some data, and then trains an hidden markov model using the JumpHMM class:

```python3
from hmmpy.jump import JumpHMM
from hmmpy.sampler import SampleHMM

# Instantiate the HMM model
hmm = JumpHMM(random_state=42)

# Instantiate the sampler with user defined HMM model parameters
hmm_params = {'mu': [0.1, -0.05],
                          'std': [0.1, 0.2],
                          'tpm': [[1-0.0021, 0.0021],
                                  [0.0120, 1-0.0120]]
             }
sampler = SampleHMM(hmm_params=hmm_params, random_state=42)

# Simulate data
observations, state_sequence = sampler.sample(n_samples=2000, n_sequences=1)  # Outputs 2000 observations and the underlying states

# Fit the model
hmm.fit(observations)

# Inspect model parameters
print(hmm.mu)
print(hmm.std)
print(hmm.tpm)
```
