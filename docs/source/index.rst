.. HMMpy documentation master file, created by
   sphinx-quickstart on Mon May  3 18:09:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HMMpy
=====

HMMpy is a Python-embedded modeling language for hidden markov models. It currently supports training of 2-state models using either maximum-likelihood or jump estimation, and uses and API that is very similar to scikit-learn.

Table of contents
------------------
.. toctree::
   :maxdepth: 2

   api

-  `Installation <#installation>`__
-  `Getting started <#getting-started>`__

Installation
------------

HMMpy is available on PyPI, and can be installed with (only available on Windows and Linux)

::

    pip install hmm-py

HMMpy has the following dependencies:

-  Python >= 3.8
-  Cython >= 0.29
-  NumPy >= 1.20.1
-  Pandas >= 1.2.0
-  SciPy >= 1.5.4
-  tqdm

Getting started
---------------

The following code samples some data, and then trains a hidden markov
model using the JumpHMM class:

.. code:: python3

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


Available models
------------------

.. autosummary::
   :nosignatures:

   hmmpy.mle.MLEHMM
   hmmpy.jump.JumpHMM
   hmmpy.sampler.SampleHMM