API Documentation
=================

This is the api reference of ``hmmpy``. The classes are structured as follows:

* hmmpy.Base: This class holds all the base methods shared across estimation methods. This includes methods related to an HMM that is already fitted (such as sampling), initialization methods etc.
* hmmpy.EMHiddenMarkov: Methods related to maximum likeleihood estimation using Expectation Maximizing (Baum-Welch) algorithm.
* hmmpy.JumpHMM: Methods related to jump estimation.

hmmpy.Base
----------
.. autoclass:: hmmpy.hidden_markov.hmm_base.BaseHiddenMarkov
   :private-members:
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:

HMMpy Gaussian
--------------
.. autoclass:: hmmpy.hidden_markov.hmm_gaussian_em.EMHiddenMarkov
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:

HMMpy Jump
-----------
.. autoclass:: hmmpy.hidden_markov.hmm_jump.JumpHMM
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members: