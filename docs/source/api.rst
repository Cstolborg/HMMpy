API Documentation
=================

This is the api reference of ``hmmpy``. The classes are structured as follows:

* hmmpy.Base: This class holds all the base methods shared across estimation methods. This includes methods related to an HMM that is already fitted (such as sampling), initialization methods etc.
* hmmpy.EMHiddenMarkov: Methods related to maximum likeleihood estimation using Expectation Maximizing (Baum-Welch) algorithm.
* hmmpy.JumpHMM: Methods related to jump estimation.

hmmpy.base
----------
.. autoclass:: hmmpy.base.BaseHiddenMarkov
   :private-members:
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:

hmmpy.gaussian
--------------
.. autoclass:: hmmpy.mle.MLEHMM
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:

hmmpy.jump
-----------
.. autoclass:: hmmpy.jump.JumpHMM
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:


hmmpy.sampler
-----------
.. autoclass:: hmmpy.sampler.SampleHMM
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members: