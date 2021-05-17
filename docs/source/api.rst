API Documentation
=================

This is the documentation of ``hmmpy``. It consists of 3 general classes for estimation and one class used for sampling. They are  structured as follows:

* hmmpy.base.BaseHiddenMarkov: This class holds all the base methods shared across estimation methods. This includes methods related to an HMM that is already fitted (such as sampling), initialization methods etc.
* hmmpy.mle.MLEHMM: Methods related to maximum likeleihood estimation using Expectation Maximizing (Baum-Welch) algorithm.
* hmmpy.jump.JumpHMM: Methods related to jump estimation.

hmmpy.base

.. autoclass:: hmmpy.base.BaseHiddenMarkov
   :private-members:
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:

hmmpy.mle
=================
.. autoclass:: hmmpy.mle.MLEHMM
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:

hmmpy.jump
=================
.. autoclass:: hmmpy.jump.JumpHMM
   :private-members:
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members:


hmmpy.sampler
=================
.. autoclass:: hmmpy.sampler.SampleHMM
   :exclude-members: set_params, get_params, _get_param_names
   :no-inherited-members: