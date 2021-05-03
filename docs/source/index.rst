.. HMMpy documentation master file, created by
   sphinx-quickstart on Mon May  3 18:09:34 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HMMpy's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
HMMpy main
===================
.. automodule:: app
   :members:

HMMpy Base
=====================
.. autoclass:: hmmpy.hidden_markov.hmm_base.BaseHiddenMarkov
   :exclude-members: set_params, get_params, _get_param_names

HMMpy Gaussian
=================
.. autoclass:: hmmpy.hidden_markov.hmm_gaussian_em.EMHiddenMarkov
   :exclude-members: set_params, get_params, _get_param_names

HMMpy Jump
===================
.. autoclass:: hmmpy.hidden_markov.hmm_jump.JumpHMM
   :exclude-members: set_params, get_params, _get_param_names

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
