from setuptools import Extension, setup
#from distutils.extension import Extension
from Cython.Build import cythonize
#from Cython.Distutils import build_ext
import numpy as np

#ext_modules = [Extension("hmm_cython", ["hmm_models/hmm_cython.pyx"])]

setup(
    name='hmm-master',
    version='0.0',
    packages=['utils', 'hmm_models'],
    url='',
    license='',
    author='Cstolborg',
    author_email='christianstolborg@gmail.com',
    description='A python module to implement Hidden Markov hmm_models for financial times series.',
    #cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize('hmm_models/hmm_cython.pyx', include_path=[np.get_include()])
    #ext_modules=cythonize('hmm_models/hmm_cython.pyx')
    #include_dirs = [numpy.get_include()]
)

