from setuptools import setup
from Cython.Build import cythonize

setup(
    name='hmm-master',
    version='0.0',
    packages=['utils', 'models'],
    url='',
    license='',
    author='Cstolborg',
    author_email='christianstolborg@gmail.com',
    description='A python module to implement Hidden Markov models for financial times series.',
    ext_modules=cythonize('models/hmm_cython.pyx')
)

