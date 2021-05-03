from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


#from distutils.extension import Extension
#from Cython.Distutils import build_ext

#ext_modules = [Extension("hmm_cython", ["hidden_markov/hmm_cython.pyx"])]

class build_ext(build_ext):
    def finalize_options(self):
        from Cython.Build import cythonize
        import numpy as np
        import numpy.distutils

        self.distribution.ext_modules[:] = cythonize("models/hidden_markov/*.pyx",language_level='3')
        # Sadly, this part needs to be done manually.
        for ext in self.distribution.ext_modules:
            for k, v in np.distutils.misc_util.get_info("npymath").items():
                setattr(ext, k, v)
            ext.include_dirs = [np.get_include()]

        super().finalize_options()

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        super().build_extensions()

setup(
    name='cstolborg',
    version='0.0.2',
    url='https://github.com/Cstolborg/hmm-master',
    author='Cstolborg',
    author_email='christianstolborg@gmail.com',
    description='A python module to implement Hidden Markov hidden_markov for financial times series.',
    packages=['utils', 'models'],
    classifiers =[
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    setup_requires=[
        'Cython',
        'numpy>=1.20.1'
    ],
    install_requires=[
        "numpy>=1.20.1",
        "scikit-learn>=0.24.0",
        "scipy>=1.5.4"
    ],
    extras_require={
        'docs': ['Sphinx', 'sphinx-gallery']
    },

    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("", [])]
    #ext_modules=cythonize('hidden_markov/hmm_cython.pyx', include_path=[np.get_include()])
    #cmdclass = {'build_ext': build_ext},
    #ext_modules=cythonize('hidden_markov/hmm_cython.pyx')
    #include_dirs = [numpy.get_include()]
)

