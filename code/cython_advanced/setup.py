"""
setup.py - Build configuration for Cython matrix multiplication module

To compile the Cython module:
    python3 setup.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="Matrix Multiply Cython Module",
    ext_modules=cythonize(
        "matrix_multiply.pyx",
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[np.get_include()],
)
