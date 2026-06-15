"""
setup_simple.py - Build configuration for simple_matrix.pyx

Run with:
    python3 setup_simple.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="Simple Matrix Multiply",
    ext_modules=cythonize("simple_matrix.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[np.get_include()],
)
