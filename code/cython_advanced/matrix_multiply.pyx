"""
matrix_multiply.pyx - Cython implementation of matrix multiplication

This is a Cython module for high-performance matrix multiplication.
Some parts are missing for students to complete.

To compile this file:
    python3 setup.py build_ext --inplace

Then use it in Python with:
    import matrix_multiply
"""

import numpy as np
cimport cython
from libc.math cimport fabs
from cython.parallel import prange

# TODO 1: Declare a typed memoryview for efficient array access
# Uncomment and complete the following line:
# cdef double[:, ::1] matrix_a_view, matrix_b_view, result_view


@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed
def matrix_multiply_python(double[:, ::1] matrix_a, double[:, ::1] matrix_b):
    """
    Pure Python-style matrix multiplication (no type hints).
    This version includes bounds checking and wrapping.
    
    Args:
        matrix_a: First input matrix (m x n)
        matrix_b: Second input matrix (n x p)
    
    Returns:
        Result matrix (m x p)
    """
    cdef int m = matrix_a.shape[0]
    cdef int n = matrix_a.shape[1]
    cdef int p = matrix_b.shape[1]
    
    # TODO 2: Create the result array using np.zeros
    # Complete the following line:
    cdef double[:, ::1] result = np.zeros((m, p))
    
    cdef int i, j, k
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                # TODO 3: Complete the matrix multiplication operation
                # This should accumulate the product result[i][j] += matrix_a[i][k] * matrix_b[k][j]
                result[i, j] += matrix_a[i, k] * matrix_b[k, j]
    
    return np.asarray(result)


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_multiply_optimized(double[:, ::1] matrix_a, double[:, ::1] matrix_b):
    """
    Optimized matrix multiplication with static type declarations.
    Uses typed C variables for better performance.
    
    Args:
        matrix_a: First input matrix (m x n)
        matrix_b: Second input matrix (n x p)
    
    Returns:
        Result matrix (m x p)
    """
    cdef int m = matrix_a.shape[0]
    cdef int n = matrix_a.shape[1]
    cdef int p = matrix_b.shape[1]
    
    cdef double[:, ::1] result = np.zeros((m, p))
    
    # TODO 4: Declare all loop variables as C integers for speed
    # Complete the following lines:
    cdef int i, j, k
    cdef double temp
    
    for i in range(m):
        # TODO 5: This inner loop could be optimized by accumulating in a temporary variable
        # Modify the inner loops to use 'temp' variable for better cache efficiency
        for j in range(p):
            temp = 0.0  # Initialize temporary accumulator
            for k in range(n):
                temp += matrix_a[i, k] * matrix_b[k, j]
            result[i, j] = temp
    
    return np.asarray(result)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.parallel(True)
def matrix_multiply_parallel(double[:, ::1] matrix_a, double[:, ::1] matrix_b):
    """
    Parallel matrix multiplication using OpenMP.
    This version uses Cython's parallel capabilities for multi-core processing.
    
    Args:
        matrix_a: First input matrix (m x n)
        matrix_b: Second input matrix (n x p)
    
    Returns:
        Result matrix (m x p)
    """
    cdef int m = matrix_a.shape[0]
    cdef int n = matrix_a.shape[1]
    cdef int p = matrix_b.shape[1]
    
    cdef double[:, ::1] result = np.zeros((m, p))
    
    cdef int i, j, k
    cdef double temp
    
    # TODO 6: Convert the outer loop to use parallel processing with prange()
    # Replace 'range' with 'prange' in the outer loop to enable parallel execution
    # (Remember to keep the original 'for i in range(m):' pattern but use prange instead)
    for i in prange(m, nogil=True):  # 'nogil=True' releases the Python GIL for speed
        for j in range(p):
            temp = 0.0
            for k in range(n):
                temp += matrix_a[i, k] * matrix_b[k, j]
            result[i, j] = temp
    
    return np.asarray(result)


# TODO 7 (Optional Challenge): Implement a block matrix multiplication function
# This would involve dividing matrices into blocks for better cache locality.
# Function signature:
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def matrix_multiply_blocked(double[:, ::1] matrix_a, double[:, ::1] matrix_b, int block_size):
#     """Block matrix multiplication for improved cache efficiency."""
#     # Your implementation here
#     pass
