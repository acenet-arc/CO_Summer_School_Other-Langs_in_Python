"""
simple_matrix.pyx - Simple Cython matrix multiplication

This is a simple student exercise for learning Cython.
There are 4 TODO items to complete.

Compile with:
    python3 setup_simple.py build_ext --inplace
"""

import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def multiply(double[:, ::1] a, double[:, ::1] b):
    """
    Multiply two matrices using Cython.
    
    Args:
        a: First matrix (m x n)
        b: Second matrix (n x p)
    
    Returns:
        Result matrix (m x p)
    """
    # TODO 1: Get the dimensions
    # Uncomment and complete:
    #  m = a.shape[0]    # rows of a
    #  n = a.shape[1]    # cols of a (= rows of b)
    #  p = b.shape[1]    # cols of b

    
    # TODO 2: Create result array
    # Uncomment and complete:
    # cdef result = np.zeros((m, p))

    # TODO 3: Declare loop variables as C integers for speed
    # Uncomment and complete:
    # int i, j, k

    # TODO 4: Complete the multiplication loop
    # Write the three nested loops with the multiplication operation
    # Hint: result[i, j] += a[i, k] * b[k, j]
    

    return np.asarray(result)
