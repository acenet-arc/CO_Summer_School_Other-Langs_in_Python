"""
CYTHON MATRIX MULTIPLICATION - STUDENT EXERCISE
================================================

This exercise teaches you how to write high-performance code using Cython.
The code contains intentional gaps (marked as TODO 1-7) that you need to complete.

FILES:
------
- matrix_multiply.pyx      : Cython implementation with missing pieces (fill these in!)
- setup.py                 : Build configuration for compiling Cython code
- test_cython_matrix.py    : Testing and timing script
- README_CYTHON.txt        : This file

GETTING STARTED:
----------------

1. First, ensure you have the required packages:
   pip install cython numpy

2. Complete the TODO items in matrix_multiply.pyx

3. Compile the Cython module:
   python3 setup.py build_ext --inplace
   
4. Run the tests:
   python3 test_cython_matrix.py

EXERCISES TO COMPLETE:
----------------------

TODO 1: Declare typed memoryviews
   Location: Top of matrix_multiply.pyx
   What: Create typed memoryview declarations for efficient array access
   Why: Typed memoryviews allow Cython to generate fast C code for array operations
   
TODO 2: Create result array using np.zeros
   Location: matrix_multiply_python() function
   What: Create a zero-initialized matrix of the correct size
   Hint: The size should be (m x p)
   
TODO 3: Complete matrix multiplication operation
   Location: matrix_multiply_python() function, innermost loop
   What: Write the accumulation operation: result[i][j] += matrix_a[i][k] * matrix_b[k][j]
   Why: This is the core computation of matrix multiplication
   
TODO 4: Declare loop variables as C integers
   Location: matrix_multiply_optimized() function
   What: Add type declarations for i, j, k variables as 'cdef int'
   Why: C integers are much faster than Python integers
   
TODO 5: Optimize with temporary accumulator
   Location: matrix_multiply_optimized() function
   What: Use the 'temp' variable to accumulate products instead of direct array access
   Why: Reduces array indexing overhead, better cache utilization
   
TODO 6: Convert to parallel processing
   Location: matrix_multiply_parallel() function
   What: Change 'for i in range(m):' to 'for i in prange(m, nogil=True):'
   Why: prange uses OpenMP to parallelize across multiple CPU cores
   
TODO 7: (Optional Challenge) Implement block matrix multiplication
   Location: End of matrix_multiply.pyx
   What: Implement blocked matrix multiplication for better cache efficiency
   Hint: Divide matrices into blocks and multiply block-by-block
   Reference: https://en.wikipedia.org/wiki/Block_matrix

KEY CYTHON CONCEPTS:
-------------------

@cython.boundscheck(False)   - Disable bounds checking (unsafe but faster)
@cython.wraparound(False)    - Disable negative indexing (unsafe but faster)
@cython.parallel(True)       - Enable OpenMP parallelization
nogil=True                   - Release Python's Global Interpreter Lock
cdef                         - Declare C variables (much faster than Python)
double[:, ::1]               - Typed memoryview for 2D arrays
prange()                     - Parallel range (like range() but parallel)

EXPECTED PERFORMANCE:
--------------------

Once completed, you should see results like:

Size 100x100:
  NumPy:              ~0.0002s
  Cython-Python:      ~0.003s
  Cython-Optimized:   ~0.001s
  Cython-Parallel:    ~0.0005s

Size 500x500:
  NumPy:              ~0.005s
  Cython-Python:      ~0.5s
  Cython-Optimized:   ~0.1s
  Cython-Parallel:    ~0.05s  (on multi-core CPU)

NumPy is still fastest because it uses highly optimized BLAS libraries,
but Cython can be competitive and is great for custom algorithms!

TROUBLESHOOTING:
----------------

"ImportError: No module named 'matrix_multiply'"
  → Run: python3 setup.py build_ext --inplace

"error: incompatible types in assignment"
  → Check that variable types match (e.g., double vs int)

"AttributeError: 'numpy.ndarray' object has no attribute 'shape'"
  → Make sure you're passing numpy arrays, not lists

"OpenMP not available"
  → The parallel version won't compile, but others will still work
  → On macOS, install: brew install llvm
  → Then add to setup.py: extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp']

LEARNING RESOURCES:
-------------------

Cython Documentation: https://cython.readthedocs.io/
NumPy arrays in Cython: https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
Parallelization: https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html

"""
