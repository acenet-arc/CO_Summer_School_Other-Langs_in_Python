"""
Matrix Multiplication Timing Program with Optional Numba JIT Compilation

This program compares:
1. Pure Python naive approach (triple-nested loop)
2. Numba JIT compiled version (with Python mode) - if numba is available
3. Numba NJIT compiled version (without Python mode) - if numba is available
4. NumPy's optimized approach

Demonstrates how Numba's JIT compilation dramatically speeds up naive implementations.

To use this program, install numba:
    pip install numba
"""

import time
import numpy as np

# Try to import numba
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: Numba not installed. Install with: pip install numba")
    print("Numba JIT versions will be skipped.\n")


# Pure Python version (no decorators)
def naive_matrix_multiply_python(matrix_a, matrix_b):
    """
    Multiply two matrices using the naive triple-nested loop approach.
    Pure Python implementation.
    
    Args:
        matrix_a: An m x n matrix (list of lists)
        matrix_b: An n x p matrix (list of lists)
        
    Returns:
        The product matrix (m x p) as a list of lists
    """
    m = len(matrix_a)
    n = len(matrix_a[0])
    p = len(matrix_b[0])
    
    result = [[0.0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result


# Numba JIT version - only define if numba is available
if HAS_NUMBA:
    @numba.jit
    def naive_matrix_multiply_numba_jit(matrix_a, matrix_b):
        """
        Multiply two matrices using Numba JIT compilation.
        This version uses @numba.jit which allows Python fallback.
        """
        m = len(matrix_a)
        n = len(matrix_a[0])
        p = len(matrix_b[0])
        
        result = np.zeros((m, p))
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
        
        return result


    # Numba NJIT version - pure numerical compilation, no Python fallback
    @numba.njit
    def naive_matrix_multiply_numba_njit(matrix_a, matrix_b):
        """
        Multiply two matrices using Numba NJIT compilation.
        This version uses @numba.njit (no-Python mode) for maximum speed.
        Requires all operations to be compilable to machine code.
        """
        m = len(matrix_a)
        n = len(matrix_a[0])
        p = len(matrix_b[0])
        
        result = np.zeros((m, p))
        
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
        
        return result


def time_function(func, matrix_a, matrix_b, warmup=True):
    """
    Time a function with optional warmup run.
    
    Args:
        func: Function to time
        matrix_a: First matrix
        matrix_b: Second matrix
        warmup: Whether to do a warmup run (important for JIT)
        
    Returns:
        Tuple of (time in seconds, result)
    """
    if warmup:
        # Warmup run - especially important for JIT compilation
        func(matrix_a, matrix_b)
    
    start_time = time.time()
    result = func(matrix_a, matrix_b)
    elapsed_time = time.time() - start_time
    
    return elapsed_time, result


def main():
    """Main function to demonstrate matrix multiplication timing."""
    
    print("=" * 75)
    print("Matrix Multiplication Timing with Numba JIT Compilation")
    print("=" * 75)
    
    # Test with different matrix sizes
    sizes = [100, 200, 300]
    
    for size in sizes:
        print(f"\n{'=' * 75}")
        print(f"Matrix Size: {size}x{size}")
        print(f"{'=' * 75}")
        
        # Generate random matrices
        np.random.seed(42)
        matrix_a_np = np.random.rand(size, size)
        matrix_b_np = np.random.rand(size, size)
        
        # Time Pure Python approach
        print("\n1. Pure Python (triple-nested loop):")
        python_time, result_python = time_function(
            naive_matrix_multiply_python,
            matrix_a_np.tolist(),
            matrix_b_np.tolist(),
            warmup=False
        )
        print(f"   Time: {python_time:.6f} seconds")
        
        # Time Numba JIT approach (if available)
        if HAS_NUMBA:
            print("\n2. Numba JIT (@numba.jit):")
            jit_time, result_jit = time_function(
                naive_matrix_multiply_numba_jit,
                matrix_a_np,
                matrix_b_np,
                warmup=True  # Warmup is important for JIT
            )
            print(f"   Time: {jit_time:.6f} seconds")
            print(f"   Speedup vs Pure Python: {python_time / jit_time:.1f}x")
            
            # Time Numba NJIT approach
            print("\n3. Numba NJIT (@numba.njit - no-Python mode):")
            njit_time, result_njit = time_function(
                naive_matrix_multiply_numba_njit,
                matrix_a_np,
                matrix_b_np,
                warmup=True  # Warmup is important for JIT
            )
            print(f"   Time: {njit_time:.6f} seconds")
            print(f"   Speedup vs Pure Python: {python_time / njit_time:.1f}x")
            print(f"   Speedup vs Numba JIT: {jit_time / njit_time:.1f}x")
            
            # Adjust print indices based on numba availability
            numpy_index = 4
        else:
            numpy_index = 2
        
        # Time NumPy approach
        print(f"\n{numpy_index}. NumPy optimized approach:")
        start_time = time.time()
        result_numpy = np.dot(matrix_a_np, matrix_b_np)
        numpy_time = time.time() - start_time
        print(f"   Time: {numpy_time:.6f} seconds")
        print(f"   Speedup vs Pure Python: {python_time / numpy_time:.1f}x")
        
        # Verify correctness
        print(f"\n{'=' * 75}")
        print("Verification (max difference from NumPy result):")
        if HAS_NUMBA:
            max_error_jit = np.max(np.abs(result_jit - result_numpy))
            max_error_njit = np.max(np.abs(result_njit - result_numpy))
            print(f"   Numba JIT: {max_error_jit:.2e}")
            print(f"   Numba NJIT: {max_error_njit:.2e}")
        
        # Summary
        print(f"\n{'=' * 75}")
        print(f"SUMMARY for {size}x{size} matrices:")
        print(f"  Pure Python: {python_time:.6f}s (baseline)")
        if HAS_NUMBA:
            print(f"  Numba JIT:   {jit_time:.6f}s ({python_time/jit_time:.1f}x faster)")
            print(f"  Numba NJIT:  {njit_time:.6f}s ({python_time/njit_time:.1f}x faster)")
        print(f"  NumPy:       {numpy_time:.6f}s ({python_time/numpy_time:.1f}x faster)")


if __name__ == "__main__":
    main()
