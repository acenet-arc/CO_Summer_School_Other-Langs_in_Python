"""
test_cython_matrix.py - Test and timing script for Cython matrix multiplication

This script tests and times the different Cython implementations.

Before running this script, compile the Cython module:
    python3 setup.py build_ext --inplace

Then run this test script:
    python3 test_cython_matrix.py
"""

import time
import numpy as np


def test_cython_implementations():
    """Test and time all Cython implementations."""
    
    print("=" * 75)
    print("Cython Matrix Multiplication - Testing and Timing")
    print("=" * 75)
    
    # Try to import the compiled Cython module
    try:
        import matrix_multiply
        print("\n✓ Cython module imported successfully!")
    except ImportError as e:
        print(f"\n✗ Error: Could not import Cython module: {e}")
        print("\nTo compile the Cython module, run:")
        print("    python3 setup.py build_ext --inplace")
        return
    
    # Test with different matrix sizes
    sizes = [100, 200, 400, 500]
    
    for size in sizes:
        print(f"\n{'=' * 75}")
        print(f"Matrix Size: {size}x{size}")
        print(f"{'=' * 75}")
        
        # Generate random test matrices
        np.random.seed(42)
        matrix_a = np.random.rand(size, size).astype(np.float64)
        matrix_b = np.random.rand(size, size).astype(np.float64)
        
        # Reference result using NumPy
        print("\nComputing reference result with NumPy...")
        start = time.time()
        result_numpy = np.dot(matrix_a, matrix_b)
        numpy_time = time.time() - start
        print(f"NumPy time: {numpy_time:.6f} seconds")
        
        # Test matrix_multiply_python
        print("\n1. Cython - Python-style (with bounds checking):")
        try:
            # Warmup
            matrix_multiply.matrix_multiply_python(matrix_a, matrix_b)
            
            # Timed run
            start = time.time()
            result_python = matrix_multiply.matrix_multiply_python(matrix_a, matrix_b)
            python_time = time.time() - start
            print(f"   Time: {python_time:.6f} seconds")
            
            # Verify correctness
            error = np.max(np.abs(result_python - result_numpy))
            print(f"   Max error vs NumPy: {error:.2e}")
            print(f"   Speedup vs NumPy: {numpy_time / python_time:.1f}x")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test matrix_multiply_optimized
        print("\n2. Cython - Optimized (with temp accumulator):")
        try:
            # Warmup
            matrix_multiply.matrix_multiply_optimized(matrix_a, matrix_b)
            
            # Timed run
            start = time.time()
            result_optimized = matrix_multiply.matrix_multiply_optimized(matrix_a, matrix_b)
            optimized_time = time.time() - start
            print(f"   Time: {optimized_time:.6f} seconds")
            
            # Verify correctness
            error = np.max(np.abs(result_optimized - result_numpy))
            print(f"   Max error vs NumPy: {error:.2e}")
            print(f"   Speedup vs NumPy: {numpy_time / optimized_time:.1f}x")
            
            # Compare to python version
            if 'python_time' in locals():
                print(f"   Speedup vs Python version: {python_time / optimized_time:.1f}x")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test matrix_multiply_parallel
        print("\n3. Cython - Parallel (with OpenMP multi-core):")
        try:
            # Warmup
            matrix_multiply.matrix_multiply_parallel(matrix_a, matrix_b)
            
            # Timed run
            start = time.time()
            result_parallel = matrix_multiply.matrix_multiply_parallel(matrix_a, matrix_b)
            parallel_time = time.time() - start
            print(f"   Time: {parallel_time:.6f} seconds")
            
            # Verify correctness
            error = np.max(np.abs(result_parallel - result_numpy))
            print(f"   Max error vs NumPy: {error:.2e}")
            print(f"   Speedup vs NumPy: {numpy_time / parallel_time:.1f}x")
            
            # Compare to other versions
            if 'python_time' in locals():
                print(f"   Speedup vs Python version: {python_time / parallel_time:.1f}x")
            if 'optimized_time' in locals():
                print(f"   Speedup vs Optimized version: {optimized_time / parallel_time:.1f}x")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Summary
        print(f"\n{'-' * 75}")
        print("SUMMARY:")
        times = {}
        if 'numpy_time' in locals():
            times['NumPy'] = numpy_time
        if 'python_time' in locals():
            times['Cython-Python'] = python_time
        if 'optimized_time' in locals():
            times['Cython-Optimized'] = optimized_time
        if 'parallel_time' in locals():
            times['Cython-Parallel'] = parallel_time
        
        for name, t in sorted(times.items(), key=lambda x: x[1]):
            speedup = min(times.values()) / t if min(times.values()) > 0 else 1
            marker = "← FASTEST" if t == min(times.values()) else ""
            print(f"  {name:20s}: {t:.6f}s {marker}")


def main():
    """Main function."""
    try:
        test_cython_implementations()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")


if __name__ == "__main__":
    main()
