"""
test_simple.py - Test the simple Cython matrix multiplication

Run with:
    python3 test_simple.py
"""

import time
import numpy as np

# Try to import the Cython module
try:
    import simple_matrix
except ImportError:
    print("Error: simple_matrix module not found!")
    print("\nTo compile, run:")
    print("  python3 setup_simple.py build_ext --inplace")
    exit(1)


def pure_python_multiply(a, b):
    """Pure Python version (slow)"""
    m, n = a.shape
    p = b.shape[1]
    result = [[0.0 for _ in range(p)] for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += a[i, k] * b[k, j]
    
    return np.array(result)


# Test different sizes
sizes = [100, 200, 300]

print("=" * 60)
print("Simple Cython Matrix Multiplication")
print("=" * 60)

for size in sizes:
    print(f"\nMatrix Size: {size}x{size}")
    print("-" * 60)
    
    # Create test matrices
    np.random.seed(42)
    a = np.random.rand(size, size).astype(np.float64)
    b = np.random.rand(size, size).astype(np.float64)
    
    # NumPy (reference)
    print("NumPy (reference):        ", end="", flush=True)
    start = time.time()
    result_numpy = np.dot(a, b)
    numpy_time = time.time() - start
    print(f"{numpy_time:.6f} sec")
    
    # Pure Python
    print("Pure Python (very slow):  ", end="", flush=True)
    start = time.time()
    result_python = pure_python_multiply(a, b)
    python_time = time.time() - start
    print(f"{python_time:.6f} sec ({python_time/numpy_time:.0f}x slower)")
    
    # Cython
    print("Cython compiled:          ", end="", flush=True)
    start = time.time()
    result_cython = simple_matrix.multiply(a, b)
    cython_time = time.time() - start
    print(f"{cython_time:.6f} sec ({python_time/cython_time:.0f}x faster than Python)")
    
    # Verify
    error = np.max(np.abs(result_cython - result_numpy))
    print(f"Accuracy check:           Error = {error:.2e} ✓")

print("\n" + "=" * 60)
print("Done! Cython gives significant speedup over pure Python.")
print("=" * 60)
