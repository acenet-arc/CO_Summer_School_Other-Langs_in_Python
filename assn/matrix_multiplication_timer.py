"""
Matrix Multiplication Timing Program

This program times how long it takes to multiply two matrices using a naive approach
(triple-nested loop) versus NumPy's optimized method.
"""

import time
import numpy as np


def naive_matrix_multiply(matrix_a, matrix_b):
    """
    Multiply two matrices using the naive triple-nested loop approach.
    
    Args:
        matrix_a: An m x n matrix (list of lists)
        matrix_b: An n x p matrix (list of lists)
        
    Returns:
        The product matrix (m x p) as a list of lists
    """
    m = len(matrix_a)           # rows in A
    n = len(matrix_a[0])        # cols in A (must equal rows in B)
    p = len(matrix_b[0])        # cols in B
    
    # Initialize result matrix with zeros
    result = [[0.0 for _ in range(p)] for _ in range(m)]
    
    # Triple nested loop - the naive approach
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result


def main():
    """Main function to demonstrate matrix multiplication timing."""
    
    print("=" * 60)
    print("Matrix Multiplication Timing Program")
    print("=" * 60)
    
    # Test with different matrix sizes
    # For simplicity, we'll use square matrices of size n x n
    sizes = [50, 100, 200, 300]
    
    for size in sizes:
        print(f"\n--- Matrix Size: {size}x{size} ---")
        
        # Generate random matrices
        np.random.seed(42)
        matrix_a_np = np.random.rand(size, size)
        matrix_b_np = np.random.rand(size, size)
        
        # Convert to list of lists for naive method
        matrix_a_list = matrix_a_np.tolist()
        matrix_b_list = matrix_b_np.tolist()
        
        # Time the naive approach
        print("\nNaive Approach (triple-nested loop):")
        start_time = time.time()
        result_naive = naive_matrix_multiply(matrix_a_list, matrix_b_list)
        naive_time = time.time() - start_time
        print(f"  Time: {naive_time:.6f} seconds")
        
        # Time NumPy's optimized approach
        print("\nNumPy optimized approach:")
        start_time = time.time()
        result_numpy = np.dot(matrix_a_np, matrix_b_np)
        numpy_time = time.time() - start_time
        print(f"  Time: {numpy_time:.6f} seconds")
        
        # Compare speedup
        speedup = naive_time / numpy_time if numpy_time > 0 else float('inf')
        print(f"\nNumPy is {speedup:.1f}x faster than naive implementation")
        
        # Verify correctness (check if results are close)
        result_naive_np = np.array(result_naive)
        max_error = np.max(np.abs(result_naive_np - result_numpy))
        print(f"Max difference between results: {max_error:.2e}")


if __name__ == "__main__":
    main()
