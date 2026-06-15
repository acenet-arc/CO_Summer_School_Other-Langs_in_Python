# Cython Matrix Multiplication - Exercise Guide

## Overview
This exercise teaches you to optimize Python code using Cython. You'll complete 7 TODOs in `matrix_multiply.pyx` to create increasingly optimized matrix multiplication implementations.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install cython numpy
```

### 2. Complete the TODOs
Edit `matrix_multiply.pyx` and fill in the missing code at each TODO location.

### 3. Compile the Module
```bash
python3 setup.py build_ext --inplace
```

### 4. Run the Tests
```bash
python3 test_cython_matrix.py
```

---

## TODO Items Explained

### TODO 1: Declare Typed Memoryviews (OPTIONAL)
**Location:** Top of file, after imports  
**Type:** Optional/Informational  
**Description:** Commented code showing how to declare memoryviews for global use  
**Why it matters:** Typed memoryviews allow Cython to generate fast C code

---

### TODO 2: Create Result Array
**Location:** `matrix_multiply_python()` function, after variable declarations  
**What to do:** Initialize a 2D array filled with zeros  
**Hints:**
- Use `np.zeros()` to create a zero-filled array
- Size should be `(m, p)` where m = rows, p = columns
- Result type should be `double[:, ::1]` (typed memoryview)

**Expected code:**
```python
cdef double[:, ::1] result = np.zeros((m, p))
```

---

### TODO 3: Matrix Multiplication Operation
**Location:** `matrix_multiply_python()` function, innermost loop  
**What to do:** Implement the core multiplication accumulation  
**Hints:**
- The operation accumulates products: `result[i, j] += matrix_a[i, k] * matrix_b[k, j]`
- This loop runs n times (the inner dimension)
- You're computing the dot product of row i with column j

**Expected code:**
```cython
result[i, j] += matrix_a[i, k] * matrix_b[k, j]
```

---

### TODO 4: Declare Loop Variables as C Integers
**Location:** `matrix_multiply_optimized()` function, after size calculations  
**What to do:** Add type declarations for loop variables  
**Hints:**
- Variables: i, j, k (already used in loops)
- Type: `cdef int`
- This makes loop operations much faster

**Expected code:**
```cython
cdef int i, j, k
cdef double temp
```

---

### TODO 5: Use Temporary Accumulator (ALREADY DONE!)
**Location:** `matrix_multiply_optimized()` function, inner loops  
**Status:** ✓ COMPLETED for you  
**Explanation:** This shows how using a temporary variable instead of directly accessing array elements can improve performance through better cache efficiency

---

### TODO 6: Convert to Parallel Processing
**Location:** `matrix_multiply_parallel()` function, outer loop  
**What to do:** Enable parallel execution using OpenMP  
**Hints:**
- Replace `for i in range(m):` with `for i in prange(m, nogil=True):`
- `prange` = parallel range (from `cython.parallel`)
- `nogil=True` releases Python's Global Interpreter Lock for speed
- This distributes the outer loop across multiple CPU cores

**Expected code:**
```cython
for i in prange(m, nogil=True):
    for j in range(p):
        temp = 0.0
        for k in range(n):
            temp += matrix_a[i, k] * matrix_b[k, j]
        result[i, j] = temp
```

---

### TODO 7: Block Matrix Multiplication (OPTIONAL CHALLENGE)
**Location:** End of `matrix_multiply.pyx`  
**Type:** Advanced/Optional  
**Challenge:** Implement blocked matrix multiplication  
**Why:** Blocks improve cache efficiency by keeping data working set in CPU cache  
**Hints:**
- Function signature provided as comment
- Divide matrices into `block_size × block_size` blocks
- Multiply blocks instead of individual elements
- Standard block size: 64, 128, or 256

---

## Key Cython Features Used

| Feature | Purpose | Speed Improvement |
|---------|---------|-------------------|
| `@cython.boundscheck(False)` | Skip array bounds checking | ~5-10% |
| `@cython.wraparound(False)` | Skip negative index handling | ~5-10% |
| `cdef double[,]` | Use C arrays instead of Python objects | ~100x |
| `cdef int` | Use C integers for loops | ~50x |
| `double[:, ::1]` | Typed memoryviews for fast access | ~10x |
| `prange()` | Parallel loop with OpenMP | ~4-8x (on 8 cores) |
| `nogil=True` | Release Python's Global Lock | Enables parallelism |

---

## Expected Results

After completing all TODOs, you should see timing like this for a 500×500 matrix:

```
NumPy:              ~0.005s (using highly optimized BLAS)
Cython-Python:      ~0.5s   (simple loop)
Cython-Optimized:   ~0.1s   (with temp accumulator, 5x speedup)
Cython-Parallel:    ~0.05s  (with OpenMP, 10x speedup on 8 cores)
```

NumPy is still faster because it uses BLAS libraries, but notice how Cython 
optimizations compound: each technique gives incremental improvements!

---

## Compilation Tips

### Verbose output (see what's happening):
```bash
python3 setup.py build_ext --inplace -v
```

### Force recompilation:
```bash
rm -rf build *.so *.c
python3 setup.py build_ext --inplace
```

### Check if OpenMP is available:
```bash
python3 -c "from Cython.Compiler.Options import get_directive_defaults; print(get_directive_defaults().get('parallel'))"
```

---

## Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ImportError: No module 'matrix_multiply'` | Not compiled | Run `python3 setup.py build_ext --inplace` |
| `error: incompatible types` | Type mismatch | Check variable declarations match usage |
| `SyntaxError` in .pyx | Cython syntax error | Use `cdef` for C variables, `def` for Python |
| `IndentationError` | Bad whitespace | Use consistent 4-space indentation |

---

## Extension Ideas

Once you complete the basic TODOs:

1. **TODO 7:** Implement the optional block multiplication
2. **Transpose optimization:** Transpose B before multiplication to improve cache efficiency
3. **SIMD:** Add `@cython.cdivision(True)` for additional speed
4. **Strassen algorithm:** Implement O(n^2.81) algorithm for very large matrices
5. **Memory profiling:** Add tracking of memory usage
6. **Different data types:** Try `float32` (single precision) for speed vs accuracy tradeoff

---

## Performance Profiling

Add timing to see which parts are slowest:

```cython
from libc.time cimport clock_t, clock, CLOCKS_PER_SEC

cdef clock_t start = clock()
# ... your code ...
cdef clock_t end = clock()
cdef double elapsed = <double>(end - start) / CLOCKS_PER_SEC
print(f"Time: {elapsed} seconds")
```

---

## Next Steps

1. ✓ Complete TODOs 1-6 (required)
2. □ Run tests and verify correctness
3. □ Compare performance across versions
4. □ Attempt TODO 7 (optional challenge)
5. □ Try extension ideas above

Good luck! Ask if you have questions about any Cython features.
