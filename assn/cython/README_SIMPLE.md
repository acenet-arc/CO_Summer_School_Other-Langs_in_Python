# Simple Cython Matrix Multiplication Exercise

A minimal, beginner-friendly Cython exercise with just 4 TODO items.

## Files

- **simple_matrix.pyx** - The exercise file (edit this!)
- **setup_simple.py** - Compilation configuration
- **test_simple.py** - Testing script

## Quick Start

### 1. Install Cython and NumPy
```bash
pip install cython numpy
```

### 2. Complete the 4 TODO items in simple_matrix.pyx
- TODO 1: Get matrix dimensions (m, n, p)
- TODO 2: Create result array with np.zeros
- TODO 3: Declare loop variables as C integers
- TODO 4: Write the multiplication loops

### 3. Compile
```bash
python3 setup_simple.py build_ext --inplace
```

### 4. Test
```bash
python3 test_simple.py
```

## Expected Output

```
Matrix Size: 100x100
NumPy (reference):        0.000250 sec
Pure Python (very slow):  0.080000 sec (320x slower)
Cython compiled:          0.000150 sec (533x faster than Python)
Accuracy check:           Error = 2.84e-15 ✓
```

## What You'll Learn

- How to write a .pyx file
- Type declarations with `cdef`
- Typed memoryviews (`double[:, ::1]`)
- How Cython can make Python 500x faster!

## The Code Structure

```cython
@cython.boundscheck(False)    # Skip bounds checking (faster)
@cython.wraparound(False)     # Skip negative indexing (faster)
def multiply(double[:, ::1] a, double[:, ::1] b):
    # Get dimensions
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    cdef int p = b.shape[1]
    
    # Create result matrix
    cdef double[:, ::1] result = np.zeros((m, p))
    
    # Declare loop variables
    cdef int i, j, k
    
    # Multiply
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i, j] += a[i, k] * b[k, j]
    
    return np.asarray(result)
```

## Common Errors

**"No module named 'simple_matrix'"**
- Run: `python3 setup_simple.py build_ext --inplace`

**"Syntax error in .pyx file"**
- Make sure you use `cdef` for C variables
- Use consistent indentation (4 spaces)

**"incompatible types"**
- Make sure dimensions are `int`: `cdef int m`
- Make sure arrays are `double[:, ::1]`
