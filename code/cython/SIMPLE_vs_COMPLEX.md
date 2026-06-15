# Cython Exercise - Simple vs Complex Comparison

## Summary

I've created TWO versions of the Cython matrix multiplication exercise:

### **SIMPLE VERSION** (Recommended for Beginners)
**Perfect for quick learning - 4 TODO items, ~140 lines total**

Files:
- `simple_matrix.pyx` - Main exercise (59 lines)
- `setup_simple.py` - Build config (13 lines)
- `test_simple.py` - Tests (78 lines)
- `README_SIMPLE.md` - Instructions
- `SIMPLE_START.py` - Quick overview script

To get started:
```bash
python3 SIMPLE_START.py              # See overview
cat README_SIMPLE.md                 # Read guide
# Edit simple_matrix.pyx and complete 4 TODOs
python3 setup_simple.py build_ext --inplace  # Compile
python3 test_simple.py               # Test
```

**4 TODO Items (Easy):**
1. Get dimensions (m, n, p)
2. Create result array
3. Declare loop variables
4. Write multiplication loops

---

### **COMPLEX VERSION** (For Advanced Learning)
**Full-featured - 7 TODO items, parallelization, multiple implementations**

Files:
- `matrix_multiply.pyx` - 3 implementations (140 lines)
- `setup.py` - Build config
- `test_cython_matrix.py` - Comprehensive tests
- `CYTHON_EXERCISE_GUIDE.md` - Detailed guide
- `README_CYTHON.txt` - Reference
- `CYTHON_QUICKSTART.py` - Overview
- `00_START_HERE.txt` - Entry point

**7 TODO Items:**
1. Memoryview declarations (optional)
2. Create result array (required)
3. Multiplication operation (required)
4. Declare loop variables (required)
5. Temporary accumulator (done - shows pattern)
6. Parallel with prange() (required)
7. Block multiplication (optional challenge)

---

## Comparison Table

| Feature | Simple | Complex |
|---------|--------|---------|
| **Number of TODO items** | 4 | 7 |
| **Total lines of code** | ~140 | ~1,100 |
| **Number of implementations** | 1 | 3 |
| **Parallelization** | No | Yes (TODO 6) |
| **Block optimization** | No | Yes (TODO 7) |
| **Documentation** | Light | Comprehensive |
| **Learning time** | ~30 min | ~2-3 hours |
| **Difficulty** | Beginner | Intermediate-Advanced |

---

## Which One to Choose?

### Choose **SIMPLE** if:
- You want to learn Cython quickly
- You're new to Cython
- You have limited time
- You want to see immediate results
- You want to understand the basics

### Choose **COMPLEX** if:
- You want comprehensive coverage
- You want to learn parallelization
- You want multiple examples
- You have time for detailed learning
- You want bonus challenges

---

## File Locations

All files are in:
```
/home/bernardj/repos/CO_Summer_School_Other-Langs_in_Python/code/
```

Simple files start with: `simple_*` or `SIMPLE_*` or `README_SIMPLE.md`

Complex files start with: `matrix_multiply*` or `CYTHON_*` or `00_START_HERE.txt`

---

## Expected Results

### Simple Version (100×100 matrix):
```
NumPy:          0.00025 sec (reference)
Pure Python:    0.08 sec
Cython:         0.0001 sec (800x faster than Python!)
```

### Complex Version (300×300 matrix):
```
NumPy:          0.001 sec (reference)
Pure Python:    2.5 sec
Cython basic:   0.5 sec (5x faster)
Cython optimized: 0.1 sec (25x faster)
Cython parallel:  0.05 sec (50x faster with multi-core)
```

---

## My Recommendation

**Start with SIMPLE version**, then if you want more:
1. Complete simple version (30 minutes)
2. See the speedup and understand the basics
3. Move to complex version for parallelization and optimization techniques
4. Try the bonus challenges

This is a better learning progression than jumping into the complex version!
