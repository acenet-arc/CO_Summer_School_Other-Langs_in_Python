#!/usr/bin/env python3
"""
QUICK START GUIDE - Cython Matrix Multiplication Exercise
=========================================================

This is a student exercise for learning Cython optimization techniques.
"""

import os
import sys

def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")

def main():
    print_section("CYTHON MATRIX MULTIPLICATION - STUDENT EXERCISE")
    
    print("This exercise teaches you to optimize Python code using Cython.")
    print("You'll fill in 7 TODO items to learn about:")
    print("  • Type declarations (cdef, typed memoryviews)")
    print("  • Cython decorators (@cython.boundscheck, etc.)")
    print("  • Performance optimization techniques")
    print("  • Parallel programming with OpenMP")
    print("  • Benchmarking and timing code")
    
    print_section("FILES YOU'LL WORK WITH")
    
    files = {
        "matrix_multiply.pyx": "Main exercise file - fill in the 7 TODO items here",
        "setup.py": "Build configuration - runs compilation",
        "test_cython_matrix.py": "Testing script - times and verifies your code",
        "CYTHON_EXERCISE_GUIDE.md": "MAIN GUIDE - read this first!",
        "README_CYTHON.txt": "Quick reference for Cython concepts",
    }
    
    for filename, description in files.items():
        status = "✓" if os.path.exists(filename) else "✗"
        print(f"  {status} {filename:30s} - {description}")
    
    print_section("GETTING STARTED - 5 STEPS")
    
    steps = [
        ("Install dependencies", "pip install cython numpy"),
        ("Read the exercise guide", "cat CYTHON_EXERCISE_GUIDE.md"),
        ("Fill in the TODO items", "Edit matrix_multiply.pyx and complete TODOs 1-6"),
        ("Compile the module", "python3 setup.py build_ext --inplace"),
        ("Run tests and compare", "python3 test_cython_matrix.py"),
    ]
    
    for i, (description, command) in enumerate(steps, 1):
        print(f"  Step {i}: {description}")
        print(f"    $ {command}\n")
    
    print_section("THE 7 TODO ITEMS")
    
    todos = [
        ("TODO 1", "Memoryview declarations", "Optional/Informational", "See guide"),
        ("TODO 2", "Create result array", "REQUIRED", "Use np.zeros((m, p))"),
        ("TODO 3", "Multiply operation", "REQUIRED", "result[i,j] += matrix_a[i,k] * matrix_b[k,j]"),
        ("TODO 4", "Declare loop vars", "REQUIRED", "cdef int i, j, k, cdef double temp"),
        ("TODO 5", "Use temp variable", "DONE FOR YOU", "Shows optimization pattern"),
        ("TODO 6", "Parallel execution", "REQUIRED", "Use prange(m, nogil=True)"),
        ("TODO 7", "Block multiply", "Optional Challenge", "Advanced optimization"),
    ]
    
    print(f"{'#':<5} {'Description':<25} {'Status':<20} {'Hint':<30}")
    print("-" * 80)
    for todo, desc, status, hint in todos:
        print(f"{todo:<5} {desc:<25} {status:<20} {hint:<30}")
    
    print_section("EXPECTED PERFORMANCE IMPROVEMENTS")
    
    print("After completing each step, timing for 500×500 matrices should show:\n")
    print("  NumPy baseline:        ~5ms   (highly optimized)")
    print("  Pure Python version:   ~500ms (no optimization)")
    print("  After TODO 2-4:        ~500ms (types only)")
    print("  After TODO 5:          ~100ms (5x speedup with temp var)")
    print("  After TODO 6:          ~50ms  (10x speedup with OpenMP)")
    print("\nEach optimization compounds - notice how the improvements add up!")
    
    print_section("KEY CONCEPTS YOU'LL LEARN")
    
    concepts = {
        "Typed memoryviews": "double[:, ::1] - fast array access",
        "@cython.boundscheck(False)": "Skip bounds checking (unsafe but fast)",
        "@cython.wraparound(False)": "Skip negative indexing (unsafe but fast)",
        "@cython.parallel(True)": "Enable OpenMP parallelization",
        "cdef int": "C integer (much faster than Python int)",
        "prange()": "Parallel range from cython.parallel",
        "nogil=True": "Release Python's Global Interpreter Lock",
    }
    
    for concept, explanation in concepts.items():
        print(f"  • {concept:<40} → {explanation}")
    
    print_section("TIPS FOR SUCCESS")
    
    tips = [
        "Read CYTHON_EXERCISE_GUIDE.md first - it has detailed explanations",
        "Look at the hints in the guide before implementing each TODO",
        "Test after each step to see performance improvements",
        "Ask questions if compilation fails - check troubleshooting guide",
        "Try the optional challenges if you finish early",
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")
    
    print_section("LEARNING OUTCOMES")
    
    print("After completing this exercise, you'll understand:")
    print("  ✓ How Cython compiles Python to C for speed")
    print("  ✓ How type declarations dramatically improve performance")
    print("  ✓ How parallelization with OpenMP works")
    print("  ✓ Performance optimization techniques and tradeoffs")
    print("  ✓ How to benchmark and compare implementations")
    print("  ✓ How to write fast numerical code in Python")
    
    print_section("NEXT STEPS")
    
    print("Ready to begin? Here's what to do:\n")
    print("  1. Open: CYTHON_EXERCISE_GUIDE.md")
    print("  2. Follow the setup instructions")
    print("  3. Edit: matrix_multiply.pyx (fill in TODOs)")
    print("  4. Compile: python3 setup.py build_ext --inplace")
    print("  5. Test: python3 test_cython_matrix.py")
    print("\nGood luck! This is a great way to learn high-performance Python!\n")

if __name__ == "__main__":
    main()
