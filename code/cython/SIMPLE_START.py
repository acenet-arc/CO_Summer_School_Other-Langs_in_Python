"""
SIMPLE CYTHON EXERCISE - Quick Start
====================================


Files needed:
  • simple_matrix.pyx       ← Edit this file (4 TODOs)
  • setup_simple.py         ← Don't edit
  • test_simple.py          ← Don't edit
  • README_SIMPLE.md        ← Read this first!
"""

import os

def main():
    print("\n" + "="*70)
    print(" SIMPLE CYTHON MATRIX MULTIPLICATION EXERCISE")
    print("="*70)
    
    print("\n📖 READ FIRST:")
    print("   cat README_SIMPLE.md")
    
    print("\n📝 EXERCISE FILE (edit this):")
    print("   simple_matrix.pyx - has 4 TODO items")
    
    print("\n4 SIMPLE TODO ITEMS:")
    print("   TODO 1: Get dimensions (3 lines)")
    print("   TODO 2: Create result array (1 line)")
    print("   TODO 3: Declare loop variables (1 line)")
    print("   TODO 4: Write multiplication loops (4 lines)")
    print("   Total: ~9 lines of code to add")
    
    print("\n⚙️  COMPILE:")
    print("   python3 setup_simple.py build_ext --inplace")
    
    print("\n🧪 TEST:")
    print("   python3 test_simple.py")
    
    print("\n📊 EXPECTED RESULT:")
    print("   Pure Python:  ~0.08 sec")
    print("   Cython:       ~0.0001 sec")
    print("   Speedup:      ~800x faster! ⚡")
    
    print("\n" + "="*70)
    print("That's it! Very simple but very powerful.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
