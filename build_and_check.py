#!/usr/bin/env python
"""
Build Cython extensions and check ModularCirc installation.

Usage:
    python build_and_check.py           # verify only
    python build_and_check.py --build   # build Cython extension, then verify
"""

import subprocess
import sys


def build_cython():
    """Build Cython extension in-place."""
    print("Building Cython extension...")
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("Build failed.")
        sys.exit(1)

    import glob
    built = (
        glob.glob("src/ModularCirc/HelperRoutines/HelperRoutinesCython*.so")
        + glob.glob("src/ModularCirc/HelperRoutines/HelperRoutinesCython*.pyd")
    )
    if built:
        print(f"Build successful: {built[0]}")
    else:
        print("Build failed - no extension found.")
        sys.exit(1)


def check_installation():
    """Check ModularCirc installation status."""
    
    print("="*70)
    print("ModularCirc Installation Verification")
    print("="*70)
    
    # Check basic import
    try:
        import ModularCirc
        print("\n✓ ModularCirc package imported successfully")
        print(f"  Version: {ModularCirc.__version__ if hasattr(ModularCirc, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"\n✗ Failed to import ModularCirc: {e}")
        return False
    
    # Check core dependencies
    print("\n" + "-"*70)
    print("Core Dependencies:")
    print("-"*70)
    
    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'numba': 'Numba (JIT compiler)'
    }
    
    all_deps_ok = True
    for module, name in dependencies.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:<25} version {version}")
        except ImportError:
            print(f"  ✗ {name:<25} NOT FOUND")
            all_deps_ok = False
    
    # Check Cython optimization
    print("\n" + "-"*70)
    print("Performance Optimizations:")
    print("-"*70)
    
    cython_available = False
    try:
        import ModularCirc.HelperRoutines.HelperRoutinesCython
        print("  ✓ Cython extensions available")
        cython_available = True
    except ImportError:
        print("  ⚠ Cython extensions not found")
        print("    → Using Numba JIT (slower setup, still functional)")
        print("    → To enable: run './build_cython.sh' or 'python setup_cython.py build_ext --inplace'")
    
    # Check if Cython is installed
    try:
        import Cython
        print(f"\n  ✓ Cython compiler available (version {Cython.__version__})")
        if not cython_available:
            print("    → Build extensions with: ./build_cython.sh")
    except ImportError:
        if not cython_available:
            print("\n  ⚠ Cython compiler not installed")
            print("    → Install with: pip install cython")
            print("    → Or: pip install '.[performance]'")
    
    # Test basic functionality
    print("\n" + "-"*70)
    print("Functionality Test:")
    print("-"*70)
    
    try:
        from ModularCirc.Models.NaghaviModel import NaghaviModelParameters
        from ModularCirc.Solver import Solver
        print("  ✓ Core modules import successfully")
        print("  ✓ Ready to run simulations")
    except Exception as e:
        print(f"  ✗ Error importing core modules: {e}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    
    if all_deps_ok:
        print("  ✓ All core dependencies installed")
    else:
        print("  ⚠ Some dependencies missing")
    
    if cython_available:
        print("  ✓ Performance optimizations ENABLED (Cython)")
    else:
        print("  ⚠ Performance optimizations NOT enabled")
        print("  → ModularCirc will work but with slower setup time")
        print("  → Recommendation: Build Cython extensions for best performance")
    
    print("\n" + "="*70)
    
    return True

if __name__ == '__main__':
    if '--build' in sys.argv:
        build_cython()
    success = check_installation()
    sys.exit(0 if success else 1)
