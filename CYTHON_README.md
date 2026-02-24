# Cythonizing ModularCirc HelperRoutines

This directory contains a Cythonized version of the `HelperRoutines` module for improved performance.

## Quick Start (Automatic Build)

**The Cython extension is now built automatically during package installation!**

```bash
# Install with Cython support (requires cython and numpy)
pip install cython numpy
pip install -e .
```

The setup will automatically detect Cython and build the extension. If Cython is not available or the build fails, the package will fall back to the Numba implementation.

## Manual Build (Development)

For quick rebuilds during development without reinstalling the entire package:

```bash
bash build_cython.sh
```

Or manually:

```bash
python setup.py build_ext --inplace
```

## Prerequisites

- Python >= 3.10
- Cython >= 3.0
- NumPy >= 1.20
- C compiler (gcc, clang, or MSVC)

## Using the Cython Module

### Option 1: Automatic (Recommended)

Add this at the top of the script:
```python
# Try to import Cython version, fall back to Numba if unavailable
try:
    from ModularCirc.HelperRoutines import resistor_model_flow
    print("Using Cythonized HelperRoutines")
except ImportError:
    # Fall back to current Numba implementation
    pass
```

### Option 2: Manual Import (where function hasn't been cythonised)

In any module that uses HelperRoutines:

```python
try:
    from ModularCirc.HelperRoutines.HelperRoutinesCython import resistor_model_flow, chamber_volume_rate_change
except ImportError:
    from ModularCirc.HelperRoutines.HelperRoutines import resistor_model_flow, chamber_volume_rate_change
```

## Performance Benefits

Cython provides:
- **No JIT compilation overhead**: Functions are pre-compiled to machine code
- **Faster setup time**: No Numba compilation delays on first run
- **C-level performance**: Direct C math library calls (`sqrt`, `exp`, `log`, etc.)
- **Type safety**: Compile-time type checking
- **GIL release**: Many functions use `nogil` for better multi-threading potential

Expected improvements:
- **Setup time**: Near-instant (no JIT compilation)
- **Runtime**: Comparable to or faster than Numba (0-15% improvement typical)
- **Memory**: Slightly lower memory footprint

## Debugging

If compilation fails, check:

1. **Compiler availability**: Ensure you have a C compiler (gcc, clang, or MSVC)
2. **NumPy headers**: Make sure NumPy is installed: `pip install numpy`
3. **Cython version**: Use Cython >= 3.0: `pip install "cython>=3.0"`

View detailed annotation (optimization opportunities):

```bash
# After building, check the generated HTML file
open src/ModularCirc/HelperRoutines/HelperRoutinesCython.html
```

## Cleanup

To remove compiled artifacts:

```bash
# Remove compiled extensions
rm -f src/ModularCirc/HelperRoutines/HelperRoutinesCython*.so
rm -f src/ModularCirc/HelperRoutines/HelperRoutinesCython*.pyd
rm -f src/ModularCirc/HelperRoutines/HelperRoutines.c

# Remove build artifacts
rm -rf build/
```

## Notes

- The `.pyx` file maintains API compatibility with the original `.py` file
- All Numba `@nb.njit` decorators are replaced with Cython equivalents
- Type annotations use Cython's static typing for maximum performance
- Functions marked `nogil` can run without the Python GIL, enabling true parallelism
