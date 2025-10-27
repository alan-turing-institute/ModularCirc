# Cythonizing ModularCirc HelperRoutines

This directory contains a Cythonized version of the `HelperRoutines` module for improved performance.

## Prerequisites

Install Cython if not already available:

```bash
pip install cython
```

## Building the Cython Extension

From the repository root, run:

```bash
python setup_cython.py build_ext --inplace
```

This will:
- Compile `src/ModularCirc/HelperRoutines.pyx` to C code
- Build the C extension as a shared library
- Place the compiled module in the source tree

## Using the Cython Module

### Option 1: Automatic (Recommended)

Add this at the top of `src/ModularCirc/HelperRoutines.py`:

```python
# Try to import Cython version, fall back to Numba if unavailable
try:
    from .HelperRoutinesCython import *
    print("Using Cythonized HelperRoutines")
except ImportError:
    # Fall back to current Numba implementation
    pass
```

### Option 2: Manual Import

In any module that uses HelperRoutines:

```python
try:
    from ModularCirc.HelperRoutinesCython import resistor_model_flow, chamber_volume_rate_change
    USE_CYTHON = True
except ImportError:
    from ModularCirc.HelperRoutines import resistor_model_flow, chamber_volume_rate_change
    USE_CYTHON = False
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
3. **Cython version**: Use Cython >= 0.29: `pip install --upgrade cython`

View detailed annotation (optimization opportunities):

```bash
# After building, check the generated HTML file
open src/ModularCirc/HelperRoutines.html
```

## Cleanup

To remove compiled artifacts:

```bash
# Remove compiled extensions
rm -f src/ModularCirc/HelperRoutinesCython*.so
rm -f src/ModularCirc/HelperRoutinesCython*.pyd
rm -f src/ModularCirc/HelperRoutinesCython.c

# Remove build artifacts
rm -rf build/
```

## Notes

- The `.pyx` file maintains API compatibility with the original `.py` file
- All Numba `@nb.njit` decorators are replaced with Cython equivalents
- Type annotations use Cython's static typing for maximum performance
- Functions marked `nogil` can run without the Python GIL, enabling true parallelism
