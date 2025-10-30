# ModularCirc Installation Options

ModularCirc supports two performance backends for the `HelperRoutines` module:

1. **Cython** (C-compiled, faster startup, recommended for production)
2. **Numba** (JIT-compiled, slower startup, easier development)

## Installation Methods

### Option 1: Install with Cython (Recommended)

For best performance with faster startup times:

```bash
# Install with Cython dependencies
pip install -e .[performance]

# Build the Cython extension
python setup.py build_ext --inplace
```

### Option 2: Install with Numba only

For development or if you encounter Cython build issues:

```bash
# Set environment variable to skip Cython build
export MODULARCIRC_USE_CYTHON=0

# Install package
pip install -e .
```

Or install normally - if Cython is not available, it will automatically fall back to Numba.

## Runtime Configuration

You can control which implementation is used at runtime with environment variables:

### Force Numba implementation

Even if Cython is installed, you can force the use of Numba:

```bash
export MODULARCIRC_FORCE_NUMBA=1
python your_script.py
```

Or in Python:

```python
import os
os.environ['MODULARCIRC_FORCE_NUMBA'] = '1'

import ModularCirc  # Will use Numba
```

### Enable verbose output

To see which implementation is being used:

```bash
export MODULARCIRC_VERBOSE=1
python your_script.py
```

Or:

```python
import os
os.environ['MODULARCIRC_VERBOSE'] = '1'

import ModularCirc
# Will print: "✓ Using Cythonized HelperRoutines" or "⚠ Using Numba HelperRoutines"
```

### Check which implementation is active

In your Python code:

```python
from ModularCirc.HelperRoutines import USING_CYTHON

if USING_CYTHON:
    print("Using fast Cython implementation")
else:
    print("Using Numba implementation")
```

## Environment Variables Summary

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `MODULARCIRC_USE_CYTHON` | 0 or 1 | 1 | Controls whether to build Cython extension during installation |
| `MODULARCIRC_FORCE_NUMBA` | 0 or 1 | 0 | Forces use of Numba implementation at runtime |
| `MODULARCIRC_VERBOSE` | 0 or 1 | 0 | Enables verbose output about which implementation is used |

## Performance Comparison

- **Cython**: No JIT compilation overhead, faster startup (~2-3x faster first run)
- **Numba**: JIT compilation on first use, slightly slower startup but similar runtime performance

For production use or when running many small simulations, Cython is recommended.
For development or when C compiler is not available, Numba is a good fallback.

## Troubleshooting

### Cython build fails

If you encounter errors building the Cython extension:

```bash
# Disable Cython and use Numba
export MODULARCIRC_USE_CYTHON=0
pip install -e .
```

### Want to rebuild Cython extension

```bash
# Clean and rebuild
python setup.py clean --all
python setup.py build_ext --inplace
```

### Verify installation

```bash
python verify_installation.py
```

This will show which implementation is active and available.
