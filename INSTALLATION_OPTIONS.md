# ModularCirc Installation Options

ModularCirc supports two performance backends for the `HelperRoutines` module:

1. **Cython** (C-compiled, faster startup, recommended for production)
2. **Numba** (JIT-compiled, slower startup, easier development)

## Installation Methods

### Option 1: Standard install (Cython built automatically)

Cython extensions are **built automatically during installation** if a C compiler is available. No extra steps required:

```bash
pip install -e .
```

To confirm which backend is active after installing:

```bash
python build_and_check.py
```

### Option 2: Skip Cython (Numba only)

If you don't have a C compiler or want to skip the Cython build:

```bash
export MODULARCIRC_USE_CYTHON=0
pip install -e .
```

### Developer rebuild (after modifying `.pyx` files)

To rebuild the Cython extension without reinstalling the whole package:

```bash
python build_and_check.py --build
```

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

### Want to rebuild Cython extension (developer)

```bash
python build_and_check.py --build
```

### Verify installation

```bash
python build_and_check.py
```

This will show which implementation is active and available.
