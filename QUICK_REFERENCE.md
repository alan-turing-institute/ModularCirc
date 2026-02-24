# ModularCirc Quick Reference: Cython vs Numba

## Quick Check

```bash
# Check which implementation you're using
python -c "from ModularCirc.HelperRoutines import USING_CYTHON; print('Cython' if USING_CYTHON else 'Numba')"
```

## Installation

| Method | Command | Use Case |
|--------|---------|----------|
| **Cython** (automatic) | `pip install -e .` | Default — built automatically if C compiler present |
| **Numba** (fallback) | `export MODULARCIRC_USE_CYTHON=0`<br>`pip install -e .` | No C compiler available, or to skip Cython |
| **Dev rebuild** | `python build_and_check.py --build` | After modifying `.pyx` files |

## Environment Variables

| Variable | Effect | When to Use |
|----------|--------|-------------|
| `MODULARCIRC_USE_CYTHON=0` | Skip Cython build during install | No C compiler, build issues |
| `MODULARCIRC_FORCE_NUMBA=1` | Use Numba at runtime | Testing, debugging, comparison |
| `MODULARCIRC_VERBOSE=1` | Show which implementation loads | Debugging, verification |

## Common Scenarios

### I don't have a C compiler
```bash
export MODULARCIRC_USE_CYTHON=0
pip install -e .
```
→ Uses Numba only, works everywhere

### I want best performance
```bash
pip install -e .
python build_and_check.py  # confirm Cython is active
```
→ Cython is built automatically if a C compiler is present

### Cython build failed
```bash
export MODULARCIRC_USE_CYTHON=0
pip install -e .
```
→ Falls back to Numba automatically

### I want to compare implementations
```bash
# Run with Cython
python my_script.py

# Run with Numba
MODULARCIRC_FORCE_NUMBA=1 python my_script.py
```
→ Easy A/B testing

### After pulling new code
```bash
python build_and_check.py --build
```
→ Rebuild Cython extension

## Verification

```bash
# Full installation check
python build_and_check.py

# Run tests with both implementations
python -m unittest discover -s tests
MODULARCIRC_FORCE_NUMBA=1 python -m unittest discover -s tests
```

## Performance Notes

- **Cython**: ~2-3x faster first run (no JIT), slightly faster overall
- **Numba**: JIT compilation overhead on first use, then similar performance
- **Use Cython for**: Production, many short runs, startup-sensitive applications
- **Use Numba for**: Development, prototyping, systems without C compiler

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: No module named 'Cython'` | `pip install cython` or use `MODULARCIRC_USE_CYTHON=0` |
| Cython build fails | Use `MODULARCIRC_USE_CYTHON=0` to skip Cython |
| Want to force Numba | Set `MODULARCIRC_FORCE_NUMBA=1` |
| Unsure which is active | Check `USING_CYTHON` flag or use `MODULARCIRC_VERBOSE=1` |

## More Information

- Full installation guide: `INSTALLATION_OPTIONS.md`
- Cython details: `CYTHON_README.md`
- Verification script: `build_and_check.py`
