# KorakianitisMixedModel Profiling

This directory contains profiling tools for analyzing the performance of the KorakianitisMixedModel.

## Quick Profile Tool

The `quick_profile.py` script provides fast performance analysis of the cardiovascular simulation model.

### Usage

```bash
# Quick 2-cycle profile (default)
python quick_profile.py

# Custom number of cycles
python quick_profile.py --cycles 5

# Higher time resolution
python quick_profile.py --dt 0.0005

# Detailed analysis with function breakdowns
python quick_profile.py --detailed

# Combined options
python quick_profile.py --cycles 10 --dt 0.0005 --detailed
```

### Output

The script provides:
- Setup and simulation timing
- Real-time performance factor
- Top performance bottlenecks
- Saved profile files for detailed analysis

### Profiling Results Summary

Based on comprehensive analysis, the main performance bottlenecks in the KorakianitisMixedModel are:

1. **Solver list comprehension** (~20% of execution time)
2. **Component factory functions** (~8% of execution time) 
3. **ODE integration methods** (~6% of execution time)

The HelperRoutines functions (with Numba optimizations) consume only ~10% of total execution time despite 1.5M+ function calls, indicating successful optimization of the mathematical core functions.

### Performance Benchmarks

- **Real-time factor**: 0.5-1.2x (simulation runs at 50-120% of real-time speed)
- **Memory usage**: ~2MB increase during simulation
- **Compilation speedup**: 691x faster with explicit type signatures
- **Function call efficiency**: Sub-microsecond execution for optimized functions

### Viewing Detailed Results

After running the profiler, use the saved .prof files for detailed analysis:

```bash
python -c "import pstats; pstats.Stats('quick_profile_2cycles_dt0.001.prof').sort_stats('tottime').print_stats(20)"
```