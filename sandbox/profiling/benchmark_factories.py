#!/usr/bin/env python3
"""
Benchmark script to test the performance improvements in ComponentFactories.py
"""

import sys
import time
import numpy as np

# Add the src path to import our module
sys.path.insert(0, '/home/mb5613/Git/ModularCirc/src')

from ModularCirc.Components._ComponentFactories import ComponentFunctionFactory, ElastanceFactory

def benchmark_factory_function(factory_method, factory_args, call_args, iterations=10000, name="Factory Function"):
    """Benchmark a factory function and its generated function calls."""
    
    print(f"\n🏃 Benchmarking {name}...")
    
    # Benchmark factory function creation
    start_time = time.perf_counter()
    for _ in range(100):  # Create functions 100 times
        func = factory_method(*factory_args)
    end_time = time.perf_counter()
    
    factory_time = (end_time - start_time) * 10  # Convert to ms per 100 calls
    print(f"   🏭 Factory creation: {factory_time:.3f} ms per 100 functions")
    
    # Create function once for actual benchmarking
    func = factory_method(*factory_args)
    
    # Warm up
    for _ in range(10):
        result = func(*call_args)
    
    # Benchmark function calls
    start_time = time.perf_counter()
    for _ in range(iterations):
        result = func(*call_args)
    end_time = time.perf_counter()
    
    call_time = (end_time - start_time) * 1000  # Convert to milliseconds
    per_call = call_time / iterations * 1000  # Convert to microseconds per call
    
    print(f"   ⏱️  {iterations:,} function calls in {call_time:.2f} ms")
    print(f"   📊 {per_call:.3f} μs per call")
    print(f"   ✅ Result: {result}")
    
    return factory_time, per_call

def benchmark_elastance_functions():
    """Benchmark elastance factory functions."""
    
    print("🧠 Testing Elastance Factory Functions")
    print("-" * 50)
    
    # Mock activation function
    def mock_af(t):
        return 0.5 * (1 + np.sin(2 * np.pi * t))
    
    # Test constant elastance
    factory_time, call_time = benchmark_factory_function(
        ElastanceFactory.gen_constant_elastance,
        (1.5, 0.3, mock_af, 10.0),  # E_act, E_pas, af, v_ref
        (0.5,),  # t
        name="gen_constant_elastance"
    )
    
    # Test elastance derivative (with optimized pre-computed constant)
    comp_E = ElastanceFactory.gen_constant_elastance(1.5, 0.3, mock_af, 10.0)
    factory_time, call_time = benchmark_factory_function(
        ElastanceFactory.gen_constant_elastance_derivative,
        (comp_E,),  # comp_E function
        (0.5,),  # t
        name="gen_constant_elastance_derivative"
    )

def main():
    """Run benchmarks for optimized factory functions."""
    
    print("🚀 ComponentFactories Performance Benchmark")
    print("=" * 60)
    print("Testing performance of optimized factory functions...")
    
    # Test basic factory functions with functools.partial optimization
    test_cases = [
        (ComponentFunctionFactory.gen_resistor_flow, (1.0,), (0.0, np.array([5.0, 2.0])), "gen_resistor_flow"),
        (ComponentFunctionFactory.gen_capacitor_dpdt, (0.2,), (0.0, np.array([2.0, 1.0])), "gen_capacitor_dpdt"),
        (ComponentFunctionFactory.gen_capacitor_pressure, (10.0, 0.2), (0.0, np.array([15.0])), "gen_capacitor_pressure"),
        (ComponentFunctionFactory.gen_simple_bernoulli_flow, (2.0, 0.1), (0.0, np.array([4.0, 2.0])), "gen_simple_bernoulli_flow"),
        # Note: gen_time_shifter creates a function that calls time_shift(t, delay, T)
        # The signature doesn't match partial() easily, so we skip this one
    ]
    
    total_factory_time = 0
    total_call_time = 0
    results = []
    
    for factory_method, factory_args, call_args, name in test_cases:
        try:
            factory_time, call_time = benchmark_factory_function(
                factory_method, factory_args, call_args, 10000, name
            )
            total_factory_time += factory_time
            total_call_time += call_time
            results.append((name, factory_time, call_time))
        except Exception as e:
            print(f"   ❌ Benchmark failed: {e}")
    
    # Test elastance functions
    benchmark_elastance_functions()
    
    print("\n" + "=" * 60)
    print("📈 FACTORY BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Functions tested: {len(results)}")
    print(f"Average factory creation time: {total_factory_time/len(results):.3f} ms per 100 functions")
    print(f"Average function call time: {total_call_time/len(results):.3f} μs per call")
    
    print(f"\n🏆 Factory Creation Performance (fastest to slowest):")
    results_sorted = sorted(results, key=lambda x: x[1])
    for name, factory_time, call_time in results_sorted:
        print(f"    {name:<35} {factory_time:.3f} ms per 100 functions")
    
    print(f"\n⚡ Function Call Performance (fastest to slowest):")
    results_sorted = sorted(results, key=lambda x: x[2])
    for name, factory_time, call_time in results_sorted:
        print(f"    {name:<35} {call_time:.3f} μs per call")
    
    print(f"\n💡 Optimization Notes:")
    print(f"   - Factory functions now use functools.partial for better performance")
    print(f"   - Numerical derivatives pre-compute constants")
    print(f"   - Elastance functions optimize arithmetic operations")
    print(f"   - All optimizations maintain backward compatibility")

if __name__ == "__main__":
    main()