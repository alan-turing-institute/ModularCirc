#!/usr/bin/env python3
"""
Test script to check which HelperRoutines functions can be Numba-compiled
and identify potential compilation issues.
"""

import sys
import numpy as np
import numba as nb
from numba import types
import traceback

# Add the src path to import our module
sys.path.insert(0, '/home/mb5613/Git/ModularCirc/src')

from ModularCirc import HelperRoutines as hr

def test_numba_compilation():
    """Test Numba compilation for HelperRoutines functions."""
    
    print("🧪 Testing Numba compilation for HelperRoutines functions...")
    print("=" * 60)
    
    # Functions to test (name, test_args, expected_result_type)
    test_functions = [
        ('resistor_model_flow', (0.0, 2.0, 1.0, 0.5), 'float'),
        ('resistor_upstream_pressure', (0.0, 1.0, 1.0, 0.5), 'float'), 
        ('resistor_model_dp', (1.0, 0.5), 'float'),
        ('grounded_capacitor_model_pressure', (0.0, 2.0, 1.0, 0.1), 'float'),
        ('grounded_capacitor_model_volume', (0.0, 10.0, 1.0, 0.1), 'float'),
        ('softplus', (1.0, 0.2), 'float'),
        ('relu_max', (1.0,), 'float'),
        ('simple_bernoulli_diode_flow', (0.0, 2.0, 1.0, 1.0, 0.0), 'float'),
        ('leaky_diode_flow', (2.0, 1.0, 0.5, 0.1), 'float'),
    ]
    
    compilation_results = {}
    
    for func_name, test_args, expected_type in test_functions:
        print(f"\n🔧 Testing {func_name}...")
        
        try:
            # Get the function
            func = getattr(hr, func_name)
            
            # Test normal execution first
            try:
                result = func(*test_args)
                print(f"   ✅ Normal execution: {result} ({type(result).__name__})")
            except Exception as e:
                print(f"   ❌ Normal execution failed: {e}")
                compilation_results[func_name] = f"Normal execution failed: {e}"
                continue
            
            # Test Numba compilation
            try:
                # Try to compile with njit
                compiled_func = nb.njit(cache=True)(func)
                compiled_result = compiled_func(*test_args)
                print(f"   ✅ Numba compilation: {compiled_result} ({type(compiled_result).__name__})")
                
                # Check if results match
                if abs(result - compiled_result) < 1e-10:
                    print(f"   ✅ Results match!")
                    compilation_results[func_name] = "SUCCESS"
                else:
                    print(f"   ⚠️  Results differ: {result} vs {compiled_result}")
                    compilation_results[func_name] = "Results differ"
                    
            except Exception as e:
                print(f"   ❌ Numba compilation failed: {e}")
                compilation_results[func_name] = f"Compilation failed: {e}"
                
        except AttributeError:
            print(f"   ❌ Function {func_name} not found")
            compilation_results[func_name] = "Function not found"
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            compilation_results[func_name] = f"Unexpected error: {e}"
    
    print("\n" + "=" * 60)
    print("📊 COMPILATION SUMMARY:")
    print("=" * 60)
    
    successful = []
    failed = []
    
    for func_name, result in compilation_results.items():
        if result == "SUCCESS":
            successful.append(func_name)
            print(f"✅ {func_name}")
        else:
            failed.append(func_name)
            print(f"❌ {func_name}: {result}")
    
    print(f"\n📈 Summary: {len(successful)} successful, {len(failed)} failed")
    
    if successful:
        print(f"\n🎯 Functions ready for Numba optimization:")
        for func in successful:
            print(f"   - {func}")
    
    if failed:
        print(f"\n🔧 Functions needing attention:")
        for func in failed:
            print(f"   - {func}: {compilation_results[func]}")
    
    return successful, failed

if __name__ == "__main__":
    test_numba_compilation()