#!/usr/bin/env python3
"""
Quick Profiling Script for KorakianitisMixedModel

Usage:
    python quick_profile.py [--cycles N] [--dt TIMESTEP] [--detailed]
    
Examples:
    python quick_profile.py                    # Quick profile (2 cycles)
    python quick_profile.py --cycles 10       # Longer simulation
    python quick_profile.py --dt 0.0005       # Higher resolution
    python quick_profile.py --detailed        # Full detailed analysis
"""
import argparse
import sys
import time
import cProfile
import pstats
import os
# Add the src directory to the path (going up two levels from sandbox/profiling/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters
from ModularCirc.Solver import Solver

def run_quick_profile(ncycles=2, dt=0.001, detailed=False):
    """
    Run a quick profile of the KorakianitisMixedModel.
    
    Args:
        ncycles: Number of cardiac cycles
        dt: Time step size
        detailed: Whether to show detailed analysis
    """
    
    print(f"Quick Profile: {ncycles} cycles, dt={dt}")
    print("-" * 50)
    
    # Setup
    time_setup_dict = {
        'name': 'QuickProfile',
        'ncycles': ncycles,
        'tcycle': 1.0,
        'dt': dt,
        'export_min': 1
    }
    
    # Initialize
    start_setup = time.time()
    parobj = KorakianitisMixedModel_parameters()
    model = KorakianitisMixedModel(
        time_setup_dict=time_setup_dict,
        parobj=parobj,
        suppress_printing=True
    )
    solver = Solver(model=model)
    solver.setup(suppress_output=True, method='LSODA', step=1)
    setup_time = time.time() - start_setup
    
    print(f"Setup time: {setup_time:.2f}s")
    
    # Profile the simulation
    pr = cProfile.Profile()
    
    start_sim = time.time()
    pr.enable()
    solver.solve()
    pr.disable()
    sim_time = time.time() - start_sim
    
    print(f"Simulation time: {sim_time:.2f}s")
    print(f"Real-time factor: {(ncycles * 1.0) / sim_time:.1f}x")
    print(f"Converged: {solver.converged}")
    print()
    
    # Analyze results
    stats = pstats.Stats(pr)
    
    if detailed:
        print("=== Top 10 Functions by Total Time ===")
        stats.sort_stats('tottime')
        stats.print_stats(10)
        print()
        
        print("=== HelperRoutines Functions ===")
        stats.sort_stats('tottime')
        stats.print_stats('HelperRoutines', 10)
        print()
    else:
        print("=== Top 5 Bottlenecks ===")
        stats.sort_stats('tottime')
        stats.print_stats(5)
        print()
    
    # Save profile for detailed analysis
    profile_filename = f'quick_profile_{ncycles}cycles_dt{dt}.prof'
    stats.dump_stats(profile_filename)
    print(f"Profile saved to: {profile_filename}")
    print(f"View details with: python -c \"import pstats; pstats.Stats('{profile_filename}').sort_stats('tottime').print_stats(20)\"")

def main():
    parser = argparse.ArgumentParser(description='Quick profiling for KorakianitisMixedModel')
    parser.add_argument('--cycles', type=int, default=2, help='Number of cardiac cycles (default: 2)')
    parser.add_argument('--dt', type=float, default=0.001, help='Time step size (default: 0.001)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    try:
        run_quick_profile(ncycles=args.cycles, dt=args.dt, detailed=args.detailed)
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()