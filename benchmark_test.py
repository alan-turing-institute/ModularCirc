#!/usr/bin/env python3
"""
Simple performance benchmark for ModularCirc optimizations
"""
import time
import numpy as np
from ModularCirc.Models.KorakianitisMixedModel import KorakianitisMixedModel
from ModularCirc.Models.KorakianitisMixedModel_parameters import KorakianitisMixedModel_parameters
from ModularCirc.Solver import Solver

def benchmark_solver():
    """Run a simple benchmark of the solver"""
    print("Running ModularCirc Solver Benchmark...")
    
    # Set up the model
    np.random.seed(42)
    
    time_setup_dict = {
        'name': 'BenchmarkTest',
        'ncycles': 20,  # Reduced for faster testing
        'tcycle': 1.0,
        'dt': 1e-3,
        'export_min': 5
    }
    
    # Initialize parameter object
    parobj = KorakianitisMixedModel_parameters()
    
    # Initialize model
    model = KorakianitisMixedModel(time_setup_dict=time_setup_dict,
                                   parobj=parobj,
                                   suppress_printing=True)
    
    # Initialize solver
    solver = Solver(model)
    solver.setup(suppress_output=True)
    
    # Time the solve operation
    start_time = time.time()
    solver.solve()
    end_time = time.time()
    
    solve_time = end_time - start_time
    print(f"Solver completed in {solve_time:.3f} seconds")
    print(f"Converged: {solver.converged}")
    print(f"Number of converged cycles: {solver.Nconv}")
    
    return solve_time

if __name__ == "__main__":
    benchmark_time = benchmark_solver()
    print(f"\nBenchmark completed: {benchmark_time:.3f} seconds")