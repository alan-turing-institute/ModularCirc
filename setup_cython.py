"""
Setup script for building Cython extensions for ModularCirc.
This compiles the HelperRoutines.pyx file into a C extension for better performance.

Usage:
    python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ModularCirc.HelperRoutines.HelperRoutinesCython",
        ["src/ModularCirc/HelperRoutines/HelperRoutines.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-ffast-math"],  # Removed -march=native for CI compatibility
        extra_link_args=["-O3"],
    )
]

setup(
    name="ModularCirc-Cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
        annotate=True,  # Generate HTML annotation files for optimization analysis
    ),
    zip_safe=False,
)
