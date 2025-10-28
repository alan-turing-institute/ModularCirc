"""
Setup script for ModularCirc package.
This handles both regular installation and optional Cython extension building.
"""

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os

# Try to build Cython extension if available
ext_modules = []
try:
    from Cython.Build import cythonize
    import numpy as np
    from setuptools import Extension
    
    extensions = [
        Extension(
            "ModularCirc.HelperRoutines.HelperRoutinesCython",
            ["src/ModularCirc/HelperRoutines/HelperRoutines.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=["-O3", "-ffast-math"],
            extra_link_args=["-O3"],
        )
    ]
    
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
        annotate=True,
    )
    print("✓ Cython extension will be built")
except ImportError:
    print("⚠️  Cython not available - skipping extension build (will use Numba fallback)")
    ext_modules = []


class BuildExtSafe(build_ext):
    """Safe build extension that doesn't fail if compilation fails"""
    
    def run(self):
        try:
            build_ext.run(self)
        except Exception as e:
            print(f"⚠️  Cython extension build failed: {e}")
            print("   Falling back to Numba implementation")
    
    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except Exception as e:
            print(f"⚠️  Failed to build {ext.name}: {e}")


setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtSafe},
)
