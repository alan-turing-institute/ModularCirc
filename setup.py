"""
Setup script for ModularCirc package.
This handles Cython extension building with graceful fallback.
Most configuration is in pyproject.toml.

Environment variables:
  MODULARCIRC_USE_CYTHON=0  - Disable Cython extension building
  MODULARCIRC_USE_CYTHON=1  - Enable Cython extension building (default if Cython available)
"""

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# Check environment variable for Cython preference
use_cython_env = os.environ.get('MODULARCIRC_USE_CYTHON', '1')
disable_cython = use_cython_env.lower() in ('0', 'false', 'no')

# Try to build Cython extension if available and not disabled
ext_modules = []
if disable_cython:
    print("⚠️  Cython extension building disabled via MODULARCIRC_USE_CYTHON")
    print("   Package will use Numba implementation only")
else:
    try:
        from Cython.Build import cythonize
        import numpy as np
        
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
                "language_level": 3,
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
    """Build extension that doesn't fail if compilation fails"""
    
    def run(self):
        try:
            build_ext.run(self)
        except Exception as e:
            print(f"⚠️  Cython extension build failed: {e}")
            print("   Package will use Numba fallback")
    
    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except Exception as e:
            print(f"⚠️  Failed to build {ext.name}: {e}")


# All other configuration is in pyproject.toml
setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtSafe},
)
