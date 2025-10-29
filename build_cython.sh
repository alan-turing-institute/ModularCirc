#!/bin/bash
# Quick rebuild script for Cython extension during development
#
# NOTE: This is OPTIONAL. The extension builds automatically with 'pip install -e .'
# Use this script only for quick rebuilds when iterating on Cython code.

set -e

echo "Building Cython extension..."
python setup.py build_ext --inplace

if [ -f src/ModularCirc/HelperRoutines/HelperRoutinesCython*.so ] || [ -f src/ModularCirc/HelperRoutines/HelperRoutinesCython*.pyd ]; then
    echo "✓ Build successful!"
    ls -lh src/ModularCirc/HelperRoutines/HelperRoutinesCython*.{so,pyd} 2>/dev/null || true
else
    echo "⚠️  Build failed - no extension found"
    exit 1
fi
