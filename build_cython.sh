#!/bin/bash
# Build script for Cythonizing ModularCirc HelperRoutines

set -e  # Exit on error

echo "================================================"
echo "Building Cython Extension for ModularCirc"
echo "================================================"

# Check if Cython is installed
if ! python -c "import Cython" 2>/dev/null; then
    echo "⚠️  Cython not found. Installing..."
    pip install cython
fi

# Check if NumPy is installed
if ! python -c "import numpy" 2>/dev/null; then
    echo "⚠️  NumPy not found. Installing..."
    pip install numpy
fi

echo ""
echo "Building Cython extension..."
python setup_cython.py build_ext --inplace

# Check if build succeeded
if [ -f src/ModularCirc/HelperRoutines/HelperRoutinesCython*.so ] || [ -f src/ModularCirc/HelperRoutines/HelperRoutinesCython*.pyd ]; then
    echo ""
    echo "✓ Build successful!"
    echo ""
    echo "Compiled extension:"
    ls -lh src/ModularCirc/HelperRoutines/HelperRoutinesCython*.{so,pyd} 2>/dev/null || true
    
    # Copy to site-packages if editable install exists
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [ -d "$SITE_PACKAGES/ModularCirc/HelperRoutines" ]; then
        echo ""
        echo "Copying extension to installed package..."
        cp src/ModularCirc/HelperRoutines/HelperRoutinesCython*.so "$SITE_PACKAGES/ModularCirc/HelperRoutines/" 2>/dev/null || true
        cp src/ModularCirc/HelperRoutines/HelperRoutinesCython*.pyd "$SITE_PACKAGES/ModularCirc/HelperRoutines/" 2>/dev/null || true
        echo "✓ Extension copied to: $SITE_PACKAGES/ModularCirc/HelperRoutines/"
    fi
    
    echo ""
    echo "To use the Cython version, your code will automatically"
    echo "detect and use it if available (no code changes needed)."
    echo ""
    echo "To verify it's being used, look for:"
    echo "  '✓ Using Cythonized HelperRoutines' message on import"
else
    echo ""
    echo "⚠️  Build completed but extension not found."
    echo "Check for compilation errors above."
    exit 1
fi

echo ""
echo "Optional: View optimization annotations"
echo "  HTML files with performance hints: src/ModularCirc/HelperRoutines/HelperRoutines.html"
