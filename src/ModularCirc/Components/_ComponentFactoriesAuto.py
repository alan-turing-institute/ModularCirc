import sys
import warnings

USING_OPTIMIZED = False
try:    
    from ._ComponentFactoriesOptimized import ComponentFunctionFactory, ElastanceFactory
    USING_OPTIMIZED = True
except:
    from ._ComponentFactories import ComponentFunctionFactory, ElastanceFactory