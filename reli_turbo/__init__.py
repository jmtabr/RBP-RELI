"""
reli_turbo -- GPU-accelerated RELI permutation testing for RBP enrichment analysis.

A CuPy/CUDA reimplementation of the RELI algorithm (Harley et al. 2018)
optimized for NVIDIA GPUs. Processes all targets in a single kernel launch
for 20-40x speedup over the C++ reference implementation.
"""

__version__ = "0.1.0"
__author__ = "RBP-RELI Project"
