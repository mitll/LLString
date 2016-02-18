"""
The :mod:'llstring.sampling' sub-package implements
basic reservoir sampling: one-pass uniform sampling of 
a large dataset.
"""
from .reservoir_sampler import ReservoirSampler
__all__ = ['ReservoirSampler']
