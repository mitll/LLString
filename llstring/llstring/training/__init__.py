"""
The :mod:'llstring.training' module implements a trainer 
which builds an IDF from raw text input (either from file or list)
"""
from .idf_trainer import IDFTrainer
__all__ = ['IDFTrainer']

