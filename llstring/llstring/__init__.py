"""
The :mod:'llstring' module implements classifiers 
based on basic string matching algorithms (Levenshtein Distance,
Jaro-Winkler Similarity and Soft TF-IDF Similarity) as well
as provides a variety of basic string processing/normalization
tools. 
"""
from pkgutil import extend_path as __extend_path
__path__ = __extend_path(__path__, __name__)
__all__ = ['matching','training','utilities']
import matching, training, utilities

