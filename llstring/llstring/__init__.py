from pkgutil import extend_path
print __path__
__path__ = extend_path(__path__, __name__)
print __path__
__all__ = ['matching','training','utilities']
import matching, training, utilities
