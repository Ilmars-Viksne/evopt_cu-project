from . import backends

# Shortcuts for easy access
try:
    from .backends import gpu
except ImportError:
    gpu = None

from .backends import cpu
