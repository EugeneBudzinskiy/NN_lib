import numpy  # Access to unwrapped numpy package

from .array_creation import empty
from .array_creation import empty_like
from .array_creation import eye
from .array_creation import identity
from .array_creation import ones
from .array_creation import ones_like
from .array_creation import zeros
from .array_creation import zeros_like
from .array_creation import full
from .array_creation import full_like
from .array_creation import array
from .array_creation import asarray
from .array_creation import copy
from .array_creation import arange
from .array_creation import linspace
from .array_creation import logspace
from .array_creation import geomspace
from .array_creation import meshgrid
from .array_creation import diag
from .array_creation import diagflat
from .array_creation import tri
from .array_creation import triu
from .array_creation import tril
from .array_creation import vander


ndarray = numpy.ndarray  # Variable for typing numpy arrays
